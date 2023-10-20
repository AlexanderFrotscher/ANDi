__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

from accelerate import Accelerator, DistributedDataParallelKwargs
import lpips
import cv2
from sklearn.metrics import average_precision_score

from diffusion import *
from modules import *
from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_val_small.csv"
    args.batch_size = 1
    args.image_size = 128

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet().to(device=device)
    ckpt = torch.load(
        "/mnt/qb/work/baumgartner/bkc035/normative-diffusion/models/232_ema_ckpt.pt"
    )

    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=128, device=device)
    dataloader = Brats_Volume(args, hist=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.81, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}

    my_lpips = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(device)

    with torch.no_grad():
        my_volume = torch.zeros(
            (
                1,
                4,
                128,
                128,
                155,
            )
        ).to("cpu")
        my_labels = (
            torch.zeros(
                (
                    1,
                    128,
                    128,
                    155,
                )
            )
            .type(torch.bool)
            .to("cpu")
        )
        for i, (image, label) in enumerate(pbar):
            image = (image * 2) - 1
            num_steps = 200
            num_inpainting = 50
            masking_threshold = 0.13
            resample_steps = 5
            size_splits = 50
            num_volumes = image.shape[0]
            num_slices = image.shape[4]

            image = torch.permute(image, (0, 4, 1, 2, 3))
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            split = torch.split(image, size_splits)
            prediction = []
            for my_tensor in split:
                pseudo_healthy = diffusion.ano_ddpm(model, my_tensor, num_steps)
                pseudo_healthy.clamp(-1,1)
                residual = (my_tensor - pseudo_healthy).abs()

                #Steps for first mask
                lpips_mask = lpips_loss(my_lpips,my_tensor, pseudo_healthy, retPerLayer=False)
                my_quantile = torch.quantile(residual,0.95)
                residual = (residual/my_quantile).clamp(0,1)
                first_mask = lpips_mask * residual
                my_mask = torch.where(first_mask > masking_threshold, torch.ones_like(first_mask), torch.zeros_like(first_mask))
                my_mask = dilate_masks(my_mask)

                image_masked = (1 - my_mask) * my_tensor

                x_inpaint = inpainting_loop(model,diffusion,device,image_masked,pseudo_healthy,my_mask,num_inpainting,resample_steps)
                residual = (my_tensor - x_inpaint.clamp(-1,1)).abs()

                prediction = residual * first_mask
                
                prediction.append(prediction.to("cpu"))

            prediction = torch.cat(prediction, dim=0)
            prediction = prediction.view(
                num_volumes,
                num_slices,
                prediction.shape[1],
                prediction.shape[2],
                prediction.shape[3],
            )
            prediction = torch.permute(prediction, (0, 2, 3, 4, 1))
            prediction = prediction.to(device)
            prediction, label = accelerator.gather_for_metrics((prediction, label))
            my_labels = torch.cat((my_labels, label.to("cpu")), dim=0)
            my_volume = torch.cat((my_volume, prediction.to("cpu")), dim=0)

        if accelerator.is_main_process:
            my_mask = torch.max(my_volume, dim=1)[0]
            my_mask = median_filter_3D(my_mask)
            my_labels = my_labels.contiguous()
            my_mask = norm_tensor(my_mask)
            my_mask = my_mask.contiguous()
            aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
            for key in dice_scores_mask:
                segmentation = torch.where(my_mask > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                dice_scores_mask[key].extend([dice(segmentation, my_labels)])

            dice_scores_mask[f"AUPRC"] = aupr
            df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
            df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/mask_3D.csv")


def lpips_loss(l_pips_sq, anomaly_img, ph_img, retPerLayer=False):
        """
        :param anomaly_img: anomaly image
        :param ph_img: pseudo-healthy image
        :param retPerLayer: whether to return the loss per layer
        :return: LPIPS loss
        """
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)

        loss_lpips = l_pips_sq(anomaly_img, ph_img, normalize=True, retPerLayer=retPerLayer)
        if retPerLayer:
            loss_lpips = loss_lpips[1][0]
        return loss_lpips

def dilate_masks(masks):
        """
        :param masks: masks to dilate
        :return: dilated masks
        """
        kernel = np.ones((3, 3), np.uint8)

        dilated_masks = torch.zeros_like(masks)
        for i in range(masks.shape[0]):
            mask = masks[i][0].detach().cpu().numpy()
            if np.sum(mask) < 1:
                dilated_masks[i] = masks[i]
                continue
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask).to(masks.device).unsqueeze(dim=0)
            dilated_masks[i] = dilated_mask

        return dilated_masks

def inpainting_loop(model, diffusion, device, masked_images, pseudo_healthy, mask, num_inpainting, resample_steps):
    model.eval()
    t = (torch.ones(pseudo_healthy.shape[0]) * num_inpainting).long().to(device)
    x_inpaint, noise = diffusion.noise_images(pseudo_healthy,t)
    with torch.no_grad():
        for i in tqdm(reversed(range(1, num_inpainting)), position=0):
            for j in range(resample_steps):
                # get healthy part of image from t-1
                if i > 1:
                    t = (torch.ones(pseudo_healthy.shape[0]) * i-1).long().to(device)
                    x_health, noise = diffusion.noise_images(masked_images,t)
                else:
                    x_health = masked_images

                # perform the denoising step for the reconstruction for the previous anomaly regions
                if i > 1:
                    t = (torch.ones(pseudo_healthy.shape[0]) * i).long().to(device)
                    predicted_noise = model(x_inpaint,t)
                    mu_t = diffusion.ddpm_mu_t(x_inpaint,predicted_noise,t)
                    beta = diffusion.beta[t][:, None, None, None]
                    noise = torch.randn_like(mu_t)
                    x_inpaint = mu_t + torch.sqrt(beta) * noise

                # combine both predictions
                x_inpaint = torch.where(mask == 1, x_inpaint, x_health)

                # apply resampling to harmonize
                if i > 1 and j < (resample_steps-1):
                    t = (torch.ones(pseudo_healthy.shape[0]) * i-1).long().to(device)
                    beta = diffusion.beta[t][:, None, None, None]
                    alpha = diffusion.alpha[t][:, None, None, None]
                    noise = torch.randn_like(x_inpaint)
                    x_inpaint = alpha.sqrt() * x_inpaint + torch.sqrt(beta) * noise
    return x_inpaint


if __name__ == "__main__":
    main()
