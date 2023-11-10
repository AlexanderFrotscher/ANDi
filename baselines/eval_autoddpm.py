__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from accelerate import Accelerator, DistributedDataParallelKwargs
import lpips
import cv2
from sklearn.metrics import average_precision_score
from skimage.filters import threshold_yen
from scipy.ndimage import generate_binary_structure

from diffusion import *
from modules import *
from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    #args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_test.csv"
    args.dataset_path = "/mnt/qb/work/baumgartner/bkc035/shifts_data/patients"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/shifts_out.csv"
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
    dataloader = MRI_Volume(args, hist=False, shift=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.1, 0.001)]
    dice_scores_mask = {i: [] for i in threshold_test}
    dice_scores_mask_median = {i: [] for i in threshold_test}
    my_auprs = {i: [] for i in ["aupr no median", "aupr"]}

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
        all_masks = []
        for i, (image, label) in enumerate(pbar):
            image = (image * 2) - 1
            num_steps = 200
            num_inpainting = 50
            masking_threshold = 0.124
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
                tripple_health = torch.zeros((my_tensor.shape[0],my_tensor.shape[1],3,my_tensor.shape[2],my_tensor.shape[3])).to(device)
                tripple_pseudo = torch.zeros((my_tensor.shape[0],my_tensor.shape[1],3,my_tensor.shape[2],my_tensor.shape[3])).to(device)
                for k in range(3):
                    tripple_health[:,:,k] = my_tensor
                    tripple_pseudo[:,:,k] = pseudo_healthy
                my_lpips_mask = torch.zeros_like(my_tensor).to(device)
                for k in range(my_tensor.shape[1]):
                    lpips_mask = lpips_loss(my_lpips,tripple_health[:,k], tripple_pseudo[:,k], retPerLayer=False)
                    my_lpips_mask[:,k] = lpips_mask[:,0]

                # calculate quantile for each image and modality individually
                my_quantile = torch.zeros_like(residual).to(device)
                for k in range(my_quantile.shape[0]):
                    for n in range(my_quantile.shape[1]):
                        my_quantile[k,n] = torch.quantile(residual[k,n],0.99)
                
                residual = (residual/my_quantile).clamp(0,1)
                first_mask = my_lpips_mask * residual
                all_masks.append(first_mask.to('cpu'))
                my_mask = torch.where(first_mask > masking_threshold, torch.ones_like(first_mask), torch.zeros_like(first_mask))
                my_mask = dilate_masks(my_mask)


                image_masked = (1 - my_mask) * my_tensor

                x_inpaint = inpainting_loop(model,diffusion,device,image_masked,pseudo_healthy,my_mask,num_inpainting,resample_steps)
                residual = (my_tensor - x_inpaint.clamp(-1,1)).abs()

                anomaly_score = residual * first_mask
                anomaly_score = anomaly_score.to('cpu')
                prediction.append(anomaly_score)

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
            if not torch.count_nonzero(my_labels[0]):
                my_labels = my_labels[1:]
                my_volume = my_volume[1:]
            my_mask = torch.max(my_volume, dim=1)[0]
            mask_median = torch.clone(my_mask)
            mask_median = median_filter_3D(mask_median, kernelsize=3)
            my_labels = my_labels.contiguous()
            my_mask = norm_tensor(my_mask)
            mask_median = norm_tensor(mask_median)

            my_mask = my_mask.contiguous()
            mask_median = mask_median.contiguous()
            aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
            my_auprs["aupr no median"].extend([aupr])
            aupr = average_precision_score(my_labels.view(-1), mask_median.view(-1))
            my_auprs["aupr"].extend([aupr])
            for key in dice_scores_mask:
                segmentation = torch.where(my_mask > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                my_mask2 = torch.where(mask_median > key, 1.0, 0.0)
                my_mask2 = my_mask2.type(torch.bool)
                dice_scores_mask[key].extend([float(x) for x in dice(segmentation, my_labels)])
                dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
                dice_scores_mask_median[key].extend([float(x) for x in dice(my_mask2, my_labels)])
                dice_scores_mask_median[key] = np.mean(np.asarray(dice_scores_mask_median[key]))

            big_segmentation = torch.zeros_like(my_mask)
            struc = generate_binary_structure(3,1)


            for j, volume in enumerate(my_mask):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                big_segmentation[j] = segmentation
            big_segmentation = bin_dilation(big_segmentation, struc)
            dice_scores_mask['yen'] = []
            dice_scores_mask['yen'].extend([float(x) for x in dice(big_segmentation, my_labels)])
            dice_scores_mask['yen'] = np.mean(np.asarray(dice_scores_mask['yen']))


            for j, volume in enumerate(mask_median):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                big_segmentation[j] = segmentation
            big_segmentation = bin_dilation(big_segmentation, struc)
            dice_scores_mask_median['yen'] = []
            dice_scores_mask_median['yen'].extend([float(x) for x in dice(big_segmentation, my_labels)])
            dice_scores_mask_median['yen'] = np.mean(np.asarray(dice_scores_mask_median['yen']))

            dice_scores_mask['AUPRC'] = np.asarray(my_auprs["aupr no median"])
            dice_scores_mask_median['AUPRC'] = np.asarray(my_auprs["aupr"])
            df_mask = pd.DataFrame(dice_scores_mask,index=[0]).T
            df_mask2 = pd.DataFrame(dice_scores_mask_median, index=[0]).T
            df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/auto_ddpm.csv")
            df_mask2.to_csv("/mnt/qb/work/baumgartner/bkc035/auto_ddpm_median.csv")
            all_masks = torch.cat(all_masks,dim=0)
            mask_threshold = np.percentile(all_masks.cpu().detach().numpy(), 99).mean()
            print(mask_threshold)


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
            mask = masks[i].detach().cpu().numpy()
            for j in range(masks.shape[1]):
                my_mask = mask[j]
                if np.sum(my_mask) < 1:
                    dilated_masks[i,j] = torch.from_numpy(my_mask).to(masks.device)
                    continue
                dilated_mask = cv2.dilate(my_mask, kernel, iterations=1)
                dilated_mask = torch.from_numpy(dilated_mask).to(masks.device)
                dilated_masks[i,j] = dilated_mask

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
                    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                    alpha_hat_minus_one = diffusion.alpha_hat[t - 1][:, None, None, None]
                    var = (beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat))
                    noise = torch.randn_like(mu_t)
                    x_inpaint = mu_t + torch.sqrt(var) * noise

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
