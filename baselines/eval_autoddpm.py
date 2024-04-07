__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import cv2
import lpips
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_yen
from sklearn.metrics import average_precision_score

from diffusion import *
from modules import *
from utils import *


def main():
    with open("../conf/eval_auto.yml", "r") as file_object:
        conf = yaml.load(file_object, Loader=yaml.SafeLoader)
        torch.manual_seed(conf["seed"])

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        device = accelerator.device
        model = UNet().to(device=device)
        ckpt = torch.load(conf["model"])

        model.load_state_dict(ckpt)
        diffusion = Diffusion(
            noise_steps=conf["noise_steps"],
            img_size=conf["size"],
            beta_start=conf["beta_start"],
            beta_end=conf["beta_end"],
            device=device,
        )
        dataloader = MRI_Volume(
            conf,
            hist=False,
            shift=(True if "shifts" in conf["dataset_path"] else False),
        )

        model, dataloader = accelerator.prepare(model, dataloader)
        pbar = tqdm(dataloader)
        threshold_test = [
            round(x, 3)
            for x in np.arange(conf["thr_start"], conf["thr_end"], conf["thr_step"])
        ]

        dice_scores = {i: [] for i in threshold_test}
        dice_scores_mf = {i: [] for i in threshold_test}
        my_auprs = {i: [] for i in ["aupr no median", "aupr"]}
        my_lpips = lpips.LPIPS(
            pretrained=True,
            net="squeeze",
            use_dropout=True,
            eval_mode=True,
            spatial=True,
            lpips=True,
        ).to(device)
        with torch.no_grad():
            all_masks = []
            my_volume = []
            my_labels = []
            for i, (image, label) in enumerate(pbar):
                image = (image * 2) - 1
                num_volumes = image.shape[0]
                num_slices = image.shape[4]

                image = torch.permute(image, (0, 4, 1, 2, 3))
                image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
                split = torch.split(image, conf["size_splits"])
                prediction = []
                for my_tensor in split:
                    pseudo_healthy = diffusion.ano_ddpm(
                        model, my_tensor, conf["num_steps"]
                    )
                    pseudo_healthy.clamp(-1, 1)
                    res = (my_tensor - pseudo_healthy).abs()

                    # Steps for first mask
                    tripple_health = torch.zeros(
                        (
                            my_tensor.shape[0],
                            my_tensor.shape[1],
                            3,
                            my_tensor.shape[2],
                            my_tensor.shape[3],
                        )
                    ).to(device)
                    tripple_pseudo = torch.zeros(
                        (
                            my_tensor.shape[0],
                            my_tensor.shape[1],
                            3,
                            my_tensor.shape[2],
                            my_tensor.shape[3],
                        )
                    ).to(device)
                    for k in range(3):
                        tripple_health[:, :, k] = my_tensor
                        tripple_pseudo[:, :, k] = pseudo_healthy
                    my_lpips_mask = torch.zeros_like(my_tensor).to(device)
                    for k in range(my_tensor.shape[1]):
                        lpips_mask = lpips_loss(
                            my_lpips,
                            tripple_health[:, k],
                            tripple_pseudo[:, k],
                            retPerLayer=False,
                        )
                        my_lpips_mask[:, k] = lpips_mask[:, 0]

                    # calculate quantile for each image and modality individually
                    my_quantile = torch.zeros_like(res).to(device)
                    for k in range(my_quantile.shape[0]):
                        for n in range(my_quantile.shape[1]):
                            my_quantile[k, n] = torch.quantile(
                                res[k, n], conf["quantile"]
                            )

                    res = (res / my_quantile).clamp(0, 1)
                    first_mask = my_lpips_mask * res
                    all_masks.append(first_mask.to("cpu"))
                    bin_mask = torch.where(
                        first_mask > conf["masking_thr"],
                        torch.ones_like(first_mask),
                        torch.zeros_like(first_mask),
                    )
                    bin_mask = dilate_masks(bin_mask)

                    image_masked = (1 - bin_mask) * my_tensor

                    x_inpaint = inpainting_loop(
                        model,
                        diffusion,
                        device,
                        image_masked,
                        pseudo_healthy,
                        bin_mask,
                        conf["num_inpainting"],
                        conf["resample"],
                    )
                    res = (my_tensor - x_inpaint.clamp(-1, 1)).abs()

                    anomaly_score = res * first_mask
                    anomaly_score = anomaly_score.to("cpu")
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
                my_labels.append(label.type(torch.bool).to("cpu"))
                my_volume.append(prediction.to("cpu"))

            if accelerator.is_main_process:
                my_volume = torch.cat(my_volume, dim=0)
                my_labels = torch.cat(my_labels, dim=0)
                if conf["max"] == True:
                    anomaly_map = torch.max(my_volume, dim=1)[0]
                else:
                    anomaly_map = torch.mean(my_volume, dim=1)

                anomaly_map_mf = torch.clone(anomaly_map)
                anomaly_map_mf = median_filter_3D(
                    anomaly_map_mf, kernelsize=conf["kernel_size"]
                )
                my_labels = my_labels.contiguous()
                anomaly_map = norm_tensor(anomaly_map)
                anomaly_map_mf = norm_tensor(anomaly_map_mf)

                anomaly_map = anomaly_map.contiguous()
                anomaly_map_mf = anomaly_map_mf.contiguous()
                aupr = average_precision_score(my_labels.view(-1), anomaly_map.view(-1))
                my_auprs["aupr no median"].extend([aupr])
                aupr = average_precision_score(
                    my_labels.view(-1), anomaly_map_mf.view(-1)
                )
                my_auprs["aupr"].extend([aupr])
                for key in dice_scores:
                    segmentation = torch.where(anomaly_map > key, 1.0, 0.0)
                    segmentation = segmentation.type(torch.bool)
                    segmentation_mf = torch.where(anomaly_map_mf > key, 1.0, 0.0)
                    segmentation_mf = segmentation_mf.type(torch.bool)
                    dice_scores[key].extend(
                        [float(x) for x in dice(segmentation, my_labels)]
                    )
                    dice_scores[key] = np.mean(np.asarray(dice_scores[key]))
                    dice_scores_mf[key].extend(
                        [float(x) for x in dice(segmentation_mf, my_labels)]
                    )
                    dice_scores_mf[key] = np.mean(np.asarray(dice_scores_mf[key]))

                yen_segmentation = torch.zeros_like(anomaly_map)
                struc = generate_binary_structure(conf["rank"], conf["connectivity"])

                for j, volume in enumerate(anomaly_map):
                    thr = threshold_yen(volume.numpy())
                    segmentation = torch.where(volume > thr, 1.0, 0.0)
                    yen_segmentation[j] = segmentation
                yen_segmentation = bin_dilation(yen_segmentation, struc)
                dice_scores["yen"] = []
                dice_scores["yen"].extend(
                    [float(x) for x in dice(yen_segmentation, my_labels)]
                )
                dice_scores["yen"] = np.mean(np.asarray(dice_scores["yen"]))

                for j, volume in enumerate(anomaly_map_mf):
                    thr = threshold_yen(volume.numpy())
                    segmentation = torch.where(volume > thr, 1.0, 0.0)
                    yen_segmentation[j] = segmentation
                yen_segmentation = bin_dilation(yen_segmentation, struc)
                dice_scores_mf["yen"] = []
                dice_scores_mf["yen"].extend(
                    [float(x) for x in dice(yen_segmentation, my_labels)]
                )
                dice_scores_mf["yen"] = np.mean(np.asarray(dice_scores_mf["yen"]))

                dice_scores["AUPRC"] = my_auprs["aupr no median"][0]
                dice_scores_mf["AUPRC"] = my_auprs["aupr"][0]
                df = pd.DataFrame.from_dict(
                    dice_scores, orient="index", columns=["value"]
                )
                df.index.name = "thr"
                df_mf = pd.DataFrame.from_dict(
                    dice_scores_mf, orient="index", columns=["value"]
                )
                df_mf.index.name = "thr"
                df.to_csv(conf["output"])
                df_mf.to_csv(conf["output_mf"])
                all_masks = torch.cat(all_masks, dim=0)
                mask_threshold = np.percentile(
                    all_masks.cpu().detach().numpy(), 99
                ).mean()
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
                dilated_masks[i, j] = torch.from_numpy(my_mask).to(masks.device)
                continue
            dilated_mask = cv2.dilate(my_mask, kernel, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask).to(masks.device)
            dilated_masks[i, j] = dilated_mask

    return dilated_masks


def inpainting_loop(
    model,
    diffusion,
    device,
    masked_images,
    pseudo_healthy,
    mask,
    num_inpainting,
    resample_steps,
):
    model.eval()
    t = (torch.ones(pseudo_healthy.shape[0]) * num_inpainting).long().to(device)
    x_inpaint, noise = diffusion.noise_images(pseudo_healthy, t)
    with torch.no_grad():
        for i in tqdm(reversed(range(1, num_inpainting)), position=0):
            for j in range(resample_steps):
                # get healthy part of image from t-1
                if i > 1:
                    t = (torch.ones(pseudo_healthy.shape[0]) * i - 1).long().to(device)
                    x_health, noise = diffusion.noise_images(masked_images, t)
                else:
                    x_health = masked_images

                # perform the denoising step for the reconstruction for the previous anomaly regions
                if i > 1:
                    t = (torch.ones(pseudo_healthy.shape[0]) * i).long().to(device)
                    predicted_noise = model(x_inpaint, t)
                    mu_t = diffusion.ddpm_mu_t(x_inpaint, predicted_noise, t)
                    beta = diffusion.beta[t][:, None, None, None]
                    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                    alpha_hat_minus_one = diffusion.alpha_hat[t - 1][
                        :, None, None, None
                    ]
                    var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                    noise = torch.randn_like(mu_t)
                    x_inpaint = mu_t + torch.sqrt(var) * noise

                # combine both predictions
                x_inpaint = torch.where(mask == 1, x_inpaint, x_health)

                # apply resampling to harmonize
                if i > 1 and j < (resample_steps - 1):
                    t = (torch.ones(pseudo_healthy.shape[0]) * i - 1).long().to(device)
                    beta = diffusion.beta[t][:, None, None, None]
                    alpha = diffusion.alpha[t][:, None, None, None]
                    noise = torch.randn_like(x_inpaint)
                    x_inpaint = alpha.sqrt() * x_inpaint + torch.sqrt(beta) * noise
    return x_inpaint


if __name__ == "__main__":
    main()
