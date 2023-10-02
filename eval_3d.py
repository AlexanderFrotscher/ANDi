__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from diffusion_cfg import *
from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/scans_val_small.csv"
    args.batch_size = 10
    args.image_size = 128

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet().to(device=device)
    ckpt = torch.load(
        "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/Brats128_full/232_ema_ckpt.pt"
     )
    
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=128, device=device)
    dataloader = Brats_Volume(args, hist=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.81, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}

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
            image, label = accelerator.gather_for_metrics((image,label))
            image = (image * 2) - 1
            num_steps = 500
            num_volumes = image.shape[0]
            num_slices = image.shape[1]

            #image = torch.permute(image,(0,4,1,2,3))
            image = image.view(-1,image.shape[2],image.shape[3],image.shape[4])
        
            #zs = diffusion.dpm_inversion(model, image[:, :, :, :, j], timestemp=num_steps)
            #zs = diffusion.dpm_encoder(model,image[:,:,:,:,j], timestemp=num_steps)
            #zs = diffusion.dpm_differences(model, image[:, :, :, :, j], timestemp=num_steps)
            zs = diffusion.differences_noise(model, image, timestemp=num_steps)

            my_mean = torch.mean(zs, dim=1)
            my_mean = my_mean.view(num_volumes,num_slices,my_mean.shape[1],my_mean.shape[2],my_mean.shape[3])
            my_mean = torch.permute(my_mean,(0,2,3,4,1))


            my_labels = torch.cat((my_labels, label.to("cpu")), dim=0)
            my_volume = torch.cat((my_volume, my_mean.to("cpu")), dim=0)

        if accelerator.is_main_process:
            my_mask = torch.max(my_volume,dim=1)[0]
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
            df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/mask_3D.csv")
            # df_mask.to_csv("./results/BraTS21/mask_one_3D.csv")


def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        for j in range(volume.shape[1]):
            volume[i, j, :, :] = medfilt2d(volume[i, j, :, :], kernel_size=kernelsize)
    return torch.Tensor(volume)


def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)



def norm_tensor(tensor):
    my_max = torch.max(tensor)
    my_min = torch.min(tensor)
    my_tensor = (tensor - my_min) / (my_max - my_min)
    return my_tensor


if __name__ == "__main__":
    main()
