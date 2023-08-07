__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from diffusion_cfg import *
from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
    #args.dataset_path = "./data/BraTS20/BraTS20_Training"
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/scans_val_small.csv"
    #args.path_to_csv = "./data/BraTS20/survival_info_02.csv"
    args.batch_size = 10
    args.image_size = 64

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet_conditional().to(device)
    ckpt = torch.load(
        "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_5/128_ema_ckpt.pt"
    )
    # ckpt = torch.load("./models/trained_models/final_no_flip/80_ema_ckpt.pt")
    # ckpt = torch.load("./models/trained_models/over_trained/248_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=64, device=device)
    dataloader = Brats_Volume(args, hist=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.5, 7, 0.1)]

    dice_scores_mask = {i: [] for i in threshold_test}
    dice_scores_mask_2 = {i: [] for i in threshold_test}
    for i, (image, label) in enumerate(pbar):
        image = (image * 2) - 1
        num_steps = 1000
        my_masks = (
            torch.zeros(
                (
                    image.shape[0],
                    len(threshold_test),
                    128,
                    128,
                    image.shape[4],
                )
            )
            .to(device)
            .type(torch.bool)
        )
        my_masks_2 = (
            torch.zeros(
                (
                    image.shape[0],
                    len(threshold_test),
                    128,
                    128,
                    image.shape[4],
                )
            )
            .to(device)
            .type(torch.bool)
        )
        for j in range(image.shape[4]):
            # xts, zs = diffusion.dpm_inversion(model, image[:, :, :, :, j], timestemp=num_steps)
            # xts, zs = diffusion.dpm_encoder(model,image[:,:,:,:,j], timestemp=num_steps)
            # xts, zs = diffusion.my_inversion_pred(model,image[:,:,:,:,j], timestemp=num_steps)
            #xts, zs = diffusion.skip_inversion(model,image[:,:,:,:,j], timestemp=num_steps,skip=25)
            #xts , zs = diffusion.skip_inversion_ind(model,image[:,:,:,:,j], timestemp=num_steps, skip=10)
            zs = diffusion.skip_inversion_dep(model, image[:,:,:,:,j], timestemp=num_steps, skip=50)

            for k, key in enumerate(dice_scores_mask):
                mask = create_mask(
                    zs, key, steps=num_steps, images=image[:, :, :, :, j]
                )
                mask_2 = create_mask_2(
                    zs, key, steps=num_steps, images=image[:, :, :, :, j]
                )
                my_masks[:, k, :, :, j] = mask
                my_masks_2[:, k, :, :, j] = mask_2
        for j, key in enumerate(dice_scores_mask):
            my_mask = my_masks[:, j, :, :, :]
            dice_scores_mask[key].extend(
                [float(x) for x in dice(my_mask, label)]
            )
            my_mask_2 = my_masks_2[:, j, :, :, :]
            dice_scores_mask_2[key].extend(
                [float(x) for x in dice(my_mask_2, label)]
            )
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
        dice_scores_mask_2[key] = np.mean(np.asarray(dice_scores_mask_2[key]))
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/mask_one_3D.csv")
    #df_mask.to_csv("./results/BraTS21/mask_one_3D.csv")
    df_mask_2 = pd.DataFrame(dice_scores_mask_2, index=[0]).T
    df_mask_2.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/mask_two_3D.csv")


def create_mask(zs, th, steps, images):
     my_mean = torch.mean(zs, dim=1) * np.sqrt(steps)
     my_mean[images[:,:,:,:] == -1] = 0
     my_resize = transforms.Resize(128, antialias=True)
     my_mask = my_resize(my_mean)
     my_mask = median_filter_2D(my_mask)
     my_mask = my_mask.to(device='cuda')
     my_mask[my_mask < th] = 0
     my_mask[my_mask != 0] = 1
     my_mask = my_mask[:,0]
     my_mask = my_mask.type(torch.bool)
     return my_mask


def create_mask_2(zs, th, steps, images):
    my_mean = torch.mean(zs, dim=1) * np.sqrt(steps)
    my_mean[images[:,:,:,:] == -1] = 0
    my_resize = transforms.Resize(128, antialias=True)
    my_mean = my_resize(my_mean)
    my_mean = median_filter_2D(my_mean)
    my_mean = my_mean.to(device='cuda')
    my_mean_1 = my_mean[:, 0]
    my_mean_2 = my_mean[:, 3]
    my_mean = (my_mean_1 + my_mean_2)*0.5
    my_mean[my_mean < th] = 0
    my_mean[my_mean != 0] = 1
    my_mean = my_mean.type(torch.bool)
    return my_mean

def create_mask_n(zs, th, steps, images):
     my_mean = torch.mean(zs, dim=1) * np.sqrt(steps)
     my_mean[images[:,:,:,:] == -1] = 0
     my_mean[my_mean < th] = 0
     my_mean[my_mean != 0] = 1
     my_mean = my_mean[:,0]
     my_resize = transforms.Resize(128, antialias=True)
     my_mask = my_resize(my_mean[None, :, :, :])
     my_mask = my_mask[0]
     my_mask[my_mask > 0.5] = 1
     my_mask = my_mask.type(torch.bool)
     return my_mask

def create_mask_2_n(zs, th, steps, images):
    my_mean = torch.mean(zs, dim=1) * np.sqrt(steps)
    my_mean[images[:,:,:,:] == -1] = 0
    my_mean_1 = my_mean[:, 0]
    my_mean_2 = my_mean[:, 3]
    my_mean = (my_mean_1 + my_mean_2)*0.5
    my_mean[my_mean < th] = 0
    my_mean[my_mean != 0] = 1
    my_resize = transforms.Resize(128, antialias=True)
    my_mean = my_resize(my_mean[None, :, :, :])
    my_mean = my_mean[0]
    my_mean[my_mean > 0.5] = 1
    my_mean = my_mean.type(torch.bool)
    return my_mean

def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        for j in range(volume.shape[1]):
            volume[i, j, :, :] = medfilt2d(volume[i, j, :, :], kernel_size=kernelsize)
    return torch.Tensor(volume)

def show_slices(slices):
    """Function to display row of image slices"""
    fig, axes = plt.subplots(1, len(slices))
    for i, (slice) in enumerate(slices):
        axes[i].imshow(slice, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
