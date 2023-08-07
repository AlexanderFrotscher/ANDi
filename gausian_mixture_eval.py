__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from diffusion_cfg import *
from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
    #args.dataset_path = "./data/BraTS20/BraTS20_Training"
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/scans_try.csv"
    #args.path_to_csv = "./data/BraTS20/survival_info_02.csv"
    args.batch_size = 1
    args.image_size = 64

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet_conditional().to(device)
    ckpt = torch.load(
        "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_5/128_ema_ckpt.pt"
    )
    #ckpt = torch.load("./models/trained_models/final_no_flip/160_ema_ckpt.pt")
    # ckpt = torch.load("./models/trained_models/over_trained/248_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=64, device=device)
    dataloader = Brats_Volume(args)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    my_resize = transforms.Resize(128, antialias=True)
    my_percentiles = [1,2,3,4,5,6,7,8,9,10]
    dice_scores_mask = {i: [] for i in my_percentiles}
    for i, (image, label) in enumerate(pbar):
        image = (image * 2) - 1
        num_steps = 1000
        skip = 50
        my_pred = torch.zeros_like(image)
        my_volume = (
            torch.zeros(
                (
                    image.shape[0],
                    int((num_steps)/skip),
                    image.shape[1],
                    image.shape[2],
                    image.shape[3],
                    image.shape[4],
                )
            )
            .to(device)
        )
        my_masks = (
            torch.zeros(
                (
                    image.shape[0],
                    len(my_percentiles),
                    128,
                    128,
                    image.shape[4]
                )
            )
            .to(device)
            .type(torch.bool)
        )
        tmp_volume = (torch.zeros((image.shape[0],len(my_percentiles),image.shape[1],image.shape[2],image.shape[3],image.shape[4])).to(device))
        for j in range(image.shape[4]):
            # xts, zs = diffusion.dpm_inversion(model, image[:, :, :, :, j], timestemp=num_steps)
            # xts, zs = diffusion.dpm_encoder(model,image[:,:,:,:,j], timestemp=num_steps)
            #xts, zs = diffusion.my_inversion_pred(model,image[:,:,:,:,j], timestemp=num_steps)
            xts, zs = diffusion.skip_inversion(model,image[:,:,:,:,j], timestemp=num_steps,skip=skip)
            #zs = diffusion.skip_inversion_dep(model,image[:,:,:,:,j], timestemp=num_steps, skip=skip)
            my_volume[:,:,:,:,:,j] = zs
        for b in range(image.shape[0]):
            #my_mask = torch.zeros_like(image[b,0,:,:,:])
            #my_mask[image[b,0,:,:,:] != -1] = 1
            #my_mask = my_mask.type(torch.bool)
            for c in range(image.shape[1]):
                my_zs = my_volume[b,:,c,:,:,:]
                num_points = my_zs.shape[0]
                #masked_values = my_mask.unsqueeze(0).repeat(num_points,1,1,1).type(torch.bool)
                #my_values = my_zs[masked_values]
                #my_shape = my_values.shape[0] / num_points
                #my_values = torch.reshape(my_values,(num_points,int(my_shape))).T
                my_values = torch.flatten(my_zs,start_dim=1).T
                my_values = my_values.cpu().numpy()
                clf = GaussianMixture(n_components=1, covariance_type="full")
                clf.fit(my_values)
                densities = clf.score_samples(my_values)
                #my_pred[b,c,:,:,:][~my_mask] = float('inf')
                #my_pred[b,c,:,:,:][my_mask] = torch.Tensor(densities).to(device)
                my_pred[b,c,:,:,:] = torch.reshape(torch.Tensor(densities).to(device),(64,64,155))
                for k, percentile in enumerate(my_percentiles):
                    my_cut = np.percentile(densities, percentile)
                    tmp_volume[b,k,c,:,:,:] = torch.where(my_pred[b,c,:,:,:] < my_cut, 1.0, 0.0)
        tmp_volume = tmp_volume[:,:,0]
        for d in range(my_masks.shape[4]):
            tmp_mask = my_resize(tmp_volume[:,:,:,:,d])
            tmp_mask[tmp_mask >= 0.5] = 1
            tmp_mask[tmp_mask != 1] = 0
            my_masks[:,:,:,:,d] = tmp_mask.type(torch.bool)
        for j, key in enumerate(dice_scores_mask):
            segmentation = my_masks[:, j, :, :, :]
            dice_scores_mask[key].extend(
                [float(x) for x in dice(segmentation, label)])
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/mask_mixture.csv")
    #df_mask.to_csv("./results/BraTS21/mask_mixture.csv")
        

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
