__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import itertools

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
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/tumor_slices_test.csv"
    args.batch_size = 16
    args.image_size = 64

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet_conditional().to(device)
    ckpt = torch.load("/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_2/40_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=64, device=device)
    dataloader = Brats20(args, eval=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_diff = [x / 100 for x in range(1, 100)]
    threshold_test = [round(x, 3) for x in np.arange(0.3, 1.6, 0.01)]
    threshold_test_up = [round(x, 3) for x in np.arange(0.9, 1.6, 0.1)]
    threshold_test_low = [round(-x, 3) for x in np.arange(0.9, 1.6, 0.1)]
    my_thresholds = list(itertools.product(threshold_test_up,threshold_test_low))

    dice_scores_diff = {i: [] for i in threshold_diff}
    dice_scores_diff_2 = {i: [] for i in threshold_diff}
    dice_scores_diff_3 = {i: [] for i in threshold_diff}
    dice_scores_mask = {i: [] for i in threshold_test}
    dice_scores_mask_2 = {(a,b):[] for (a,b) in my_thresholds}
    my_resize = transforms.Resize(128, antialias=True)
    #my_transform = transforms.Lambda(lambda x: (x * 2) - 1)
    for i, (image, label) in enumerate(pbar):
        #image = my_transform(image).to(device)
        #label = label.to(device)
        label = label[:, 0, :, :].type(torch.uint8)
        num_steps = 1000
        xts, zs = diffusion.dpm_inversion(model, image, num_steps)
        my_images_one = diffusion.guide_restoration(
            model, xts[:, 0:250], zs[:, 0:250], cfg_scale=0, noise_scale=0.3
        )
        my_images_two = diffusion.guide_restoration(
            model, xts[:, 0:150], zs[:, 0:150], cfg_scale=0, noise_scale=0.4
        )
        my_images_three = diffusion.guide_restoration(
            model, xts[:, 0:200], zs[:, 0:200], cfg_scale=0, noise_scale=0.5
        )
        #plot_images(image,mode='L')
        #plot_images(my_images,mode='L')
        for key in dice_scores_diff:
            my_masks = create_difference(image, my_images_one, threshold=key)
            my_masks = my_resize(my_masks)
            dice_scores_diff[key].extend([float(x) for x in dice(my_masks, label)])
            my_masks = create_difference(image, my_images_two, threshold=key)
            my_masks = my_resize(my_masks)
            dice_scores_diff_2[key].extend([float(x) for x in dice(my_masks, label)])
            my_masks = create_difference(image, my_images_three, threshold=key)
            my_masks = my_resize(my_masks)
            dice_scores_diff_3[key].extend([float(x) for x in dice(my_masks, label)])
        for key in dice_scores_mask:
            mask = diffusion.create_mask(zs, num_steps, threshold=key)
            mask = my_resize(mask)
            dice_scores_mask[key].extend([float(x) for x in dice(mask, label)])
        for key in dice_scores_mask_2:
            mask = two_th_mask(zs,key)
            mask = my_resize(mask)
            dice_scores_mask_2[key].extend([float(x) for x in dice(mask, label)])
    for key in dice_scores_diff:
        dice_scores_diff[key] = np.mean(np.asarray(dice_scores_diff[key]))
        dice_scores_diff_2[key] = np.mean(np.asarray(dice_scores_diff_2[key]))
        dice_scores_diff_3[key] = np.mean(np.asarray(dice_scores_diff_3[key]))
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
    for key in dice_scores_mask_2:
        dice_scores_mask_2[key] = np.mean(np.asarray(dice_scores_mask_2[key]))
    df_diff = pd.DataFrame(dice_scores_diff, index=[0]).T
    df_diff.index.rename("threshold", inplace=True)
    df_diff_2 = pd.DataFrame(dice_scores_diff_2, index=[0]).T
    df_diff_2.index.rename("threshold", inplace=True)
    df_diff_3 = pd.DataFrame(dice_scores_diff_3, index=[0]).T
    df_diff_3.index.rename("threshold", inplace=True)
    df = pd.concat([df_diff,df_diff_2],axis=1)
    df = pd.concat([df,df_diff_3],axis=1)
    df.columns = ['0.3 - 250','0.4 - 150','0.5 - 200']
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask.index.rename("threshold", inplace=True)
    df_mask2 = pd.DataFrame(dice_scores_mask_2, index=[0]).T
    df.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/difference_score.csv")
    df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/mask_one.csv")
    df_mask2.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/mask_two.csv")
    """
    my_images = []
    my_images.append(np.array(Image.open('./data/BraTS20/my_test_0_A(2).jpg'))[:,:,0])
    for i in range(1,4):
        my_images.append(np.array(Image.open(f'./data/BraTS20/my_test_{i}.jpg')))
    img = torch.stack([torch.from_numpy(x) for x in my_images], dim=0).unsqueeze(dim=0)
    img = img.float()
    img = my_transforms(img[0].to(device))
    img = img[None,:,:,:]
    xts, zs = diffusion.dpm_inversion(model, img, 500)
    zs_mean = torch.mean(zs, dim=1)
    my_slices = []
    for i in range(zs_mean.shape[1]):
        my_slices.append(zs_mean[:,i,:,:][0].cpu())
    show_slices(my_slices)
    mask = diffusion.create_mask(zs,500)
    show_slices([mask[0].cpu()])
    """


def show_slices(slices):
    """Function to display row of image slices"""
    fig, axes = plt.subplots(1, len(slices))
    for i, (slice) in enumerate(slices):
        axes[i].imshow(slice, cmap="gray")
    plt.show()

def two_th_mask(zs,my_th):
    upper = my_th[0]
    lower = my_th[1]
    my_mean = torch.mean(zs,dim=1) * np.sqrt(1000)
    my_mean = torch.where((my_mean<lower)|(my_mean>upper),1.0,0.0)
    my_mean = torch.mean(my_mean,dim=1)
    my_mean[my_mean>0] = 1
    return my_mean



if __name__ == "__main__":
    main()
