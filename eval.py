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
    #args.dataset_path = './data/BraTS20'
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/tumor_slices_val.csv"
    #args.path_to_csv = './data/BraTS20/tumor_slices_small.csv'
    args.batch_size = 16
    args.image_size = 64

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet_conditional().to(device)
    ckpt = torch.load("/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/BraTS21_6/80_ema_ckpt.pt")
    #ckpt = torch.load("./models/trained_models/80_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=64, device=device)
    dataloader = Brats20(args, eval=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    #threshold_diff = [x / 100 for x in range(1, 100)]
    threshold_test = [round(x,3) for x in np.arange(0.7,2.2,0.1)]
    #threshold_test_1 = [round(x, 3) for x in np.arange(1.0, 1.6, 0.1)]
    #threshold_test_2 = [round(-x, 3) for x in np.arange(2.2, 2.8, 0.1)]
    #threshold_test_3 = [round(x, 3) for x in np.arange(3.0, 3.5, 0.1)]
    #threshold_test_3_m = [round(-x, 3) for x in np.arange(2.7, 3.3, 0.1)]
    #threshold_test_4 = [round(x, 3) for x in np.arange(2.3, 2.8, 0.1)]
    #threshold_test_1 = [round(x, 3) for x in np.arange(0.9, 1.5, 0.1)]
    #threshold_test_2 = [round(-x, 3) for x in np.arange(1.4, 2.3, 0.1)]
    #threshold_test_3 = [round(x, 3) for x in np.arange(1.7, 2.3, 0.1)]
    #threshold_test_3_m = [round(-x, 3) for x in np.arange(1.0, 1.7, 0.1)]
    #threshold_test_4 = [round(x, 3) for x in np.arange(0.9, 1.5, 0.1)]
    #my_thresholds = list(itertools.product(threshold_test_1,threshold_test_2,threshold_test_3,threshold_test_3_m,threshold_test_4))

    #dice_scores_diff = {i: [] for i in threshold_diff}
    #dice_scores_diff_2 = {i: [] for i in threshold_diff}
    dice_scores_mask = {i: [] for i in threshold_test}
    #dice_scores_mask_3 = {(a,b,c,d,e):[] for (a,b,c,d,e) in my_thresholds}
    my_resize = transforms.Resize(128, antialias=True)
    #my_transform = transforms.Lambda(lambda x: (x * 2) - 1)
    scaling = torch.linspace(0.4,1.0,1000).to(device)
    my_scaling = diffusion.beta * scaling
    my_scaling = my_scaling.to(device)
    for i, (image, label) in enumerate(pbar):
        #image = my_transform(image).to(device)
        label = label.to(device)
        label = label[:, 0, :, :].type(torch.uint8)
        num_steps = 1000
        xts, zs = diffusion.dpm_inversion(model, image,timestemp=num_steps, scaling=my_scaling)
        #my_mean = torch.mean(zs,dim=1) * np.sqrt(1000)
        #my_mean_1 =  my_mean[:,0]
        #my_mean_2 =  my_mean[:,3]
        #my_mean = (my_mean_1 + my_mean_2) * 0.5
        #show_slices([my_mean[0].cpu(),my_mean[1].cpu(),my_mean[2].cpu()])
        #my_images_one = diffusion.guide_restoration(
        #    model, xts[:, 0:150], zs[:, 0:150], cfg_scale=0, noise_scale=0.45
        #)


        #for key in dice_scores_diff:
        #    my_masks = create_difference(image, my_images_one, threshold=key)
        #    my_masks = my_resize(my_masks)
        #    dice_scores_diff[key].extend([float(x) for x in dice(my_masks, label)])
            #my_masks = create_difference(image, my_images_two, threshold=key)
            #my_masks = my_resize(my_masks)
            #dice_scores_diff_2[key].extend([float(x) for x in dice(my_masks, label)])

        for key in dice_scores_mask:
            mask = create_mask_2(zs, key,steps=num_steps)
            mask = my_resize(mask)
            dice_scores_mask[key].extend([float(x) for x in dice(mask, label)])
        #for key in dice_scores_mask_3:
        #    mask = create_mask_3(zs,key,steps=num_steps)
        #    mask = my_resize(mask)
        #    dice_scores_mask_3[key].extend([float(x) for x in dice(mask, label)])
    #for key in dice_scores_diff:
    #    dice_scores_diff[key] = np.mean(np.asarray(dice_scores_diff[key]))
    #    dice_scores_diff_2[key] = np.mean(np.asarray(dice_scores_diff_2[key]))
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
    #for key in dice_scores_mask_3:
    #    dice_scores_mask_3[key] = np.mean(np.asarray(dice_scores_mask_3[key]))
    #df_diff = pd.DataFrame(dice_scores_diff, index=[0]).T
    #df_diff.index.rename("threshold", inplace=True)
    #df_diff_2 = pd.DataFrame(dice_scores_diff_2, index=[0]).T
    #df_diff_2.index.rename("threshold", inplace=True)
    #df = pd.concat([df_diff,df_diff_2],axis=1)
    #df.columns = ['0.3 - 150','0.05 - 1000']
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    #df_mask2 = pd.DataFrame(dice_scores_mask_3, index=[0]).T
    #df.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/difference_score.csv")
    df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/mask_one.csv")
    #df_mask2.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/mask_two.csv")
    #df_diff.to_csv("./results/BraTS21/difference_score.csv")
    #df_mask.to_csv("./results/BraTS21/mask_one.csv")
    #df_mask2.to_csv("./results/BraTS21/mask_two.csv")
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

def create_mask_2(zs,th,steps):
    my_mean = torch.mean(zs,dim=1) * np.sqrt(steps)
    my_mean_1 =  my_mean[:,0]
    my_mean_2 =  my_mean[:,3]
    my_mean = (my_mean_1 + my_mean_2) * 0.5
    my_mean[my_mean < th] = 0
    my_mean[my_mean != 0] = 1
    return my_mean.type(torch.uint8)

def create_mask_3(zs,my_th, steps):
    th_m1 = my_th[0]
    th_m2 = my_th[1]
    th_m3 = my_th[2]
    th_m3_m = my_th[3]
    th_m4 = my_th[4]
    my_mean = torch.mean(zs,dim=1) * np.sqrt(steps)
    my_mean[:,0] = torch.where(my_mean[:,0]>th_m1,1.0,0.0)
    my_mean[:,1] = torch.where(my_mean[:,1]<th_m2,1.0,0.0)
    mask_3_p = torch.where(my_mean[:,2]>th_m3,1.0,0.0)
    mask_3_m = torch.where(my_mean[:,2]<th_m3_m,1.0,0.0)
    mask_3 = mask_3_p + mask_3_m
    mask_3[mask_3>0] = 1
    my_mean[:,2] = mask_3
    my_mean[:,3] = torch.where(my_mean[:,3]>th_m4,1.0,0.0)
    my_mean = torch.mean(my_mean,dim=1)
    my_mean[my_mean>0.25] = 1
    return my_mean



if __name__ == "__main__":
    main()
