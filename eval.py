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
    args.dataset_path = "./data/BraTS20"
    args.path_to_csv = "./data/BraTS20/tumor_slices_small.csv"
    args.batch_size = 3
    args.image_size = 64

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet_conditional().to(device)
    ckpt = torch.load("./models/trained_models/1000ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=64, device=device)
    dataloader = Brats20(args, eval=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_diff = [x / 100 for x in range(1, 100)]
    threshold_test = [round(x, 3) for x in np.arange(0.2, 2, 0.01)]
    dice_scores_diff = {i: [] for i in threshold_diff}
    dice_scores_mask = {i: [] for i in threshold_test}
    my_transform = transforms.Lambda(lambda x: (x * 2) - 1)
    for i, (image, label) in enumerate(pbar):
        image = my_transform(image).to(device)
        label = label.to(device)
        label = label[:, 0, :, :].type(torch.uint8)
        num_steps = 1000
        xts, zs = diffusion.dpm_inversion(model, image, num_steps)
        my_images = diffusion.guide_restoration(
            model, xts[:, 0:150], zs[:, 0:150], cfg_scale=0, noise_scale=0.4
        )
        # plot_images(image,mode='L')
        # plot_images(my_images,mode='L')
        for key in dice_scores_diff:
            my_masks = create_difference(image, my_images, threshold=key)
            dice_scores_diff[key].extend([float(x) for x in dice(my_masks, label)])
        for key in dice_scores_mask:
            mask = diffusion.create_mask(zs, num_steps, threshold=key)
            dice_scores_mask[key].extend([float(x) for x in dice(mask, label)])
    for key in dice_scores_diff:
        dice_scores_diff[key] = np.mean(np.asarray(dice_scores_diff[key]))
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
    df_diff = pd.DataFrame(dice_scores_diff, index=[0]).T
    df_diff.index.rename("threshold", inplace=True)
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask.index.rename("threshold", inplace=True)
    df_diff.to_csv("./results/BraTS21/diff_score.csv")
    df_mask.to_csv("./results/BraTS21/mask_score.csv")
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


if __name__ == "__main__":
    main()
