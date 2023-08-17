__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from dae import *
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
    args.image_size = 128
    args.channels = 4

    accelerator = Accelerator()
    device = accelerator.device
    model = UNet(args.channels,args.channels,depth=4,wf=6,padding=True).to(device)
    ckpt = torch.load(
        "/mnt/lustre/baumgartner/bkc035/normative-diffusion/models/DAE/12_ckpt.pt"
     )
    model.load_state_dict(ckpt)
    dataloader = Brats_Volume(args, hist=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_diff = [x / 1000 for x in range(1, 1000)]
    dice_scores_mask = {i: [] for i in threshold_diff}
    model.eval()
    with torch.no_grad():
        my_volume = torch.zeros(
            (
                1,
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
            my_labels = torch.cat((my_labels, label.to("cpu")), dim=0)
            tmp_volume = torch.zeros(
                (
                    image.shape[0],
                    128,
                    128,
                    image.shape[4],
                )
            ).to(device)
            for j in range(image.shape[4]):
                my_img = model(image[:,:,:,:,j])
                mask = image[:,:,:,:,j].sum(dim=1, keepdim=True) > 0.01
                # Erode the mask a bit to remove some of the reconstruction errors at the edges.
                mask = (F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.95)

                my_diff = ((image[:,:,:,:,j] - my_img) * mask).abs().mean(dim=1)
                #my_diff = median_filter_2D(my_diff)
                tmp_volume[:, :, :, j] = my_diff

            my_volume = torch.cat((my_volume, tmp_volume.to("cpu")), dim=0)
        my_labels = my_labels[1:].contiguous()
        my_volume = median_filter_3D(my_volume)
        my_mask = my_volume[1:].contiguous()
        aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
        for key in dice_scores_mask:
            segmentation = torch.where(my_mask > key, 1.0, 0.0)
            segmentation = segmentation.type(torch.bool)
            dice_scores_mask[key].extend([dice(segmentation, my_labels)])

        dice_scores_mask[f"AUPRC"] = aupr
        df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
        df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/dae_result.csv")
        # df_mask.to_csv("./results/BraTS21/mask_one_3D.csv")


def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = medfilt2d(volume[i], kernel_size=kernelsize)
    return torch.Tensor(volume)


def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def show_slices(slices):
    """Function to display row of image slices"""
    fig, axes = plt.subplots(1, len(slices))
    for i, (slice) in enumerate(slices):
        axes[i].imshow(slice, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
