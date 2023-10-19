__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import os
import random

import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def plot_images(images, mode="RGB"):
    if mode == "L":
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu(),
            cmap="gray",
        )
        plt.show()
    else:
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu()
        )
        plt.show()


def save_images(images, path, mode="RGB", **kwargs):
    if mode == "L":  # mode L is greyscale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)


def upload_images(images, mode="RGB", **kwargs):
    if mode == "L":
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    return ndarr


class BratsDataset(Dataset):
    """
    This class is based on https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder and
    loads individual slices that have to be given by a .csv file.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        my_transforms: transforms,
        dataset_path: str,
        image_size: int,
        hist: bool,
    ):
        self.df = df
        self.transforms = my_transforms
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "BraTS21ID"]
        images = []
        slice = self.df.loc[idx, "Slice"]
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, id_ + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, id_ + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj[:, :, slice], dtype=int)
        mask[mask >= 1] = 1  # mask contains the labels 1, 2, and 4
        mask = torch.from_numpy(mask)
        mask = mask[None, :, :]
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())
        img = img[:, :, :, slice]
        img = self.transforms(img)
        my_transform = transforms.Resize(self.image_size, antialias=True)
        mask = my_transform(mask)
        return img, mask


class BratsDataVolume(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        image_size: int,
        hist: bool,
    ):
        self.df = df
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "BraTS21ID"]
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, id_ + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, id_ + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=float)
        mask[mask >= 1] = 1
        mask = torch.from_numpy(mask)
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())

        start_range = 0
        end_range = 155
        volume = torch.zeros(
            img.shape[0], self.image_size, self.image_size, end_range - start_range
        )
        my_mask = torch.zeros(self.image_size, self.image_size, end_range - start_range)
        my_transform = transforms.Resize(self.image_size, antialias=True)
        for i in range(start_range, end_range):
            volume[:, :, :, i - start_range] = my_transform(img[None, :, :, :, i])
            my_mask[:, :, i - start_range] = my_transform(mask[None, None, :, :, i])
        my_mask[my_mask > 0.5] = 1
        my_mask[my_mask != 1] = 0
        my_mask = my_mask.type(torch.bool)
        return volume, my_mask


class preload_dataset(Dataset):
    def __init__(self, my_images: list, my_transforms: transforms):
        self.transforms = my_transforms
        self.images = my_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transforms(img)
        return img


def Brats21(args, preload=False, eval=False, hist=False):
    if eval == True:
        my_transforms = transforms.Compose(
            [
                transforms.Resize(args.image_size, antialias=True),
            ]
        )
    else:
        my_transforms = transforms.Compose(
            [
                transforms.Resize(args.image_size, antialias=True),
                # transforms.RandomHorizontalFlip(0.4),
            ]
        )
    if preload == True:
        df = pd.read_csv(args.path_to_csv)
        root_path = args.dataset_path
        ids = df.loc[:, "BraTS21ID"]
        my_slices = []
        data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
        for id in ids:
            images = []
            mask_path = os.path.join(root_path, id, id + "_seg.nii.gz")
            mask = np.asarray(nib.load(mask_path).dataobj, dtype=int)
            for data_type in data_types:
                img_path = os.path.join(args.dataset_path, id, id + data_type)
                img = np.asarray(nib.load(img_path).dataobj, dtype=float)
                images.append(img)

            if hist == True:
                img = np.stack([x for x in images])
                img = hist_norm(img)

            else:
                img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
                img = normalize_volume(img.float())

            mask[mask >= 1] = 1
            for i in range(img.shape[3]):
                my_slice = img[0, :, :, i]
                my_mask = mask[:, :, i]
                if torch.count_nonzero(my_slice) and 1 not in my_mask:
                    my_slices.append(img[:, :, :, i])
        dataset = preload_dataset(my_slices, my_transforms)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, shuffle=True
        )
    else:
        df = pd.read_csv(args.path_to_csv)
        dataset = BratsDataset(
            df, my_transforms, args.dataset_path, args.image_size, hist=hist
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, shuffle=False
        )
    return dataloader


def Brats_Volume(args, hist=False):
    df = pd.read_csv(args.path_to_csv)
    dataset = BratsDataVolume(df, args.dataset_path, args.image_size, hist=hist)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=1, shuffle=False
    )
    return dataloader


def normalize_volume(images):
    """
    This function is adapted from https://github.com/AntanasKascenas/DenoisingAE
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = torch.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99

    return images


def hist_norm(images):
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :]
        mask = np.zeros_like(i_)
        mask[i_ > 0] = 1
        i_ = i_ / np.max(i_)
        i_ = ex.equalize_hist(i_.astype(np.float32), mask=mask, nbins=256)
        i_ *= mask
        images[modality, :, :, :] = i_
    return torch.Tensor(images)


def dice_stitch(pred, target):
    pred_sum = pred.view(-1).sum()
    target_sum = target.view(-1).sum()
    intersection = pred.view(-1).float() @ target.view(-1).float()
    dice = (2 * intersection) / (pred_sum + target_sum)
    return dice

def dice(pred, truth):
    num = 2 * ((pred * truth).sum(dim=(1, 2, 3)).type(torch.float))
    den = (pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))).type(torch.float)
    return num / den


def coarse_noise(n, channels, device, noise_size=16, noise_std=0.2, image_size=128):
    noise = torch.normal(
        mean=torch.zeros(n, channels, noise_size, noise_size), std=noise_std
    ).to(device)
    noise = F.interpolate(
        noise,
        size=(image_size, image_size),
        mode="bilinear",
        antialias=False,
        align_corners=True,
    )
    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(image_size))
    roll_y = random.choice(range(image_size))
    noise = torch.roll(noise, shifts=[roll_x, roll_y], dims=[2, 3])
    return noise


def pyramid_noise_like(n, channels, device, image_size=128, discount=0.8):
    u = transforms.Resize(image_size, antialias=True)
    noise = torch.randn((n, channels, image_size, image_size)).to(device)
    w = image_size
    h = image_size
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(n, channels, w, h).to(device)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# dynamic normalisation
def clamp_to_spatial_quantile(x: torch.Tensor, p: float):
    b, c, *spatial = x.shape
    quantile = torch.quantile(torch.abs(x).view(b, c, -1), p, dim=-1, keepdim=True)
    quantile = torch.max(quantile, torch.ones_like(quantile))
    quantile_broadcasted, _ = torch.broadcast_tensors(quantile.unsqueeze(-1), x)
    return (
        torch.min(torch.max(x, -quantile_broadcasted), quantile_broadcasted)
        / quantile_broadcasted
    )


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


def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))


def make_dicts(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
