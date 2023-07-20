__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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


def cifar_10(args):
    transform_train = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(0.4),
            transforms.ToTensor(),  # divide by 255
        ]
    )
    ds_train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, num_workers=2, shuffle=True
    )
    return dl_train


"""
This class is based on https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder
"""


class BratsDataset(Dataset):
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
        # age = self.df.loc[idx, "Age"]
        slice = self.df.loc[idx, "Slice"]
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, id_ + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, id_ + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj[:, :, slice], dtype=int)
        mask[mask == 1] = 1
        mask[mask == 2] = 1
        mask[mask == 4] = 1
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
        my_transform = transforms.Resize(128, antialias=True)
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
        self.data_types =  ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

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
        mask[mask == 1] = 1
        mask[mask == 2] = 1
        mask[mask == 4] = 1
        mask = torch.from_numpy(mask)
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())
        volume = torch.zeros(
            img.shape[0],
            self.image_size,
            self.image_size,
            img.shape[3]
        )
        my_mask = torch.zeros(
            128,
            128,
            mask.shape[2]
        )
        my_transform_1 = transforms.Resize(self.image_size, antialias=True)
        my_transform_2 = transforms.Resize(128, antialias=True)
        for i in range(img.shape[3]):
            volume[:,:,:,i] = my_transform_1(img[None, :, :, :, i])
            my_mask[:,:,i] = my_transform_2(mask[None, None, :, :, i])
        my_mask[my_mask > 0.5] = 1
        my_mask[my_mask !=1] = 0
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


"""
This function is from https://github.com/AntanasKascenas/DenoisingAE
"""


def normalize_volume(images):
    """
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
        i_ = images[modality, :, :,:]
        mask = np.zeros_like(i_)
        mask[i_ > 0] = 1
        i_ = i_ / np.max(i_)
        i_ = ex.equalize_hist(i_.astype(np.float32), mask=mask, nbins=256)
        i_ *= mask
        images[modality,:,:,:] = i_
    return torch.Tensor(images)


def preprocess_mask(mask):
    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1
    return mask_WT


def dice(pred, target):
    pred_sum = torch.flatten(pred,1).sum(dim=1)
    target_sum = torch.flatten(target,1).sum(dim=1)
    intersection = torch.flatten(pred,1).float() * torch.flatten(target,1).float()
    dice = (2 * intersection.sum(dim=1)) / (pred_sum + target_sum)
    return dice



def Brats21(args, preload=False, eval=False, hist=True):
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
                img = torch.stack(
                    [torch.from_numpy(x) for x in images], dim=0)
                img = normalize_volume(img.float())

            mask = preprocess_mask(mask)
            for i in range(img.shape[3]):
                my_slice = img[0, :, :, i]
                my_mask = mask[:, :, i]
                num_zeros = np.count_nonzero(my_slice == 0)
                if num_zeros < 54000 and 1 not in my_mask:
                    my_slices.append(img[:, :, :, i])
        dataset = preload_dataset(my_slices, my_transforms)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=2, shuffle=True
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


def Brats_Volume(args, hist = True):
    df = pd.read_csv(args.path_to_csv)
    dataset = BratsDataVolume(df, args.dataset_path, args.image_size, hist=hist)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return dataloader

def make_dicts(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
