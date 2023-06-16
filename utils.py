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
            # transforms.Lambda(
            #    lambda x: (x * 2) - 1
            # ),  # bring to [-1,1] but does not work on windows
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    ):
        self.df = df
        self.transforms = my_transforms
        self.dataset_path = dataset_path
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "Brats20ID"]
        images = []
        age = self.df.loc[idx, "Age"]
        slice = self.df.loc[idx, "Slice"]
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, id_ + data_type)
            img = np.asarray(nib.load(img_path).dataobj[:, :, slice], dtype=float)
            images.append(img)

        img = torch.stack([torch.from_numpy(x) for x in images], dim=0).unsqueeze(dim=0)
        img = self.normalize(img)
        img = img[0].float()
        img = self.transforms(img)

        return img, age

    def normalize(self, images):
        """
        Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
        """
        for modality in range(images.shape[1]):
            i_ = images[:, modality, :, :].reshape(-1)
            i_ = i_[i_ > 0]
            p_99 = torch.quantile(i_, 0.99)
            images[:, modality, :, :] /= p_99

        return images


class preload_dataset(Dataset):
    def __init__(self, my_images: list, my_transforms: transforms):
        self.transforms = my_transforms
        self.images = my_images

    def __len__(self):
        return len(self.labels)

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


def preprocess_mask(mask):
    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1
    return mask_WT


def Brats20(args, preload=False, my_shuffle = True):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(args.image_size, antialias=True),
            transforms.RandomHorizontalFlip(0.4),
            transforms.Lambda(
                lambda x: (x * 2) - 1
            ),  # bring to [-1,1] but does not work on windows
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

            img = torch.stack([torch.from_numpy(x) for x in images], dim=0).unsqueeze(
                dim=0
            )
            img = normalize_volume(img[0].float())
            mask = preprocess_mask(mask)
            for i in range(img.shape[3]):
                my_slice = img[0, :, :, i]
                my_mask = mask[:, :, i]
                num_zeros = np.count_nonzero(my_slice == 0)
                if num_zeros < 54000 and 1 not in my_mask:
                    my_slices.append(img[:, :, :, i])
        dataset = preload_dataset(my_slices, my_transforms)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, shuffle=my_shuffle
        )
    else:
        df = pd.read_csv(args.path_to_csv)
        dataset = BratsDataset(df, my_transforms, args.dataset_path)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, shuffle=my_shuffle
        )
    return dataloader


def make_dicts(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
