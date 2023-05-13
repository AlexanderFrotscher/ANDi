__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import os

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def plot_images(images):
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


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def upload_images(images, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    return ndarr


def get_data(args):
    """
    For local data
    """
    my_transforms = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: (x * 2) - 1), does not work on windows with enumerate
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(args.dataset_path, transform=my_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def cifar_10(args):
    transform_train = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(0.4),
            transforms.ToTensor(),  # divide by 255
            transforms.Lambda(lambda x: (x * 2) - 1),  # bring to [-1,1] but does not work on windows
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ds_train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, num_workers=2, shuffle=True
    )
    return dl_train


def make_dicts(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
