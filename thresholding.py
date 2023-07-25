import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from skimage.measure import label, regionprops
from torch.nn.modules.utils import _pair, _quadruple
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = (
        "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
    )
    #args.dataset_path = "./data/BraTS20/BraTS20_Training"
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/scans_test.csv"
    #args.path_to_csv = "./data/BraTS20/survival_info_01.csv"
    args.batch_size = 20
    args.image_size = 128
    accelerator = Accelerator()
    device = accelerator.device

    dataloader = Brats_Volume(args, hist=True)
    dataloader = accelerator.prepare(dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.8, 0.9, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}

    for i, (image, label) in enumerate(pbar):
        image = (image * 2) - 1
        for key in dice_scores_mask:
            my_mask = torch.where(image > key, 1.0, 0.0)
            my_mask = my_mask[:, 0].type(torch.bool).to(device)
            dice_scores_mask[key].extend([float(x) for x in dice(my_mask, label)])
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))

    # use best threshold with connected_components
    my_scores = []
    my_thresh = max(dice_scores_mask, key=dice_scores_mask.get)
    for i, (image, label) in enumerate(pbar):
        image = (image * 2) - 1
        my_mask = torch.where(image > key, image, 0.0)
        my_mask = median_filter_tensor(my_mask)
        my_mask[my_mask != 0] = 1
        my_mask = connected_components_3d(my_mask)
        my_mask = my_mask.type(torch.bool).to(device)
        my_scores.extend([float(x) for x in dice(my_mask, label)])
    my_dice = np.asarray(my_scores).mean()

    dice_scores_mask[f"{my_thresh}_cc"] = my_dice
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    #df_mask.to_csv("./results/Threshold_results/dice_scores.csv")
    df_mask.to_csv("/mnt/lustre/baumgartner/bkc035/data/BraTS2021/dice_threshold.csv")


def dice(pred, target):
    pred_sum = torch.flatten(pred, 1).sum(dim=1)
    target_sum = torch.flatten(target, 1).sum(dim=1)
    intersection = torch.flatten(pred, 1).float() * torch.flatten(target, 1).float()
    dice = (2 * intersection.sum(dim=1)) / (pred_sum + target_sum)
    return dice


def connected_components_3d(volume):
    # shape [b, d, h, w], treat every sample in batch independently
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Connected components")
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop["filled_area"] <= 20:
                volume[i, cc_volume == prop["label"]] = 0
    return torch.Tensor(volume)


def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        for j in range(volume.shape[3]):
            volume[i, :, :, j] = medfilt2d(volume[i, :, :, j], kernel_size=kernelsize)
    return torch.Tensor(volume)

def median_filter_tensor(volume, kernelsize=5):
    for j in range(volume.shape[4]):
        volume[:, :, :, :, j] = median_pool(volume[:, :, :, :, j], kernel_size=kernelsize, padding=2)
    return torch.Tensor(volume[:,0])


def median_pool(x, kernel_size=3, stride=1, padding=0):
    k = _pair(kernel_size)
    stride = _pair(stride)
    padding = _quadruple(padding)

    x = F.pad(x, padding, mode="reflect")
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

    return x


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
        self.data_types = ["_flair.nii.gz"]

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
        # mask_path = os.path.join(self.dataset_path, id_, "anomaly_segmentation.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=float)
        mask[mask >= 0.9] = 1
        mask[mask != 1] = 0
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
        my_mask = torch.zeros(128, 128, end_range - start_range)
        my_transform_1 = transforms.Resize(self.image_size, antialias=True)
        my_transform_2 = transforms.Resize(128, antialias=True)
        for i in range(start_range, end_range):
            volume[:, :, :, i - start_range] = my_transform_1(img[None, :, :, :, i])
            my_mask[:, :, i - start_range] = my_transform_2(mask[None, None, :, :, i])
        my_mask[my_mask > 0.5] = 1
        my_mask[my_mask != 1] = 0
        my_mask = my_mask.type(torch.bool)
        return volume, my_mask


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
        i_ = images[modality, :, :, :]
        mask = np.zeros_like(i_)
        mask[i_ > 0] = 1
        i_ = i_ / np.max(i_)
        i_ = ex.equalize_hist(i_.astype(np.float32), mask=mask, nbins=256)
        i_ *= mask
        images[modality, :, :, :] = i_
    return torch.Tensor(images)


def Brats_Volume(args, hist=True):
    df = pd.read_csv(args.path_to_csv)
    dataset = BratsDataVolume(df, args.dataset_path, args.image_size, hist=hist)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, shuffle=False
    )
    return dataloader


if __name__ == "__main__":
    main()
