import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
from accelerate import Accelerator
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
    #args.dataset_path = "./data/BraTS20"
    args.path_to_csv = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/scans_val_small.csv"
    #args.path_to_csv = "./data/BraTS20/survival_info_02.csv"
    args.batch_size = 10
    accelerator = Accelerator()
    device = accelerator.device

    dataloader = Brats_Volume(args, hist=True)
    dataloader = accelerator.prepare(dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.85, 0.9, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}

    for i, (image, label) in enumerate(pbar):
        image = (image * 2) - 1
        for key in dice_scores_mask:
            my_mask = torch.where(image > key,1.0,0.0)
            my_mask = connected_components_3d(my_mask)
            my_mask = my_mask.type(torch.bool).to(device)
            dice_scores_mask[key].extend([float(x) for x in dice(my_mask, label)])
    for key in dice_scores_mask:
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    #df_mask.to_csv("./results/Threshold_results/dice_scores.csv")
    df_mask.to_csv(
        "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/dice_threshold.csv"
    )


def dice(pred, truth):
    num = 2 * ((pred * truth).sum(dim=(1, 2, 3)).type(torch.float))
    den = (pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))).type(torch.float)
    return num / den

def connected_components_3d(volume):
    # shape [b, d, h, w], treat every sample in batch independently
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Connected components")
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop['filled_area'] <= 20:
                volume[i, cc_volume == prop['label']] = 0
    return torch.Tensor(volume)


class BratsDataVolume(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        hist: bool,
    ):
        self.df = df
        self.dataset_path = dataset_path
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
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=int)
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
        img = img.squeeze()
        img = img[:,:,15:125].permute(2,0,1)
        mask = mask.squeeze()
        mask = mask[:,:,15:125].permute(2,0,1)
        mask = mask.type(torch.bool)
        return img, mask


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
        i_ = ex.equalize_hist(i_.astype(np.uint16), mask=mask)
        i_ *= mask
        images[modality,:,:,:] = i_
    return torch.Tensor(images)

def Brats_Volume(args, hist = True):
    df = pd.read_csv(args.path_to_csv)
    dataset = BratsDataVolume(df, args.dataset_path, hist=hist)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return dataloader

if __name__ == "__main__":
    main()