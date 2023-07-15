__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torchvision import  transforms

def preprocess_mask(mask):
    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1
    mask_WT = torch.from_numpy(mask_WT)
    mask_WT = mask_WT[None, :, :]
    my_transform = transforms.Resize(128, antialias=True)
    mask_WT = my_transform(mask_WT)
    mask_WT[mask_WT > 0.5] = 1
    mask_WT[mask_WT != 1] = 0
    return mask_WT


df = pd.read_csv(
    "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/scans_val.csv"
)
root_path = "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data"
ids = df.loc[:, "BraTS21ID"]
tumor_ids = []
tumor_slices = []
healthy_ids = []
healthy_slices = []
for id in ids:
    mask_path = os.path.join(root_path, id, id + "_seg.nii.gz")
    mask = np.asarray(nib.load(mask_path).dataobj, dtype=int)
    img_path = os.path.join(root_path, id, id + "_flair.nii.gz")
    img = np.asarray(nib.load(img_path).dataobj, dtype=float)
    mask = preprocess_mask(mask)
    for i in range(img.shape[2]):
        num_zeros = np.count_nonzero(img[:, :, i] == 0)
        if num_zeros < 54000:
            if 1 in mask[:, :, i]:
                tumor_ids.append(id)
                tumor_slices.append(i)
            else:
                healthy_ids.append(id)
                healthy_slices.append(i)

tumor_dict = {"BraTS21ID": tumor_ids, "Slice": tumor_slices}
healthy_dict = {"BraTS21ID": healthy_ids, "Slice": healthy_slices}
df_tumor = pd.DataFrame(tumor_dict)
df_tumor.to_csv(
    "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/tumor_slices_val.csv",
    index=False,
)
df_healthy = pd.DataFrame(healthy_dict)
df_healthy.to_csv(
    "/mnt/lustre/baumgartner/bkc035/data/BraTS2021/BraTS2021_Training_Data/healthy_slices_val.csv",
    index=False,
)
