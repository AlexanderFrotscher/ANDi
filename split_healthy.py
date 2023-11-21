__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm.auto import tqdm

# import torch


def preprocess_mask(mask):
    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1
    return mask_WT


df = pd.read_csv(
    "/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data/splits/scans_train.csv"
)
# df = pd.read_csv('D:/DokumenteD/Uni/Programme/Python/Deep Learning/data/BraTS21/scans_myimage.csv')
root_path = "/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data"
# root_path = 'D:/DokumenteD/Uni/Programme/Python/Deep Learning/data/BraTS21/BraTS21_val'
os.makedirs(
    "/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data/healthy_slices",
    exist_ok=True,
)


def normalize_volume(images):
    """
    This function is adapted from https://github.com/AntanasKascenas/DenoisingAE
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = np.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99
    return images


# )
# ids = df.loc[:, "BraTS21ID"]
# healthy_ids = []
# healthy_slices = []
# i_ = 0
# for id in tqdm(ids):
#     mask_path = os.path.join(root_path, id, id + "_seg.nii.gz")
#     mask = np.asarray(nib.load(mask_path).dataobj, dtype=int)
#     img_path = os.path.join(root_path, id, id + "_flair.nii.gz")
#     img_flair = np.asarray(nib.load(img_path).dataobj, dtype=float)
#     img_path = os.path.join(root_path, id, id + "_t1.nii.gz")
#     img_t1 = np.asarray(nib.load(img_path).dataobj, dtype=float)
#     img_path = os.path.join(root_path, id, id + "_t1ce.nii.gz")
#     img_t1ce = np.asarray(nib.load(img_path).dataobj, dtype=float)
#     img_path = os.path.join(root_path, id, id + "_t2.nii.gz")
#     img_t2 = np.asarray(nib.load(img_path).dataobj, dtype=float)
#     mask = preprocess_mask(mask)
#     for i in range(img.shape[2]):
#         my_slice = img_flair[:, :, i]
#         my_mask = mask[:, :, i]
#         # my_slice = torch.tensor(my_slice)
#         if np.count_nonzero(my_slice) and 1 not in my_mask:
#             healthy_ids.append(id)
#             healthy_slices.append(i)
#             healthy_slices_values = np.stack(
#                 [img_flair, img_t1, img_t1ce, img_t2], axis=0
#             )

#     return images

#             np.save(
#                 f"/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data/healthy_slices/{i_:05d}.npy",
#                 healthy_slices_values,
#             )
#             i_ += 1

ids = df.loc[:, "BraTS21ID"]
data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
i__ = 0
with tqdm(ids) as pbar:
    for id in pbar:
        images = []
        mask_path = os.path.join(root_path, id, id + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=int)
        for data_type in data_types:
            img_path = os.path.join(root_path, id, id + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

            img = np.stack(images, axis=0)
            img = normalize_volume(img.astype(np.float32))

        mask[mask >= 1] = 1
        j__ = 0
        for i in range(img.shape[3]):
            my_slice = img[0, :, :, i]
            my_mask = mask[:, :, i]
            if np.count_nonzero(my_slice) and 1 not in my_mask:
                j__ += 1

                np.save(
                    f"/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data/healthy_slices/{i__:05d}.npy",
                    img[:, :, :, i],
                )
                i__ += 1
        pbar.set_description(f"{i__:05d}, shape {img.shape[3]}, healthy {j__}")

print(f"Number of healthy slices: {i__:05d}")


# healthy_dict = {"BraTS21ID": healthy_ids, "Slice": healthy_slices}
# df_healthy = pd.DataFrame(healthy_dict)
# df_healthy.to_csv(
#     "/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data/splits/healthy_slices_train.csv",
#     index=False,
# )

# df_healthy.to_csv(
#     "/mnt/qb/macke/jkapoor83/brats_data/BraTS2021_Training_Data/splits/healthy_slices_train.csv",
#     index=False,
# )
