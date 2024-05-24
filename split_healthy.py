__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import os
import pickle

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm


def normalize_volume(images):
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = np.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99
    return images


def main(args):
    split_and_save(args)


def split_and_save(args):
    df = pd.read_csv(args.input_file)
    root_path = args.data_set
    os.makedirs(f"{args.data_set}/healthy_slices", exist_ok=True)
    ids = df.loc[:, df.columns[0]]
    my_transform = transforms.Resize(args.resolution, antialias=True)
    global_ids = []
    global_slices = []
    data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
    pbar = tqdm(ids)
    map_size = 0
    for id in pbar:
        map_size += (
            sum(
                d.stat().st_size
                for d in os.scandir(os.path.join(args.data_set, id))
                if d.is_file()
            )
            * 2
        )
    env = lmdb.open(str(f"{args.data_set}/healthy_slices"), map_size=map_size)
    with env.begin(write=True) as txn:
        num = 0
        for id in pbar:
            my_ids = []
            healthy_slices = []
            my_slices = []
            images = []
            mask_path = os.path.join(root_path, id, id + "_seg.nii.gz")
            mask = np.asarray(nib.load(mask_path).dataobj, dtype=int)
            for data_type in data_types:
                img_path = os.path.join(args.data_set, id, id + data_type)
                img = np.asarray(nib.load(img_path).dataobj, dtype=float)
                images.append(img)
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())
            mask[mask >= 1] = 1
            for i in range(img.shape[3]):
                my_slice = img[0, :, :, i]
                my_mask = mask[:, :, i]
                if (
                    torch.count_nonzero(my_slice) and 1 not in my_mask
                ):  # filter out empty and slices containing an anomaly
                    my_ids.append(id)
                    healthy_slices.append(i)
                    my_slices.append(img[None, :, :, :, i])
            my_slices = torch.cat(my_slices)
            my_slices = my_transform(my_slices).numpy()
            for data in my_slices:
                key = f"{num:08}"
                num += 1
                txn.put(key.encode("ascii"), pickle.dumps(data))
            global_ids.extend(my_ids)
            global_slices.extend(healthy_slices)
    env.close()
    healthy_dict = {df.columns[0]: global_ids, "Slice": global_slices}
    df_healthy = pd.DataFrame(healthy_dict)
    df_healthy.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the data into healthy slices.")
    parser.add_argument(
        "-d",
        "--data_set",
        type=str,
        required=True,
        metavar="",
        help="The folder that contains the MRI-Volumes containing anomalies.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        metavar="",
        help="The .csv that specifies the volumes to split into healthy slices.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        metavar="",
        help="The .csv path for the file that specifies the healthy slices.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        metavar="",
        help="The resolution of the stored slices, if save flag is set to true.",
    )
    args = parser.parse_args()
    main(args)
