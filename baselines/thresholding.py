__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import os
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from sklearn.metrics import average_precision_score
from skimage.filters import threshold_yen
from scipy.ndimage import generate_binary_structure

from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    #args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_test.csv"
    args.dataset_path = "/mnt/qb/work/baumgartner/bkc035/shifts_data/patients"
    #args.dataset_path = "/mnt/qb/baumgartner/rawdata/shifts_registered/patients"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/shifts_in.csv"
    args.image_size = 128
    device = "cpu"

    len_df = pd.read_csv(args.path_to_csv)
    len_df = len(len_df)
    args.batch_size = len_df

    dataloader = MRI_Volume(args, hist=True, shift=True)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.85, 1, 0.001)]
    dice_scores_mask = {i: [] for i in threshold_test}
    dice_scores_mask_median = {i: [] for i in threshold_test}
    my_auprs = {i: [] for i in ["aupr no median", "aupr"]}

    for i, (image, label) in enumerate(pbar):
        image = image.to(device)
        label = label.to(device)
        my_volume = image[:, 0]
        median_volume = torch.clone(my_volume)
        median_volume = median_filter_3D(median_volume,kernelsize=5)
        my_volume = my_volume.contiguous()
        median_volume = median_volume.contiguous()
        aupr = average_precision_score(label.view(-1), my_volume.view(-1))
        my_auprs["aupr no median"].extend([aupr])
        aupr = average_precision_score(label.view(-1), median_volume.view(-1))
        my_auprs["aupr"].extend([aupr])
        for key in dice_scores_mask:
            my_mask = torch.where(my_volume > key, 1.0, 0.0)
            my_mask = my_mask.type(torch.bool).to(device)
            my_mask2 = torch.where(median_volume > key, 1.0, 0.0)
            my_mask2 = my_mask2.type(torch.bool).to(device)
            dice_scores_mask[key].extend([float(x) for x in dice(my_mask, label)])
            dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
            dice_scores_mask_median[key].extend([float(x) for x in dice(my_mask2, label)])
            dice_scores_mask_median[key] = np.mean(np.asarray(dice_scores_mask_median[key]))
        
        
        # calculate threshold without greedy search
        big_segmentation = torch.zeros_like(my_volume)
        struc = generate_binary_structure(3,1)
        for j, volume in enumerate(my_volume):
            thr = threshold_yen(volume.numpy())
            segmentation = torch.where(volume > thr, 1.0, 0.0)
            big_segmentation[j] = segmentation
        big_segmentation = bin_dilation(big_segmentation, struc)
        dice_scores_mask['yen'] = []
        dice_scores_mask['yen'].extend([float(x) for x in dice(big_segmentation, label)])
        dice_scores_mask['yen'] = np.mean(np.asarray(dice_scores_mask['yen']))


        for j, volume in enumerate(median_volume):
            thr = threshold_yen(volume.numpy())
            segmentation = torch.where(volume > thr, 1.0, 0.0)
            big_segmentation[j] = segmentation
        big_segmentation = bin_dilation(big_segmentation, struc)
        dice_scores_mask_median['yen'] = []
        dice_scores_mask_median['yen'].extend([float(x) for x in dice(big_segmentation, label)])
        dice_scores_mask_median['yen'] = np.mean(np.asarray(dice_scores_mask_median['yen']))


   
    dice_scores_mask['AUPRC'] = np.asarray(my_auprs["aupr no median"])
    dice_scores_mask_median['AUPRC'] = np.asarray(my_auprs["aupr"])
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask2 = pd.DataFrame(dice_scores_mask_median, index=[0]).T
    df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/thr_result.csv")
    df_mask2.to_csv("/mnt/qb/work/baumgartner/bkc035/median_thr_result.csv")


if __name__ == "__main__":
    main()
