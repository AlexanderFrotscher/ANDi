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

from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    #args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_val.csv"
    args.dataset_path = "/mnt/qb/work/baumgartner/bkc035/shifts_data/patients"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/shifts.csv"
    args.image_size = 128
    device = "cpu"

    len_df = pd.read_csv(args.path_to_csv)
    len_df = len(len_df)
    args.batch_size = len_df

    dataloader = MRI_Volume(args, hist=True, shift=True)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.8, 1.0, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}
    my_auprs = {i: [] for i in ["aupr no post", "aupr post"]}

    for i, (image, label) in enumerate(pbar):
        image = image.to(device)
        label = label.to(device)
        my_volume = image[:, 0].contiguous()
        aupr = average_precision_score(label.view(-1), my_volume.view(-1))
        my_auprs["aupr no post"].extend([aupr])
        for key in dice_scores_mask:
            my_mask = torch.where(my_volume > key, 1.0, 0.0)
            my_mask = my_mask.type(torch.bool).to(device)
            dice_scores_mask[key].extend([float(x) for x in dice(my_mask, label)])
            dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))

    # use best threshold with post-processing
    my_score = []
    my_thresh = max(dice_scores_mask, key=dice_scores_mask.get)
    for i, (image, label) in enumerate(pbar):
        image = image.to(device)
        label = label.to(device)
        my_volume = image[:, 0]
        #my_mask = median_filter_3D(my_volume,kernelsize=3)
        my_mask = my_volume.contiguous()
        aupr = average_precision_score(label.view(-1), my_mask.view(-1))
        my_auprs["aupr post"].extend([aupr])
        my_mask = torch.where(my_mask > my_thresh, 1.0, 0.0)
        my_mask = connected_components_3d(my_mask)
        my_mask = my_mask.type(torch.bool).to(device)
        my_score.extend([float(x) for x in dice(my_mask, label)])
        my_score = np.mean(np.asarray(my_score))

    dice_scores_mask[f"{my_thresh}_cc"] = my_score
    for key in my_auprs:
        dice_scores_mask[key] = np.asarray(my_auprs[key])

    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/thr_result.csv")


if __name__ == "__main__":
    main()
