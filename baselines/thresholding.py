__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import os
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import yaml
from sklearn.metrics import average_precision_score
from skimage.filters import threshold_yen
from scipy.ndimage import generate_binary_structure

from utils import *


def main():
    with open("../conf/eval_thr.yml", "r") as file_object:
        conf = yaml.load(file_object, Loader=yaml.SafeLoader)
        torch.manual_seed(conf["seed"])
        device = "cpu"

        len_df = pd.read_csv(conf["path_to_csv"])
        len_df = len(len_df)
        conf["batch_size"] = len_df

        dataloader = MRI_Volume(
            conf,
            hist=True,
            shift=(True if "shifts" in conf["dataset_path"] else False),
        )

        pbar = tqdm(dataloader)
        threshold_test = [
            round(x, 3)
            for x in np.arange(conf["thr_start"], conf["thr_end"], conf["thr_step"])
        ]

        dice_scores = {i: [] for i in threshold_test}
        dice_scores_mf = {i: [] for i in threshold_test}
        my_auprs = {i: [] for i in ["aupr no median", "aupr"]}

        for i, (image, label) in enumerate(pbar):
            image = image.to(device)
            label = label.to(device)
            anomaly_map = image[:, 0]
            anomaly_map_mf = torch.clone(anomaly_map)
            anomaly_map_mf = median_filter_3D(
                anomaly_map_mf, kernelsize=conf["kernel_size"]
            )
            anomaly_map = anomaly_map.contiguous()
            anomaly_map_mf = anomaly_map_mf.contiguous()
            aupr = average_precision_score(label.view(-1), anomaly_map.view(-1))
            my_auprs["aupr no median"].extend([aupr])
            aupr = average_precision_score(label.view(-1), anomaly_map_mf.view(-1))
            my_auprs["aupr"].extend([aupr])
            for key in dice_scores:
                segmentation = torch.where(anomaly_map > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                segmentation_mf = torch.where(anomaly_map_mf > key, 1.0, 0.0)
                segmentation_mf = segmentation_mf.type(torch.bool)
                dice_scores[key].extend([float(x) for x in dice(segmentation, label)])
                dice_scores[key] = np.mean(np.asarray(dice_scores[key]))
                dice_scores_mf[key].extend(
                    [float(x) for x in dice(segmentation_mf, label)]
                )
                dice_scores_mf[key] = np.mean(np.asarray(dice_scores_mf[key]))

            yen_segmentation = torch.zeros_like(anomaly_map)
            struc = generate_binary_structure(conf["rank"], conf["connectivity"])

            for j, volume in enumerate(anomaly_map):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                yen_segmentation[j] = segmentation
            yen_segmentation = bin_dilation(yen_segmentation, struc)
            dice_scores["yen"] = []
            dice_scores["yen"].extend([float(x) for x in dice(yen_segmentation, label)])
            dice_scores["yen"] = np.mean(np.asarray(dice_scores["yen"]))

            for j, volume in enumerate(anomaly_map_mf):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                yen_segmentation[j] = segmentation
            yen_segmentation = bin_dilation(yen_segmentation, struc)
            dice_scores_mf["yen"] = []
            dice_scores_mf["yen"].extend(
                [float(x) for x in dice(yen_segmentation, label)]
            )
            dice_scores_mf["yen"] = np.mean(np.asarray(dice_scores_mf["yen"]))

        dice_scores["AUPRC"] = my_auprs["aupr no median"][0]
        dice_scores_mf["AUPRC"] = my_auprs["aupr"][0]
        df = pd.DataFrame.from_dict(dice_scores, orient="index", columns=["value"])
        df.index.name = "thr"
        df_mf = pd.DataFrame.from_dict(
            dice_scores_mf, orient="index", columns=["value"]
        )
        df_mf.index.name = "thr"
        df.to_csv(conf["output"])
        df_mf.to_csv(conf["output_mf"])


if __name__ == "__main__":
    main()
