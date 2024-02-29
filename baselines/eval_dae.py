__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import yaml
from accelerate import Accelerator
from dae_unet import *
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_yen
from sklearn.metrics import average_precision_score

from utils import *


def main():
    with open("../conf/eval_dae.yml", "r") as file_object:
        conf = yaml.load(file_object, Loader=yaml.SafeLoader)
        torch.manual_seed(conf["seed"])
        accelerator = Accelerator()
        device = accelerator.device
        model = UNet(
            conf["channels"],
            conf["channels"],
            conf["depth"],
            conf["wf"],
            padding=conf["padding"],
        ).to(device)
        ckpt = torch.load(conf["model"])

        model.load_state_dict(ckpt)

        dataloader = MRI_Volume(
            conf,
            hist=False,
            shift=(True if "shifts" in conf["dataset_path"] else False),
        )

        model, dataloader = accelerator.prepare(model, dataloader)
        pbar = tqdm(dataloader)
        threshold_test = [
            round(x, 3)
            for x in np.arange(conf["thr_start"], conf["thr_end"], conf["thr_step"])
        ]

        dice_scores = {i: [] for i in threshold_test}
        dice_scores_mf = {i: [] for i in threshold_test}
        my_auprs = {i: [] for i in ["aupr no median", "aupr"]}

        with torch.no_grad():
            my_volume = []
            my_labels = []
            for i, (image, label) in enumerate(pbar):
                num_volumes = image.shape[0]
                num_slices = image.shape[4]

                image = torch.permute(image, (0, 4, 1, 2, 3))
                image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
                split = torch.split(image, conf["size_splits"])
                residuals = []
                for my_tensor in split:
                    pseudo_img = model(my_tensor)
                    # Erode the mask a bit to remove some of the reconstruction errors at the edges.
                    mask = my_tensor.sum(dim=1, keepdim=True) > 0.01
                    mask = (
                        F.avg_pool2d(
                            mask.float(),
                            kernel_size=conf["pool_size"],
                            stride=conf["stride"],
                            padding=conf["pool_pad"],
                        )
                        > conf["mask_thr"]
                    )
                    res = ((my_tensor - pseudo_img) * mask).abs()
                    residuals.append(res.to("cpu"))

                residuals = torch.cat(residuals, dim=0)

                residuals = residuals.view(
                    num_volumes,
                    num_slices,
                    residuals.shape[1],
                    residuals.shape[2],
                    residuals.shape[3],
                )
                residuals = torch.permute(residuals, (0, 2, 3, 4, 1))
                residuals = residuals.to(device)
                residuals, label = accelerator.gather_for_metrics((residuals, label))
                my_labels.append(label.type(torch.bool).to("cpu"))
                my_volume.append(residuals.to("cpu"))

            if accelerator.is_main_process:
                my_volume = torch.cat(my_volume, dim=0)
                my_labels = torch.cat(my_labels, dim=0)
                if conf["max"] == True:
                    anomaly_map = torch.max(my_volume, dim=1)[0]
                else:
                    anomaly_map = torch.mean(my_volume, dim=1)

                anomaly_map_mf = torch.clone(anomaly_map)
                anomaly_map_mf = median_filter_3D(
                    anomaly_map_mf, kernelsize=conf["kernel_size"]
                )
                my_labels = my_labels.contiguous()
                anomaly_map = norm_tensor(anomaly_map)
                anomaly_map_mf = norm_tensor(anomaly_map_mf)

                anomaly_map = anomaly_map.contiguous()
                anomaly_map_mf = anomaly_map_mf.contiguous()
                aupr = average_precision_score(my_labels.view(-1), anomaly_map.view(-1))
                my_auprs["aupr no median"].extend([aupr])
                aupr = average_precision_score(
                    my_labels.view(-1), anomaly_map_mf.view(-1)
                )
                my_auprs["aupr"].extend([aupr])
                for key in dice_scores:
                    segmentation = torch.where(anomaly_map > key, 1.0, 0.0)
                    segmentation = segmentation.type(torch.bool)
                    segmentation_mf = torch.where(anomaly_map_mf > key, 1.0, 0.0)
                    segmentation_mf = segmentation_mf.type(torch.bool)
                    dice_scores[key].extend(
                        [float(x) for x in dice(segmentation, my_labels)]
                    )
                    dice_scores[key] = np.mean(np.asarray(dice_scores[key]))
                    dice_scores_mf[key].extend(
                        [float(x) for x in dice(segmentation_mf, my_labels)]
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
                dice_scores["yen"].extend(
                    [float(x) for x in dice(yen_segmentation, my_labels)]
                )
                dice_scores["yen"] = np.mean(np.asarray(dice_scores["yen"]))

                for j, volume in enumerate(anomaly_map_mf):
                    thr = threshold_yen(volume.numpy())
                    segmentation = torch.where(volume > thr, 1.0, 0.0)
                    yen_segmentation[j] = segmentation
                yen_segmentation = bin_dilation(yen_segmentation, struc)
                dice_scores_mf["yen"] = []
                dice_scores_mf["yen"].extend(
                    [float(x) for x in dice(yen_segmentation, my_labels)]
                )
                dice_scores_mf["yen"] = np.mean(np.asarray(dice_scores_mf["yen"]))

                dice_scores["AUPRC"] = my_auprs["aupr no median"][0]
                dice_scores_mf["AUPRC"] = my_auprs["aupr"][0]
                df = pd.DataFrame.from_dict(
                    dice_scores, orient="index", columns=["value"]
                )
                df.index.name = "thr"
                df_mf = pd.DataFrame.from_dict(
                    dice_scores_mf, orient="index", columns=["value"]
                )
                df_mf.index.name = "thr"
                df.to_csv(conf["output"])
                df_mf.to_csv(conf["output_mf"])


if __name__ == "__main__":
    main()
