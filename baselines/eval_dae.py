__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import argparse
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


from accelerate import Accelerator
from dae_unet import *
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
    args.batch_size = 1
    args.image_size = 128
    args.channels = 4

    accelerator = Accelerator()
    device = accelerator.device
    model = UNet(args.channels, args.channels, depth=4, wf=6, padding=True).to(device)
    ckpt = torch.load(
        "/mnt/qb/work/baumgartner/bkc035/normative-diffusion/baselines/models/DAE/2_ckpt.pt"
    )
    model.load_state_dict(ckpt)
    dataloader = MRI_Volume(args, hist=False,shift=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_diff = [x / 1000 for x in range(50, 200)]
    dice_scores_mask = {i: [] for i in threshold_diff}
    model.eval()
    with torch.no_grad():
        my_volume = torch.zeros(
            (
                1,
                4,
                128,
                128,
                155,
            )
        ).to("cpu")
        my_labels = (
            torch.zeros(
                (
                    1,
                    128,
                    128,
                    155,
                )
            )
            .type(torch.bool)
            .to("cpu")
        )
        for i, (image, label) in enumerate(pbar):
            num_volumes = image.shape[0]
            num_slices = image.shape[4]
            size_splits = 155

            image = torch.permute(image, (0, 4, 1, 2, 3))
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            split = torch.split(image, size_splits)
            prediction = []
            for my_tensor in split:
                pseudo_img = model(my_tensor)
                # Erode the mask a bit to remove some of the reconstruction errors at the edges.
                mask = my_tensor.sum(dim=1, keepdim=True) > 0.01
                mask = (
                    F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2)
                    > 0.95
                )
                my_diff = ((my_tensor - pseudo_img) * mask).abs()
                prediction.append(my_diff.to("cpu"))

            prediction = torch.cat(prediction, dim=0)

            prediction = prediction.view(
                num_volumes,
                num_slices,
                prediction.shape[1],
                prediction.shape[2],
                prediction.shape[3],
            )
            prediction = torch.permute(prediction, (0, 2, 3, 4, 1))
            prediction = prediction.to(device)
            prediction, label = accelerator.gather_for_metrics((prediction, label))
            my_labels = torch.cat((my_labels, label.to("cpu")), dim=0)
            my_volume = torch.cat((my_volume, prediction.to("cpu")), dim=0)

        if accelerator.is_main_process:
            if not torch.count_nonzero(my_labels[0]):
                my_labels = my_labels[1:]
                my_volume = my_volume[1:]
            my_volume = torch.max(my_volume, dim=1)[0]
            #my_volume = torch.mean(my_volume,dim=1)
            my_labels = my_labels.contiguous()
            # my_volume = median_filter_2D(my_volume)
            # my_volume = my_dilation(my_volume,kernelsize=2)
            # my_volume = norm_tensor(my_volume)
            my_mask = my_volume.contiguous()
            aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
            for key in dice_scores_mask:
                segmentation = torch.where(my_mask > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                dice_scores_mask[key].extend([float(x) for x in dice(segmentation, my_labels)])
                dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))

            dice_scores_mask[f"AUPRC"] = aupr
            df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
            df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/dae_result.csv")


if __name__ == "__main__":
    main()
