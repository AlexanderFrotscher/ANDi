__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

from accelerate import Accelerator, DistributedDataParallelKwargs
from sklearn.metrics import average_precision_score

from diffusion import *
from modules import *
from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_val_small.csv"
    args.batch_size = 1
    args.image_size = 128

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet().to(device=device)
    ckpt = torch.load(
        "/mnt/qb/work/baumgartner/bkc035/normative-diffusion/models/232_ema_ckpt.pt"
    )

    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=128, device=device)
    dataloader = Brats_Volume(args, hist=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.81, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}

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
            image = (image * 2) - 1
            num_steps = 250
            size_splits = 50
            num_volumes = image.shape[0]
            num_slices = image.shape[4]

            image = torch.permute(image, (0, 4, 1, 2, 3))
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            split = torch.split(image, size_splits)
            prediction = []
            for my_tensor in split:
                pseudo_image = diffusion.ano_ddpm(model, my_tensor, num_steps)
                diff = (my_tensor - pseudo_image) ** 2
                prediction.append(diff.to("cpu"))

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
            my_mask = torch.max(my_volume, dim=1)[0]
            my_mask = median_filter_3D(my_mask)
            my_labels = my_labels.contiguous()
            my_mask = norm_tensor(my_mask)
            my_mask = my_mask.contiguous()
            aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
            for key in dice_scores_mask:
                segmentation = torch.where(my_mask > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                dice_scores_mask[key].extend([dice(segmentation, my_labels)])

            dice_scores_mask[f"AUPRC"] = aupr
            df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
            df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/mask_3D.csv")


if __name__ == "__main__":
    main()
