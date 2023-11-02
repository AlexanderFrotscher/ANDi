__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

from accelerate import Accelerator, DistributedDataParallelKwargs
from sklearn.metrics import average_precision_score

from diffusion import *
from modules import *
from utils import *


def main():
    torch.manual_seed(77)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_test.csv"
    args.batch_size = 1
    args.image_size = 128

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet().to(device=device)
    ckpt = torch.load(
        "/mnt/qb/work/baumgartner/bkc035/normative-diffusion/models/pyramid/232_ema_ckpt.pt"
    )

    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=128, device=device)
    dataloader = Brats_Volume(args, hist=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.15, 0.001)]
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
            num_steps = 200
            size_splits = 50
            num_volumes = image.shape[0]
            num_slices = image.shape[4]

            image = torch.permute(image, (0, 4, 1, 2, 3))
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            split = torch.split(image, size_splits)
            zs_list = []
            for my_tensor in split:
                zs = diffusion.dpm_differences(model, my_tensor, start=100, stop=num_steps, pyramid=False).to('cpu')
                # zs = diffusion.skip_differences(model, my_tensor, start = 100, stop = num_steps, skip=25, pyramid=False).to('cpu')
                # zs = diffusion.differences_noise(model, my_tensor, start = 100, stop = num_steps, pyramid=False).to('cpu')
                zs_list.append(zs)

            zs_list = torch.cat(zs_list, dim=0)
            my_mean = gmean(zs_list,dim=1)

            my_mean = my_mean.view(
                num_volumes,
                num_slices,
                my_mean.shape[1],
                my_mean.shape[2],
                my_mean.shape[3],
            )
            my_mean = torch.permute(my_mean, (0, 2, 3, 4, 1))
            my_mean = my_mean.to(device)
            my_mean, label = accelerator.gather_for_metrics((my_mean, label))
            my_labels = torch.cat((my_labels, label.to("cpu")), dim=0)
            my_volume = torch.cat((my_volume, my_mean.to("cpu")), dim=0)

        if accelerator.is_main_process:
            if not torch.count_nonzero(my_labels[0]):
                my_labels = my_labels[1:]
                my_volume = my_volume[1:]
            my_mask = torch.max(my_volume, dim=1)[0]
            my_mask = median_filter_3D(my_mask)
            my_mask = my_dilation(my_mask,kernelsize=3)
            my_labels = my_labels.contiguous()
            my_mask = norm_tensor(my_mask)
            my_mask = my_mask.contiguous()
            aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
            for key in dice_scores_mask:
                segmentation = torch.where(my_mask > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                dice_scores_mask[key].extend([float(x) for x in dice(segmentation, my_labels)])
                dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))

            dice_scores_mask[f"AUPRC"] = aupr
            df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
            df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/mask_3D.csv")


if __name__ == "__main__":
    main()
