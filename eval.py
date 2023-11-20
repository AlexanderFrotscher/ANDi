__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse

from accelerate import Accelerator, DistributedDataParallelKwargs
from sklearn.metrics import average_precision_score
from skimage.filters import threshold_yen
from scipy.ndimage import generate_binary_structure
from diffusion import *
from modules import *
from utils import *


def main(args):
    torch.manual_seed(73)
    if args.data == "brats":
        args.dataset_path = "data/brats"
        args.path_to_csv = "data/brats/splits/scans_test.csv"
    elif args.data == "shifts1":
        args.dataset_path = "data/shifts_registered/patients"
        args.path_to_csv = "data/shifts_registered/shifts_in.csv"
    elif args.data == "shifts2":
        args.dataset_path = "data/shifts_registered/patients"
        args.path_to_csv = "data/shifts_registered/shifts_out.csv"
    else:
        raise ValueError("Dataset not found")
    # args.dataset_path = "/mnt/qb/work/baumgartner/bkc035/shifts_data/patients"
    # args.dataset_path = "/mnt/qb/baumgartner/rawdata/shifts_registered/patients"
    # args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/shifts_out.csv"
    args.batch_size = 1
    args.image_size = 128

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet().to(device=device)
    print(f"loading {args.model}")
    ckpt = torch.load(f"data/brats/models/{args.model}/232_ema_ckpt.pt")

    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=128, device=device)
    print("Loading dataset {}...".format(args.data))
    dataloader = MRI_Volume(
        args, hist=False, shift=(True if "shifts" in args.data else False)
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.10, 0.001)]
    dice_scores_mask = {i: [] for i in threshold_test}
    dice_scores_mask_median = {i: [] for i in threshold_test}
    my_auprs = {i: [] for i in ["aupr no median", "aupr"]}

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
            size_splits = 200
            num_volumes = image.shape[0]
            num_slices = image.shape[4]

            image = torch.permute(image, (0, 4, 1, 2, 3))
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            split = torch.split(image, size_splits)
            zs_list = []
            for my_tensor in split:
                zs = diffusion.dpm_differences(
                    model, my_tensor, start=75, stop=num_steps, pyramid=True
                ).to("cpu")
                # zs = diffusion.skip_differences(model, my_tensor, start = 100, stop = num_steps, skip=25, pyramid=False).to('cpu')
                # zs = diffusion.differences_noise(model, my_tensor, start = 100, stop = num_steps, pyramid=False).to('cpu')
                zs_list.append(zs)

            zs_list = torch.cat(zs_list, dim=0)
            my_mean = gmean(zs_list, dim=1)
            # my_mean = torch.mean(zs_list,dim=1)

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
            # break

        if accelerator.is_main_process:
            if not torch.count_nonzero(my_labels[0]):
                my_labels = my_labels[1:]
                my_volume = my_volume[1:]
            my_mask = torch.max(my_volume, dim=1)[0]
            mask_median = torch.clone(my_mask)
            mask_median = median_filter_3D(mask_median, kernelsize=5)
            my_labels = my_labels.contiguous()
            my_mask = norm_tensor(my_mask)
            mask_median = norm_tensor(mask_median)

            my_mask = my_mask.contiguous()
            mask_median = mask_median.contiguous()
            aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
            my_auprs["aupr no median"].extend([aupr])
            aupr = average_precision_score(my_labels.view(-1), mask_median.view(-1))
            my_auprs["aupr"].extend([aupr])
            for key in dice_scores_mask:
                segmentation = torch.where(my_mask > key, 1.0, 0.0)
                segmentation = segmentation.type(torch.bool)
                my_mask2 = torch.where(mask_median > key, 1.0, 0.0)
                my_mask2 = my_mask2.type(torch.bool)
                dice_scores_mask[key].extend(
                    [float(x) for x in dice(segmentation, my_labels)]
                )
                dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
                dice_scores_mask_median[key].extend(
                    [float(x) for x in dice(my_mask2, my_labels)]
                )
                dice_scores_mask_median[key] = np.mean(
                    np.asarray(dice_scores_mask_median[key])
                )

            big_segmentation = torch.zeros_like(my_mask)
            struc = generate_binary_structure(3, 1)

            for j, volume in enumerate(my_mask):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                big_segmentation[j] = segmentation
            big_segmentation = bin_dilation(big_segmentation, struc)
            dice_scores_mask["yen"] = []
            dice_scores_mask["yen"].extend(
                [float(x) for x in dice(big_segmentation, my_labels)]
            )
            dice_scores_mask["yen"] = np.mean(np.asarray(dice_scores_mask["yen"]))

            for j, volume in enumerate(mask_median):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                big_segmentation[j] = segmentation
            big_segmentation = bin_dilation(big_segmentation, struc)
            dice_scores_mask_median["yen"] = []
            dice_scores_mask_median["yen"].extend(
                [float(x) for x in dice(big_segmentation, my_labels)]
            )
            dice_scores_mask_median["yen"] = np.mean(
                np.asarray(dice_scores_mask_median["yen"])
            )

            dice_scores_mask["AUPRC"] = np.asarray(my_auprs["aupr no median"])
            dice_scores_mask_median["AUPRC"] = np.asarray(my_auprs["aupr"])
            df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
            df_mask2 = pd.DataFrame(dice_scores_mask_median, index=[0]).T
            df_mask.to_csv(f"{args.model}_{args.data}_pyr_brats_mask_3D.csv")
            df_mask2.to_csv(f"{args.model}_{args.data}_pyr_brats_mask_median.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DDPM-Pyramid")
    parser.add_argument("--data", type=str, default="brats")
    args = parser.parse_args()
    main(args)
