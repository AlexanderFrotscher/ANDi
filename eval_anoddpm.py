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
from baselines.simplex_noise import generate_simplex_noise


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

    args.batch_size = 1
    args.image_size = 128

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    model = UNet().to(device=device)
    print(f"loading anoddpm")
    ckpt = torch.load(f"./328_ema_ckpt.pt")

    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=1000, img_size=128, device=device)
    print("Loading dataset {}...".format(args.data))
    dataloader = MRI_Volume(
        args, hist=False, shift=(True if "shifts" in args.data else False)
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    pbar = tqdm(dataloader)
    threshold_test = [round(x, 3) for x in np.arange(0.01, 0.5, 0.01)]
    dice_scores_mask = {i: [] for i in threshold_test}
    dice_scores_mask_median3 = {i: [] for i in threshold_test}
    dice_scores_mask_median5 = {i: [] for i in threshold_test}
    my_auprs = {i: [] for i in ["aupr no median", "aupr median3", "aupr median5"]}

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

    precomputed_noise = torch.load("data/simplex_noise.pt")
    precomputed_noise = (
        precomputed_noise.to("cpu").view(1, 1, 10000, 128, 128).squeeze(0)
    )
    print("precomputed noise loaded", precomputed_noise.shape)

    for i, (image, label) in enumerate(pbar):
        image = (image * 2) - 1
        num_steps = 250
        size_splits = 200
        num_volumes = image.shape[0]
        num_slices = image.shape[4]

        image = torch.permute(image, (0, 4, 1, 2, 3))
        image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        split = torch.split(image, size_splits)
        zs_list = []
        for my_tensor in split:
            with torch.no_grad():
                pseudo_healthy = diffusion.ano_ddpm(
                    model,
                    my_tensor,
                    num_steps,
                    simplex=True,
                    precomputed_noise=precomputed_noise,
                )
            # break
            residual = (my_tensor - pseudo_healthy) ** 2
            zs_list.append(residual)

        zs_list = torch.cat(zs_list, dim=0)
        my_mean = zs_list
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
        # if i > 0:
        #     break

    # postprocessing
    if not torch.count_nonzero(my_labels[0]):
        my_labels = my_labels[1:]
        my_volume = my_volume[1:]

    # modality max
    my_mask = torch.max(my_volume, dim=1)[0]

    # median filter
    mask_median3 = torch.clone(my_mask)
    mask_median3 = median_filter_3D(mask_median3, kernelsize=3)
    mask_median5 = torch.clone(my_mask)
    mask_median5 = median_filter_3D(mask_median5, kernelsize=5)

    # normalize
    my_mask = norm_tensor(my_mask)
    mask_median3 = norm_tensor(mask_median3)
    mask_median5 = norm_tensor(mask_median5)

    # contiguous
    my_labels = my_labels.contiguous()
    my_mask = my_mask.contiguous()
    mask_median3 = mask_median3.contiguous()
    mask_median5 = mask_median5.contiguous()

    # dice scores for different thresholds
    for key in dice_scores_mask:
        segmentation = torch.where(my_mask > key, 1.0, 0.0)
        segmentation = segmentation.type(torch.bool)
        dice_scores_mask[key].extend([float(x) for x in dice(segmentation, my_labels)])
        dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))

        segmentation3 = torch.where(mask_median3 > key, 1.0, 0.0)
        segmentation3 = segmentation3.type(torch.bool)
        dice_scores_mask_median3[key].extend(
            [float(x) for x in dice(segmentation3, my_labels)]
        )
        dice_scores_mask_median3[key] = np.mean(
            np.asarray(dice_scores_mask_median3[key])
        )

        segmentation5 = torch.where(mask_median5 > key, 1.0, 0.0)
        segmentation5 = segmentation5.type(torch.bool)
        dice_scores_mask_median5[key].extend(
            [float(x) for x in dice(segmentation5, my_labels)]
        )
        dice_scores_mask_median5[key] = np.mean(
            np.asarray(dice_scores_mask_median5[key])
        )

    # calculate AUPRC
    aupr = average_precision_score(my_labels.view(-1), my_mask.view(-1))
    my_auprs["aupr no median"].extend([aupr])
    dice_scores_mask["AUPRC"] = np.asarray(my_auprs["aupr no median"])

    aupr3 = average_precision_score(my_labels.view(-1), mask_median3.view(-1))
    my_auprs["aupr median3"].extend([aupr3])
    dice_scores_mask_median3["AUPRC"] = np.asarray(my_auprs["aupr median3"])

    aupr5 = average_precision_score(my_labels.view(-1), mask_median5.view(-1))
    my_auprs["aupr median5"].extend([aupr5])
    dice_scores_mask_median5["AUPRC"] = np.asarray(my_auprs["aupr median5"])

    # yen thresholding
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
    print("no median yen", dice_scores_mask["yen"])

    big_segmentation = torch.zeros_like(mask_median3)
    struc = generate_binary_structure(3, 1)

    for j, volume in enumerate(mask_median3):
        thr = threshold_yen(volume.numpy())
        segmentation = torch.where(volume > thr, 1.0, 0.0)
        big_segmentation[j] = segmentation
    big_segmentation = bin_dilation(big_segmentation, struc)
    dice_scores_mask_median3["yen"] = []
    dice_scores_mask_median3["yen"].extend(
        [float(x) for x in dice(big_segmentation, my_labels)]
    )
    dice_scores_mask_median3["yen"] = np.mean(
        np.asarray(dice_scores_mask_median3["yen"])
    )
    print("median3 yen", dice_scores_mask_median3["yen"])

    big_segmentation = torch.zeros_like(mask_median5)
    struc = generate_binary_structure(3, 1)

    for j, volume in enumerate(mask_median5):
        thr = threshold_yen(volume.numpy())
        segmentation = torch.where(volume > thr, 1.0, 0.0)
        big_segmentation[j] = segmentation
    big_segmentation = bin_dilation(big_segmentation, struc)
    dice_scores_mask_median5["yen"] = []
    dice_scores_mask_median5["yen"].extend(
        [float(x) for x in dice(big_segmentation, my_labels)]
    )
    dice_scores_mask_median5["yen"] = np.mean(
        np.asarray(dice_scores_mask_median5["yen"])
    )
    print("median5 yen", dice_scores_mask_median5["yen"])

    # convert to pandas
    df_mask = pd.DataFrame(dice_scores_mask, index=[0]).T
    df_mask_median3 = pd.DataFrame(dice_scores_mask_median3, index=[0]).T
    df_mask_median5 = pd.DataFrame(dice_scores_mask_median5, index=[0]).T

    # save to csv
    df_mask.to_csv(f"{args.data}_anoddpm_mask_nomedian.csv")
    df_mask_median3.to_csv(f"{args.data}_anoddpm_mask_median3.csv")
    df_mask_median5.to_csv(f"{args.data}_anoddpm_mask_median5.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="brats")
    args = parser.parse_args()
    main(args)
