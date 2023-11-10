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
from skimage.filters import threshold_yen
from scipy.ndimage import generate_binary_structure

from utils import *


def main():
    torch.manual_seed(73)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #args.dataset_path = "/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data"
    #args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/scans_val.csv"
    args.dataset_path = "/mnt/qb/work/baumgartner/bkc035/shifts_data/patients"
    args.path_to_csv = "/mnt/qb/work/baumgartner/bkc035/shifts_out.csv"
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
    dice_scores_mask_median = {i: [] for i in threshold_diff}
    my_auprs = {i: [] for i in ["aupr no median", "aupr"]}
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
            my_mask = torch.max(my_volume, dim=1)[0]
            mask_median = torch.clone(my_mask)
            mask_median = median_filter_3D(mask_median, kernelsize=3)
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
                dice_scores_mask[key].extend([float(x) for x in dice(segmentation, my_labels)])
                dice_scores_mask[key] = np.mean(np.asarray(dice_scores_mask[key]))
                dice_scores_mask_median[key].extend([float(x) for x in dice(my_mask2, my_labels)])
                dice_scores_mask_median[key] = np.mean(np.asarray(dice_scores_mask_median[key]))

            big_segmentation = torch.zeros_like(my_mask)
            struc = generate_binary_structure(3,1)


            for j, volume in enumerate(my_mask):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                big_segmentation[j] = segmentation
            big_segmentation = bin_dilation(big_segmentation, struc)
            dice_scores_mask['yen'] = []
            dice_scores_mask['yen'].extend([float(x) for x in dice(big_segmentation, my_labels)])
            dice_scores_mask['yen'] = np.mean(np.asarray(dice_scores_mask['yen']))


            for j, volume in enumerate(mask_median):
                thr = threshold_yen(volume.numpy())
                segmentation = torch.where(volume > thr, 1.0, 0.0)
                big_segmentation[j] = segmentation
            big_segmentation = bin_dilation(big_segmentation, struc)
            dice_scores_mask_median['yen'] = []
            dice_scores_mask_median['yen'].extend([float(x) for x in dice(big_segmentation, my_labels)])
            dice_scores_mask_median['yen'] = np.mean(np.asarray(dice_scores_mask_median['yen']))

            dice_scores_mask['AUPRC'] = np.asarray(my_auprs["aupr no median"])
            dice_scores_mask_median['AUPRC'] = np.asarray(my_auprs["aupr"])
            df_mask = pd.DataFrame(dice_scores_mask,index=[0]).T
            df_mask2 = pd.DataFrame(dice_scores_mask_median, index=[0]).T
            df_mask.to_csv("/mnt/qb/work/baumgartner/bkc035/dae_result.csv")
            df_mask2.to_csv("/mnt/qb/work/baumgartner/bkc035/dae_median.csv")


if __name__ == "__main__":
    main()
