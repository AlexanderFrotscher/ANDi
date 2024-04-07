__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import argparse
import os
import shutil
from glob import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from registrator import MRIRegistrator


class ShiftsHandler:
    def __init__(self, args):
        if args.register:
            self.register_Shifts(args)
        elif args.norm:
            self.histogram_matching(args)
        else:
            self.prepare_Shifts(args)

    @staticmethod
    def prepare_Shifts(args):
        """This method generates a patient folder for the Shifts data sets.
        Every patient has a seperate id folder that contains all files.

        Parameters
        ----------
        args : _type_
            The input files needed
        """
        # First over all possible data sets manually, e.g., msseg, best, ljubljana has to be args.data_set
        parent = args.data_set.split("/")[0:-1]
        data_set_name = args.data_set.split("/")[-1]
        parent = "/".join(parent)
        target = parent + "/patients"
        folders = glob(f"{args.data_set}/*")
        for folder in folders:  # filter out unsupervised of msseg
            if folder.find("unsu") > -1:
                pass
            else:
                sub_folders = glob(f"{folder}/*")
                my_folder = sub_folders[0]
                list_patients = os.listdir(
                    my_folder
                )  # get number of patients in split of data set
                patients = [x.split("_")[0] for x in list_patients]

                # Iterate over all modalities, masks, etc.
                for sub in sub_folders:
                    sub = sub.replace('\\', '/')
                    folder_name = sub.split("/")[-2]
                    if sub.find("indiv") > -1:  # filter out the individual annotators
                        pass
                    else:
                        for patient in patients:
                            scan = glob(f"{sub}/{patient}_*")[0]
                            final_folder = os.path.join(
                                target, f"{data_set_name}_{folder_name}_{patient}"
                            )
                            os.makedirs(final_folder, exist_ok=True)
                            shutil.copy(scan, final_folder)

    @staticmethod
    def register_Shifts(args):
        print("Registering Shifts_MS")

        # Get all files
        files = glob(f"{args.data_set}/*/*_T1_*")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        template_path = args.template
        registrator = MRIRegistrator(template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path[: path.rfind("T1")]
            # Transform T2 image
            path = base + "T2_isovox.nii.gz"
            save_path = base + "t2.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="float32",
            )
            # Transform FLAIR image
            path = base + "FLAIR_isovox.nii.gz"
            save_path = base + "flair.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="float32",
            )

            # Transform T1CE image
            path = base + "T1CE_isovox.nii.gz"
            save_path = base + "t1ce.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="float32",
            )

            # Transform segmentation
            path = base + "gt_isovox.nii.gz"
            save_path = base + "seg.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short",
            )

    @staticmethod
    def histogram_matching(args):
        id_ = args.input_volume.split("/")[-1]
        data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
        brats_images = []
        for data_type in data_types:
            img_path = os.path.join(args.input_volume, id_ + data_type)
            img = sitk.GetImageFromArray(
                np.asarray(nib.load(img_path).dataobj, dtype=float)
            )
            brats_images.append(img)

        patients = glob(f"{args.data_set}/*")
        for patient in patients:  # filter out unsupervised of msseg
            files = glob(f"{patient}/*")
            # Iterate over all modalities
            for file in files:
                if file.find("_flair.nii.gz") > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(
                        np.asarray(volume.dataobj, dtype=float)
                    )
                    transformed = sitk.GetArrayFromImage(
                        sitk.HistogramMatching(img, brats_images[0])
                    )
                    save_nii(file, transformed, volume.affine, dtype="float32")
                elif file.find("_t1.nii.gz") > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(
                        np.asarray(volume.dataobj, dtype=float)
                    )
                    transformed = sitk.GetArrayFromImage(
                        sitk.HistogramMatching(img, brats_images[1])
                    )
                    save_nii(file, transformed, volume.affine, dtype="float32")
                elif file.find("_t1ce.nii.gz") > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(
                        np.asarray(volume.dataobj, dtype=float)
                    )
                    transformed = sitk.GetArrayFromImage(
                        sitk.HistogramMatching(img, brats_images[2])
                    )
                    save_nii(file, transformed, volume.affine, dtype="float32")
                elif file.find("_t2.nii.gz") > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(
                        np.asarray(volume.dataobj, dtype=float)
                    )
                    transformed = sitk.GetArrayFromImage(
                        sitk.HistogramMatching(img, brats_images[3])
                    )
                    save_nii(file, transformed, volume.affine, dtype="float32")
                else:
                    pass


def save_nii(path, img, affine, dtype):
    nib.save(nib.Nifti1Image(img.astype(dtype), affine), path)


def select_data(args):
    ShiftsHandler(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the Shifts data set.")
    parser.add_argument(
        "-d",
        "--data_set",
        type=str,
        required=True,
        metavar="",
        help="The folder that contains the MRI-Volumes for the individual data sets, e.g., MSSEG folder.",
    )
    parser.add_argument(
        "-n",
        "--norm",
        type=bool,
        default=False,
        metavar="",
        help="Use histogram matching on the volumes in the folder given with the -d flag. The source volumes (patient id folder) has to be given with the -i flag.",
    )
    parser.add_argument(
        "-i",
        "--input_volume",
        type=str,
        metavar="",
        help="Path to the source volume used for histogram matching.",
    )
    parser.add_argument(
        "-r",
        "--register",
        type=bool,
        default=False,
        metavar="",
        help="Register the volumes in the data set folder given with -d to the SRI atlas. The template is given with the -t flag.",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        metavar="",
        help="Path to the T1 template (T1_brain.nii) of the SRI atlas.",
    )
    args = parser.parse_args()
    select_data(args)
