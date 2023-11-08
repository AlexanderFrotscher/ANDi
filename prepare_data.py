__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


"""
This code is inspired by @FeliMe download_data.py
https://github.com/FeliMe/brain_sas_baseline
"""

import argparse
import os
import shutil
from glob import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from registrator import MRIRegistrator


class ShiftsHandler():
    def __init__(self, args):
        
        if args.register:
            self.register_Shifts(args)
        elif args.hist:
            self.histogram_matching(args)
        else:
            self.prepare_Shifts(args)

    @staticmethod
    def prepare_Shifts(args):
        """Puts all the files in one patient folder"""

        # First over all possible data sets manually, e.g., msseg, best, ljubljana has to be args.dataset_path
        parent = args.dataset_path.split('/')[0:-1]
        data_set_name = args.dataset_path.split('/')[-1]
        parent = '/'.join(parent)
        target = parent + '/patients'
        folders = glob(f'{args.dataset_path}/*')
        for folder in folders:  # filter out unsupervised of msseg
            if folder.find('unsu') > -1:
                pass
            else:
                sub_folders = glob(f'{folder}/*')
                my_folder = sub_folders[0]
                list_patients = os.listdir(my_folder) # get number of patients in split of data set
                patients = [x.split('_')[0] for x in list_patients]

                # Iterate over all modalities, masks, etc.
                for sub in sub_folders:
                    folder_name = sub.split('\\')[-2]
                    if sub.find('indiv') > -1:  # filter out the individual annotators
                        pass
                    else:
                        for patient in patients:
                            scan = glob(f"{sub}/{patient}_*")[0]
                            final_folder = os.path.join(target,f'{data_set_name}_{folder_name}_{patient}')
                            os.makedirs(final_folder,exist_ok=True)
                            shutil.copy(scan,final_folder)


    @staticmethod
    def register_Shifts(args):
        print("Registering Shifts_MS")

        # Get all files
        files = glob(f"{args.dataset_path}/*/*_T1_*")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        template_path = os.path.join('/mnt/qb/work/baumgartner/bkc035', 'BrainAtlas/sri24_spm8/templates/T1_brain.nii')
        registrator = MRIRegistrator(template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path [:path.rfind("T1")]
            # Transform T2 image
            path = base + "T2_isovox.nii.gz"
            save_path = base + "t2.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='float32'
            )
            # Transform FLAIR image
            path = base + "FLAIR_isovox.nii.gz"
            save_path = base + "flair.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='float32'
            )

            # Transform T1CE image
            path = base + "T1CE_isovox.nii.gz"
            save_path = base + "t1ce.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='float32'
            )

            # Transform segmentation
            path = base + "gt_isovox.nii.gz"
            save_path = base + "seg.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )

    @staticmethod
    def histogram_matching(args):
        """Matches the histograms to a source volume for all modalities"""
        id_ = 'BraTS2021_00000'
        root_path = '/mnt/qb/baumgartner/rawdata/BraTS2021_Training_Data'
        data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz","_t2.nii.gz"]
        brats_images = []
        for data_type in data_types:
            img_path = os.path.join(root_path,id_, id_ + data_type)
            img = sitk.GetImageFromArray(np.asarray(nib.load(img_path).dataobj,dtype=float))
            brats_images.append(img)

        patients = glob(f'{args.dataset_path}/*')
        for patient in patients:  # filter out unsupervised of msseg
            files = glob(f'{patient}/*')
            # Iterate over all modalities
            for file in files:
                if file.find('_flair.nii.gz') > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(np.asarray(volume.dataobj,dtype=float))
                    transformed = sitk.GetArrayFromImage(sitk.HistogramMatching(img, brats_images[0]))
                    save_nii(file,transformed,volume.affine,dtype='float32')
                elif file.find('_t1.nii.gz') > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(np.asarray(volume.dataobj,dtype=float))
                    transformed = sitk.GetArrayFromImage(sitk.HistogramMatching(img, brats_images[1]))
                    save_nii(file,transformed,volume.affine,dtype='float32')
                elif file.find('_t1ce.nii.gz') > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(np.asarray(volume.dataobj,dtype=float))
                    transformed = sitk.GetArrayFromImage(sitk.HistogramMatching(img, brats_images[2]))
                    save_nii(file,transformed,volume.affine,dtype='float32')
                elif file.find('_t2.nii.gz') > -1:
                    volume = nib.load(file)
                    img = sitk.GetImageFromArray(np.asarray(volume.dataobj,dtype=float))
                    transformed = sitk.GetArrayFromImage(sitk.HistogramMatching(img, brats_images[3]))
                    save_nii(file,transformed,volume.affine,dtype='float32')
                else:
                    pass
                        

def save_nii(path, img, affine, dtype):
        nib.save(nib.Nifti1Image(img.astype(dtype), affine), path)

def select_data(args):
    if args.dataset == 'Shifts':
        ShiftsHandler(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'Shifts'
    args.dataset_path = '/mnt/qb/work/baumgartner/bkc035/shifts_data/patients'
    #args.dataset_path = './data/Shifts_MS/patients'
    args.register = False
    args.hist = True
    # Add this to handle ~ in path variables
    if args.dataset_path:
        args.dataset_path = os.path.expanduser(args.dataset_path)

    select_data(args)
