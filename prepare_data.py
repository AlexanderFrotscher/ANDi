import argparse
from glob import glob
import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tqdm import tqdm

from registrator import MRIRegistrator


class WMHHandler():
    def __init__(self, args):
        """ Download data from https://wmh.isi.uu.nl/data/
        Your args.dataset_path should contain the following elements now:
        Amsterdam_GE3T.zip
        Singapore.zip
        Utrecht.zip
        """
        if args.dataset_path is None:
            args.dataset_path = os.path.join('./data', 'WMH')

        if args.skull_strip:
            self.skull_stripWMH(args)
        elif args.register:
            self.register_WMH(args)
        else:
            self.prepare_WMH(args)

    @staticmethod
    def prepare_WMH(args):
        """Puts all the files in the correct folder and renames them correctly"""

        # Unzip files
        zip_files = ['Amsterdam_GE3T.zip', 'Singapore.zip', 'Utrecht.zip']
        zip_files = [os.path.join(args.dataset_path, z) for z in zip_files]

        for zip_file in zip_files:
            # If file is missing, throw an error
            if not os.path.exists(zip_file):
                raise RuntimeError(f"No file found at {zip_file}")

            # Extract zip
            print(f"Extracting {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(args.dataset_path)

        # Rename anomaly segmentation files
        seg_files = glob(f"{args.dataset_path}/*/*/wmh.nii.gz")
        for seg_file in seg_files:
            target = seg_file[:seg_file.rfind("wmh.nii.gz")] + 'orig/anomaly_segmentation_unregistered.nii.gz'
            shutil.copy(seg_file, target)

    @staticmethod
    def skull_stripWMH(args):
        # Get list of all files
        files = glob(f"{args.dataset_path}/*/*/orig/T1.nii.gz")
        print(f"Found {len(files)} files.")

        # Run ROBEX
        #strip_skull_ROBEX(files)

        files_stripped = glob(f"{args.dataset_path}/*/*/orig/T1_stripped.nii.gz")
        print(f"Found {len(files_stripped)} stripped files.")
        for fi in files_stripped:
            # Strip FLAIR based on T1
            folder = '/'.join(fi.split('/')[:-1])
            f_flair = os.path.join(folder, "FLAIR.nii.gz")
            f_flair_stripped = os.path.join(folder, "FLAIR_stripped.nii.gz")

            # Load files
            data = nib.load(f_flair, keep_file_open=False)
            flair = data.get_fdata(caching='unchanged', dtype=np.float32).astype(np.short)
            t1 = nib.load(fi, keep_file_open=False).get_fdata(
                          caching='unchanged', dtype=np.float32).astype(np.short)

            # Strip FLAIR
            flair_stripped = flair * np.where(t1 > 0, 1, 0)

            # Save stripped FLAIR image
            nib.save(nib.Nifti1Image(flair_stripped.astype(np.short), data.affine), f_flair_stripped)


    @staticmethod
    def register_WMH(args):
        print("Registering WMH")

        # Get all files
        files = glob(f"{args.dataset_path}/*/*/orig/T1_stripped.nii.gz")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        # template_path = os.path.join(DATAROOT, 'BrainAtlases/mni_icbm152_nlin_sym_09a/t1_stripped.nii')
        template_path = os.path.join('./data', 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')
        # registrator = SitkRegistrator(template_path)
        registrator = MRIRegistrator(template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path [:path.rfind("T1")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform FLAIR image
            path = base + "FLAIR_stripped.nii.gz"
            save_path = base + "FLAIR_stripped_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )
            # Transform segmentation
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii.gz")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )


class ShiftsHandler():
    def __init__(self, args):
        
        if args.register:
            self.register_Shifts(args)
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
        template_path = os.path.join('./data', 'BrainAtlas/sri24_spm8/templates/T1_brain.nii')
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


def select_data(args):
    if args.dataset == 'WMH':
        WMHHandler(args)
    elif args.dataset == 'Shifts':
        ShiftsHandler(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'Shifts'
    args.dataset_path = '/mnt/qb/work/baumgartner/bkc035/shifts_data/patients'
    #args.dataset_path = None
    args.register = True
    # Add this to handle ~ in path variables
    if args.dataset_path:
        args.dataset_path = os.path.expanduser(args.dataset_path)

    select_data(args)