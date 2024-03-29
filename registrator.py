"""
Code from @FeliMe https://github.com/FeliMe/brain_sas_baseline/tree/main/utils/registrator.py
"""

import multiprocessing
import os
from time import time

import SimpleITK as sitk
from dipy.align.imaffine import AffineMap
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    TranslationTransform3D,
)
from dipy.viz import regtools
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm



class MRIRegistrator:
    def __init__(
        self,
        template_path=None,
        brain_mask_path=None,
        nbins=32,
        sampling_proportion=None,
        level_iters=[100, 10, 5],
        sigmas=[3.0, 1.0, 1.0],
        factors=[4, 2, 1],
        verbose=False,
        rotate = True
    ):
        """Class for a registrator to perform an affine registration based on
        mutual information with dipy.

        Args:
            template_path (str): path to the brain atlas file in NiFTI format
                                (.nii or .nii.gz)
            brain_mask_path (str): path to the brain mask for the template used
                                for skull stripping. Use None if no skull
                                stripping is needed for the template
            nbins (int): number of bins to be used to discretize the joint and
                        marginal probability distribution functions (PDF) to
                        calculate the mutual information.
            sampling_proportion (int): Value from 1 to 100. Percentage of voxels
                                    used to calculate the PDF. None is 100%
            level_iters (list of int): Number of optimization iterations at each
                                    resolution in the gaussian pyramid.
            sigmas (list of float): Standard deviations of the gaussian smoothing
                                    kernels in the pyramid.
            factors (list of int): Inverse rescaling factors for pyramid levels.
        """
        if not os.path.exists(template_path):
            raise RuntimeError("Download SRI ATLAS from https://www.nitrc.org/projects/sri24/")
        template_data = nib.load(template_path)
        self.template = template_data.get_fdata()
        self.template_affine = template_data.affine

        if self.template.ndim == 4:
            self.template = self.template.squeeze(-1)

        if rotate == True:
            self.template = np.rot90(self.template,2,axes=(0,1))
            center = np.array(self.template.shape) / 2.0
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = center
            rotation_matrix = np.diag([-1, -1, 1, 1])
            self.template_affine = np.dot(np.dot(self.template_affine, translation_matrix), rotation_matrix)

        if brain_mask_path is not None:
            mask = nib.load(brain_mask_path).get_fdata()
            self.template = self.template * mask

        self.nbins = nbins
        self.sampling_proportion = sampling_proportion
        self.level_iters = level_iters
        self.sigmas = sigmas
        self.factors = factors

        self.verbose = verbose

    def _print(self, str):
        if self.verbose:
            print(str)

    @staticmethod
    def save_nii(f, img, affine, dtype):
        nib.save(nib.Nifti1Image(img.astype(dtype), affine), f)

    @staticmethod
    def load_nii(path, dtype='short'):
        # Load file
        data = nib.load(path, keep_file_open=False)
        volume = data.get_fdata(caching='unchanged',
                                dtype=np.float32).astype(np.dtype(dtype))
        affine = data.affine
        return volume, affine

    @staticmethod
    def overlay(template, moving, transformer=None):
        # Matplotlib params
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['image.interpolation'] = 'nearest'

        if transformer is not None:
            moving = transformer.transform(moving)
        regtools.overlay_slices(template, moving, None,
                                0, 'Template', 'Moving')
        regtools.overlay_slices(template, moving, None,
                                1, 'Template', 'Moving')
        regtools.overlay_slices(template, moving, None,
                                2, 'Template', 'Moving')
        plt.show()

    def transform(self, img, save_path, transformation, affine, dtype):
        """Transform a scan given a transformation and save it.

        Args:
            path (str): Path to scan
            save_path (str): Path so save transformed scan
            transformation (AffineMap): Affine transformation map
            affine
            dtype (str): numpy datatype of transformed scan
        """
        # Save maybe
        if save_path is not None:
            img, _ = self.load_nii(img, dtype='float32')

        transformed = transformation.transform(img)

        # Find data type to save
        if (transformed - transformed.astype(np.short)).sum() == 0.:
            dtype = np.short
        else:
            dtype = np.dtype('<f4')

        self.save_nii(
            f=save_path,
            img=transformed,
            affine=affine,
            dtype=dtype
        )

    def register_batch(self, moving_list, num_cpus=None):
        """Register a list of NiFTI files and save the registration result
        with a '_registered.nii' suffix"""

        # Set number of cpus used
        num_cpus = os.cpu_count() if num_cpus is None else num_cpus

        # Split list into batches
        moving_batches = [list(p) for p in np.array_split(
            moving_list, num_cpus) if len(p) > 0]
        print(f"Using {len(moving_batches)} CPU cores for registration")

        # Start multiprocessing
        with multiprocessing.Pool(processes=num_cpus) as pool:
            temp = pool.starmap(
                self._register_batch,
                zip(moving_batches, range(len(moving_batches)))
            )

        transformations = {}
        for t in temp:
            transformations = {**transformations, **t}

        return transformations

    def _register_batch(self, moving_list, i_process):
        """Don't call yourself"""
        t_start = time()
        transformations = {}
        for i, path in enumerate(moving_list):
            save_path = path.split('_')[0:-2]
            save_path = '_'.join(save_path)
            save_path = save_path + '_t1.nii'
            if path.endswith('.gz'):
                save_path += '.gz'
            _, transformation = self(moving=path, save_path=save_path)

            transformations[path] = transformation

            print(f"Process {i_process} finished {i + 1} of"
                  f" {len(moving_list)} in {time() - t_start:.2f}s")
        return transformations

    def __call__(self, moving, moving_affine=None, save_path=None, show=False):
        """Register a scan

        Args:
            moving (np.array): 3D volume of a scan
            moving_affine (np.array): 4x4 affine transformation of volume2world
            show (bool): Plot the result
        """
        # Start timer
        t_start = time()

        # Maybe load moving image
        if isinstance(moving, str):
            moving, moving_affine = self.load_nii(moving, dtype="<f4")

        # First resample moving image to same resolution
        identity = np.eye(4)
        affine_map = AffineMap(identity,
                               self.template.shape, self.template_affine,
                               moving.shape, moving_affine)

        # Center of mass transform
        c_of_mass = transform_centers_of_mass(self.template, self.template_affine,
                                              moving, moving_affine)

        # Affine registration
        metric = MutualInformationMetric(self.nbins, self.sampling_proportion)
        affreg = AffineRegistration(metric=metric,
                                    level_iters=self.level_iters,
                                    sigmas=self.sigmas,
                                    factors=self.factors,
                                    verbosity=1 if self.verbose else 0)
        # 3D translational only transform
        self._print("3D translational transform")
        translation3d = TranslationTransform3D()
        translation = affreg.optimize(self.template, moving,
                                      translation3d, None,
                                      self.template_affine, moving_affine,
                                      starting_affine=c_of_mass.affine)

        # 3D rigid transform
        self._print("3D rigid transform")
        rigid3d = RigidTransform3D()
        rigid = affreg.optimize(self.template, moving, rigid3d, None,
                                self.template_affine, moving_affine,
                                starting_affine=translation.affine)

        # 3D affine transform
        self._print("3D affine transform")
        affine3d = AffineTransform3D()
        affine = affreg.optimize(self.template, moving, affine3d,
                                 None, self.template_affine,
                                 moving_affine,
                                 starting_affine=rigid.affine)

        registered = affine.transform(moving)
        transformation = affine

        self._print(f"Time for registration: {time() - t_start:.2f}s")

        if show:
            self.overlay(self.template, registered)
            plt.show()

        # Save maybe
        if save_path is not None:
            # Select the right datatype
            if np.abs(registered - registered.astype(np.short)).sum() == 0:
                dtype = 'short'
            else:
                dtype = "<f4"
            # Save
            self.save_nii(
                f=save_path,
                img=registered,
                affine=self.template_affine,
                dtype=dtype
            )

        return registered, transformation


class SitkRegistrator:
    def __init__(
        self,
        template_path=None,
    ):
        # Load fixed image
        if not os.path.exists(template_path):
            raise RuntimeError("Download SRI ATLAS from https://www.nitrc.org/projects/sri24/")
        self.FixedImage = sitk.ReadImage(template_path)


    def register_batch(self, moving_list):
        """Register a list of files"""
        transformations = {}
        for path in tqdm(moving_list):
            save_path = path.split('nii')[0][:-1] + '_registered.nii'
            if path.endswith('.gz'):
                save_path += '.gz'
            _, transformation = self(path, save_path=save_path)

            transformations[path] = transformation
        return transformations

    @staticmethod
    def transform(img, transformParameterMap, save_path=None):
        """Transform an image based on a known transformation

        Args:
            img (str or SimpleITK.SimpleITK.Image): Moving image
            transformParameterMap (SimpleITK.SimpleITK.ParameterMap)
            save_path (None or str): Rath to save transformed image

        Returns:
            resImage (SimpleITK.SimpleITK.Image): Transformed image
        """
        if isinstance(img, str):
            img = sitk.ReadImage(img)

        # Define transformation object
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)

        # Turn logging to console off
        transformixImageFilter.LogToConsoleOff()

        # Transform moving image
        transformixImageFilter.SetMovingImage(img)
        transformixImageFilter.Execute()
        resImage = transformixImageFilter.GetResultImage()

        # Save maybe
        if save_path is not None:
            sitk.WriteImage(resImage, save_path)

        return resImage

    def __call__(self, img, save_path=None, transform="affine"):
        """Transform an image based on a fixed atlas

        Args:
            img (str or SimpleITK.SimpleITK.Image): Moving image
            save_path (None or str): Rath to save transformed image
            transform (str): Type of transformation

        Returns:
            resImage (SimpleITK.SimpleITK.Image): Transformed image
            transformParameterMap (SimpleITK.SimpleITK.ParameterMap)
        """
        # Select registration method
        elastixImageFilter = sitk.ElastixImageFilter()

        # Turn logging to console off
        elastixImageFilter.LogToConsoleOff()

        # Set fixed image
        elastixImageFilter.SetFixedImage(self.FixedImage)

        # Set moving image
        if isinstance(img, str):
            img = sitk.ReadImage(img)
        elastixImageFilter.SetMovingImage(img)

        # Register moving image
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap(transform))
        elastixImageFilter.Execute()
        resImage = elastixImageFilter.GetResultImage()

        # Save maybe
        if save_path is not None:
            # Determine dtype
            short_img = sitk.GetArrayFromImage(sitk.Cast(img, sitk.sitkInt16))
            if np.abs(sitk.GetArrayFromImage(img) - short_img).sum() == 0:
                img = sitk.Cast(img, sitk.sitkInt16)
            # Write
            sitk.WriteImage(resImage, save_path)

        # Get transforms
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()

        return resImage, transformParameterMap
