import numpy as np
from scipy.ndimage import zoom
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm


def resize_array(arr, new_size=(128, 128)):
    """Resize a numpy array using scipy zoom."""
    zoom_factors = (1, new_size[0] / arr.shape[1], new_size[1] / arr.shape[2])
    return zoom(arr, zoom_factors, order=1)


def process_file(file_path, output_dir):
    """Process a single .npy file and save it to the output directory."""
    arr = np.load(file_path)
    resized_arr = resize_array(arr)
    np.save(os.path.join(output_dir, os.path.basename(file_path)), resized_arr)


def parallel_process_files(input_dir, output_dir, max_workers=12):
    """Process all .npy files in the input directory in parallel and save them to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npy")
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Using tqdm for progress bar
        list(
            tqdm(
                executor.map(process_file, file_paths, [output_dir] * len(file_paths)),
                total=len(file_paths),
            )
        )


# Parameters
input_dir = "/scratch_local/jkapoor83-4570786/BraTS2021_Training_Data/healthy_slices"  # Update with your input folder path
output_dir = "/mnt/qb/work/macke/jkapoor83/brats_data_slices"  # Update with your output folder path
os.makedirs(output_dir, exist_ok=True)

parallel_process_files(input_dir, output_dir)
