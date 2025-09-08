import os
import re
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import skvideo.io
import scipy.io
import numpy as np
from skimage import color
from skimage.io import imsave
from skimage.transform import resize

AVAILABLE_GT_SYMBOL = "ü"
AVAILABLE_GT_ROW = "LV Wall Ground-truth Segmentation Masks"
DATASET_INFO_FILENAME = r"C:\Users\Administrator\Downloads\HMC-QU\HMC_QU.xlsx"
IMG_RELATIVE_PATH = "HMC-QU Echos\HMC-QU Echos"
GT_RELATIVE_PATH = "LV Ground-truth Segmentation Masks"


def resize_image(image, size):
    """Resize image to the given size."""
    return resize(image, size, anti_aliasing=True, preserve_range=True)


def process_and_save_images(
    patient_list: List[str], path: Path, dataset_info: pd.DataFrame, image_dir: Path, label_dir: Path, image_size: int
):
    """Process and save images and labels to folders.

    Args:
        patient_list: List of patient IDs.
        path: Path to the raw data.
        dataset_info: DataFrame containing dataset information.
        image_dir: Directory to save processed images.
        label_dir: Directory to save processed labels.
        image_size: Size to resize the images.
    """
    for patient in patient_list:
        # Read image and ground truth
        img = skvideo.io.vread(str(path / IMG_RELATIVE_PATH / (patient + ".avi")))
        gt = scipy.io.loadmat(str(path / GT_RELATIVE_PATH / ("Mask_" + patient + ".mat")))['predicted']

        # Retrieve reference frame and end of cycle
        reference_frame = dataset_info.loc[patient]['Reference Frame']
        end_of_cycle = dataset_info.loc[patient]['End of Cycle']

        # Adjust frames (convert from MATLAB 1-based index to Python 0-based index)
        img = img[reference_frame - 1:end_of_cycle]

        # Process and save each frame
        for frame_id, frame in enumerate(img):
            # Resize and convert to grayscale
            frame_resized = color.rgb2gray(resize_image(frame, (image_size, image_size)))

            # Save processed frame
            frame_filename = f"{patient}_{frame_id:04d}.jpg"
            imsave(str(image_dir / frame_filename), frame_resized.astype(np.uint8))

        # Process and save ground truth masks
        for frame_id, mask in enumerate(gt):
            mask_resized = resize_image(mask, (image_size, image_size))
            mask_normalized = (mask_resized * 255).astype(np.uint8)  # Normalize mask to 0-255 range
            # Save processed mask
            mask_filename = f"{patient}_{frame_id:04d}.jpg"
            imsave(str(label_dir / mask_filename), mask_normalized.astype(np.uint8))


def main():
    """Main function to parse arguments and process the dataset."""
    parser = argparse.ArgumentParser(
        description="Script to process medical images and ground truth into image and label folders."
    )
    parser.add_argument("--path", type=Path, default=r'C:\Users\Administrator\Downloads\HMC-QU', help="Path to the raw dataset folder.")
    parser.add_argument("--output", type=Path, default=r'C:\Users\Administrator\Desktop\HMC-QU', help="Path to save the processed images and labels.")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize the images and labels.")

    args = parser.parse_args()

    # Load dataset information
    dataset_info = pd.read_excel(args.path / DATASET_INFO_FILENAME, index_col="ECHO")

    columns = [
        "SEG1",
        "SEG2",
        "SEG3",
        "SEG5",
        "SEG6",
        "SEG7",
        "Reference Frame",
        "End of Cycle",
        "LV Wall Ground-truth Segmentation Masks",
    ]
    dataset_info.columns = columns
    dataset_info = dataset_info[dataset_info[AVAILABLE_GT_ROW] == AVAILABLE_GT_SYMBOL]  # Only keep samples with GT

    # Create output directories
    image_dir = args.output / "images"
    label_dir = args.output / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Process and save images and labels
    patient_list = list(dataset_info.index.values)
    process_and_save_images(patient_list, args.path, dataset_info, image_dir, label_dir, args.image_size)


if __name__ == "__main__":
    main()
