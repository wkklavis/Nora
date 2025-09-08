import os
import nibabel as nib
import numpy as np
from PIL import Image


def process_nifti_files(base_dir, output_dir):
    # Define paths for images and labels folders
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    # Create directories if they do not exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Traverse all patient folders
    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        if os.path.isdir(patient_path):
            for file_name in os.listdir(patient_path):
                if file_name.endswith(".nii.gz"):
                    file_path = os.path.join(patient_path, file_name)

                    # Load the NIfTI file
                    nifti_img = nib.load(file_path)
                    data = nifti_img.get_fdata()

                    # Extract the first frame if it's a multi-frame file
                    if data.ndim == 3:
                        image_data = data[:, :, 0]
                    else:
                        image_data = data

                    # Normalize the image to 0-255 only for gt files
                    if "gt" in file_name.lower():
                        image_data = np.where(image_data == 2, 255, 0)

                    # Convert to PIL Image
                    image = Image.fromarray(image_data.astype(np.uint8))

                    # Rotate image 90 degrees clockwise
                    image = image.rotate(-90, expand=True)

                    # Resize image to 256x256
                    image = image.resize((256, 256))

                    # Determine output path based on file type
                    if "gt" in file_name.lower():
                        save_path = os.path.join(labels_dir, file_name.replace(".nii.gz", ".jpg"))
                    else:
                        save_path = os.path.join(images_dir, file_name.replace(".nii.gz", ".jpg"))

                    # Save the image
                    image.save(save_path)


if __name__ == "__main__":
    base_directory = r"C:\Users\Administrator\Downloads\CAMUS_public\database_nifti"
    output_directory = r"C:\Users\Administrator\Desktop\CAMUS15"
    process_nifti_files(base_directory, output_directory)
