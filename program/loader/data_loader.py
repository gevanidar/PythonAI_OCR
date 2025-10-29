import os
import numpy as np
from PIL import Image


def load_data_from_file(folder_path, filename, image_size):
    """
    Load the image data from the file name and convert it to a flat array using numpy.

    Args:
        folder_path (str): The path to the folder containing the file name filename
        filename (str): The name of the file
        image_size (Tuple(int, int)): New width and height of the image (used for downsizing the images)
    """
    image_path = os.path.join(folder_path, filename)

    img = Image.open(image_path).convert("L")

    img = img.resize(image_size)

    return np.array(img).flatten()


def load_data_from_root_folder(root_folder):
    """
    Load images in subfolders of root_folder

    Data folder strubture is expected to be 'root_folder/subfolder/images'
    Where each subfolder has the name of the label.

    Args:
        root_folder (str): The base folder containing all the labeled subfolders
    """

    images = []
    labels = []

    image_size = (16, 16)  # Resize he images to a lower size 16x16 pixels

    for image_folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, image_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".jpg"):
                continue

            try:
                img_array = load_data_from_file(folder_path, filename, image_size)
                label = image_folder

                # Add the image and the corresponding label
                images.append(img_array)
                labels.append(label)

            except Exception as e:
                print(f"Error when processing the file {filename}: {e}")
    return np.array(images), np.array(labels)
