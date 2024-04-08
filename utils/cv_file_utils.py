import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import logging

IMAGE_EXTENSIONS = [".png", ".jpg", "JPG", ".jpeg", ".tif'", ".tiff", ".bmp"]


####################################################################################################
# Load a displacement map from the specified path
####################################################################################################
def load_displacement_map(d_map_path):
    """
    Load a glyph image from the specified path.
    """
    image = cv2.imread(d_map_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        logging.warning(f"Could not load {d_map_path}")
        return None

    # Normalize the pixel values to the range [0, 1]
    normalized_image = cv2.normalize(
        image,
        None,
        alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F)

    # Round the values to three decimal places
    rounded_image = np.round(normalized_image, 3)

    return rounded_image


def load_displacement_maps_from_directory(path):
    """
    Load displacement maps from the specified directory
    """

    displacement_maps = []

    # Iterate through each subdirectory in the path
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            full_dir_path = os.path.join(subdir, dir)

            # Load the displacement maps
            displacement_maps += load_displacement_maps(full_dir_path)

    print(f'Number of Ground Displacement Maps: {len(displacement_maps)}')

    return displacement_maps


def load_displacement_maps(path, target_size=(512, 512)):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")

    displacement_maps = []
    map_paths = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.tif'))
    map_paths = sorted(map_paths)

    for map_path in tqdm(map_paths, desc="Loading displacement maps"):
        # Load displacement map as a grayscale image
        displacement_map = load_displacement_map(map_path)

        # Resize displacement map while preserving aspect ratio
        # resized_map = resize_with_aspect_ratio(displacement_map, target_size)

        displacement_maps.append(displacement_map)

        logging.info(f"File: {map_path} loaded successfully.")

    return displacement_maps


def resize_with_aspect_ratio(image, target_size=(512, 512), background_value=0):
    """
    Resize the image while preserving the aspect ratio
    """

    # Compute the aspect ratio of the image and the target size
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Compute scaling factors and new dimensions
    scaling_factor = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)

    if scaling_factor > 1:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if len(image.shape) == 3:
        # Color image
        new_image = np.full((target_h, target_w, image.shape[2]), background_value, dtype=image.dtype)
    else:
        # Grayscale image
        new_image = np.full((target_h, target_w), background_value, dtype=image.dtype)

    # Calculate where to place the resized image in the new_image array
    top_left_x = (target_w - new_w) // 2
    top_left_y = (target_h - new_h) // 2

    # Place the resized image onto the new_image array
    if len(image.shape) == 3:
        new_image[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w, :] = resized_image
    else:
        new_image[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_image

    return new_image


def save_paired_images(input_d_map, target_d_map, input_path, target_path, pair_index):
    """
    Save a pair of depth images (x and y) to the specified paths.
    """
    # Construct unique filenames for the x and y images
    filename_input = f"i_pair_{pair_index}.png"
    filename_target = f"t_pair_{pair_index}.png"

    # Rescale the images from [0, 1] to [0, 255]
    input_d_map_rescaled = (input_d_map * 255).astype(np.uint8)
    target_d_map_rescaled = (target_d_map * 255).astype(np.uint8)

    # Save the images
    cv2.imwrite(os.path.join(input_path, filename_input), input_d_map_rescaled)
    cv2.imwrite(os.path.join(target_path, filename_target), target_d_map_rescaled)


def validate_directories(paths):
    # Validate and create directories
    for path in paths:
        print(f"Validating directory: {path}")

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.access(path, os.W_OK):
            raise Exception(f"Directory {path} is not writable.")

    print('Directories validated successfully.')


def get_image_paths(directory):
    return [
        os.path.join(directory, fname)

        for fname in sorted(os.listdir(directory))

        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ]


def load_inscriptions_images(path, target_size=(512, 512)):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")

    inscriptions_images = []
    images_paths = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg')) + glob.glob(
        os.path.join(path, '*.JPG'))

    for image_path in tqdm(images_paths, desc="Loading inscriptions images"):
        try:
            inscription_image = cv2.imread(image_path)
            inscription_image = cv2.cvtColor(inscription_image, cv2.COLOR_BGR2RGB)

            if inscription_image is None:
                logging.warning(f"Could not load {image_path}")
                continue

            # Resize displacement map while preserving aspect ratio
            # resized_image = resize_with_aspect_ratio(inscription_image, target_size)

            # Append the image along with its path
            inscriptions_images.append((image_path, inscription_image))

            logging.info(f"File: {image_path} loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")

    return inscriptions_images
