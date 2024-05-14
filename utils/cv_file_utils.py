import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import logging
import torch

IMAGE_EXTENSIONS = [".png", ".jpg", "JPG", ".jpeg", ".tif'", ".tiff", ".bmp"]


def load_displacement_maps_from_directory(path, preprocess=False, resize=False):
    """
    Load displacement maps from the specified directory
    """

    displacement_maps = []

    # Iterate through each subdirectory in the path
    for subdir, dirs, files in os.walk(path):
        for dir in sorted(dirs):
            full_dir_path = os.path.join(subdir, dir)

            # Load the displacement maps
            displacement_maps += load_displacement_maps(full_dir_path, preprocess=preprocess, resize=resize)

    print(f'Number of Ground Displacement Maps: {len(displacement_maps)}')

    return displacement_maps


def load_crack_displacement_maps_from_directory(path, preprocess=False):
    """
    Load crack displacement maps from the specified directory
    """

    # Get all crack displacement maps
    crack_d_map_paths = get_image_paths(path)

    crack_d_maps = []

    for crack_d_map_path in crack_d_map_paths:
        # Load a crack displacement map
        crack_d_map = load_displacement_map(crack_d_map_path, preprocess=preprocess)

        crack_d_maps.append(crack_d_map)

    return crack_d_maps


def load_displacement_maps(path, preprocess=False, resize=False, apply_clahe=False):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")

    displacement_maps = []

    # Collect all .png and .tif files and sort them by filename
    map_paths = sorted(glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.tif')),
                       key=lambda x: os.path.basename(x))

    for map_path in tqdm(map_paths, desc="Loading displacement maps"):
        # Load displacement map as a grayscale image
        displacement_map = load_displacement_map(
            map_path, preprocess=preprocess, resize=resize, apply_clahe=apply_clahe)

        displacement_maps.append(displacement_map)

        logging.info(f"File: {map_path} loaded successfully.")

    return displacement_maps


def load_displacement_map(d_map_path, preprocess=False, resize=False, apply_clahe=False):
    """
    Load a glyph image from the specified path.
    """
    d_map = cv2.imread(d_map_path, cv2.IMREAD_GRAYSCALE)

    if d_map is None:
        logging.warning(f"Could not load {d_map_path}")
        return None

    if preprocess:
        d_map = preprocess_displacement_map(d_map, apply_clahe=apply_clahe)

    if resize:
        d_map = resize_and_pad(d_map)

    return d_map


def load_displacement_map_as_tensor(d_map_path):
    """
    Loads a displacement map image from the file path and returns it as a torch tensor.

    :param d_map_path: The path to the image file
    :return: The image as a PIL Image
    """

    image = cv2.imread(d_map_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the pixel values to the range [0, 1]
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return torch.from_numpy(image).float().unsqueeze(0)


def transform_displacement_map_to_tensor(image):
    """
    Converts a displacement map image into a torch tensor, normalizing the data.

    :param image: The image array
    :return: Normalized tensor
    """
    image_tensor = torch.from_numpy(image).float()

    # Normalize the image tensor to [0, 1] if it's not already
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

    return image_tensor.unsqueeze(0)


def preprocess_displacement_map(d_map, apply_clahe=False):
    """
    Preprocess the displacement map for training.
    """
    # Convert to 8-bit image, necessary for CLAHE
    # if d_map.dtype != np.uint8:
    #     # Normalize the image to 0-255 and convert to uint8
    #     d_map = cv2.normalize(d_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    #     d_map = np.uint8(d_map)

    # Median Filtering for Noise Reduction
    d_map = cv2.medianBlur(d_map, 5)

    # Histogram Equalization for Contrast Enhancement
    # d_map = cv2.equalizeHist(d_map)

    # Apply CLAHE for contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=5.0,
            tileGridSize=(8, 8))

        d_map = clahe.apply(d_map)

    # Normalizing the pixel values to the range [0, 1]
    d_map = cv2.normalize(d_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Round the values to three decimal places
    d_map = np.round(d_map, 2)

    return d_map


def resize_and_pad(img, target_size=(512, 512)):
    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Set the canvas with the correct dtype from the start
    canvas = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized

    return canvas


# def resize_with_aspect_ratio_2(image, target_size=(512, 512), default_background_value=0):
#     """
#     Resize the image while preserving the aspect ratio, with dynamic background calculation
#     and enhanced interpolation method choice.
#
#     :param image: Input 2D numpy array (single-channel or multi-channel).
#     :param target_size: Desired output size as a tuple (height, width).
#     :param default_background_value: Default background pixel value used for padding.
#     :return: Resized image as a numpy array with the specified target size.
#     """
#
#     # Compute the aspect ratio of the image and the target size
#     h, w = image.shape[:2]
#     target_h, target_w = target_size
#
#     # Compute scaling factors and new dimensions
#     scaling_factor = min(target_w / w, target_h / h)
#     new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
#
#     # Determine interpolation method
#     if scaling_factor > 1:
#         interpolation = cv2.INTER_LANCZOS4  # Enhanced method for upsampling
#     else:
#         interpolation = cv2.INTER_AREA  # Best choice for downsampling
#
#     # Resize the image
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
#
#     # Dynamic background value calculation based on edge pixels
#     if len(image.shape) == 3:
#         edge_pixels = np.concatenate([image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]], axis=0)
#         background_value = np.median(edge_pixels, axis=0)  # Median is more robust to outliers
#     else:
#         edge_pixels = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
#         background_value = np.median(edge_pixels)  # Median value of the edges
#
#     # Prepare the new image array with the calculated background value
#     if len(image.shape) == 3:
#         new_image = np.full((target_h, target_w, image.shape[2]), background_value, dtype=image.dtype)
#     else:
#         new_image = np.full((target_h, target_w), background_value, dtype=image.dtype)
#
#     # Calculate the placement position
#     top_left_x = (target_w - new_w) // 2
#     top_left_y = (target_h - new_h) // 2
#
#     # Place the resized image onto the new_image array
#     new_image[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_image
#
#     return new_image


def save_paired_images(input_d_map, target_d_map, input_path, target_path, set_index, pair_index):
    """
    Save a pair of depth images (x and y) to the specified paths.
    """
    # Construct unique filenames for the x and y images
    filename_input = f"i_pair_{set_index}_{pair_index}.png"
    filename_target = f"t_pair_{set_index}_{pair_index}.png"

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


####################################################################################################
# Color Image Utilities
# - Load inscriptions images
####################################################################################################
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
