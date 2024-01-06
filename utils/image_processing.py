import os
import sys
import cv2
import glob
import numpy as np
import configparser
import logging
import imgaug.augmenters as iaa

from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL         import Image

import albumentations as A

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif'", ".tiff", ".bmp"]

# project_path = "./"

# # Read the config file
# config = configparser.ConfigParser()
# config.read(project_path + 'config.ini')

# dataset_size='small'

# x_training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_X_TRAINING_DATASET_PATH']
# y_training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_Y_TRAINING_DATASET_PATH']

# Define a pipeline of augmentation operations for enhancing displacement maps
enhacement_augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.0, 0.0), contrast_limit = (-0.1, 0.1), always_apply=True),
    A.Sharpen(alpha = (0.8, 1.0), lightness = (1.0, 1.0), always_apply=True),
    A.Emboss(alpha = (0.9, 1.0), strength = (0.9, 1.0), always_apply = True)
])

# Define a pipeline of augmentation operations for damaging displacement maps
damaging_augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.0, 0.0), contrast_limit = (-0.1, 0.1), always_apply=True),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.RandomGamma(gamma_limit=(50, 150), p=0.5),
    A.Sharpen(alpha = (0.5, 0.8), lightness = (1.0, 1.0), always_apply=True),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=2, fill_value=0, p=0.5),
])

####################################################################################################
# Function to get image from paths
####################################################################################################
def get_image_paths(directory):
    return [
        os.path.join(directory, fname)

        for fname in sorted(os.listdir(directory))
        
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ]

####################################################################################################
# Add blur
# # Randomly choose between 3x3, 5x5, 7x7 kernel sizes
####################################################################################################
def add_blur(image, kernel_size=3, sigma=0, preserve_edges=True):
    if preserve_edges:
        # Use bilateral filtering to preserve edges. 
        # Adjust sigmaColor and sigmaSpace.
        blurred_map = cv2.bilateralFilter(image, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        # Apply standard Gaussian blur, 
        # allowing sigma to be calculated from the kernel size if sigma=0
        blurred_map = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    
    return blurred_map

####################################################################################################
    """
    Applies adaptive Gaussian blur to an inscription map to enhance eroded letters 
    while preserving essential details.
    
    Parameters:
    - image: The input map of the ancient inscription.
    - max_kernel_size: The maximum size of the Gaussian kernel for the blur.

    Returns:
    - The blurred inscription map.
    """
####################################################################################################
def add_adaptive_blur(image, kernel_size=3, low_threshold=50, high_threshold=150):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Create a copy of the map to work on
    blurred_image = gray_image.copy()

    # Detect edges to preserve them; the threshold values may be adjusted for displacement maps
    edges = cv2.Canny(gray_image, threshold1=low_threshold, threshold2=high_threshold)

    # Invert edges to create a mask for areas to blur
    mask = cv2.bitwise_not(edges)

    # Use adaptiveThreshold to identify regions with uniform intensity
    adaptive_mask = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, kernel_size, 2
    )

    # Combine masks to exclude edges and uniform regions from blurring
    combined_mask = cv2.bitwise_or(adaptive_mask, mask)
    
    # Apply a Gaussian blur on the entire image
    full_blur = cv2.GaussianBlur(blurred_image, (kernel_size, kernel_size), 0)

    # Where combined_mask is not set, replace blurred_image with full_blur
    blurred_image = np.where(combined_mask[..., None], blurred_image, full_blur)

    return blurred_image.squeeze().copy()


####################################################################################################
# Add noise
####################################################################################################
def add_noise(image, intensity = 0.5):
    # Randomly choose between Gaussian and Salt & Pepper noise
    noise_type = np.random.choice(['gaussian', 's&p'])

    if noise_type == 'gaussian':
        mean = 0
        sigma = intensity  # Adjust intensity based on the map's range
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = image + gauss
    else:  
        # Salt & Pepper
        s_vs_p = np.random.uniform(0.3, 0.7)
        amount = intensity / 10  # Adjust to be lower for displacement maps
        noisy = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy[tuple(coords)] = image.max()

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy[tuple(coords)] = image.min()

    # Clip to ensure the noisy image is within the valid range
    noisy = np.clip(noisy, image.min(), image.max()).astype(image.dtype)

    return noisy

####################################################################################################
# Add erosion
####################################################################################################
def erode_image(image, kernel_size_range=(2, 14), intensity=1.0, kernel_shape=cv2.MORPH_ELLIPSE, iterations_range=(2, 3)):
    # Validate and adapt kernel size range based on image size
    max_kernel_size = min(image.shape[:2]) // 2
    kernel_size_range = (max(1, kernel_size_range[0]), min(max_kernel_size, kernel_size_range[1]))
    iterations = np.random.randint(*iterations_range)
    
    # Determine kernel size based on intensity
    kernel_size = np.random.randint(*kernel_size_range)
    kernel_size = int(kernel_size * intensity)

    # Define erosion kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply erosion
    eroded = cv2.erode(image, kernel, iterations=iterations)

    return eroded

####################################################################################################
# Add dilate image
# 
####################################################################################################
def dilate_image(image, kernel_size_range=(1, 5), intensity=1.0, inscription_mask=None, iterations=1):
    # Randomly choose kernel size
    kernel_size = np.random.randint(*kernel_size_range)

    # Define dilation 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # If an inscription mask is provided, apply dilation selectively
    if inscription_mask is not None:
        masked_image = cv2.bitwise_and(image, image, mask=inscription_mask)

        dilated = cv2.dilate(masked_image, kernel, iterations=iterations)

        # Combine the dilated inscriptions with the original image
        dilated = cv2.max(image, dilated)
    else:
        dilated = cv2.dilate(image, kernel, iterations=iterations)
    
    return dilated

####################################################################################################
# The function takes an image as input and returns a skewed version of the image.
# Referece: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
####################################################################################################
def skew_image(image, x_skew_range=(0.7, 1.3), y_skew_range=(0.7, 1.5)):
    # Skew image
    rows, cols     = image.shape
    # Randomly choose skew factors
    x_skew = np.random.uniform(*x_skew_range)
    y_skew = np.random.uniform(*y_skew_range)

    pts1           = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2           = np.float32([[50 * x_skew, 50 * y_skew], [200 * x_skew, 50], [50, 200 * y_skew]])

    M              = cv2.getAffineTransform(pts1, pts2)
    skewed         = cv2.warpAffine(image, M, (cols, rows))
    return skewed

####################################################################################################
# stretch image
####################################################################################################
def stretch_image(image, x_factor_range=(0.8, 1.3), y_factor_range=None):
    # Generate factors
    x_factor = np.random.uniform(*x_factor_range)
    y_factor = np.random.uniform(*y_factor_range) if y_factor_range else x_factor

    # Stretch image
    return cv2.resize(image, None, fx=x_factor, fy=y_factor, interpolation=cv2.INTER_AREA)


####################################################################################################
# adjust brightness
####################################################################################################
def adjust_brightness(image, brightness_factor_range=(0.7, 1.3)):
    # Randomly choose brightness factor
    brightness_factor = np.random.uniform(*brightness_factor_range)

    # Adjust brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * brightness_factor, 0, 1).astype(np.float32)
    return cv2.merge((h, s, v))

####################################################################################################
# Convert to grayscale
####################################################################################################
def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale

####################################################################################################
# Simulate cracks
####################################################################################################
def simulate_cracks(image,  num_cracks_range=(2, 5), max_length=120, crack_types=["hairline", "wide", "branching"]):
    if image is None:
        logging.warning("Received None image in simulate_cracks")
        return None  # Or handle the None case differently based on your logic

    # Ensure the image is in grayscale for simplicity
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Number and types of cracks
    num_cracks = np.random.randint(*num_cracks_range)

    for _ in range(num_cracks):
        crack_type = np.random.choice(crack_types)
        if crack_type == "hairline":
            thickness = 2
        elif crack_type == "wide":
            thickness = np.random.randint(3, 8)
        elif crack_type == "branching":
            thickness = np.random.randint(2, 4)
            # Add branching crack logic here
            # Draw additional lines starting from random points on the main crack

        # Start point for the crack
        x_start, y_start = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])

        # Simulate crack propagation
        for _ in range(np.random.randint(1, max_length)):
            # Crack step size and angle
            #length = np.random.uniform(20, 100)  # Adjust as needed
            step_length = np.random.randint(3, 9)
            angle = np.random.uniform(0, 2 * np.pi)
            x_end = int(x_start + step_length * np.cos(angle))
            y_end = int(y_start + step_length * np.sin(angle))

            # Draw the crack segment
            cv2.line(image, (x_start, y_start), (x_end, y_end), (0), thickness)

            # Update start point
            x_start, y_start = x_end, y_end
            
            # Randomly decide if the crack should branch or stop
            if np.random.rand() < 0.1:  # 10% chance to branch or stop
                if np.random.rand() < 0.5:  # 50% of those 10% to branch
                    # Start a new crack segment from the current point
                    x_start, y_start = x_end, y_end
                else:
                    break  # Stop the crack propagation

    return image

####################################################################################################
# Simulate erosion
####################################################################################################
def simulate_erosion(image, erosion_intensity_range=(1, 5), max_erosion_iters=3):
    # Ensure the image is in grayscale for simplicity
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    erosion_intensity = np.random.randint(*erosion_intensity_range)
    erosion_iterations = np.random.randint(1, max_erosion_iters)

    # Define erosion kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_intensity, erosion_intensity))

    # Apply erosion
    eroded_image = cv2.erode(image, kernel, iterations=erosion_iterations)

    return eroded_image

####################################################################################################
# Simulate text fading
####################################################################################################
def simulate_text_fading(image, num_areas_range=(1, 5), area_size_range=(10, 40), fading_intensity_range=(0.3, 0.9)):
    # Convert image to float32 for processing
    faded_image = image.astype(np.float32)

    num_areas = np.random.randint(*num_areas_range)

    for _ in range(num_areas):
        w, h = np.random.randint(*area_size_range, size=2)
        x, y = np.random.randint(0, max(1, image.shape[1] - w)), np.random.randint(0, max(1, image.shape[0] - h))
        fading_intensity = np.random.uniform(*fading_intensity_range)

        # Creating a radial gradient
        y_indices, x_indices = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 1 - (distance / max_dist)

        # Apply fading with gradient
        faded_image[y:y+h, x:x+w] *= fading_intensity + (1 - fading_intensity) * gradient

    # Convert back to original data type
    faded_image = np.clip(faded_image, 0, 255).astype(image.dtype)

    return faded_image




####################################################################################################
# Simulate weathering fffects
####################################################################################################
def simulate_weathering_effects(image):
    # Placeholder for weathering simulation logic
    return image

####################################################################################################
# Simulate bumps and scratches
####################################################################################################
def simulate_bumps_and_scratches(image, intensity=0.5, scratches=True, bumps=False):
    # Create a copy of the image to work on
    simulated_image = image.copy()

    # Calculate the number of features based on intensity
    num_bumps     = int(6 * intensity)
    num_scratches = int(6 * intensity)

    if bumps == True:
        # Adding bumps
        for _ in range(num_bumps):
            x, y      = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            radius    = np.random.randint(1, 6)
            bump_mask = np.zeros_like(image)

            cv2.circle(bump_mask, (x, y), radius, (0), -1)

            bump_blurred    = cv2.GaussianBlur(bump_mask, (0, 0), radius / 2)
            simulated_image = cv2.addWeighted(simulated_image, 1, bump_blurred, intensity, 0)

    if scratches == True:
        # Adding scratches
        for _ in range(num_scratches):
            x_start, y_start = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            x_end, y_end     = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            thickness        = np.random.randint(1, 3)

            cv2.line(simulated_image, (x_start, y_start), (x_end, y_end), (0), thickness)

    return simulated_image

####################################################################################################
# Simulate discoloration and texture
####################################################################################################
def simulate_discoloration_and_texture(image, discoloration_intensity=0.5, texture_intensity=0.9, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Discoloration
    if discoloration_intensity > 0:
        discoloration_mask = np.random.uniform(1 - discoloration_intensity, 1, size=image.shape)
        image = (image * discoloration_mask).astype(np.uint8)

    # Texture Alteration
    if texture_intensity > 0:
        num_regions = np.random.randint(1, 4)
        for _ in range(num_regions):
            x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            w, h = np.random.randint(10, 30), np.random.randint(10, 30)
            blur_size = int(7 * texture_intensity)
            blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size  # ensuring odd size for Gaussian kernel
            smoothed_region = cv2.GaussianBlur(image[y:y+h, x:x+w], (blur_size, blur_size), 0)
            image[y:y+h, x:x+w] = smoothed_region

    return image

####################################################################################################
# Add resize with aspect ratio
####################################################################################################
def resize_with_aspect_ratio(image, target_size=(512, 512), background_value=0):
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


####################################################################################################
# Sharpen image
####################################################################################################
def sharp_imgage(image):
    # Define a sequence of augmentations
    seq = iaa.Sequential([
        # Beneficial for enhancing contrast in images
        #iaa.AllChannelsHistogramEqualization(),

        # Improve contrast
        iaa.ContrastNormalization(1.1), 
        
        # Adjusting brightness and contrast nonlinearly
        iaa.GammaContrast(0.9), # 1.5

        # Emphasizing edges the in displacement map, 
        iaa.Sharpen(alpha=1, lightness=(1.1)),

        # Enhance texture 
        iaa.Emboss(alpha=(0.4, 0.7), strength=1.0),
    ])

    # Apply augmentations
    augmented_image = seq(image=image)

    return augmented_image


####################################################################################################
# Apply imgaug augmentations to an image
# 
# Reference: https://imgaug.readthedocs.io/en/latest/index.html
####################################################################################################
def apply_imgaug_augmentations(image):
    # Define a sequence of augmentations
    seq = iaa.Sequential([
        # Apply two to four of the following augmentations
        iaa.SomeOf((2, 4), [
            # Apply affine transformations to simulate real-world variations in 
            # orientation and perspective
            iaa.Affine(
                scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-5, 5),
                shear=(-2, 2)
            ),
            
            # Beneficial for enhancing contrast in images
            iaa.AllChannelsHistogramEqualization(),

            # Improve contrast
            iaa.ContrastNormalization((0.75, 1.5)), 

            # Adjusting brightness and contrast nonlinearly
            iaa.GammaContrast((0.5, 1.5)),

            # Adjusting brightness and contrast nonlinearly
            iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6)),

            # Emphasizing edges and textures in displacement maps, 
            # aiding in the enhancement of eroded or faded inscriptions.
            iaa.Sharpen(alpha=(0.4, 1.0), lightness=(0.75, 1)),

            # Enhance texture
            iaa.Emboss(alpha=(0.4, 1.0), strength=(0.5, 2.0)),

            # Local distortions
            iaa.PiecewiseAffine(scale=(0.01, 0.05))

            # Can be beneficial to simulate different viewing angles
            #iaa.PerspectiveTransform(scale=(0.01, 0.15))

            # Simulate missing parts
            #iaa.CoarseDropout((0.02, 0.1), size_percent=(0.02, 0.25))
        ], random_order=True)
    ])

    # Apply augmentations
    augmented_image = seq(image=image)

    return augmented_image


def augment_image(image, pipeline=enhacement_augmentation_pipeline):
    return pipeline(image=image)['image']

####################################################################################################
# Randomly select a degradation level
# More weight on lighter degradation levels
# Levels: 1 (light), 2 (moderate), 3 (heavy)
####################################################################################################
def degradation_level_selector(levels=[1, 2, 3, 4], probabilities=[0.3, 0.3, 0.2, 0.2]):
    return np.random.choice(levels, p = probabilities)


####################################################################################################
# Apply degradations based on the selected level
####################################################################################################
def apply_degradation_based_on_level(image, level):
    # Check if input is a valid image array
    if not isinstance(image, np.ndarray) or len(image.shape) < 2:
        raise ValueError("Input must be a 2D or 3D numpy array")

    if level == 1:
        # Apply lighter degradations
        funcs = [
            #(add_noise, {'intensity': 0.5}),
            (add_blur, {'kernel_size': 3, 'preserve_edges': True}), # Apply a general blur to the map
            (erode_image, {'kernel_size_range': (2, 8), 'intensity': 0.5, 'kernel_shape': cv2.MORPH_ELLIPSE, 'iterations_range': (1, 3)}),
        ]
    elif level == 2:
        # Apply moderate degradations
        funcs = [
            #(add_noise, {'intensity': 0.5}), 
            (add_blur, {'kernel_size': 5, 'preserve_edges': False}), # Apply a general blur to the map
            (erode_image, {'kernel_size_range': (3, 8), 'intensity': 0.6, 'kernel_shape': cv2.MORPH_ELLIPSE, 'iterations_range': (2, 3)}),
            (dilate_image, {'kernel_size_range': (2, 5), 'intensity': 1.0, 'inscription_mask': None, 'iterations': 1}),
            (simulate_cracks, {'num_cracks_range': (3, 6), 'max_length': 120, 'crack_types': ['branching', 'wide', 'hairline']}), # Heavier crack simulation
        ]
    elif level == 3:
        # Apply heavier degradations
        funcs = [
            #(add_noise, {'intensity': 0.4}),
            (add_blur, {'kernel_size': 5, 'preserve_edges': False}), # Apply a general blur to the map
            (erode_image, {'kernel_size_range': (3, 8), 'intensity': 0.8, 'kernel_shape': cv2.MORPH_ELLIPSE, 'iterations_range': (1, 3)}),
            #(dilate_image, {'kernel_size_range': (3, 6), 'intensity': 1.0, 'inscription_mask': None, 'iterations': 1}),
            #(stretch_image, {'x_factor_range': (0.6, 1.3), 'y_factor_range': (0.6, 1.3)}),
            #(skew_image, {'x_skew_range': (0.6, 1.3), 'y_skew_range': (0.6, 1.3)}),
            #(simulate_discoloration_and_texture, {'discoloration_intensity': 0.2, 'texture_intensity': 0.2}),
            (simulate_cracks, {'num_cracks_range': (3, 6), 'max_length': 120, 'crack_types': ['branching', 'wide']}), # Heavier crack simulation
            (simulate_text_fading, {'num_areas_range': (1, 4), 'area_size_range': (10, 40), 'fading_intensity_range': (0.1, 0.3)}),
            (simulate_bumps_and_scratches, {'intensity': 0.7, 'scratches': True}) # Heavier bump and scratch simulation
        ]
    elif level == 4: 
        funcs = [
            (augment_image, {'pipeline': damaging_augmentation_pipeline})
        ]
    else:
        raise ValueError(f"Invalid degradation level: {level}")

    # Apply the selected degradations
    for func, params in funcs:
        image = func(image, **params)
    
    # Apply imgaug augmentations to emphasize features relevant to text and structural integrity
    #image = apply_imgaug_augmentations(image)

    return image


####################################################################################################
# The function takes a path to a directory as input and returns a list of images.
####################################################################################################
def load_displacement_maps(path, target_size=(512, 512)):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")
    
    displacement_maps = []
    map_paths         = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.tif'))
    map_paths         = sorted(map_paths)

    for map_path in tqdm(map_paths, desc="Loading displacement maps"):
        # Load displacement map as a grayscale image
        displacement_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        if displacement_map is None:
            logging.warning(f"Could not load {map_path}")
            continue
        
        # Resize displacement map while preserving aspect ratio
        resized_map = resize_with_aspect_ratio(displacement_map, target_size)

        displacement_maps.append(resized_map)
        logging.info(f"File: {map_path} loaded successfully.")

    return displacement_maps


####################################################################################################
# The function takes a path to a directory as input and returns a list of images.
####################################################################################################
def load_inscriptions_images(path, target_size=(512, 512)):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")
    
    inscriptions_images = []
    images_paths         = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))

    for image_path in tqdm(images_paths, desc="Loading inscriptions images"):
        try:
            inscription_image = cv2.imread(image_path)
            inscription_image = cv2.cvtColor(inscription_image, cv2.COLOR_BGR2RGB)

            if inscription_image is None:
                logging.warning(f"Could not load {image_path}")
                continue
            
            # Resize displacement map while preserving aspect ratio
            #resized_image = resize_with_aspect_ratio(inscription_image, target_size)

            # Append the image along with its path
            inscriptions_images.append((image_path, inscription_image))

            logging.info(f"File: {image_path} loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
    
    return inscriptions_images


####################################################################################################
# Display samples of images
####################################################################################################
def display_images(images, num_cols=4, img_size=(200, 200), titles=None, cmap='gray', first_n=20):
    num_images = len(images)

    if num_images == 0:
        print("No images to display.")
        return

    if num_images > first_n:
        images = images[:first_n]
        num_images = len(images)
        print(f"Displaying first {first_n} images.")

    # Calculate the number of rows required in the grid
    num_rows = int(num_images / num_cols) + int(num_images % num_cols > 0)

    # Calculate dynamic figure size (each subplot of size 2x2)
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        img = images[i]

        # Resize and convert image
        small_img_norm      = cv2.resize(img, img_size)
        #small_img_norm = cv2.normalize(small_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        plt.subplot(num_rows, num_cols, i + 1)

        if cmap == 'gray':
            # Display grayscale image
            plt.imshow(small_img_norm, cmap=cmap)
        else:
            # Display image
            plt.imshow(small_img_norm)
        
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')

    plt.show()


####################################################################################################
# Validate Intensity Distribution
####################################################################################################
def validate_intensity_distribution(real_images, synthetic_images):
    synthetic_intensities = [img.mean() for img in synthetic_images]
    real_intensities      = [img.mean() for img in real_images]

    # Compare distributions, e.g., using Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(synthetic_intensities, real_intensities)

    if p_value < 0.05:
        print("Distributions differ significantly.")
        return False
    else:
        print("No significant difference in distributions.")
        return True


####################################################################################################
# Save save_paired_image
####################################################################################################
def save_paired_image(input_image, target_image, x_path, y_path, batch_index, pair_index):
    # Construct unique filenames for the x and y images
    filename_x = f"input_batch_{batch_index}_pair_{pair_index}.png"
    filename_y = f"target_batch_{batch_index}_pair_{pair_index}.png"

    # Save the images
    cv2.imwrite(os.path.join(x_path, filename_x), input_image)
    cv2.imwrite(os.path.join(y_path, filename_y), target_image)


# def load_pairs_of_est_ground(input_displacement_maps_path, target_displacement_maps_path):
#     input_displacement_maps   = []
#     target_displacement_maps      = []

#     # Load ground displacement maps
#     input_displacement_maps  += load_displacement_maps(input_displacement_maps_path)
#     target_displacement_maps += load_displacement_maps(target_displacement_maps_path)

#     # Display ground images
#     display_images(input_displacement_maps, first_n=5)

#     # Generate synthetic enhanced displacement maps from ground displacement maps
#     for i, displacement_map in enumerate(target_displacement_maps):
#         # Sharpen the estimated displacement map
#         target_d_map = augment_image(target_displacement_maps[i], pipeline=enhacement_augmentation_pipeline)

#         for j in range(10):
#             #enhanced_displacement_map_v1 = sharp_imgage(est_displacement_map)
#             input_d_map = augment_image(input_displacement_maps[i], pipeline=damaging_augmentation_pipeline)

#             # Display images
#             #display_images([enhanced_est_d_map, ground_d_map], first_n=5, titles=["Estimated", "Ground"], num_cols=3)

#             save_paired_image(
#                 input_d_map, 
#                 target_d_map, 
#                 x_training_dataset_path, 
#                 y_training_dataset_path, 
#                 i, j)

####################################################################################################
# Test displacement maps from directory
####################################################################################################
def test_displacement_maps_generation(ground_displacement_maps_path, estimated_displacement_maps_path):
    validation = False
    ground_displacement_maps   = []
    est_displacement_maps      = []
    enhanced_displacement_maps = []

    # Load ground displacement maps
    ground_displacement_maps += load_displacement_maps(ground_displacement_maps_path)
    est_displacement_maps    += load_displacement_maps(estimated_displacement_maps_path)

    # Display ground images
    display_images(ground_displacement_maps, first_n=5)

    # Generate synthetic enhanced displacement maps from ground displacement maps
    for i, displacement_map in enumerate(ground_displacement_maps):

        # Estimated displacement map
        
        # Ground displacement map
        ground_displacement_map = augment_image(ground_displacement_maps[i], pipeline=enhacement_augmentation_pipeline)

        for j in range(10):
            # Sharpen the estimated displacement map
            #enhanced_displacement_map_v1 = sharp_imgage(est_displacement_map)
            enhanced_est_displacement_map = augment_image(est_displacement_maps[i], pipeline=damaging_augmentation_pipeline)

            enhanced_displacement_maps.append(enhanced_est_displacement_map)

            # Display images
            display_images([enhanced_est_displacement_map, ground_displacement_map], first_n=5, titles=["Estimated", "Ground"], num_cols=3)


    #Generate synthetic damaged displacement maps from ground displacement maps
    for i, displacement_map in enumerate(ground_displacement_maps):
        synthetic_damaged_displacement_maps = []

        for j in range(5):
            # Select test degradation level
            level = 2

            # Apply degradations based on the selected level
            damaged_displacement_map_v1 = apply_degradation_based_on_level(displacement_map.copy(), level)
            damaged_displacement_map_v2 = damaging_augmentation_pipeline(image=displacement_map.copy())['image']

            # Validate intensity distribution
            validate_intensity_distribution([displacement_map], [damaged_displacement_map_v1])

            # Display image
            display_images([damaged_displacement_map_v1, damaged_displacement_map_v2, displacement_map], titles=["Damaged v1", "Damaged v2", "Original"])

            # Add to list of synthetic displacement maps
            synthetic_damaged_displacement_maps.append(damaged_displacement_map_v1)
        
        # Display images
        display_images(synthetic_damaged_displacement_maps)

    return validation

def apply_depth_to_rgb(rgb_image, depth_map):
    """
    Apply the enhanced depth map to the RGB image as a shading layer.
    
    Args:
    - rgb_image: Original RGB image as a NumPy array.
    - depth_map: Enhanced depth map as a NumPy array.

    Returns:
    - shaded_rgb: RGB image with depth shading applied.
    """

    # Normalize depth map
    depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Apply depth map as shading to RGB image
    shaded_rgb = rgb_image * depth_map_normalized[..., np.newaxis]

    return shaded_rgb


def test_depth_to_rgb(rgb_image_path, depth_map_path):
    # Load your RGB image and depth map
    rgb_image = cv2.imread(rgb_image_path, cv2.COLOR_BGR2RGB)  # Replace with your image path
    #rgb_image = Image.open(rgb_image_path).convert("RGB")
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)  # Replace with your depth map path

    # Enhance the depth map
    #depth_map = augment_image(depth_map, pipeline=enhacement_augmentation_pipeline)

    #rgb_image_np = np.array(rgb_image)

    # Resize depth map to match the size of the RGB image
    depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))

    # Apply the depth map to the RGB image
    shaded_rgb_image = apply_depth_to_rgb(rgb_image, depth_map)

    # Save or display the result
    #cv2.imwrite('shaded_rgb_image.jpg', shaded_rgb_image)  # Save the shaded image

    # Dispay results
    display_images([rgb_image, depth_map, shaded_rgb_image], 
                   titles=["RGB", "Depth", "Shaded RGB"], 
                   num_cols=3, img_size=(100, 100), cmap='color')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Show original image
    # axes[0].imshow(rgb_image)
    # axes[0].set_title('Original Image')
    # axes[0].axis('off')

    # # Show estimated image
    # axes[1].imshow(depth_map, cmap='gray')
    # axes[1].set_title('Estimated Image')
    # axes[1].axis('off')

    # # Show restored image
    # axes[2].imshow(shaded_rgb_image)
    # axes[2].set_title('Restored Image')
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()
    
####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Read parameters from a configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get the path to the test dataset directory
    rgb_images      = config['DEFAULT']['RGB_TEST_DATASET_PATH']
    ground_d_m_path = config['DEFAULT']['GROUND_TEST_DATASET_PATH']
    est_d_m_path    = config['DEFAULT']['EST_TEST_DATASET_PATH']
    dh_d_m_path    = config['DEFAULT']['DH_TEST_DATASET_PATH']

    # Test displacement maps generation
    #load_pairs_of_est_ground(ground_d_m_path, est_d_m_path)

    # test depth to rgb
    test_depth_to_rgb(rgb_images + 'KAI_214_L10-11_2.png', dh_d_m_path + 'KAI_214_L10-11_2.png')

    print("Prevalidation successful.")
