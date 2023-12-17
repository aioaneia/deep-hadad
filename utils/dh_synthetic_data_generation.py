import os
import sys
import cv2 
import glob
import numpy as np
import configparser
import logging
import multiprocessing

from matplotlib import pyplot as plt

# Read parameters from a configuration file
config = configparser.ConfigParser()
config.read('config.ini')

####################################################################################################
# 'DEFAULT' paths section
####################################################################################################
project_path                 = config['DEFAULT']['PROJECT_PATH']
displacement_maps_path       = config['DEFAULT']['DISPLACEMENT_DATASET_PATH']
test_displacement_maps_path  = config['DEFAULT']['TEST_DATASET_PATH']

training_dataset_path        = config['DEFAULT']['TRAINING_DATASET_PATH']
x_training_dataset_path      = config['DEFAULT']['X_TRAINING_DATASET_PATH']
y_training_dataset_path      = config['DEFAULT']['Y_TRAINING_DATASET_PATH']

# training_dataset_path   = config['DEFAULT']['TRAINING_DATASET_MEDIUM_PATH']
# x_training_dataset_path = config['DEFAULT']['X_TRAINING_DATASET_MEDIUM_PATH']
# y_training_dataset_path = config['DEFAULT']['Y_TRAINING_DATASET_MEDIUM_PATH']

paths = [
    training_dataset_path,
    x_training_dataset_path,
    y_training_dataset_path
]

# Constants for data generation
num_pairs = 100  # Number of synthetic pairs per image
batch_size = 5  # Batch size for parallel processing

####################################################################################################
# Add project path to sys.path
# Import image processing functions
####################################################################################################

# Add project path to sys.path
sys.path.append(project_path)

# Import image processing functions
import utils.image_processing as ip


####################################################################################################
# Validate and create directories
####################################################################################################
def validate_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.access(path, os.W_OK):
            raise Exception(f"Directory {path} is not writable.")
    print('Directories validated successfully.')

####################################################################################################
# Save save_paired_image
####################################################################################################
def save_paired_image(x_depth_image, y_depth_image, x_path, y_path, batch_index, pair_index):
    # Construct unique filenames for the x and y images
    filename_x = f"x_batch_{batch_index}_pair_{pair_index}.png"
    filename_y = f"y_batch_{batch_index}_pair_{pair_index}.png"

    # Save the images
    cv2.imwrite(os.path.join(x_path, filename_x), x_depth_image)
    cv2.imwrite(os.path.join(y_path, filename_y), y_depth_image)

####################################################################################################
# Generate an image pair for specified image
# Returns a tuple of (intact, damaged) images
####################################################################################################
def generate_pair_for_image(image):
    if image is None:
        raise ValueError("Input image is None")

    # Select degradation level
    level = ip.degradation_level_selector()

    # Apply degradations based on the selected level
    damaged = ip.apply_degradation_based_on_level(image.copy(), level)

    # Ensure that a valid image is returned
    if damaged is None:
        raise ValueError("Degradation function returned None")

    # Return the pair of images (intact, damaged)
    return image, damaged

####################################################################################################
# Generate synthetic displacement maps
#  - For each displacement map, generate num_pairs of intact-damaged pairs
#  - Save the pairs to the specified directories
####################################################################################################
def generate_synthetic_maps(displacement_maps, num_pairs=10):
    # Iterate through each displacement map
    for i in range(len(displacement_maps)):
        # Iterate through each pair
        for j in range(num_pairs):
            # Generate intact-damaged pair
            x_depth_image, y_depth_image = generate_pair_for_image(displacement_maps[i])

            # Save intact-damaged pair
            save_paired_image(
                x_depth_image, y_depth_image, 
                x_training_dataset_path, y_training_dataset_path, 
                i, j
            )
    
    logging.info("Data generation completed.")


####################################################################################################
# Generate synthetic displacement maps in parallel
####################################################################################################
def generate_synthetic_maps_batch(batch_data):
    # Unpack batch data
    displacement_maps_batch, num_pairs, batch_index = batch_data

    # Iterate through each displacement map
    for i, displacement_map in enumerate(displacement_maps_batch):
        for j in range(num_pairs):
            # Generate intact-damaged pair
            x_depth_image, y_depth_image = generate_pair_for_image(displacement_map)

            # Save intact-damaged pair with unique identifiers
            save_paired_image(
                x_depth_image, y_depth_image, 
                x_training_dataset_path, y_training_dataset_path, 
                batch_index, j + i * num_pairs
            )


####################################################################################################
# Use multiprocessing to generate the displacement maps in parallel
####################################################################################################
def generate_data_in_parallel(displacement_maps, num_pairs=100, batch_size=5):
    # Split displacement maps into batches
    batches = [displacement_maps[i:i + batch_size] for i in range(0, len(displacement_maps), batch_size)]

    # Prepare parameters for each batch
    params = [(batch, num_pairs, batch_index) for batch_index, batch in enumerate(batches)]

    # Use multiprocessing pool to process each batch
    with multiprocessing.Pool() as pool:
        pool.map(generate_synthetic_maps_batch, params)

    logging.info("Data generation completed.")


####################################################################################################
# Validate the generated data
# - Check if the quality of generated images is acceptable
####################################################################################################
def prevalidate_generated_data(path):
    validation = ip.test_displacement_maps_generation(path);

    if validation:
        print("Prevalidation successful.")
    else:
        print("Prevalidation failed.")
    
    return validation

####################################################################################################
# Validate the generated data
####################################################################################################
def validate_generated_data():
    # Placeholder for the actual implementation
    pass

####################################################################################################
# Load displacement maps from directory
####################################################################################################
def load_displacement_maps_from_directory(path):
    ground_displacement_maps = []
    x_displacement_maps      = []

    # Iterate through each subdirectory in the path
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            full_dir_path = os.path.join(subdir, dir)
            # Load the displacement maps
            ground_displacement_maps += ip.load_displacement_maps(full_dir_path)

    print(f'Number of Ground Displacement Maps: {len(ground_displacement_maps)}')

    # Generate x synthetic displacement maps
    for i, displacement_map in enumerate(ground_displacement_maps):
        # Add the original displacement map
        x_displacement_maps.append(displacement_map)

        # Sharpen the image and add it to the list
        #x_displacement_maps.append(ip.sharp_imgage(displacement_map.copy()))
    
    print(f'Number of Synthetic Displacement Maps: {len(x_displacement_maps)}')

    return x_displacement_maps


####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Validate and create directories
    validate_directories(paths)

    # Load the displacement images
    displacement_maps = load_displacement_maps_from_directory(displacement_maps_path)

    # Generate synthetic displacement maps in parallel
    generate_data_in_parallel(displacement_maps, num_pairs, batch_size)

    # Validate the generated data
    validate_generated_data()
