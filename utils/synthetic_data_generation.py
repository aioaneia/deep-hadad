import os
import sys
import cv2 
import logging
import multiprocessing
import configparser

# Add project path to sys.path
sys.path.append('./')

# Import image processing functions
from utils import image_processing as ip 

###############################
# Global variables
###############################
project_path            = None
displacement_maps_path  = None
x_training_dataset_path = None
y_training_dataset_path = None
paths                   = None
num_pairs               = None
batch_size              = None


####################################################################################################
# 'DEFAULT' paths section
# - Constants for data generation
####################################################################################################
def init_default_paths(project_dir='./', dataset_size='medium'):
    global project_path, displacement_maps_path, x_training_dataset_path, y_training_dataset_path
    global paths, num_pairs, batch_size

    # Read the config file
    config = configparser.ConfigParser()
    config.read(project_dir + 'config.ini')

    # Path to the project root directory
    project_path = project_dir

    # Path to the displacement maps dataset 
    displacement_maps_path  = project_path + config['DEFAULT']['DISPLACEMENT_DATASET_PATH']

    # Paths to the generated data directories 
    training_dataset_path   = project_path + config['DEFAULT'][f'{dataset_size.upper()}_TRAINING_DATASET_PATH']
    x_training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_X_TRAINING_DATASET_PATH']
    y_training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_Y_TRAINING_DATASET_PATH']

    # Paths to be validated 
    paths = [training_dataset_path, x_training_dataset_path, y_training_dataset_path]

    # Number of synthetic pairs per image
    num_pairs = int(config['DEFAULT']['NUM_OF_PAIRS'])

    # Batch size for parallel processing
    batch_size = int(config['DEFAULT']['BATCH_SIZE'])

####################################################################################################
# Validate and create directories
####################################################################################################
def validate_directories(paths):
    # Validate and create directories
    for path in paths:
        print(f"Validating directory: {path}")

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
# - Type: 'degradation' or 'enhancement'
####################################################################################################
def generate_pair_for_image(image, type='enhancement'):
    if image is None:
        raise ValueError("Input image is None")

    # Select degradation level
    level = ip.degradation_level_selector()

    # Apply degradations based on the selected level
    input_image = ip.apply_degradation_based_on_level(image.copy(), level)

    # Sharpen the image and add it to the list
    target_image = ip.sharp_imgage(image.copy())

    # Ensure that a valid image is returned
    if input_image is None or target_image is None:
        raise ValueError("Function returned None")

    # Return the pair of images
    return input_image, target_image


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
            # Generate real-enhanced pair
            x_depth_image, y_depth_image = generate_pair_for_image(displacement_maps[i], type='enhancement')

            # Save real-enhanced pair
            save_paired_image(
                x_depth_image, 
                y_depth_image, 
                x_training_dataset_path, 
                y_training_dataset_path, 
                i, j
            )

            # generate real-degraded pair
            #x_depth_image, y_depth_image = generate_pair_for_image(displacement_maps[i], type='degradation')

    
    logging.info("Data generation completed.")


####################################################################################################
# Generate synthetic displacement maps in parallel
####################################################################################################
def generate_synthetic_maps_batch(batch_data):
    # Unpack batch data
    displacement_maps_batch, num_pairs, batch_index, x_path, y_path = batch_data

    # Iterate through each displacement map
    for i, displacement_map in enumerate(displacement_maps_batch):
        for j in range(num_pairs):
            # Generate intact-damaged pair
            x_depth_image, y_depth_image = generate_pair_for_image(displacement_map)

            # Save intact-damaged pair with unique identifiers
            save_paired_image(
                x_depth_image, y_depth_image, 
                x_path, y_path, batch_index, j + i * num_pairs)

    logging.info(f"Batch {batch_index} completed.")


####################################################################################################
# Display sample pairs
#  - Display num_pairs of intact-damaged pairs
####################################################################################################
def display_sample_pairs(num_pairs=10):
    # Get images from path
    intact_image_paths  = ip.get_image_paths(x_training_dataset_path)
    damaged_image_paths = ip.get_image_paths(y_training_dataset_path)

    print(f"Number of X Generated Images: {len(intact_image_paths)}")
    print(f"Number of Y Generated Images: {len(damaged_image_paths)}")

    assert len(intact_image_paths) == len(damaged_image_paths), "Number of intact and damaged images must be the same"

    image_pairs = []

    # Iterate through each displacement map
    for i in range(num_pairs):
        x_depth_image = cv2.imread(intact_image_paths[i], cv2.IMREAD_GRAYSCALE)
        y_depth_image = cv2.imread(damaged_image_paths[i], cv2.IMREAD_GRAYSCALE)

        image_pairs.append(x_depth_image)
        image_pairs.append(y_depth_image)

    # Display the pair
    ip.display_images(image_pairs, num_cols=2, img_size=(200, 200))


####################################################################################################
# Use multiprocessing to generate the displacement maps in parallel
####################################################################################################
def generate_data_in_parallel(displacement_maps, num_pairs=100, batch_size=10, x_path=x_training_dataset_path, y_path=y_training_dataset_path):
    # Split displacement maps into batches
    batches = [displacement_maps[i:i + batch_size] for i in range(0, len(displacement_maps), batch_size)]

    # Prepare parameters for each batch 
    params = [(batch, num_pairs, batch_index, x_path, y_path) for batch_index, batch in enumerate(batches)]

    # Use multiprocessing pool to process each batch in parallel 
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

    # Initialize paths and global variables
    init_default_paths(dataset_size='small')

    print(f"Project Path:            {project_path}")
    print(f"Displacement Maps Path:  {displacement_maps_path}")
    print(f"X Training Dataset Path: {x_training_dataset_path}")
    print(f"Y Training Dataset Path: {y_training_dataset_path}")

    # Validate and create directories
    validate_directories(paths)

    # Load the displacement images
    displacement_maps = load_displacement_maps_from_directory(displacement_maps_path)

    # Generate synthetic displacement maps in parallel
    generate_data_in_parallel(displacement_maps, num_pairs, batch_size, x_training_dataset_path, y_training_dataset_path)

    # Validate the generated data
    validate_generated_data()

    # Display sample pairs
    display_sample_pairs(num_pairs=7)
