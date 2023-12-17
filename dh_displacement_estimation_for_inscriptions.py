
import os
import sys
from turtle import width
import cv2
import glob
import re
import numpy as np
import configparser
import logging

import torch
import urllib.request
import matplotlib.pyplot as plt

# Read parameters from a configuration file
config = configparser.ConfigParser()
config.read('config.ini')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

####################################################################################################
# 'DEFAULT' paths section
####################################################################################################
project_path               = config['DEFAULT']['PROJECT_PATH']
inscriptions_dataset_path  = config['DEFAULT']['INSCRIPTIONS_DATASET_PATH']
displacement_maps_path     = config['DEFAULT']['DISPLACEMENT_DATASET_PATH']

paths = [
    displacement_maps_path,
]

####################################################################################################
# Import Image Processing functions
####################################################################################################
# Add project path to sys.path
sys.path.append(project_path)

# Import image processing functions
import utils.image_processing as ip


####################################################################################################
# Load Zoe model for depth estimation
####################################################################################################
def load_zoe():

    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

    repo = "isl-org/ZoeDepth"

    # Zoe_N
    #model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    # Zoe_K
    #model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

    # Zoe_NK
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

    zoe = model_zoe_nk.to(device)

    return zoe

####################################################################################################
# Prdict depth using Zoe model
####################################################################################################
def predict_depth(zoe_model, image, invert=True):
    # as 16-bit PIL Image
    prediction = zoe_model.infer_pil(image, output_type="tensor") 

    # Convert to numpy array and normalize
    depth_np = prediction.cpu().numpy()

    # Normalize the depth image to 16-bit scale
    depth_normalized = cv2.normalize(depth_np, None, 0, 65535, cv2.NORM_MINMAX)

    # Invert depth values
    depth_inverted = 65535 - depth_normalized if invert else depth_normalized

    # Convert to 16-bit unsigned integer
    depth_image = np.uint16(depth_inverted)

    return depth_image


####################################################################################################
# Load Inscriptions images from directory
# Returns a list of tuples (image_path, image)
####################################################################################################
def load_inscriptions_images_from_directory(path, dir_regex=r'.*'):
    # Initialize the list of inscriptions images
    inscriptions_images = []

    # Compile the regular expression for efficiency
    dir_pattern = re.compile(dir_regex)

    # Iterate through each subdirectory in the path
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            # Check if the directory name matches the regex
            if dir_pattern.match(dir):
                full_dir_path = os.path.join(subdir, dir)

                try:
                    inscriptions_images += ip.load_inscriptions_images(full_dir_path)
                except Exception as e:
                    logging.error(f"Error loading images from {full_dir_path}: {e}")

    # Display the number of images loaded
    print(f'Number of inscriptions images: {len(inscriptions_images)}')

    # Extracting only the image data from each tuple
    images_only = [img for _, img in inscriptions_images]
    
    # Display images in a grid layout 
    ip.display_images(images_only, num_cols=6, img_size=(100, 100))

    return inscriptions_images

####################################################################################################
# Save the depth estimation
####################################################################################################
def save_depth_image(depth_image, original_image_path, inscriptions_dataset_path, displacement_maps_base_path, index):
    # Compute the relative path of the original image with respect to the inscriptions dataset path
    relative_path = os.path.relpath(original_image_path, inscriptions_dataset_path)

    # Derive the directory path for saving the depth image
    save_dir_path = os.path.join(displacement_maps_base_path, os.path.dirname(relative_path))

    print(f"Saving depth image to {save_dir_path}")

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # Construct unique filenames for the depth images
    filename = f"depth_{index}.png"

    # Save the image
    cv2.imwrite(os.path.join(save_dir_path, filename), depth_image)

####################################################################################################
# Generate depth images
#  - For each inscription image, generate a depth estimation
#  - Save the depth estimations in the specified directories
####################################################################################################
def generate_depth_estimations(inscriptions_images, depth_model, displacement_maps_path, invert=True):
    depth_estimations = []

    for i, (inscription_image_path, inscription_image) in enumerate(inscriptions_images):
        # Predict depth
        output = predict_depth(depth_model, inscription_image, invert=invert)

        # Save the depth estimation in the same subdirectory as the original image
        save_depth_image(output, inscription_image_path, inscriptions_dataset_path, displacement_maps_path, i)

        # Add the depth estimation to the list
        depth_estimations.append(output)

    # Display the depth estimations
    ip.display_images(depth_estimations, num_cols=9, img_size=(100, 100), cmap='gray_r')

    logging.info("Depth estimation generation completed.")


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
# Main function
# - Load MiDaS model and transforms for depth estimation
# - Load inscriptions images from directory (dataset)
# - Generate depth estimations for each image in the dataset
# - Validate the generated data (optional)
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Validate and create directories
    validate_directories(paths)

    # Load the displacement images
    inscriptions_images = load_inscriptions_images_from_directory(inscriptions_dataset_path, dir_regex=r'Karatepe Ort 1')

    # Extracting only the image data from each tuple
    images_only = [img for _, img in inscriptions_images]

    # Load Zoe
    zoe = load_zoe()

    # Generate synthetic displacement maps
    generate_depth_estimations(inscriptions_images, zoe, displacement_maps_path, invert=False)


####################################################################################################
# Test Zoe
####################################################################################################
def test_zoe():
    # Load Zoe
    zoe = load_zoe()

    # Load the displacement images
    inscriptions_images = load_inscriptions_images_from_directory(inscriptions_dataset_path)  

    # Extracting only the image data from each tuple
    images_only = [img for _, img in inscriptions_images]

    for i, img in enumerate(images_only):
        # Predict depth
        depth_pil = predict_depth(zoe, img)

        # show the depth map
        plt.imshow(depth_pil, cmap="gray")

        plt.show()
