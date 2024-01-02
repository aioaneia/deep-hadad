import sys
sys.path.append('./')

import cv2
import numpy as np
import glob
import os

import matplotlib.pyplot  as plt
import torch

from torchvision.transforms import ToPILImage, Compose, Resize, Lambda, ToTensor

from PIL         import Image
from torchvision import datasets

# import the networks
from core.DHadadGenerator     import DHadadGenerator
from core.DHadadDiscriminator import DHadadDiscriminator

# Constants
PROJECT_PATH          = './'
test_dataset_path     = PROJECT_PATH + "data/test_dataset/"
original_dataset_path = PROJECT_PATH + "data/test_dataset/Org"

IMAGE_EXTENSIONS      = [".png", ".jpg", ".tif"]

MODEL_PATH   = PROJECT_PATH + 'models/'
MODEL_NAME   = 'dh_delta_model_ep_19_a0.05_b0.10_g0.35_d0.25_e0.05_z0.15_e0.05.pth'

transform = Compose([
  Resize((512, 512)),
  ToTensor()
])

to_pil = ToPILImage()

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model(model, model_path):
    """
    Loads the model weights from the specified path
    :param model: The model to load the weights into
    :param model_path: The path to the model weights
    :return: The model with the loaded weights
    """
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    return model


def load_dh_generator():
    """
    Loads the generator for the DHadad model
    :return: The generator
    """
    # Constants
    gen_in_channels  = 1  # grayscale images, 3 for RGB images
    gen_out_channels = 1  # to generate grayscale restored images, change as needed

    # Instantiate the generator with the specified channel configurations
    generator = DHadadGenerator(gen_in_channels, gen_out_channels)

    return generator


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

# Get the paths of all images in a directory
def get_image_paths(directory):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory, '*' + ext)))

    return sorted(image_paths)

# Load the dataset
def load_dataset(dataset_path):
    # Get images from path
    image_paths  = get_image_paths(dataset_path)

    return image_paths


def predict_depth(zoe_model, image, invert=True):
    """
    Predicts the depth of the image
    :param zoe_model: The Zoe model to use for depth estimation
    :param image: The image to estimate the depth for
    :return: The depth image
    """
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

def generate_restored_image(generator, test_image_tensor, invert_pixel_values=False):
    """
    Generates the restored image from the test image
    :param generator: The generator to use for restoring the image
    :param test_image_tensor: The test image
    :return: The restored image
    """

    # Turn off gradients for testing
    with torch.no_grad():
        # Add a batch dimension and move to the GPU if needed
        image_tensor = test_image_tensor.unsqueeze(0)

        # Generate the restored image and remove the batch dimension
        restored_image = generator(image_tensor).squeeze(0).cpu()

    # Invert the pixel values
    if invert_pixel_values:
        restored_image = 1 - restored_image 
    
    return restored_image


def generate_restored_images(generator, depth_model, dataset, invert_pixel_values=False):
    """
    Generates the restored images from the test dataset
    :param generator: The generator to use for restoring the images
    :param test_dataset: The test dataset
    :return: The restored images
    """

    # Generate the restored images
    table_images = []

    for image_path in dataset:
        # Load the image
        image_pil = Image.open(image_path).convert("RGB")

        # Predict the depth
        # as 16-bit PIL Image
        prediction = depth_model.infer_pil(image_pil, output_type="tensor") 

        # Invert depth values
        prediction = 1 - prediction

        prediction_pil = to_pil(prediction)

        # Transform the image
        image_tensor = transform(prediction_pil).to(device)

        # Generate the restored image
        restored_image = generate_restored_image(generator, image_tensor, invert_pixel_values)

            # Add the restored image to the list
        table_images.append({
            'org_image':    image_pil,
            'ground_truth': prediction_pil,
            'est_image':    prediction_pil,
            'res_label':    to_pil(restored_image)
        })

        print("Generated restored image for label: {}".format(image_path))

    return table_images


def plot_images(table_images, cmap='gray'):
    """
    Plots the test images and the restored images in a grid.
    Each row represents a different image and each column represents a category.
    :param table_images: A list of dictionaries containing the images and their labels
    :return: None
    """

    num_images = len(table_images)

    if num_images == 0:
        print("No images to display.")
        return
    
    # Create the figure and axes
    # The number of columns is 4, one for each image
    # The number of rows is the number of images
    # The figure size is 10 x 5 * num_images
    fig, axes = plt.subplots(num_images, 4, figsize=(10, 2 * num_images))

    # Plot the images from the table
    for i, table_image in enumerate(table_images):
        # Show original image
        axes[i, 0].imshow(table_image['org_image'])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Show ground truth image
        axes[i, 1].imshow(table_image['ground_truth'], cmap=cmap)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Show estimated image
        axes[i, 2].imshow(table_image['est_image'], cmap=cmap)
        axes[i, 2].set_title('Estimated Image')
        axes[i, 2].axis('off')

        # Show restored image
        axes[i, 3].imshow(table_image['res_label'], cmap=cmap)
        axes[i, 3].set_title('Restored Image')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

def save_images(table_images):
    """
    Saves the test image and the restored image
    :param test_image_pil: The test image
    :param restored_image_pil: The restored image
    :return: None
    """

    # Save the images
    #restored_image_pil.save(PROJECT_PATH + "data/test_dataset/X_image.png")
    pass

# Main function for testing
if __name__ == "__main__":
    # Load the DHadad generator
    generator = load_dh_generator()
    
    # Load the model weights
    load_model(generator, MODEL_PATH + MODEL_NAME)

    # Load Zoe depth estimation model
    zoe = load_zoe()

    # Load the test dataset
    test_dataset = load_dataset(original_dataset_path)

    # Generate the restored image
    table_images = generate_restored_images(generator, zoe, test_dataset, invert_pixel_values=True)

    # Plot the images
    plot_images(table_images)

    # Save the images
    save_images(table_images)