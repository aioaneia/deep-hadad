import sys

sys.path.append('./')

import matplotlib.pyplot as plt
import torch
import cv2
import glob
import os
import functools
import torch.nn as nn

from torchvision.transforms import ToPILImage, Compose, Resize, Lambda, ToTensor

from PIL import Image

from models.UnetGenerator import UnetGenerator

# PyTorch version
print("PyTorch version: " + torch.__version__)

# Constants
PROJECT_PATH = './'
test_dataset_path = PROJECT_PATH + "data/test_dataset/"
IMAGE_EXTENSIONS = [".png", ".jpg", ".tif"]

MODEL_PATH = PROJECT_PATH + 'trained_models/'
MODEL_NAME = 'dh_depth_model_ep_18_l0.60_s0.25_a0.10_d0.15_s0.50_g0.50.pth'

transform = Compose([
    Resize((512, 512)),
    Lambda(lambda x: x.convert('L')),
    ToTensor(),
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


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


def load_dh_generator():
    """
    Loads the generator for the DHadad model
    :return: The generator
    """
    # Constants
    gen_in_channels = 1  # grayscale images, 3 for RGB images
    gen_out_channels = 1  # to generate grayscale restored images, change as needed

    # Get the normalization layer
    norm_layer = get_norm_layer(norm_type='instance')

    # Instantiate the generator with the specified channel configurations
    generator = UnetGenerator(
        gen_in_channels,
        gen_out_channels,
        num_downs=7,
        ngf=160,
        norm_layer=norm_layer,
        use_dropout=False
    ).to(device)

    generator.apply(generator.initialize_weights)

    return generator


# Get the paths of all images in a directory
def get_image_paths(directory):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory, '*' + ext)))

    return sorted(image_paths)


# Load the dataset
def load_dataset(dataset_path):
    # Get images from path
    image_paths = get_image_paths(dataset_path)

    return image_paths


def generate_restored_image(generator, test_image_tensor, invert_pixel_values=True):
    """
    Generates the restored image from the test image
    :param generator: The generator to use for restoring the image
    :param test_image_tensor: The test image
    :return: The restored image
    """

    # Turn off gradients for testing
    with torch.no_grad():
        # Add a batch dimension and move to the GPU if needed
        broken_image = test_image_tensor.unsqueeze(0)

        # Generate the restored image and remove the batch dimension
        restored_image = generator(broken_image).squeeze(0).cpu()

        # Normalize the image to the range [0, 1]
        restored_image = (restored_image - restored_image.min()) / (restored_image.max() - restored_image.min())

    # Invert the pixel values
    if invert_pixel_values == True:
        restored_image = 1 - restored_image

    return restored_image


def generate_restored_images(generator, test_dataset):
    """
    Generates the restored images from the test dataset
    :param generator: The generator to use for restoring the images
    :param test_dataset: The test dataset
    :return: The restored images
    """

    # Generate the restored images
    table_images = []

    for image_path in test_dataset:
        # Load the depth map
        # image = Image.open(image_path).convert("L")
        d_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert the NumPy array to a PIL Image
        d_map_pil = Image.fromarray(d_map)

        # Transform the image
        image_tensor = transform(d_map_pil).to(device)

        # Generate the restored image
        restored_image = generate_restored_image(generator, image_tensor, invert_pixel_values=False)

        # Add the restored image to the list
        table_images.append({
            'org_image': d_map,
            'ground_truth': d_map,
            'est_image': d_map,
            'res_image': to_pil(restored_image)
        })

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
    fig, axes = plt.subplots(num_images, 4, figsize=(12, 3 * num_images))

    # Plot the images from the table
    for i, table_image in enumerate(table_images):
        # Show original image
        axes[i, 0].imshow(table_image['org_image'], cmap=cmap)
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
        axes[i, 3].imshow(table_image['res_image'], cmap=cmap)
        axes[i, 3].set_title('Restored Image')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

    for i, table_image in enumerate(table_images):
        # plot the pair images one by one
        fig, axes = plt.subplots(1, 2, figsize=(12, 12))

        # Show estimated image
        axes[0].imshow(table_image['est_image'], cmap=cmap)
        axes[0].set_title('Estimated Image')
        axes[0].axis('off')

        # Show restored image
        axes[1].imshow(table_image['res_image'], cmap=cmap)
        axes[1].set_title('Restored Image')
        axes[1].axis('off')

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
    # restored_image_pil.save(PROJECT_PATH + "data/test_dataset/X_image.png")
    pass


# Function to load model
# def load_pix2pix_model(model_path):
#     # Get the image shape
#     image_shape = (512, 512, 1)
#
#     # Instantiate the generator
#     generator = get_generator(image_shape)
#
#     # Instantiate the discriminator
#     discriminator = get_discriminator(image_shape)
#
#     # Instantiate the GAN model
#     model = GAN(discriminator, generator)
#
#     model.load_state_dict(torch.load(model_path), strict=False)
#
#     model.eval()
#
#     return model


if __name__ == "__main__":
    # Later, to load the model (assuming the GAN class definition is available):
    # pix2pix_model = load_pix2pix_model(MODEL_PATH + GAN_MODEL_NAME)
    # Get the generator from the GAN model
    # generator = pix2pix_model.generator

    # Load the DHadad generator
    generator = load_dh_generator()
    generator = load_model(generator, MODEL_PATH + MODEL_NAME)

    # Load the test dataset
    test_dataset = load_dataset(test_dataset_path)

    # Generate the restored image
    table_images = generate_restored_images(generator, test_dataset)

    # Plot the images
    plot_images(table_images)

    # Save the images
    save_images(table_images)
