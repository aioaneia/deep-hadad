import functools

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage, Compose, ToTensor

import utils.cv_file_utils as file_utils
from models.UnetGenerator import UnetGenerator

# PyTorch version
print("PyTorch version: " + torch.__version__)

# Constants
PROJECT_PATH = './'
test_dataset_path = PROJECT_PATH + "data/test_dataset/Synthetic Damaged Glyphs/"
# test_dataset_path = PROJECT_PATH + "data/test_dataset/Real Damaged Glyphs/"
IMAGE_EXTENSIONS = [".png", ".jpg", ".tif"]

MODEL_PATH = PROJECT_PATH + 'trained_models/'

MODEL_NAME = 'dh_depth_model_ep_3_l1.00_s0.50_l0.10_a0.05_g0.10.pth'

transform = Compose([
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
        ngf=128,
        norm_layer=norm_layer,
        use_dropout=True
    ).to(device)

    generator.apply(generator.initialize_weights)

    return generator


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

    for test_d_map in test_dataset:

        # Resize displacement map while preserving aspect ratio
        resized_map = file_utils.resize_and_pad(test_d_map)

        # Transform the resized_map to a tensor
        d_map_tensor = transform(resized_map)

        # Generate the restored image
        restored_image = generate_restored_image(generator, d_map_tensor, invert_pixel_values=False)

        # Add the restored image to the list
        table_images.append({
            'org_image': resized_map,
            'ground_truth': resized_map,
            'est_image': resized_map,
            'res_image': to_pil(restored_image)
        })

    return table_images


def plot_qualitative_table(image_table, cmap='gray'):
    """
    Plots the test images and the restored images in a grid.
    Each row represents a different image and each column represents a category.
    :param table_images: A list of dictionaries containing the images and their labels
    :return: None
    """

    num_images = len(image_table)

    if num_images == 0:
        print("No images to display.")
        return

    # Create the figure and axes
    # The number of columns is 4, one for each image
    # The number of rows is the number of images
    # The figure size is 10 x 5 * num_images
    fig, axes = plt.subplots(num_images, 4, figsize=(12, 3 * num_images))

    # Plot the images from the table
    for i, table_image in enumerate(image_table):
        # Show original image
        axes[i, 0].imshow(table_image['org_image'], cmap=cmap)
        axes[i, 0].set_title('Textured Image')
        axes[i, 0].axis('off')

        # Show ground truth image
        axes[i, 1].imshow(table_image['ground_truth'], cmap=cmap)
        axes[i, 1].set_title('Displacement Ground Truth')
        axes[i, 1].axis('off')

        # Show restored image
        axes[i, 3].imshow(table_image['res_image'], cmap=cmap)
        axes[i, 3].set_title('Restored Image')
        axes[i, 3].axis('off')

        # Show estimated image
        axes[i, 2].imshow(table_image['est_image'], cmap=cmap)
        axes[i, 2].set_title('Textured Restored Image')
        axes[i, 2].axis('off')

    plt.tight_layout()

    plt.show()

    for i, table_image in enumerate(image_table):
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


if __name__ == "__main__":
    # Load the DHadad generator
    generator = load_dh_generator()
    generator = load_model(generator, MODEL_PATH + MODEL_NAME)

    # Load the test dataset
    test_dataset = file_utils.load_displacement_maps(test_dataset_path, preprocess=True)

    # Generate the restored image
    table_images = generate_restored_images(generator, test_dataset)

    # Plot the images
    plot_qualitative_table(table_images)
