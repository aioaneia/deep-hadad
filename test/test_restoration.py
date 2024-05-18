import functools

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage, Compose, ToTensor

import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils
import utils.cv_convert_utils as conv_utils

from models.UnetGenerator import UnetGenerator

print("PyTorch version: " + torch.__version__)

PROJECT_PATH = '../'

LARGE_D_MAP_PATH = PROJECT_PATH + "data/test_dataset/Real Damaged Inscriptions/KAI_214_d_map_2.png"

MODEL_PATH = PROJECT_PATH + 'trained_models/dh_depth_model_ep_40_l0.40_s0.80_a0.20_g0.40_s0.15.pth'

transform = Compose([
    ToTensor(),
])

to_pil = ToPILImage()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(model, model_path):
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    return model


def get_norm_layer(norm_type='instance'):
    """
        Return a normalization layer
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
        use_dropout=False
    ).to(device)

    generator.apply(generator.initialize_weights)

    return generator


def restore_d_map(generator, d_map_tensor, invert_pixel_values=True):
    """
    Generates the restored image from the image

    :param generator: The generator to use for restoring
    :param d_map_tensor: The displacement map for restoring
    :return: The restored image
    """

    # Turn off gradients for testing
    with torch.no_grad():
        # Add a batch dimension and move to the GPU if needed
        broken_image = d_map_tensor.unsqueeze(0)

        # Generate the restored image and remove the batch dimension
        restored_image = generator(broken_image).squeeze(0).cpu()

        # Normalize the image to the range [0, 1]
        restored_image = (restored_image - restored_image.min()) / (restored_image.max() - restored_image.min())

    if invert_pixel_values:
        restored_image = 1 - restored_image

    return restored_image



def restore_large_displacement_map(generator, d_map, crop_size=(512, 512), overlap=(5, 5), invert_pixel_values=False, test_stitch=False):
    """
    Generates the restored image for a large displacement map by cropping, processing, and stitching patches.

    :param generator: The generator to use for restoring the images
    :param d_map: The displacement map (2D NumPy array or image)
    :param crop_size: Tuple representing the height and width of the patches
    :param overlap: Tuple for height and width overlap between adjacent patches
    :param invert_pixel_values: Boolean to determine whether to invert pixel values
    :param test_stitch: Boolean to determine whether to stitch patches without reconstruction
    :return: The restored full displacement map
    """

    height, width = d_map.shape
    patch_height, patch_width = crop_size
    overlap_h, overlap_w = overlap

    # Initialize a large zero matrix to hold the restored image or test stitch image
    restored_d_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # Iterate through the displacement map using overlapping patches
    for y in range(0, height, patch_height - overlap_h):
        for x in range(0, width, patch_width - overlap_w):
            # Ensure we don't go out of bounds
            y_end = min(y + patch_height, height)
            x_end = min(x + patch_width, width)

            # Crop the patch from the large displacement map
            patch = d_map[y:y_end, x:x_end]

            # Pad the patch to ensure it has the dimensions of crop_size
            padded_patch = np.pad(
                patch,
                ((0, patch_height - patch.shape[0]), (0, patch_width - patch.shape[1])),
                mode='constant', constant_values=0
            )

            if test_stitch:
                restored_patch = padded_patch
            else:
                # Transform the padded patch to a tensor
                patch_tensor = transform(padded_patch).unsqueeze(0).to(device)

                # Turn off gradients for testing
                with torch.no_grad():
                    restored_patch_tensor = generator(patch_tensor).squeeze(0).cpu()

                    # Normalize the output
                    restored_patch = (restored_patch_tensor - restored_patch_tensor.min()) / (
                            restored_patch_tensor.max() - restored_patch_tensor.min())

                    restored_patch = restored_patch.numpy()

                # Invert pixel values if needed
                if invert_pixel_values:
                    restored_patch = 1 - restored_patch

                # Remove extra dimensions explicitly and ensure it matches the slice
                restored_patch = np.squeeze(restored_patch)

                if restored_patch.ndim != 2:
                    raise ValueError("Patch has incorrect number of dimensions after squeezing.")

            # Reshape the patch to match the expected slice dimensions exactly
            restored_patch = restored_patch[:y_end - y, :x_end - x]

            # Blend the restored patch into the full displacement map
            restored_d_map[y:y_end, x:x_end] += restored_patch
            weight_map[y:y_end, x:x_end] += 1

    # Normalize the final restored map by the weight map to manage overlaps
    restored_d_map /= np.maximum(weight_map, 1)  # Avoid division by zero

    return restored_d_map


if __name__ == "__main__":
    # Load the DHadad generator
    generator = load_dh_generator()

    generator = load_model(generator, MODEL_PATH)

    # Load the large displacement map
    d_map = file_utils.load_displacement_map(LARGE_D_MAP_PATH, preprocess=True)

    plot_utils.plot_displacement_map(d_map, title=f'Displacement Map', cmap='gray')  # Other cmaps: 'viridis'

    # Convert the displacement map to a mesh
    # d_map_vertices, d_map_faces = conv_utils.displacement_map_to_mesh(d_map)

    # plot_utils.plotly_mesh(d_map_vertices, d_map_faces, title=f"3D Surface Mesh")

    # Generate the restored image
    restored_d_map = restore_large_displacement_map(
        generator,
        d_map,
        crop_size=(512, 512),
        overlap=(5, 5),
        invert_pixel_values=False,
        test_stitch=False
    )

    plot_utils.plot_displacement_map(restored_d_map, title=f'Displacement Map', cmap='gray')

    # Convert the restored displacement map to a mesh
    # d_map_vertices, d_map_faces = conv_utils.displacement_map_to_mesh(restored_d_map)

    # plot_utils.plotly_mesh(d_map_vertices, d_map_faces, title=f"3D Surface Mesh")
