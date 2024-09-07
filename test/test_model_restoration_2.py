import functools
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage, Compose, ToTensor

import utils.cv_file_utils as fu
import utils.plot_utils as plot_utils

from models.Spade2Generator import Spade2Generator


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

print("PyTorch version: " + torch.__version__)

PROJECT_PATH = '../'

LARGE_D_MAP_PATH = PROJECT_PATH + "data/test_dataset/Real Damaged Inscriptions/KAI_214_d_map_1.png"

model_1 = 'dh_model_ep_17_l10.50_s1.50_m1.00_g3.00_t0.10_f2.00_a1.00.pth'

MODEL_PATH = PROJECT_PATH + 'trained_models/' + model_1

transform = Compose([
    ToTensor(),
])

to_pil = ToPILImage()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(model, model_path):
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint)

    return model.eval()


def load_dh_generator():
    """
    Loads the generator for the DHadad model
    :return: The generator
    """

    """ Loads the generator for the DHadad model """
    return Spade2Generator(
        input_nc=1,
        output_nc=1,
        label_nc=1,
        ngf=64,
        n_downsampling=3,
        n_blocks=8
    ).to(device)


def restore_d_map(generator, d_map_tensor, invert_pixel_values=True):
    """
    Generates the restored image from the image
    """

    with torch.no_grad():
        # Add a batch dimension and move to the GPU if needed
        broken_image = d_map_tensor.unsqueeze(0)
        segmap = torch.zeros_like(broken_image)

        # Generate the restored image and remove the batch dimension
        restored_image = generator(broken_image, segmap).squeeze(0).cpu()

        # Normalize the image to the range [0, 1]
        restored_image = (restored_image - restored_image.min()) / (restored_image.max() - restored_image.min())

    if invert_pixel_values:
        restored_image = 1 - restored_image

    return restored_image


def restore_large_displacement_map(
        generator,
        displacement_map,
        crop_size=(512, 512),
        overlap=(5, 5),
        apply_clahe=False,
        invert_pixel_values=False,
        test_stitch=False):
    """
    Generates the restored image for a large displacement map by cropping, processing, and stitching patches.
    """

    height, width = displacement_map.shape
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
            patch = displacement_map[y:y_end, x:x_end]

            # Pad the patch to ensure it has the dimensions of crop_size
            padded_patch = np.pad(
                patch,
                ((0, patch_height - patch.shape[0]), (0, patch_width - patch.shape[1])),
                mode='constant', constant_values=0
            )

            padded_patch = fu.preprocess_displacement_map(padded_patch, apply_clahe=apply_clahe)

            if test_stitch:
                restored_patch = padded_patch
            else:
                # Transform the padded patch to a tensor
                patch_tensor = transform(padded_patch).unsqueeze(0).to(device)

                # Turn off gradients for testing
                with torch.no_grad():
                    segmap = torch.zeros_like(patch_tensor)

                    restored_patch_tensor = generator(patch_tensor, segmap).squeeze(0).cpu()

                    # Normalize the output to the range [0, 1]
                    restored_patch_tensor = (restored_patch_tensor - restored_patch_tensor.min()) / (
                            restored_patch_tensor.max() - restored_patch_tensor.min())

                    # Convert the tensor to a numpy array
                    restored_patch = restored_patch_tensor.numpy()

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
    # Load the DHadad generator architecture
    generator = load_dh_generator()

    # Load the generator weights
    generator = load_model(generator, MODEL_PATH)

    # Load the large displacement map without preprocessing
    d_map = fu.load_displacement_map(
        LARGE_D_MAP_PATH,
        preprocess=False,
        resize=False,
        apply_clahe=True
    )

    # Load the large displacement map without preprocessing
    d_map_enhanced = fu.load_displacement_map(
        LARGE_D_MAP_PATH,
        preprocess=True,
        resize=False,
        apply_clahe=False
    )

    # Plot the original displacement map without preprocessing
    plot_utils.plot_displacement_map(
        d_map,
        title=f'Displacement Map',
        cmap='gray'  # Other cmaps: 'viridis'
    )

    # Plot the enhanced displacement map with image processing
    plot_utils.plot_displacement_map(
        d_map_enhanced,
        title=f'Displacement Map',
        cmap='gray'  # Other cmaps: 'viridis'
    )

    # Generate restored image for the original displacement map
    restored_d_map = restore_large_displacement_map(
        generator,
        d_map,
        crop_size=(512, 512),
        overlap=(0, 0),
        apply_clahe=True,
        invert_pixel_values=False,
        test_stitch=False
    )

    # Plot the restored image
    plot_utils.plot_displacement_map(
        restored_d_map,
        title='Restored Displacement Map',
        cmap='gray',  # Other cmaps: 'viridis', 'coolwarm',
        save_plot=True
    )


    # Apply viridis colormap to the restored displacement map
    restored_d_map_viridis_ = plot_utils.apply_viridis_colormap(
        restored_d_map,
        title='Restored Displacement Map (Viridis)',
        save_image=True
    )

    # Save the restored displacement map to a file
    fu.save_displacement_map(
        restored_d_map,
        '../data/plots/',
        'restored_d_map.png'
    )

    # Normalize the restored displacement map to the range [0, 1]
    # restored_d_map = en.apply_histogram_equalization(restored_d_map)
    restored_d_map = fu.preprocess_displacement_map(restored_d_map, apply_clahe=False)

    # Plot the restored image
    plot_utils.plot_displacement_map(
        restored_d_map,
        title='Restored Displacement Map',
        cmap='gray',  # Other cmaps: 'viridis', 'coolwarm',
        save_plot=True
    )

    # Apply Sobel filter to the image
    # sobel_image = en.apply_sobel(d_map.copy())
    #
    # # Plot the Sobel image
    # plot_utils.plot_displacement_map(
    #     sobel_image,
    #     title='Sobel for Restored Displacement Map',
    #     cmap='viridis',  # Other cmaps: 'viridis', 'coolwarm
    #     save_plot=True
    # )

    # Convert the restored displacement map to a Point Cloud
    # point_cloud = convert_utils.displacement_map_to_point_cloud(d_map)

    # Plot the Point Cloud
    # plot_utils.plot_point_cloud(point_cloud, title='Displacement Map Point Cloud')

    # Plot the Point Cloud
    # plot_utils.plotly_point_cloud(point_cloud, title='Restored Displacement Map Point Cloud')
