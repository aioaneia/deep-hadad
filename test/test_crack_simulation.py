import cv2
import numpy as np
from matplotlib import pyplot as plt

import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

import simulation.crack_simulation as crack_simulation


project_path = '../'

glyph_d_map_path = '../data/test_dataset/Real Glyphs/test_1.png'

crack_d_map_dataset_path = '../data/masks_dataset/'


def test_apply_mask():
    """
    Test the crack simulation function.
    """

    original_map = file_utils.load_displacement_map(
        glyph_d_map_path,
        preprocess=False,
        resize=False,
        apply_clahe=False
    )

    plot_utils.plot_displacement_map(
        original_map,
        title='Real 3D Geometry',
        cmap='gray'
    )

    # Load and preprocess the displacement map of the glyph to be cracked
    d_map = file_utils.load_displacement_map(
        glyph_d_map_path,
        preprocess=True,
        resize=True,
        apply_clahe=False
    )

    # Load the displacement maps of the cracks to be simulated
    crack_d_map_paths = file_utils.get_image_paths(crack_d_map_dataset_path)

    # reverse the crack_d_map_paths
    crack_d_map_paths = crack_d_map_paths[::-1]

    # Limit the number of crack displacement maps to be used for simulation
    # crack_d_map_paths = crack_d_map_paths[:7]

    # Simulate the cracks on the glyph
    for crack_d_map_path in crack_d_map_paths:
        crack_d_map = file_utils.load_displacement_map(
            crack_d_map_path,
            preprocess=True,
            resize=False
        )

        # Simulate the crack on the glyph
        syn_cracked_glyph_d_map = crack_simulation.apply_mask(d_map, crack_d_map)

        # Display the simulated crack on the glyph
        plot_utils.plot_displacement_map(
            syn_cracked_glyph_d_map,
            title='Crack 3D Geometry',
            cmap='gray'
        )

        # plot_utils.plot_displacement_map_geometry_in_3d(
        #     syn_cracked_glyph_d_map,
        #     title='Crack 3D Geometry',
        #     cmap='gray'
        # )

        # plot_comparison(d_map, syn_cracked_glyph_d_map, d_map - syn_cracked_glyph_d_map,
        #                 ['Original Glyph', 'Simulated Crack', 'Difference'])

    assert True


def test_crack_simulation():
    """
    Test the crack simulation function.
    """

    original_map = file_utils.load_displacement_map(
        glyph_d_map_path,
        preprocess=False,
        resize=False,
        apply_clahe=False
    )

    plot_utils.plot_displacement_map(
        original_map,
        title='Real 3D Geometry',
        cmap='gray'
    )

    # Load and preprocess the displacement map of the glyph to be cracked
    d_map = file_utils.load_displacement_map(
        glyph_d_map_path,
        preprocess=True,
        resize=True,
        apply_clahe=False
    )

    # Load the displacement maps of the cracks to be simulated
    crack_d_map_paths = file_utils.get_image_paths(crack_d_map_dataset_path)

    # reverse the crack_d_map_paths
    crack_d_map_paths = crack_d_map_paths[::-1]

    # Limit the number of crack displacement maps to be used for simulation
    # crack_d_map_paths = crack_d_map_paths[:7]

    # Simulate the cracks on the glyph
    for crack_d_map_path in crack_d_map_paths:
        crack_d_map = file_utils.load_displacement_map(
            crack_d_map_path,
            preprocess=True,
            resize=False
        )

        random_crack_d_map_path = np.random.choice(crack_d_map_paths)

        random_crack_d_map = file_utils.load_displacement_map(
            random_crack_d_map_path,
            preprocess=True,
            resize=False
        )

        # Combine the crack maps
        target_shape = d_map.shape
        # crack_d_map = combine_crack_maps([crack_d_map, random_crack_d_map], target_shape)

        # Simulate the crack on the glyph
        syn_cracked_glyph_d_map = crack_simulation.simulate_crack(d_map, crack_d_map)

        # Display the simulated crack on the glyph
        plot_utils.plot_displacement_map(
            syn_cracked_glyph_d_map,
            title='Crack 3D Geometry',
            cmap='gray'
        )

        # plot_utils.plot_displacement_map_geometry_in_3d(
        #     syn_cracked_glyph_d_map,
        #     title='Crack 3D Geometry',
        #     cmap='gray'
        # )

        # plot_comparison(d_map, syn_cracked_glyph_d_map, d_map - syn_cracked_glyph_d_map,
        #                 ['Original Glyph', 'Simulated Crack', 'Difference'])

    assert True


def combine_crack_maps(crack_maps, target_shape):
    """
    Combine multiple crack maps by taking the maximum value at each pixel location.
    Ensures all crack maps are resized to target_shape.
    :param crack_maps: List of 2D numpy arrays with depth values of the cracks
    :param target_shape: Tuple (height, width) to resize crack maps to
    :return: Combined crack map
    """

    resized_crack_maps = [
        cv2.resize(crack_map, target_shape[::-1], interpolation=cv2.INTER_NEAREST) for crack_map in crack_maps
    ]

    combined_crack_map = np.max(resized_crack_maps, axis=0)

    return combined_crack_map


def plot_comparison(original, simulated, difference, titles, cmap='coolwarm'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, data, title in zip(axes, [original, simulated, difference], titles):
        im = ax.imshow(data, cmap=cmap, interpolation='nearest')

        ax.set_title(title)

        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)

    plt.show()


def test_poison_blending():
    d_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=True, resize=False)

    crack_d_map_paths = file_utils.get_image_paths(crack_d_map_dataset_path)

    for crack_d_map_path in crack_d_map_paths:
        crack_d_map = file_utils.load_displacement_map(crack_d_map_path, preprocess=True, resize=False)

        syn_cracked_glyph_d_map = crack_simulation.simulate_crack_with_poisson_blending(d_map, crack_d_map)

        plot_utils.plot_displacement_map(syn_cracked_glyph_d_map, title='Crack 3D Geometry')

    assert True


if __name__ == "__main__":
    """
    Test the crack simulation function.
    """

    # test_crack_simulation()

    test_apply_mask()
