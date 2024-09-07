
import matplotlib.pyplot as plt
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

from torchvision.transforms.v2 import ToPILImage
from generation.synthetic_glyph_generation import SyntheticDatasetGenerator

import simulation.augmentation_utils as aug_utils

# Paths to the dataset and glyph displacement map
project_path                 = '../'

glyphs_for_testing_path      = [
    # '../data/test_dataset/Real Preserved Glyphs/test_t00.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t01.png',
    '../data/test_dataset/Real Glyphs/test_1.png',
    # '../data/test_dataset/Real Glyphs/test_2.png',
    # '../data/test_dataset/Real Glyphs/depthmap_1.png',
    # '../data/test_dataset/Real Glyphs/depthmap_2.png',
    # '../data/test_dataset/Real Glyphs/depthmap_3.png',
    # '../data/test_dataset/Real Glyphs/depthmap_4.png',
    # '../data/test_dataset/Real Glyphs/depthmap_5.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t03.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t3.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t4.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t5.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t6.png',
    # '../data/test_dataset/Real Preserved Glyphs/test_t7.png',
]

displacement_maps_path       = project_path + 'data/glyphs_dataset/preserved_glyphs/displacement_maps/'
crack_d_map_dataset_path     = project_path + 'data/cracks_dataset/'
masks_dataset_path     = project_path + 'data/masks_dataset/'
input_training_dataset_path  = project_path + 'data/training_dataset/X/'
target_training_dataset_path = project_path + 'data/training_dataset/Y/'

# glyph_vertices, glyph_faces = conv_utils.displacement_map_to_mesh(glyph_d_map)

to_pil = ToPILImage()


def test_synthetic_generation():
    glyph_d_map = file_utils.load_displacement_map(
        glyphs_for_testing_path[0],
        preprocess=True, resize=False
    )

    # Plot the original image
    plot_utils.plot_displacement_map(glyph_d_map, title='Original Displacement Map of G(t)', cmap='gray')

    # Plot the original image in 3D
    plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title='Original 3D Geometry of G(t)')

    # Initialize the generator
    generator = SyntheticDatasetGenerator(
        displacement_maps_path,
        crack_d_map_dataset_path,
        masks_dataset_path,
        input_training_dataset_path,
        target_training_dataset_path)

    dataset = None

    # Generate synthetic displacement maps from the original glyph displacement maps
    dataset = generator.generate_synthetic_input_target_pairs(
        dataset_size = 100,
        image_size   = (256, 256),
        save_dataset = True
    )

    for i, glyph_path in enumerate(glyphs_for_testing_path):
        glyph_d_map = file_utils.load_displacement_map(
            glyph_path,
            preprocess=True,
            resize=True
        )

        # Generate synthetic displacement maps from the original glyph displacement map
        dataset = generator.generate_pairs_from_d_map(
            glyph_d_map,
            dataset_size=700,
            save_dataset=True,
            set_index=i,
            d_map_size=(256, 256)
        )

    # Plot a cluster of input synthetic displacement maps (damaged displacement maps)
    plot_displacement_map_cluster(
        dataset['input'],
        title='Generated Input Displacement Maps',
        cmap='gray'
    )

    # Plot a cluster of target synthetic displacement maps (preserved displacement maps)
    plot_displacement_map_cluster(
        dataset['target'],
        title='Generated Target Displacement Maps',
        cmap='gray'
    )


def plot_displacement_map_cluster(displacement_maps, title='Displacement Map Cluster', cmap='viridis'):
    """
    Plot a cluster of displacement maps

    :param displacement_maps: List of displacement maps
    :param title: Title of the plot
    :param cmap: Color map
    """
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))

    for i in range(2):
        for j in range(5):
            index = i * 5 + j
            axs[i, j].imshow(to_pil(displacement_maps[index]), cmap=cmap)
            axs[i, j].set_title(f'Displacement Map {index}')
            axs[i, j].axis('off')

    fig.suptitle(title)
    plt.show()


def test_augmentation():
    for i, glyph_path in enumerate(glyphs_for_testing_path):
        glyph_d_map = file_utils.load_displacement_map(
            glyph_path,
            preprocess=True,
            resize=False
        )

        # check if the preserved_d_map size is larger than image_size
        if glyph_d_map.shape[0] > 512 and glyph_d_map.shape[1] > 512:
            for i in range(10):
                augmented_d_map = aug_utils.augment_preserved_glyph_image(glyph_d_map.copy())

                # Plot the augmented image
                plot_utils.plot_displacement_map(augmented_d_map, title='Augmented Displacement Map', cmap='gray')


        # Generate synthetic displacement maps from the original glyph displacement map

if __name__ == "__main__":

    # Test the synthetic generation of displacement maps
    test_synthetic_generation()

    # Test the augmentation of displacement maps
    # test_augmentation()





