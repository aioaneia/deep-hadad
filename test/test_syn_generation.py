import matplotlib.pyplot as plt
from torchvision.transforms.v2 import ToPILImage

import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils
from generation.synthetic_glyph_generation import SyntheticDatasetGenerator

project_path = '../'

glyph_d_map_path = '../data/test_dataset/Real Preserved Glyphs/test_1.png'

displacement_maps_path = project_path + 'data/glyph_dataset/preserved_glyphs/displacement_maps/'

crack_d_map_dataset_path = project_path + 'data/masks_dataset/'

input_training_dataset_path = project_path + 'data/training_dataset/X/'

target_training_dataset_path = project_path + 'data/training_dataset/Y/'

# glyph_vertices, glyph_faces = conv_utils.displacement_map_to_mesh(glyph_d_map)

to_pil = ToPILImage()


def test_synthetic_generation():
    glyph_d_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=True, resize=False)

    # Plot the original image
    plot_utils.plot_displacement_map(glyph_d_map, title='Original Displacement Map of G(t)', cmap='gray')

    # Plot the original image in 3D
    plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title='Original 3D Geometry of G(t)')

    # Initialize the generator
    generator = SyntheticDatasetGenerator(
        displacement_maps_path,
        crack_d_map_dataset_path,
        input_training_dataset_path,
        target_training_dataset_path)

    # dataset = generator.generate_synthetic_input_target_pairs(size=300, save_dataset=True)

    # Generate synthetic displacement maps
    dataset = generator.generate_pairs_from_d_map(glyph_d_map, limit=1000, save_dataset=True, set_index=1)

    # Plot a cluster of input synthetic displacement maps (damaged displacement maps)
    plot_displacement_map_cluster(dataset['input'], title='Generated Input Displacement Maps', cmap='gray')

    # Plot a cluster of target synthetic displacement maps (preserved displacement maps)
    plot_displacement_map_cluster(dataset['target'], title='Generated Target Displacement Maps', cmap='gray')


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


if __name__ == "__main__":
    test_synthetic_generation()





