
import cv2
import logging
import configparser

import simulation.glyph_erosion_simulation as glyph_erosion

import utils.cv_file_utils as file_utils
import utils.plot_utils as plt_utils
import utils.cv_convert_utils as conv_utils
import utils.plot_utils as plot_utils

project_path = '../'
dataset_size = 'small'
crack_d_map_path = '../data/crack_depth_map.png'
crack_d_map = file_utils.load_displacement_map(crack_d_map_path)

# Read the config file
config = configparser.ConfigParser()
config.read(project_path + 'config/config.ini')

displacement_maps_path = project_path + config['DEFAULT']['DISPLACEMENT_GLYPH_DATASET_PATH']

training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_TRAINING_DATASET_PATH']

input_training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_X_TRAINING_DATASET_PATH']

target_training_dataset_path = project_path + config['DEFAULT'][f'{dataset_size.upper()}_Y_TRAINING_DATASET_PATH']

num_pairs = int(config['DEFAULT']['NUM_OF_PAIRS'])

# Paths to be validated
paths = [training_dataset_path, input_training_dataset_path, target_training_dataset_path]


def generate_synthetic_d_maps(d_maps, nr_pairs_per_d_map=3):
    """
    Generate synthetic displacement maps
        - For each displacement map, generate num_pairs of intact-damaged pairs
        - Save the pairs to the specified directories
    """
    count = 1

    # Iterate through each displacement map
    for i in range(len(d_maps)):
        target_d_map = d_maps[i].copy()

        # Simulate erosion, cracks, and missing parts for the well-preserved glyph
        syn_eroded_glyph_d_maps, syn_cracked_glyph_d_maps, syn_eroded_point_clouds = (
            generate_damage_simulated_data_for_d_map(target_d_map, crack_d_map))

        # Concatenate the synthetic eroded and cracked glyph displacement map arrays
        syn_eroded_cracked_glyph_d_maps = syn_eroded_glyph_d_maps + syn_cracked_glyph_d_maps

        # Create pairs of damaged/repaired displacement maps for
        # each synthetic displacement map from syn_eroded_cracked_glyph_d_maps
        for input_d_map in syn_eroded_cracked_glyph_d_maps:

            # Save input/target pair
            file_utils.save_paired_images(
                input_d_map,
                target_d_map,
                input_training_dataset_path,
                target_training_dataset_path,
                count)

            count += 1

    logging.info("Data generation completed.")

    return


def generate_damage_simulated_data_for_d_map(d_map, crack_d_mask):
    """
    Simulate erosion, cracks, and missing parts for a single displacement map.
    """
    max_iterations = 3
    syn_eroded_glyph_d_maps = []
    syn_cracked_glyph_d_maps = []
    syn_eroded_point_clouds = []

    # Starting from the well-preserved glyph displacement map, iterate over the following damage simulation steps:
    # - Erosion
    # - Cracks
    # - Missing parts
    # Each step should be visualized in 3D and as a heatmap image.
    # The elevation difference between the glyph and its surroundings should be measured after each step.
    for i in range(max_iterations):
        # Simulate erosion of the glyph displacement map
        syn_eroded_d_map = glyph_erosion.simulate_cv2_erosion(
            d_map,
            kernel_size_range=(14, 18),
            intensity=1.0,
            iterations=i + 1)
        syn_eroded_glyph_d_maps.append(syn_eroded_d_map)

        # Simulate cracks in the glyph displacement map
        syn_cracked_glyph_d_map = glyph_erosion.simulate_crack(d_map, crack_d_mask)
        syn_cracked_glyph_d_maps.append(syn_cracked_glyph_d_map)

        # Convert the eroded glyph displacement map to a point cloud
        eroded_point_cloud = conv_utils.displacement_map_to_point_cloud(syn_eroded_d_map)
        syn_eroded_point_clouds.append(eroded_point_cloud)

    return syn_eroded_glyph_d_maps, syn_cracked_glyph_d_maps, syn_eroded_point_clouds


def display_sample_pairs(num_pairs=10):
    # Get images from path
    intact_image_paths = file_utils.get_image_paths(input_training_dataset_path)
    damaged_image_paths = file_utils.get_image_paths(target_training_dataset_path)

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
    plt_utils.display_images(image_pairs, num_cols=2, img_size=(200, 200))


####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Project Path:            {project_path}")
    print(f"Displacement Maps Path:  {displacement_maps_path}")
    print(f"X Training Dataset Path: {input_training_dataset_path}")
    print(f"Y Training Dataset Path: {target_training_dataset_path}")

    # Validate and create directories
    file_utils.validate_directories(paths)

    # Load the displacement images
    displacement_maps = file_utils.load_displacement_maps_from_directory(displacement_maps_path)

    # # Generate synthetic displacement maps
    generate_synthetic_d_maps(displacement_maps, num_pairs)

    # Display sample pairs
    display_sample_pairs(num_pairs=7)
