import logging
import random

import numpy as np
import torch

import simulation.crack_simulation as crack_simulation
import simulation.elastic_simulation as elastic_simulation
import simulation.erosion_simulation as erosion_simulation
import simulation.weathering_simulation as weathering_simulation
import simulation.augmentation_utils as aug_utils

from utils.SyntheticDataset import SyntheticDataset as SyntheticDataset

import utils.cv_file_utils as file_utils


class SyntheticDatasetGenerator:
    def __init__(self,
                 displacement_maps_path,
                 cracks_dataset_path,
                 masks_dataset_path,
                 input_training_dataset_path,
                 target_training_dataset_path):

        self.displacement_maps_path       = displacement_maps_path
        self.cracks_dataset_path          = cracks_dataset_path
        self.masks_dataset_path           = masks_dataset_path
        self.input_training_dataset_path  = input_training_dataset_path
        self.target_training_dataset_path = target_training_dataset_path

        self.paths = [
            input_training_dataset_path,
            target_training_dataset_path
        ]

        file_utils.validate_directories(self.paths)

        self.preserved_d_maps = file_utils.load_displacement_maps_from_directory(
            displacement_maps_path,
            preprocess=True,
            resize=True
        )

        self.crack_d_maps = file_utils.load_crack_displacement_maps_from_directory(
            cracks_dataset_path,
            preprocess=True
        )

        self.mask_d_maps = file_utils.load_crack_displacement_maps_from_directory(
            masks_dataset_path,
            preprocess=True
        )


    def get_real_input_target_pairs(self, image_size=(256, 256), save_dataset=False):
        """
        Load the real displacement maps and create pairs of intact-damaged displacement maps
        """

        data_sets = {
            'input': [],
            'target': [],
            'segmap': []
        }

        set_index = 0

        real_damaged_d_maps = file_utils.load_displacement_maps_from_directory(
            'data/glyphs_dataset/damaged_glyphs/input_d_maps/',
            preprocess=True)

        real_preserved_d_maps = file_utils.load_displacement_maps_from_directory(
            'data/glyphs_dataset/damaged_glyphs/target_d_maps/',
            preprocess=True)

        # Iterate through the real displacement maps and create pairs of intact-damaged displacement maps
        for i in range(len(real_damaged_d_maps)):
            input_d_map_pair = real_damaged_d_maps[i]
            target_d_map_pair = real_preserved_d_maps[i]

            # resize the displacement maps to the target size
            # input_d_map_pair = file_utils.resize_and_pad_depth_map(input_d_map_pair, target_size=image_size)
            # target_d_map_pair = file_utils.resize_and_pad_depth_map(target_d_map_pair, target_size=image_size)

            input_d_map_pair_tensor = file_utils.transform_displacement_map_to_tensor(input_d_map_pair)
            target_d_map_pair_tensor = file_utils.transform_displacement_map_to_tensor(target_d_map_pair)

            data_sets['input'].append(input_d_map_pair_tensor)
            data_sets['target'].append(target_d_map_pair_tensor)

            # Create segmentation maps (random for testing)
            for _ in range(len(data_sets['input'])):
                # Simple uniform map or a random segmentation map
                segmap = np.ones((1, *image_size))  # Replace with actual segmentation logic
                data_sets['segmap'].append(torch.tensor(segmap, dtype=torch.float32))  # Correct key used here

            if save_dataset:
                file_utils.save_paired_images(
                    input_d_map_pair,
                    target_d_map_pair,
                    self.input_training_dataset_path,
                    self.target_training_dataset_path,
                    set_index,
                    1)

            set_index += 1

        dataset_generator = SyntheticDataset(data_sets['input'], data_sets['target'], data_sets['segmap'])

        return dataset_generator

    def generate_synthetic_input_target_pairs(self,
                                              dataset_size=700,  # limit of the number of synthetic displacement maps
                                              image_size=(256, 256),  # size of the displacement map
                                              save_dataset=False):
        """
        Generate synthetic datasets for training the GAN model
            - For each preserved displacement map, generate pairs of intact-damaged pairs
            - Save the pairs to the specified directories
        """

        data_sets = {
            'input': [],
            'target': [],
            'segmap': []
        }

        set_index = 0

        for preserved_d_map in self.preserved_d_maps:
            data_set = self.generate_pairs_from_d_map(
                preserved_d_map,
                image_size,
                dataset_size,
                save_dataset,
                set_index
            )

            # Append the generated data to the dataset
            data_sets['input'].extend(data_set['input'])
            data_sets['target'].extend(data_set['target'])

            set_index += 1

            # check if the preserved_d_map size is larger than image_size
            if preserved_d_map.shape[0] > 700 and preserved_d_map.shape[1] > 700:
                # print the preserved displacement map size
                print(f"Preserved displacement map size: {preserved_d_map.shape}")
                print(preserved_d_map.shape[0])
                print(preserved_d_map.shape[1])

                for i in range(3):
                    augmented_d_map = aug_utils.augment_preserved_glyph_image(preserved_d_map.copy())

                    aug_data_set = self.generate_pairs_from_d_map(
                        augmented_d_map,
                        image_size,
                        dataset_size / 5,
                        save_dataset,
                        set_index
                    )

                    # Append the generated data to the dataset
                    data_sets['input'].extend(aug_data_set['input'])
                    data_sets['target'].extend(aug_data_set['target'])

                    # Create segmentation maps
                    for _ in range(len(aug_data_set['input'])):
                        # Simple uniform map or a random segmentation map
                        segmap = np.ones((1, *image_size))  # Replace with actual segmentation logic
                        data_sets['segmap'].append(torch.tensor(segmap, dtype=torch.float32))

                    set_index += 1

            # Create segmentation maps (random for testing)
            for _ in range(len(data_set['input'])):
                # Simple uniform map or a random segmentation map
                segmap = np.ones((1, *image_size))  # Replace with actual segmentation logic
                data_sets['segmap'].append(torch.tensor(segmap, dtype=torch.float32))  # Correct key used here

        dataset_generator = SyntheticDataset(data_sets['input'], data_sets['target'], data_sets['segmap'])

        return dataset_generator

    def generate_pairs_from_d_map(self,
                                  d_map,  # preserved displacement map
                                  d_map_size,  # size of the displacement map
                                  dataset_size,  # limit of the number of synthetic displacement maps to generate
                                  save_dataset=False,  # save the dataset to the specified directories
                                  set_index=0):
        dataset = {
            'input': [],
            'target': []
        }
        pair_index = 1

        # Generate synthetic damaged displacement maps for the preserved displacement map
        syn_damaged_d_maps = self.generate_damage_simulations_for_d_map(
            d_map,
            erosion_iterations=3,
            dataset_size=dataset_size
        )

        target_d_map_pair = file_utils.resize_and_pad_depth_map(d_map, target_size=d_map_size)

        # Create displacement map pairs
        for syn_damaged_d_map in syn_damaged_d_maps:
            # resize the displacement maps to the target size
            input_d_map_pair  = file_utils.resize_and_pad_depth_map(syn_damaged_d_map, target_size=d_map_size)

            input_d_map_pair_tensor = file_utils.transform_displacement_map_to_tensor(input_d_map_pair)
            target_d_map_pair_tensor = file_utils.transform_displacement_map_to_tensor(target_d_map_pair)

            dataset['input'].append(input_d_map_pair_tensor)
            dataset['target'].append(target_d_map_pair_tensor)

            if save_dataset:
                file_utils.save_paired_images(
                    input_d_map_pair,
                    target_d_map_pair,
                    self.input_training_dataset_path,
                    self.target_training_dataset_path,
                    set_index,
                    pair_index)

            pair_index += 1

        print(f"Generated synthetic displacement map pair set: {set_index}")

        return dataset

    def generate_damage_simulations_for_d_map(self,
                                              d_map,
                                              erosion_iterations=3,
                                              dataset_size=400):
        """
        Generate synthetic displacement maps by simulating damage on the glyph displacement map
        """

        syn_d_maps = []

        # ----------------- Apply Erosion simulation ----------------- #
        for i in range(erosion_iterations):
            syn_eroded_d_map = erosion_simulation.simulate_cv2_erosion(
                d_map,
                kernel_size_range=(10, 16),
                intensity=1.0,
                iterations=i
            )
            syn_eroded_d_map = weathering_simulation.water_erosion_channels(
                syn_eroded_d_map,
                num_channels=1,
                depth=0.2
            )

            syn_d_maps.append(syn_eroded_d_map)
        # ----------------- End erosion simulation ----------------- #

        # ----------------- Apply patina simulation ----------------- #
        thickness = 0.1
        coverage  = 0.1
        for i in range(2):
            syn_weathered_d_map = weathering_simulation.patina_formation(
                d_map,
                thickness=thickness,
                coverage=coverage
            )
            syn_weathered_d_map = weathering_simulation.water_erosion_channels(
                syn_weathered_d_map,
                num_channels=2,
                depth=0.25
            )

            syn_d_maps.append(syn_weathered_d_map)

            thickness += 0.02
            coverage  += 0.05
        # ----------------- End patina simulation ----------------- #

        # ----------------- Apply biological growth simulation ----------------- #
        coverage  = 0.1
        thickness = 0.3
        for i in range(2):
            syn_weathered_d_map = weathering_simulation.biological_growth(
                d_map,
                coverage  = coverage,
                thickness = thickness
            )
            syn_weathered_d_map = weathering_simulation.water_erosion_channels(
                syn_weathered_d_map,
                num_channels=2,
                depth=0.2
            )
            syn_d_maps.append(syn_weathered_d_map)

            coverage  += 0.1
            thickness += 0.1
        # ----------------- End biological growth simulation ----------------- #

        # ----------------- Apply water erosion simulation ----------------- #
        syn_weathered_d_map = weathering_simulation.water_erosion_channels(
            d_map,
            num_channels=5,
            depth=0.4
        )
        syn_d_maps.append(syn_weathered_d_map)

        # ----------------- End water erosion simulation ----------------- #

        # ----------------- Apply Crack simulation ----------------- #
        crack_d_maps = self.get_crack_d_maps(20)
        mask_d_maps = self.get_masks_d_maps(15)

        syn_crack_d_maps = []

        for syn_d_map in syn_d_maps:
            if (len(syn_d_maps) + len(syn_crack_d_maps)) >= dataset_size:
                break

            for crack_d_map in crack_d_maps:
                syn_crack_d_map = crack_simulation.simulate_crack(
                    syn_d_map,
                    crack_d_map
                )

                syn_crack_d_map = elastic_simulation.apply_elastic_transform_2d(
                    syn_crack_d_map,
                    alpha=200,
                    sigma=10
                )

                syn_crack_d_maps.append(syn_crack_d_map)

            for mask_d_map in mask_d_maps:
                syn_mask_d_map = crack_simulation.apply_mask(
                    syn_d_map,
                    mask_d_map
                )

                syn_mask_d_map = elastic_simulation.apply_elastic_transform_2d(
                    syn_mask_d_map,
                    alpha=180,
                    sigma=10
                )

                syn_crack_d_maps.append(syn_mask_d_map)

        syn_d_maps.extend(syn_crack_d_maps)

        return syn_d_maps

    def get_crack_d_maps(self, size):
        size = min(size, len(self.crack_d_maps))

        return random.sample(self.crack_d_maps, size)

    def get_masks_d_maps(self, size):
        size = min(size, len(self.mask_d_maps))

        return random.sample(self.mask_d_maps, size)


####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_path = '../'

    # displacement_maps_path = project_path + 'data/glyph_dataset/preserved_glyphs/displacement_maps/'
    #
    # crack_d_map_dataset_path = project_path + 'data/masks_dataset/'
    #
    # input_training_dataset_path = project_path + 'data/training_dataset/X/'
    #
    # target_training_dataset_path = project_path + 'data/training_dataset/Y/'
    #
    # generator = SyntheticDatasetGenerator(
    #     displacement_maps_path,
    #     crack_d_map_dataset_path,
    #     input_training_dataset_path,
    #     target_training_dataset_path)
    #
    # dataset = generator.generate_synthetic_input_target_pairs(
    #     dataset_size=700,
    #     image_size=(512, 512),
    #     save_dataset=True
    # )
