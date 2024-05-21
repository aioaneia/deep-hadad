import logging

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
                 crack_d_map_dataset_path,
                 input_training_dataset_path,
                 target_training_dataset_path):

        self.displacement_maps_path = displacement_maps_path
        self.crack_d_map_dataset_path = crack_d_map_dataset_path
        self.input_training_dataset_path = input_training_dataset_path
        self.target_training_dataset_path = target_training_dataset_path
        self.paths = [input_training_dataset_path, target_training_dataset_path]

        file_utils.validate_directories(self.paths)

        self.preserved_d_maps = file_utils.load_displacement_maps_from_directory(
            displacement_maps_path,
            preprocess=True)

        self.crack_d_maps = file_utils.load_crack_displacement_maps_from_directory(
            crack_d_map_dataset_path,
            preprocess=True)

    def generate_synthetic_input_target_pairs(self, size=500, save_dataset=False):
        """
        Generate synthetic datasets for training the GAN model
            - For each preserved displacement map, generate pairs of intact-damaged pairs
            - Save the pairs to the specified directories
        """

        data_sets = {
            'input': [],
            'target': []
        }

        set_index = 0

        for preserved_d_map in self.preserved_d_maps:
            data_set = self.generate_pairs_from_d_map(preserved_d_map, size, save_dataset, set_index)

            # Append the generated data to the dataset
            data_sets['input'].extend(data_set['input'])
            data_sets['target'].extend(data_set['target'])

            set_index += 1

            # check if the preserved_d_map size is larger than 512x512
            if preserved_d_map.shape[0] > 512 and preserved_d_map.shape[1] > 512:
                for i in range(1):
                    augmented_d_map = aug_utils.augment_preserved_glyph_image(preserved_d_map)

                    aug_data_set = self.generate_pairs_from_d_map(augmented_d_map, size/2, save_dataset, set_index)

                    # Append the generated data to the dataset
                    data_sets['input'].extend(aug_data_set['input'])
                    data_sets['target'].extend(aug_data_set['target'])

                set_index += 1

        dataset_generator = SyntheticDataset(data_sets['input'], data_sets['target'])

        return dataset_generator

    def generate_pairs_from_d_map(self, d_map, limit, save_dataset=False, set_index=0):
        dataset = {
            'input': [],
            'target': []
        }
        pair_index = 1

        # Generate synthetic damaged displacement maps for the preserved displacement map
        syn_damaged_d_maps = self.generate_damage_simulations_for_d_map(
            d_map, erosion_iterations=4, morphology_iterations=11, limit=limit
        )

        # Create displacement map pairs
        for syn_damaged_d_map in syn_damaged_d_maps:
            input_d_map_pair = syn_damaged_d_map
            target_d_map_pair = d_map

            # resize the displacement maps to 512x512
            input_d_map_pair = file_utils.resize_and_pad(input_d_map_pair, (512, 512))
            target_d_map_pair = file_utils.resize_and_pad(target_d_map_pair, (512, 512))

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

    def generate_damage_simulations_for_d_map(self, d_map, erosion_iterations=5, morphology_iterations=12, limit=1000):
        """
        Generate synthetic displacement maps by simulating damage on the glyph displacement map
        :param d_map: The glyph displacement map
        :param erosion_iterations: The number of erosion iterations
        :param morphology_iterations: The number of morphology iterations
        :param limit: The maximum number of synthetic displacement maps to generate
        :return: A list of synthetic displacement maps
        """

        syn_d_maps = []

        crack_d_maps = self.get_random_crack_displacement_maps(30)

        # ----------------- Apply erosion simulation ----------------- #
        for i in range(erosion_iterations):
            if len(syn_d_maps) >= limit:
                break

            # Simulate eroded displacement maps
            syn_eroded_d_map = erosion_simulation.simulate_cv2_erosion(
                d_map,
                kernel_size_range=(14, 20),
                intensity=1.0,
                iterations=i)

            syn_d_maps.append(syn_eroded_d_map)

            # Simulate crack displacement maps
            for crack_d_map in crack_d_maps:
                if len(syn_d_maps) >= limit:
                    break

                syn_cracked_glyph_d_map = crack_simulation.simulate_crack(syn_eroded_d_map, crack_d_map)

                # ----------------- Apply Elastic Deformation simulation ----------------- #
                elastic_deformed = elastic_simulation.apply_elastic_transform_2d(
                    syn_cracked_glyph_d_map,
                    alpha=150, sigma=10
                )

                syn_d_maps.append(elastic_deformed)
        # ----------------- End erosion simulation ----------------- #

        # ----------------- Apply morphology simulation ----------------- #
        # The number of iterations for top hat transform to
        max_iterations = morphology_iterations
        min_iterations = 4

        # iterate backwards from max_iterations to min_iterations
        for i in range(max_iterations, min_iterations, -1):
            if len(syn_d_maps) >= limit:
                break

            # Simulate erosion of the glyph displacement map
            syn_eroded_d_map = erosion_simulation.top_hat_transform(
                d_map,
                kernel_size_range=(12, 18),
                intensity=1.0,
                iterations=i)

            syn_d_maps.append(syn_eroded_d_map)
        # ----------------- End morphology simulation ----------------- #

        # ----------------- Apply Surface Roughness simulation ----------------- #
        if len(syn_d_maps) < limit:
            rough_d_map = weathering_simulation.surface_roughness(
                d_map, scale=0.1, octaves=6, persistence=0.5, lacunarity=2.0)
            syn_d_maps.append(rough_d_map)
        # ----------------- End Surface Roughness simulation ----------------- #

        # ----------------- Apply Weathering simulation ----------------- #
        erosion_sizes = [20, 30, 40]

        for erosion_size in erosion_sizes:
            if len(syn_d_maps) >= limit:
                break

            # Simulate weathering of the glyph displacement map
            syn_weathered_d_map = weathering_simulation.simulate_erosion_weathering(
                d_map,
                erosion_size=erosion_size,
                weathering_intensity=0.5,
                curvature_threshold=10
            )

            syn_d_maps.append(syn_weathered_d_map)
        # ----------------- End Weathering simulation ----------------- #

        # ----------------- Apply Thermal Erosion simulation ----------------- #
        thermal_iterations = [20, 40, 60, 80]

        for i in thermal_iterations:
            if len(syn_d_maps) >= limit:
                break

            syn_thermal_eroded_d_map = weathering_simulation.simulate_thermal_erosion_2(
                d_map,
                iterations=i,
                crack_threshold=0.05,
                smoothing_iterations=7,
                smoothing_kernel_size=(7, 7)
            )

            syn_d_maps.append(syn_thermal_eroded_d_map)

            # ----------------- Apply erosion simulation to the thermal eroded displacement map ----------------- #
            for i in range(erosion_iterations):
                if len(syn_d_maps) >= limit:
                    break

                syn_eroded_d_map = erosion_simulation.simulate_cv2_erosion(
                    syn_thermal_eroded_d_map,
                    kernel_size_range=(14, 20),
                    intensity=1.0,
                    iterations=i)

                syn_d_maps.append(syn_eroded_d_map)

                # Simulate crack displacement maps
                for crack_d_map in crack_d_maps:
                    syn_cracked_eroded_d_map = crack_simulation.simulate_crack(
                        syn_eroded_d_map, crack_d_map
                    )

                    syn_d_maps.append(syn_cracked_eroded_d_map)

            # ----------------- End erosion simulation to the thermal eroded displacement map ----------------- #

            # ----------------- Apply Elastic Deformation simulation ----------------- #
            elastic_deformed = elastic_simulation.apply_elastic_transform_2d(
                syn_thermal_eroded_d_map,
                alpha=300, sigma=10
            )

            if len(syn_d_maps) < limit:
                syn_d_maps.append(elastic_deformed)
            # ----------------- End Elastic Deformation simulation ----------------- #
        # ----------------- End Thermal Erosion simulation ----------------- #

        return syn_d_maps

    def get_random_crack_displacement_maps(self, size):
        """
        Extract random crack displacement maps from the crack displacement map list
        """

        # get the first size crack displacement maps
        crack_d_maps = self.crack_d_maps[:size]

        return crack_d_maps


####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_path = '../'

    displacement_maps_path = project_path + 'data/glyph_dataset/preserved_glyphs/displacement_maps/'

    crack_d_map_dataset_path = project_path + 'data/masks_dataset/'

    input_training_dataset_path = project_path + 'data/training_dataset/X/'

    target_training_dataset_path = project_path + 'data/training_dataset/Y/'

    generator = SyntheticDatasetGenerator(
        displacement_maps_path,
        crack_d_map_dataset_path,
        input_training_dataset_path,
        target_training_dataset_path)

    dataset = generator.generate_synthetic_input_target_pairs(
        size=100,
        save_dataset=True)
