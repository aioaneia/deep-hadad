import random

import simulation.crack_simulation as crack_simulation
import simulation.elastic_simulation as elastic_simulation
import simulation.erosion_simulation as erosion_simulation

from utils.SyntheticDataset import SyntheticDataset as SyntheticDataset

import utils.cv_file_utils as file_utils
import utils.image_processing as img_utils


class SyntheticDatasetGenerator:
    def __init__(self, displacement_maps_path, cracks_dataset_path, masks_dataset_path,
                 input_training_dataset_path, target_training_dataset_path):
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

        # Do not resize the displacement maps
        self.preserved_d_maps = file_utils.load_displacement_maps_from_directory(
            displacement_maps_path, preprocess=True, resize=False)

        self.crack_d_maps = file_utils.load_crack_displacement_maps_from_directory(
            cracks_dataset_path, preprocess=True)

        self.mask_d_maps = file_utils.load_crack_displacement_maps_from_directory(
            masks_dataset_path, preprocess=True)

    def generate_synthetic_input_target_pairs(self, dataset_size=700, image_size=(256, 256), save_dataset=False):
        """Generate synthetic datasets for training the GAN model"""
        data_sets = {'input': [], 'target': []}

        set_index = 0
        for preserved_d_map in self.preserved_d_maps:
            data_set = self.generate_pairs_from_d_map(preserved_d_map, image_size, dataset_size,
                                                      save_dataset, set_index)

            data_sets['input'].extend(data_set['input'])
            data_sets['target'].extend(data_set['target'])
            set_index += 1

            print("================Augmented Data================")
            patches = []
            if preserved_d_map.shape[0] > 1024 and preserved_d_map.shape[1] > 1024:
                patches = img_utils.extract_patches(preserved_d_map, patch_size=(512, 512), overlap=0.4)
                context_patches = img_utils.extract_patches(preserved_d_map, patch_size=(768, 768), overlap=0.5)
                patches.extend(context_patches[:5])
            elif preserved_d_map.shape[0] > 512 and preserved_d_map.shape[1] > 512:
                patches = img_utils.extract_patches(preserved_d_map, patch_size=(384, 384), overlap=0.3)
            elif preserved_d_map.shape[0] > 384 and preserved_d_map.shape[1] > 384:
                patches = img_utils.extract_patches(preserved_d_map, patch_size=(256, 256), overlap=0.3)

            print(f"Number of patches: {len(patches)}")

            # Filter out low-information patches
            filtered_patches = []
            for patch in patches:
                # Skip patches that are mostly uniform (likely empty background)
                if patch.std() < 0.02:  # Adjust threshold based on your data
                    continue
                filtered_patches.append(patch)

            print(f"Number of patches after filter: {len(filtered_patches)}")

            patches = filtered_patches
            # keep max 10 patches
            patches = patches[:5]

            for patch in patches:
                # Generate damage for each patch
                patch_dataset = self.generate_pairs_from_d_map(patch, image_size, 20,
                                                               save_dataset, set_index)

                data_sets['input'].extend(patch_dataset['input'])
                data_sets['target'].extend(patch_dataset['target'])
                set_index += 1

            print("================Augmented Data End================")

        dataset_generator = SyntheticDataset(data_sets['input'], data_sets['target'])

        return dataset_generator

    def generate_pairs_from_d_map(self, d_map, d_map_size, dataset_size, save_dataset=False, set_index=0):
        dataset = {'input': [], 'target': []}
        pair_index = 1

        syn_damaged_d_maps = self.generate_damage_simulations_for_d_map(
            d_map,
            erosion_iterations=1,
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

    def generate_damage_simulations_for_d_map(self, d_map, erosion_iterations=3, dataset_size=100):
        """Generate synthetic displacement maps by simulating damage on the glyph displacement map"""
        syn_d_maps = []

        # ----------------- Apply Erosion simulation ----------------- #
        for i in range(erosion_iterations):
            syn_eroded_d_map = erosion_simulation.simulate_cv2_erosion(
                d_map,
                kernel_size_range=(6, 10),
                intensity=0.5,
                iterations=i
            )

            syn_d_maps.append(syn_eroded_d_map)
        # ----------------- End erosion simulation ----------------- #

        # ----------------- Apply Crack simulation ----------------- #
        crack_d_maps = self.get_crack_d_maps(30)
        mask_d_maps = self.get_masks_d_maps(2)

        syn_crack_d_maps = []

        for syn_d_map in syn_d_maps:
            if (len(syn_d_maps) + len(syn_crack_d_maps)) >= dataset_size:
                break

            for crack_d_map in crack_d_maps:
                syn_crack_d_map = crack_simulation.simulate_crack(syn_d_map, crack_d_map)

                syn_crack_d_map = elastic_simulation.apply_elastic_transform_2d(
                    syn_crack_d_map,
                    alpha=30,
                    sigma=8
                )

                syn_crack_d_maps.append(syn_crack_d_map)

            for mask_d_map in mask_d_maps:
                syn_mask_d_map = crack_simulation.apply_mask(syn_d_map, mask_d_map)

                syn_mask_d_map = elastic_simulation.apply_elastic_transform_2d(
                    syn_mask_d_map,
                    alpha=30,
                    sigma=8
                )

                syn_crack_d_maps.append(syn_mask_d_map)

        syn_d_maps.extend(syn_crack_d_maps)

        # return a subset of the synthetic displacement
        syn_d_maps = random.sample(syn_d_maps, dataset_size)

        return syn_d_maps

    def get_crack_d_maps(self, size):
        size = min(size, len(self.crack_d_maps))

        return random.sample(self.crack_d_maps, size)

    def get_masks_d_maps(self, size):
        size = min(size, len(self.mask_d_maps))

        return random.sample(self.mask_d_maps, size)

