
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

import simulation.crack_simulation as crack_simulation


project_path = '../'

glyph_d_map_path = '../data/test_dataset/Real Glyphs/test_3.png'

crack_d_map_dataset_path = '../data/masks_dataset/'


def test_crack_simulation():
    original_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=False, resize=False, apply_clahe=False)

    plot_utils.plot_displacement_map(original_map, title='Rel 3D Geometry', cmap='gray')

    d_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=True, resize=False, apply_clahe=False)

    crack_d_map_paths = file_utils.get_image_paths(crack_d_map_dataset_path)

    crack_d_map_paths = crack_d_map_paths[:10]

    for crack_d_map_path in crack_d_map_paths:
        crack_d_map = file_utils.load_displacement_map(crack_d_map_path, preprocess=True, resize=False)

        syn_cracked_glyph_d_map = crack_simulation.simulate_crack(d_map, crack_d_map)

        plot_utils.plot_displacement_map(syn_cracked_glyph_d_map, title='Crack 3D Geometry', cmap='gray')

    assert True


def test_poison_blending():
    d_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=True, resize=False)

    crack_d_map_paths = file_utils.get_image_paths(crack_d_map_dataset_path)

    # crack_d_map_paths = crack_d_map_paths[:10]

    for crack_d_map_path in crack_d_map_paths:
        crack_d_map = file_utils.load_displacement_map(crack_d_map_path, preprocess=True, resize=False)

        syn_cracked_glyph_d_map = crack_simulation.simulate_crack_with_poisson_blending(d_map, crack_d_map)

        plot_utils.plot_displacement_map(syn_cracked_glyph_d_map, title='Crack 3D Geometry')

    assert True


if __name__ == "__main__":
    """
    Test the crack simulation function.
    """

    test_crack_simulation()
