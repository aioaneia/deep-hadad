
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

import simulation.erosion_simulation as erosion_simulation


glyph_d_map_path = '../data/test_dataset/Real Glyphs/test_3.png'
crack_d_map_dataset_path = '../data/masks_dataset/'


def test_erosion_simulation():
    original_map = file_utils.load_displacement_map(
        glyph_d_map_path,
        preprocess=False,
        resize=False,
        apply_clahe=False
    )

    plot_utils.plot_displacement_map(
        original_map,
        title='Rel 3D Geometry',
        cmap='gray'
    )

    d_map = file_utils.load_displacement_map(
        glyph_d_map_path,
        preprocess=True,
        resize=False
    )

    max_iterations = 5

    for i in range(max_iterations):
        # Simulate erosion of the glyph displacement map
        syn_eroded_d_map = erosion_simulation.simulate_cv2_erosion(
            d_map,
            kernel_size_range=(14, 18),
            intensity=1.0,
            iterations=i + 1)

        # Plot the eroded displacement map
        plot_utils.plot_displacement_map(syn_eroded_d_map, title='Eroded Displacement Map', cmap='gray')
        plot_utils.plot_displacement_map_geometry_in_3d(syn_eroded_d_map, title='Eroded 3D Geometry')

    assert True


def test_top_hat_transform():
    d_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=True, resize=False)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map', cmap='gray')
    plot_utils.plot_displacement_map_geometry_in_3d(d_map, title='Well Preserved 3D Geometry')

    # The number of iterations for top hat transform to
    max_iterations = 11
    min_iterations = 2

    # iterate backwards from max_iterations to min_iterations
    for i in range(max_iterations, min_iterations, -1):
        syn_eroded_d_map = erosion_simulation.top_hat_transform(
            d_map,
            kernel_size_range=(14, 18),
            intensity=1.0,
            iterations=i)

        plot_utils.plot_displacement_map(syn_eroded_d_map, title='Eroded Displacement Map'+str(i), cmap='gray')
        plot_utils.plot_displacement_map_geometry_in_3d(syn_eroded_d_map, title='Eroded 3D Geometry' + str(i))

    assert True


if __name__ == "__main__":
    #test_erosion_simulation()

    test_top_hat_transform()

