
import simulation.weathering_simulation as weathering_simulation
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

project_path = '../'
dataset_size = 'small'
glyph_d_map_path = '../data/test_dataset/Real Glyphs/test_1.png'
crack_d_map_dataset_path = '../data/masks_dataset/'


def water_erosion_channels():
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')
    plot_utils.plot_displacement_map_geometry_in_3d(d_map, title='Weathered 3D Geometry')

    syn_weathered_d_map = weathering_simulation.water_erosion_channels(
        d_map,
        num_channels=5,
        depth=0.5
    )

    # Plot the weathered displacement map
    plot_utils.plot_displacement_map(syn_weathered_d_map, title='Weathered Displacement Map')
    # plot_utils.plot_heatmap_from_displacement_map(syn_weathered_d_map, title='Weathered Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(syn_weathered_d_map, title='Weathered 3D Geometry')

    assert True


if __name__ == "__main__":

    water_erosion_channels()
