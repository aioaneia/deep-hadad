import configparser
import utils.cv_file_utils as cv_utils
import utils.plot_utils as plot_utils
import utils.cv_convert_utils as conv_utils

import scripts.synthetic_glyph_generation as syn_glyph_gen

project_path = '../'
dataset_size = 'small'

glyph_d_map_path = '../data/preserved_beth_glyph.png'
crack_d_map_path = '../data/masks_dataset/crack_depth_map.png'

config = configparser.ConfigParser()
config.read(project_path + 'config/config.ini')

displacement_maps_path = project_path + config['DEFAULT']['DISPLACEMENT_GLYPH_DATASET_PATH']


def test_for_one_d_map():
    # Load a well-preserved glyph displacement map
    glyph_d_map = cv_utils.load_displacement_map(glyph_d_map_path)
    crack_d_map = cv_utils.load_displacement_map(crack_d_map_path)

    # Simulate erosion, cracks, and missing parts for the well-preserved glyph
    syn_eroded_glyph_d_maps, syn_cracked_glyph_d_maps, syn_eroded_point_clouds = (
        syn_glyph_gen.generate_damage_simulated_data_for_d_map(glyph_d_map, crack_d_map))

    # Plots for the well-preserved glyph
    plot_utils.plot_heatmap_from_displacement_map(glyph_d_map, title='Heatmap of G(t)')
    plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title="3D Geometry of G(t)")
    point_cloud = conv_utils.displacement_map_to_point_cloud(glyph_d_map)
    plot_utils.plot_point_cloud(point_cloud, title="Point Cloud of G(t)")

    # Plot the first 6 images for the synthetic eroded glyph displacement maps
    syn_eroded_glyph_d_maps = syn_eroded_glyph_d_maps[:6]
    plot_utils.plot_heatmap_from_displacement_maps(syn_eroded_glyph_d_maps, title='Heatmap of G_e')
    plot_utils.plot_displacement_maps_geometry_in_3d(syn_eroded_glyph_d_maps, title='3D Geometry of G_e')
    plot_utils.plot_point_clouds(syn_eroded_point_clouds, title='Point Cloud of G_e')

    # Plot the first 6 images for the synthetic cracked glyph displacement maps
    syn_cracked_glyph_d_maps = syn_cracked_glyph_d_maps[:6]
    plot_utils.plot_heatmap_from_displacement_maps(syn_cracked_glyph_d_maps, title='Heatmap of G_c')

    # # Simulate erosion
    # eroded_point_cloud = simulate_gaussian_erosion(point_cloud, erosion_iterations=10, smoothing_sigma=1.0)
    # # Simulate cracks
    # cracked_point_cloud = simulate_cracks(eroded_point_cloud, crack_fraction=0.1, crack_depth=0.2)

    # plot_point_cloud_2(eroded_point_cloud, title="Eroded Point Cloud")
    # plot_point_cloud_2(cracked_point_cloud, title="Eroded and Cracked Point Cloud")
    #
    # # Measure the elevation difference between the glyph and its surroundings
    # glyph_mask = glyph_d_map > 0.5
    # elevation_difference = measure_glyph_elevation_difference(glyph_d_map, glyph_mask)
    # print(f"Elevation difference: {elevation_difference}")

    assert True


####################################################################################################
# Main function
# - Load a good glyph image
####################################################################################################
if __name__ == "__main__":
    # test with one depth map

    # plot using the functions in utils/plot_utils.py
    plot_utils.visualize_with_vtk(glyph_d_map_path)

    # test_for_one_d_map()
