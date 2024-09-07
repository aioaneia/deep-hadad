
import simulation.weathering_simulation as weathering_simulation
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

project_path = '../'
dataset_size = 'small'
glyph_d_map_path = '../data/test_dataset/Real Glyphs/test_1.png'
crack_d_map_dataset_path = '../data/masks_dataset/'


def test_surface_roughness():
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')

    # Generate surface roughness on the displacement map
    rough_d_map = weathering_simulation.surface_roughness(d_map, scale=0.1, octaves=6, persistence=0.5, lacunarity=2.0)

    # Plot the rough displacement map
    plot_utils.plot_displacement_map(rough_d_map, title='Rough Displacement Map')
    # plot_utils.plot_heatmap_from_displacement_map(rough_d_map, title='Rough Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(rough_d_map, title='Rough 3D Geometry')

    assert True


def test_simulate_hydraulic_erosion(iterations=10):
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')

    # Simulate hydraulic erosion of the glyph displacement map
    syn_hydraulic_eroded_d_map = weathering_simulation.simulate_hydraulic_erosion(d_map, iterations=iterations)

    # Plot the hydraulic eroded displacement map
    plot_utils.plot_displacement_map(syn_hydraulic_eroded_d_map, title='Hydraulic Eroded Displacement Map')
    # plot_utils.plot_heatmap_from_displacement_map(syn_hydraulic_eroded_d_map, title='Hydraulic Eroded Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(syn_hydraulic_eroded_d_map, title='Hydraulic Eroded 3D Geometry')

    assert True


def test_simulate_thermal_erosion():
    d_map = file_utils.load_displacement_map(glyph_d_map_path, preprocess=True, resize=False)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')

    thermal_iterations = [15, 20, 30, 40, 50, 55, 60, 65]

    for i in thermal_iterations:
        syn_thermal_eroded_d_map = weathering_simulation.simulate_thermal_erosion_2(
            d_map,
            iterations=i,
            crack_threshold=0.05,
            smoothing_iterations=5,
            smoothing_kernel_size=(9, 9)
        )

        # Plot the thermal eroded displacement map
        plot_utils.plot_displacement_map(syn_thermal_eroded_d_map, title='Thermal Eroded Displacement Map')

        plot_utils.plot_displacement_map_geometry_in_3d(syn_thermal_eroded_d_map, title='Thermal Eroded 3D Geometry')

    assert True


def test_patina_formation():
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')
    plot_utils.plot_displacement_map_geometry_in_3d(d_map, title='Weathered 3D Geometry')

    syn_weathered_d_map = weathering_simulation.patina_formation(
        d_map,
        thickness=0.7,
        coverage=0.7
    )

    # Plot the weathered displacement map
    plot_utils.plot_displacement_map(syn_weathered_d_map, title='Weathered Displacement Map')
    # plot_utils.plot_heatmap_from_displacement_map(syn_weathered_d_map, title='Weathered Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(syn_weathered_d_map, title='Weathered 3D Geometry')

    assert True


def test_biological_growth():
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')
    plot_utils.plot_displacement_map_geometry_in_3d(d_map, title='Weathered 3D Geometry')

    syn_weathered_d_map = weathering_simulation.biological_growth(
        d_map,
        coverage=0.1,
        thickness=1.0
    )

    # Plot the weathered displacement map
    plot_utils.plot_displacement_map(syn_weathered_d_map, title='Weathered Displacement Map')
    # plot_utils.plot_heatmap_from_displacement_map(syn_weathered_d_map, title='Weathered Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(syn_weathered_d_map, title='Weathered 3D Geometry')

    assert True


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

    # test_surface_roughness()

    # test_weathering_simulation()

    # test_simulate_hydraulic_erosion(iterations=200)

    # test_simulate_thermal_erosion()
