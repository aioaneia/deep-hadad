
import cv2
import numpy as np
import matplotlib.pyplot as plt
import vtk

PLOTS_PATH = '../plots/'


def visualize_with_vtk(depth_map_path):
    # Read the image data from a file
    reader = vtk.vtkPNGReader()
    reader.SetFileName(depth_map_path)

    # Create a mapper that will map the image data to geometry
    mapper = vtk.vtkImageResliceMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    # Create an actor that uses the mapper to render the data
    actor = vtk.vtkImageActor()
    actor.SetMapper(mapper)

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor that allows us to interact with the visualization
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Background color dark blue

    # Start the visualization
    render_window.Render()
    render_window_interactor.Start()


def plot_displacement_map_geometry_in_3d(depth_map, title="Glyph Geometry", cmap='coolwarm'):
    """
    Plot the glyph geometry in 3D based on its depth map.
    The x and y axes represent the pixel coordinates
    The z-axis represents the elevation of the glyph.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(depth_map.shape[1])
    y = np.arange(depth_map.shape[0])

    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, depth_map, cmap=cmap, edgecolor='w', linewidth=0, antialiased=True)

    ax.set_xlabel('Pixel Coordinate X')
    ax.set_ylabel('Pixel Coordinate Y')
    ax.set_zlabel('Elevation')
    ax.set_title(title)

    # Set the viewing angle for better visualization
    # ax.view_init(
    #     elev=45, # Elevation angle in the z plane. Rotates the plot vertically.
    #     azim=90 # Azimuth angle in the x,y plane. Rotates the plot horizontally.
    # )

    # Save the plot as a PNG image in the plots folder
    plt.savefig(f'{PLOTS_PATH}{title}.png', bbox_inches='tight')
    plt.show()


def plot_displacement_maps_geometry_in_3d(depth_maps, title="3D Geometry of G", cmap='coolwarm'):
    """
    Plot the glyph geometry in 3D based on its depth maps.
    The x and y axes represent the pixel coordinates.
    The z-axis represents the elevation of the glyph.
    """
    num_maps = len(depth_maps)
    fig = plt.figure(figsize=(10 * num_maps, 8))

    for i, depth_map in enumerate(depth_maps):
        ax = fig.add_subplot(1, num_maps, i + 1, projection='3d')
        x = np.arange(depth_map.shape[1])
        y = np.arange(depth_map.shape[0])

        x, y = np.meshgrid(x, y)

        surf = ax.plot_surface(x, y, depth_map, cmap=cmap, edgecolor='w', linewidth=0, antialiased=True)

        ax.set_xlabel('Pixel Coordinate X')
        ax.set_ylabel('Pixel Coordinate Y')
        ax.set_zlabel('Elevation')
        ax.set_title(f'{title} ({i + 1})')

        # Set the viewing angle for better visualization
        # ax.view_init(elev=45, azim=90)

        # Adding a color bar to each subplot to make it easier to interpret the elevation data
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')

    plt.savefig(f'{PLOTS_PATH}{title}(i).png', bbox_inches='tight')
    plt.show()


def plot_heatmap_from_displacement_map(displacement_map, title='Displacement Map'):
    """
    Creates a heatmap image from a displacement map.
    """

    plt.figure(figsize=(10, 8))
    plt.imshow(displacement_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Elevation')
    plt.title(title)
    plt.axis('off')  # Hide axis for clarity in the article
    plt.savefig(f'{title}.png', bbox_inches='tight')

    plt.savefig(f'{PLOTS_PATH}{title}.png', bbox_inches='tight')
    plt.show()


def plot_heatmap_from_displacement_maps(displacement_maps, title='Heatmap of G'):
    """
    Creates heatmap images from multiple displacement maps.
    """

    num_maps = len(displacement_maps)
    fig, axes = plt.subplots(1, num_maps, figsize=(10 * num_maps, 8))

    # Ensure that axes is always iterable (important if there's only one subplot)
    if num_maps == 1:
        axes = [axes]

    for i, displacement_map in enumerate(displacement_maps):
        im = axes[i].imshow(displacement_map, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'{title} ({i + 1})')
        axes[i].axis('off')

        # Add a color bar for each subplot
        # fig.colorbar(im, ax=axes[i], orientation='vertical')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}{title}(i).png', bbox_inches='tight')

    plt.show()


####################################################################################################
# Plotting Point Clouds
####################################################################################################
def plot_point_cloud(point_cloud, title="Point Cloud of G", cmap='coolwarm'):
    """
    Plot a point cloud from a 3D numpy array.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud with elevation values as colors
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap=cmap)

    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title(title)

    # Set the viewing angle for better visualization
    # ax.view_init(
    #     elev=45,
    #     azim=90
    # )

    plt.savefig(f'{PLOTS_PATH}{title}.png', bbox_inches='tight')

    plt.show()


def plot_point_clouds(point_clouds, title="Point Cloud Visualization", cmap='coolwarm'):
    """
    Plot multiple point clouds in a single figure.
    """
    num_clouds = len(point_clouds)
    fig = plt.figure(figsize=(10 * num_clouds, 8))

    for i, point_cloud in enumerate(point_clouds):
        ax = fig.add_subplot(1, num_clouds, i + 1, projection='3d')  # Add a new 3D subplot for each point cloud
        # Plot the point cloud with elevation values as colors
        sc = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap=cmap)
        # Set the axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title(f'{title} {i + 1}')

        # Optional: Add color bar if needed
        # fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='Elevation')

        # Set the viewing angle for better visualization
        # ax.view_init(elev=45, azim=90)

    plt.savefig(f'{PLOTS_PATH}{title}(i).png', bbox_inches='tight')

    plt.show()


####################################################################################################
# Display Images
####################################################################################################
def display_images(images, num_cols=4, img_size=(200, 200), titles=None, cmap='gray', first_n=20):
    num_images = len(images)

    if num_images == 0:
        print("No images to display.")
        return

    if num_images > first_n:
        images = images[:first_n]
        num_images = len(images)
        print(f"Displaying first {first_n} images.")

    # Calculate the number of rows required in the grid
    num_rows = int(num_images / num_cols) + int(num_images % num_cols > 0)

    # Calculate dynamic figure size (each subplot of size 2x2)
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        img = images[i]

        # Resize and convert image
        small_img_norm = cv2.resize(img, img_size)
        # small_img_norm = cv2.normalize(small_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        plt.subplot(num_rows, num_cols, i + 1)

        if cmap == 'gray':
            # Display grayscale image
            plt.imshow(small_img_norm, cmap=cmap)
        else:
            # Display image
            plt.imshow(small_img_norm)

        if titles is not None and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')

    plt.show()