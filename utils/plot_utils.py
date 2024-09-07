
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go


PLOTS_PATH = '../data/plots/'


def apply_viridis_colormap(displacement_map, title="Displacement Map", save_image=False):
    """
    Apply Viridis colormap to a displacement map and save it as an image.
    """
    # Apply the Viridis colormap
    colormap = plt.get_cmap('viridis')
    colored_map = colormap(displacement_map)

    # Convert the colormap to an image
    colored_image = (colored_map[:, :, :3] * 255).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(colored_image)

    if save_image:
        image.save(f'{PLOTS_PATH}{title}.png', format='PNG')

    return image


def plot_displacement_map(
        displacement_map,
        title="Displacement Map",
        cmap='viridis',
        save_plot=False):
    """
    Plot a displacement map as a heatmap.
    """

    plt.figure(figsize=(10, 8))
    plt.imshow(displacement_map, cmap=cmap, interpolation='nearest')

    plt.colorbar(label='Elevation')
    plt.title(title)
    plt.axis('off')  # Hide axis for clarity in the article

    if save_plot:
        plt.savefig(f'{PLOTS_PATH}{title}.png', bbox_inches='tight')

    plt.show()


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
def plot_mesh(vertices, faces, title="3D Surface Mesh of G", cmap='coolwarm'):
    """
    Plot a 3D surface mesh from vertices and faces.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap=cmap)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Elevation')
    plt.title(title)
    plt.show()


def plotly_mesh(vertices, faces, title="3D Surface Mesh of G", color='grey'):
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        hoverinfo='x+y+z+text',
        text=[f'Vertex {i}: ({x:.2f}, {y:.2f}, {z:.2f})' for i, (x, y, z) in
              enumerate(zip(vertices[:, 0], vertices[:, 1], vertices[:, 2]))],
        color=color,
        opacity=0.70
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Elevation'
        )
    )

    fig = go.Figure(data=[mesh], layout=layout)

    fig.update_layout(scene_aspectmode='auto')

    # fig.update_layout(
    #     scene_aspectmode='manual',
    #     scene_aspectratio=dict(x=1, y=1, z=0.5),
    #     scene=dict(
    #         xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
    #         yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
    #         zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)")
    #     )
    # )

    # fig.update_layout(
    #     scene_camera=dict(
    #         up=dict(x=0, y=0, z=1),
    #         center=dict(x=0, y=0, z=0),
    #         eye=dict(x=1.25, y=1.25, z=1.25)
    #     ),
    #     scene=dict(
    #         xaxis=dict(showspikes=True),
    #         yaxis=dict(showspikes=True),
    #         zaxis=dict(showspikes=True)
    #     )
    # )

    fig.show()


def plotly_mesh_with_texture(vertices, faces, texture_img, title="3D Surface Mesh with Texture"):
    print("Image shape:", texture_img.shape)  # Debug: Check image shape
    print("First pixel RGB values:", texture_img[0, 0])  # Debug: Check first pixel values

    # Increase contrast and brightness for testing
    # alpha = 2.0  # Contrast control
    # beta = 50  # Brightness control
    # texture_img = cv2.convertScaleAbs(texture_img, alpha=alpha, beta=beta)

    texture_colors = texture_img[
        np.clip((vertices[:, 1] * (texture_img.shape[0] - 1)).astype(int), 0, texture_img.shape[0] - 1),
        np.clip((vertices[:, 0] * (texture_img.shape[1] - 1)).astype(int), 0, texture_img.shape[1] - 1)
    ]

    vertex_color = [
        'rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) if texture_img.dtype == np.float32 else 'rgb({},{},{})'.format(r, g, b)
        for r, g, b in texture_colors.reshape(-1, 3)
    ]

    print("Sample vertex colors:", vertex_color[:10])  # Debug: Check some vertex colors

    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=vertex_color,
        opacity=1
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Elevation'
        )
    )

    fig = go.Figure(data=[mesh], layout=layout)
    fig.update_layout(scene_aspectmode='auto')
    fig.show()


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


def plotly_point_cloud(vertices, title="Point Cloud of G", cmap='gray'):
    point_cloud = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=vertices[:, 2],
            colorscale=cmap,
            opacity=0.8
        )
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Elevation',
            aspectmode='data'
        )
    )

    fig = go.Figure(data=[point_cloud], layout=layout)

    fig.update_layout(scene_aspectmode='auto')

    fig.show()


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

