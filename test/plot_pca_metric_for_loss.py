import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from models import DHadadLossFunctions as LossF
from utils import cv_file_utils as file_utils


def load_and_preprocess_maps(path_list):
    """
    Load and preprocess a list of displacement maps ordered by the paths

    :param path_list: List of paths to displacement maps
    """

    path_list.sort()

    # Load the displacement maps and sort them by the paths
    displacement_maps = [file_utils.load_displacement_map_as_tensor(path) for path in path_list]

    return displacement_maps


def plot_pca_matrix(damaged_maps, preserved_maps, sample_size=100):
    # Reduce the number of samples for visualization if too large
    damaged_sample = damaged_maps[:sample_size]
    preserved_sample = preserved_maps[:sample_size]

    # Flatten the displacement maps and reduce dimensionality
    damaged_flattened = [d_map.flatten() for d_map in damaged_sample]
    preserved_flattened = [p_map.flatten() for p_map in preserved_sample]

    # Combine the flattened displacement maps
    data = torch.stack(damaged_flattened + preserved_flattened)

    # Applying PCA to reduce dimensions to a manageable number for plotting
    pca = PCA(n_components=5)
    data_reduced = pca.fit_transform(data)

    # Create scatter matrix plot
    fig = go.Figure(data=go.Splom(
        dimensions=[dict(label=f'PCA {i+1}', values=data_reduced[:, i]) for i in range(data_reduced.shape[1])],
        text=['Damaged' if i < len(damaged_sample) else 'Preserved' for i in range(len(data_reduced))],
        marker=dict(color=['blue' if i < len(damaged_sample) else 'red' for i in range(len(data_reduced))])
    ))

    fig.update_layout(title_text="Scatter Matrix of Displacement Maps (PCA Reduced)")
    fig.show()


if __name__ == "__main__":
    # Load paths and data as per previous implementation
    syn_damaged_d_maps_paths = file_utils.get_image_paths("../data/training_dataset/X")
    syn_preserved_d_maps_paths = file_utils.get_image_paths("../data/training_dataset/Y")

    # Load and preprocess synthetic damaged displacement maps
    syn_damaged_d_maps = load_and_preprocess_maps(syn_damaged_d_maps_paths)
    # Load and preprocess synthetic preserved displacement maps
    syn_preserved_d_maps = load_and_preprocess_maps(syn_preserved_d_maps_paths)

    # Load and preprocess real damaged displacement maps
    real_damaged_d_maps = file_utils.load_displacement_maps_from_directory("../data/glyph_dataset/damaged_glyphs/input_d_maps")

    # Load and preprocess real preserved displacement maps
    real_preserved_d_maps = file_utils.load_displacement_maps_from_directory("../data/glyph_dataset/damaged_glyphs/target_d_maps")

    plot_pca_matrix(syn_damaged_d_maps, syn_preserved_d_maps, sample_size=50)

    print("Done")
