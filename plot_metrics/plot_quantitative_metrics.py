
import numpy as np
import pandas as pd
import plotly.express as px

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


def transform_to_tensor(displacement_maps):
    """
    Transform a list of displacement maps to tensors
    :param displacement_maps: List of displacement maps
    :return: List of tensors
    """
    return [file_utils.transform_displacement_map_to_tensor(map_tensor) for map_tensor in displacement_maps]


def compute_loss_metrics(damaged_data, reference_data, loss_fn):
    """ Compute specified loss between damaged data and reference data. """
    losses = []

    for damaged_map, ref_map in zip(damaged_data, reference_data):
        damaged_map_tensor = damaged_map.unsqueeze(0)  # Assuming damaged_map is a tensor
        ref_map_tensor = ref_map.unsqueeze(0)  # Assuming ref_map is a tensor

        loss = loss_fn(damaged_map_tensor, ref_map_tensor).item()

        losses.append(loss)

    return losses


def plot_quantitative_metric(synthetic_damage_data, real_damage_data, synthetic_reference_data, real_reference_data):
    """
    Plot a scatter plot comparing SSIM and L1 Loss for synthetic and real data
    """
    l1_loss = LossF.DHadadLossFunctions.l1_loss
    ssim_loss = LossF.DHadadLossFunctions.ssim_loss

    # Compute losses for synthetic data
    print("Computing synthetic data losses...")
    print("Synthetic data size: ", len(synthetic_damage_data),
          "Synthetic reference size: ", len(synthetic_reference_data))

    synthetic_l1_losses = compute_loss_metrics(synthetic_damage_data, synthetic_reference_data, l1_loss)
    synthetic_ssim_losses = compute_loss_metrics(synthetic_damage_data, synthetic_reference_data, ssim_loss)

    # Compute losses for real data
    print("Computing real data losses...")
    print("Real data size: ", len(real_damage_data),
          "Real reference size: ", len(real_reference_data))

    real_l1_losses = compute_loss_metrics(real_damage_data, real_reference_data, l1_loss)
    real_ssim_losses = compute_loss_metrics(real_damage_data, real_reference_data, ssim_loss)

    # Prepare data for plotting
    data = {
        'SSIM': list(synthetic_ssim_losses) + list(real_ssim_losses),
        'L1_Loss': list(synthetic_l1_losses) + list(real_l1_losses),
        'Category': ['Synthetic Damage'] * len(synthetic_ssim_losses) + ['Real Damage'] * len(real_ssim_losses)
    }

    df = pd.DataFrame(data)

    # Visualize this data
    fig = px.scatter(
        df,
        x='SSIM',
        y='L1_Loss',
        color='Category',
        title='Comparison of SSIM and L1 Loss in Synthetic and Real Data',
        labels={'SSIM': 'Structural Similarity (SSIM)', 'L1_Loss': 'L1 Loss'},
        color_discrete_map={'Synthetic Damage': 'purple', 'Real Damage': 'black'}
    )

    fig.show()


def plot_ideal_quantitative_metric():
    """
    Plot a scatter plot comparing SSIM and L1 Loss for synthetic data
    """
    # Simulating some data
    np.random.seed(42)

    data = {
        'SSIM': np.random.normal(loc=0.8, scale=0.1, size=500),  # High SSIM values, close to 1
        'L1_Loss': np.random.normal(loc=0.2, scale=0.05, size=500),  # Simulated L1 loss values
        'Category': ['Synthetic Damage' if i % 2 == 0 else 'Real Damage' for i in range(500)]  # Categorical data
    }

    df = pd.DataFrame(data)

    # Visualize this data
    fig = px.scatter(
        df,
        x='SSIM',
        y='L1_Loss',
        color='Category',
        title='Comparison of SSIM and L1 Loss in Synthetic Data',
        labels={'SSIM': 'Structural Similarity (SSIM)', 'L1_Loss': 'L1 Loss'},
        color_discrete_map={'Synthetic Damage': 'purple', 'Real Damage': 'black'})

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
    real_damaged_d_maps = file_utils.load_displacement_maps_from_directory(
        "../data/glyph_dataset/damaged_glyphs/input_d_maps", preprocess=True, resize=True)
    real_damaged_d_maps = transform_to_tensor(real_damaged_d_maps)

    # Load and preprocess real preserved displacement maps
    real_preserved_d_maps = file_utils.load_displacement_maps_from_directory(
        "../data/glyph_dataset/damaged_glyphs/target_d_maps", preprocess=True, resize=True)
    real_preserved_d_maps = transform_to_tensor(real_preserved_d_maps)

    plot_quantitative_metric(syn_damaged_d_maps, real_damaged_d_maps, syn_preserved_d_maps, real_preserved_d_maps)

    # plot_ideal_quantitative_metric()

    print("Done")
