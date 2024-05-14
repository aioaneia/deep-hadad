
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def calculate_metrics(damaged_maps, preserved_maps):
    metrics = {
        'L1 Loss': [],
        'SSIM Loss': [],
        'Geometric Consistency Loss': [],
        'Sharpness Loss': [],
        'Edge Loss': []
    }

    for damaged_map, preserved_map in zip(damaged_maps, preserved_maps):
        damaged_map = damaged_map.unsqueeze(0)  # Add batch dimension
        preserved_map = preserved_map.unsqueeze(0)  # Add batch dimension

        metrics['L1 Loss'].append(LossF.DHadadLossFunctions.l1_loss(damaged_map, preserved_map).item())
        metrics['SSIM Loss'].append(LossF.DHadadLossFunctions.ssim_loss(damaged_map, preserved_map).item())
        metrics['Geometric Consistency Loss'].append(LossF.DHadadLossFunctions.geometric_consistency_loss(damaged_map, preserved_map).item())
        metrics['Sharpness Loss'].append(LossF.DHadadLossFunctions.sharpness_loss(damaged_map, preserved_map).item())
        metrics['Edge Loss'].append(LossF.DHadadLossFunctions.edge_loss(damaged_map, preserved_map).item())

    return metrics


def plot_distribution_histograms(metrics):
    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=[f"{key} Distribution" for key in metrics.keys()])
    row = 1

    for key, values in metrics.items():
        fig.add_trace(go.Histogram(x=values, name=key), row=row, col=1)
        row += 1

    fig.update_layout(height=1500, width=800, title_text="Metrics Distribution")
    fig.show()


def ideal_metric_for_distribution_metrics():
    ideal_metrics = {
        'L1 Loss': np.random.normal(loc=0.1, scale=0.05, size=100),
        'SSIM Loss': np.random.normal(loc=0.05, scale=0.02, size=100),
        'Geometric Consistency Loss': np.random.normal(loc=0.01, scale=0.005, size=100),
        'Sharpness Loss': np.random.normal(loc=0.2, scale=0.1, size=100),
        'Edge Loss': np.random.normal(loc=0.1, scale=0.05, size=100)
    }

    fig = make_subplots(rows=5, cols=1)

    for i, (key, values) in enumerate(ideal_metrics.items(), start=1):
        fig.add_trace(go.Histogram(x=values, name=key), row=i, col=1)

    fig.update_layout(height=1500, title_text="Ideal Metrics Distribution")
    fig.show()


if __name__ == "__main__":
    # Load paths and data as per previous implementation
    syn_damaged_d_maps_paths = file_utils.get_image_paths("../data/training_dataset/X")
    syn_preserved_d_maps_paths = file_utils.get_image_paths("../data/training_dataset/Y")

    syn_damaged_d_maps = load_and_preprocess_maps(syn_damaged_d_maps_paths)

    syn_preserved_d_maps = load_and_preprocess_maps(syn_preserved_d_maps_paths)

    metrics = calculate_metrics(syn_damaged_d_maps, syn_preserved_d_maps)

    plot_distribution_histograms(metrics)

    ideal_metric_for_distribution_metrics()

    print("Done")
