import math

import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F
from plotly.subplots import make_subplots

from models import DHadadLossFunctions as LossF
from utils import cv_file_utils as file_utils


def load_and_preprocess_maps(path_list):
    """
    Load and preprocess a list of displacement maps ordered by the paths

    :param path_list: List of paths to displacement maps
    """
    path_list.sort()

    print(path_list)

    # Load the displacement maps and sort them by the paths
    displacement_maps = [file_utils.load_displacement_map_as_tensor(path) for path in path_list]

    return displacement_maps


def plot_loss_behaviors(damaged_maps, preserved_maps):
    loss_functions = {
        'L1 Loss': LossF.DHadadLossFunctions.l1_loss,
        'SSIM Loss': LossF.DHadadLossFunctions.ssim_loss,
        'MSE Loss': F.mse_loss,
        'LPIPS Loss': LossF.DHadadLossFunctions.lpips_loss,
        'Geometric Consistency Loss': LossF.DHadadLossFunctions.geometric_consistency_loss,
        'Sharpness Loss': LossF.DHadadLossFunctions.sharpness_loss,
        'Edge Loss': LossF.DHadadLossFunctions.edge_loss,
        #'PSNR': psnr
    }

    # Calculate the number of rows and columns needed based on the number of loss functions
    num_rows = int(math.ceil(len(loss_functions) / 2))
    num_cols = 2 if len(loss_functions) > 1 else 1

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=list(loss_functions.keys()))

    for i, (label, loss_fn) in enumerate(loss_functions.items()):
        losses = []

        for damaged_map, preserved_map in zip(damaged_maps, preserved_maps):
            damaged_map_tensor = damaged_map.unsqueeze(0)
            preserved_map_tensor = preserved_map.unsqueeze(0)

            loss = loss_fn(damaged_map_tensor, preserved_map_tensor).item()

            losses.append(loss)

        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.add_trace(go.Scatter(x=list(range(len(damaged_maps))), y=losses, mode='markers', name=label),
                      row=row, col=col)

    fig.update_layout(height=800, width=1000, title_text="Loss Behaviors")
    fig.show()


def plot_ideal_loss_behaviors(num_samples):
    ideal_metrics = {
        'L1 Loss': np.random.normal(loc=0.05, scale=0.02, size=num_samples),  # Low variance near zero
        'SSIM Loss': np.random.normal(loc=0.02, scale=0.01, size=num_samples),  # Low as it is a loss
        'MSE Loss': np.random.normal(loc=0.05, scale=0.02, size=num_samples),
        'LPIPS Loss': np.random.normal(loc=0.1, scale=0.05, size=num_samples),
        'Geometric Consistency Loss': np.random.normal(loc=0.01, scale=0.005, size=num_samples),
        'Sharpness Loss': np.random.normal(loc=0.1, scale=0.05, size=num_samples),
        'Edge Loss': np.random.normal(loc=0.1, scale=0.05, size=num_samples),
    }

    num_rows = int(np.ceil(len(ideal_metrics) / 2))
    num_cols = 2 if len(ideal_metrics) > 1 else 1

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=list(ideal_metrics.keys()))

    for i, (label, values) in enumerate(ideal_metrics.items()):
        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.add_trace(go.Scatter(x=list(range(num_samples)), y=values, mode='markers', name=label),
                      row=row, col=col)

    fig.update_layout(height=800, width=1000, title_text="Ideal Loss Behaviors")
    fig.show()


def simulate_ideal_metric_evolution(num_samples, initial_mean, final_mean, initial_std, final_std):
    """ Generate metrics showing a progression from initial to final values over epochs. """
    means = np.linspace(initial_mean, final_mean, num_samples)
    stds = np.linspace(initial_std, final_std, num_samples)
    values = np.array([np.random.normal(mean, std, 1)[0] for mean, std in zip(means, stds)])
    return values


def plot_ideal_metrics_evolution(num_epochs=100):
    fig = make_subplots(rows=5, cols=1, subplot_titles=[
        'L1 Loss Over Epochs', 'SSIM Loss Over Epochs',
        'Geometric Consistency Loss Over Epochs', 'Sharpness Loss Over Epochs',
        'Edge Loss Over Epochs'])

    # Metrics showing improvements over epochs
    l1_loss = simulate_ideal_metric_evolution(num_epochs, 0.3, 0.05, 0.1, 0.02)
    ssim_loss = simulate_ideal_metric_evolution(num_epochs, 0.2, 0.01, 0.1, 0.02)
    geometric_consistency_loss = simulate_ideal_metric_evolution(num_epochs, 0.2, 0.01, 0.1, 0.02)
    sharpness_loss = simulate_ideal_metric_evolution(num_epochs, 0.2, 0.05, 0.1, 0.02)
    edge_loss = simulate_ideal_metric_evolution(num_epochs, 0.2, 0.05, 0.1, 0.02)

    # Adding traces for each metric
    fig.add_trace(go.Scatter(y=l1_loss, mode='lines', name='L1 Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ssim_loss, mode='lines', name='SSIM Loss'), row=2, col=1)
    fig.add_trace(go.Scatter(y=geometric_consistency_loss, mode='lines', name='Geometric Consistency Loss'), row=3, col=1)
    fig.add_trace(go.Scatter(y=sharpness_loss, mode='lines', name='Sharpness Loss'), row=4, col=1)
    fig.add_trace(go.Scatter(y=edge_loss, mode='lines', name='Edge Loss'), row=5, col=1)

    fig.update_layout(height=1200, width=600, title_text="Ideal Metric Evolution Over Epochs")
    fig.show()


if __name__ == "__main__":
    syn_damaged_d_maps_paths = file_utils.get_image_paths("../data/training_dataset/X")
    syn_preserved_d_maps_paths = file_utils.get_image_paths("../data/training_dataset/Y")

    syn_damaged_d_maps = load_and_preprocess_maps(syn_damaged_d_maps_paths)

    syn_preserved_d_maps = load_and_preprocess_maps(syn_preserved_d_maps_paths)

    plot_ideal_metrics_evolution()

    plot_loss_behaviors(syn_damaged_d_maps, syn_preserved_d_maps)

    plot_ideal_loss_behaviors(200)

    print("Done")
