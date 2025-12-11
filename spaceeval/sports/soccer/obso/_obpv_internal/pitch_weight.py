import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import sqrt, pi

def sigmoid(x, a, b):
    """
    Computes the sigmoid function with scaling and shifting parameters.

    Parameters:
    x (float or np.ndarray): Input value(s).
    a (float): Scaling parameter.
    b (float): Shifting parameter.

    Returns:
    np.ndarray: Sigmoid function output.
    """
    return 1 / (1 + np.exp(-(x - b) / a))

def generate_heatmap_data(field_dimen=(105.0, 68.0), n_grid_cells_x=50):
    """
    Generate heatmap data for a soccer field based on Gaussian distribution
    and sigmoid scaling factors.

    Parameters:
    field_dimen (tuple): Dimensions of the soccer field (length, width).
    n_grid_cells_x (int): Number of grid cells along the x-axis.

    Returns:
    tuple:
        - np.ndarray: 2D array of pitch values representing the heatmap.
        - np.ndarray: x-axis grid values.
        - np.ndarray: y-axis grid values.
    """

    plt.rcParams.update({'font.size': 15})
    # Calculate grid dimensions and spacing
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y

    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2 + dx / 2
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2 + dy / 2

    # Compute sigmoid weights and scales for x-axis
    scale_weights = sigmoid(xgrid, 30, -15)
    scales = 34 * scale_weights + 34
    x_weights = sigmoid(xgrid, 30, -15)

    # Compute Gaussian distribution for y-axis
    y_center = ygrid[len(ygrid) // 2]
    y_values = np.array([norm.pdf(ygrid, loc=y_center, scale=scale) for scale in scales])

    # Normalize and apply x_weights
    norm_factors = 1 / (scales * sqrt(2 * pi))
    pitch_value = (y_values / norm_factors[:, None]) * x_weights[:, None]

    return pitch_value.T, xgrid, ygrid