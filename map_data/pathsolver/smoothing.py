from collections.abc import Callable

import numpy as np
from shapely.geometry import LineString


def smooth_path(
    path: np.ndarray,
    collision_check_func: Callable[[LineString], bool] | None = None,
    weight_data: float = 0.5,
    weight_smooth: float = 0.3,
    tolerance: float = 0.001,
) -> np.ndarray:
    """
    Gradient descent path smoothing.

    Parameters
    ----------
    path : np.ndarray
        The path to smooth as an (N, 2) or (N, 3) array.
    collision_check_func : callable, optional
        A function that takes a LineString and returns True if it collides with obstacles.
        If provided, smoothing will revert to the original path if a collision is detected.
    weight_data : float
        How much to weigh the original path points.
    weight_smooth : float
        How much to weigh the smoothness.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        The smoothed path.
    """
    new_path = np.copy(path)
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):
            for j in range(len(path[i])):
                aux = new_path[i][j]
                new_path[i][j] += weight_data * (path[i][j] - new_path[i][j]) + weight_smooth * (
                    new_path[i - 1][j] + new_path[i + 1][j] - 2.0 * new_path[i][j]
                )
                change += abs(aux - new_path[i][j])

        # Check for collisions after each iteration
        if collision_check_func and collision_check_func(LineString(new_path)):
            return path  # Return original path if smoothing failed safely

    return new_path
