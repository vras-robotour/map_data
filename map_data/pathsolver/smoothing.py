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

    Each sweep is a vectorised Jacobi update: every interior point is relaxed from the
    *previous* sweep's neighbours at once. ``collision_check_func`` is called once per sweep.

    Parameters
    ----------
    path : np.ndarray
        The path to smooth as an (N, 2) or (N, 3) array.
    collision_check_func : callable, optional
        Returns True if the LineString collides. On a collision the last
        collision-free sweep is returned.
    weight_data : float
        How much to weigh the original path points.
    weight_smooth : float
        How much to weigh the smoothness.
    tolerance : float
        Convergence tolerance, compared against the total absolute movement of a sweep.

    Returns
    -------
    np.ndarray
        The smoothed path.

    """
    new_path = np.copy(path)
    best_path = path  # original is assumed collision-free
    change = tolerance
    while change >= tolerance:
        # Paths shorter than three points have no interior: the sweep is a no-op.
        interior = new_path[1:-1]
        delta = weight_data * (path[1:-1] - interior) + weight_smooth * (
            new_path[:-2] + new_path[2:] - 2.0 * interior
        )
        # Assign rather than `+=`: an integer path truncates instead of raising on the float cast.
        new_path[1:-1] = interior + delta
        change = float(np.abs(delta).sum())

        if collision_check_func:
            if collision_check_func(LineString(new_path)):
                return best_path
            best_path = np.copy(new_path)

    return new_path
