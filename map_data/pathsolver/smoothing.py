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
    *previous* sweep's neighbours at once. The original scalar loop was Gauss-Seidel
    (it read the already-updated ``i - 1`` point within the same sweep), which converges
    in fewer sweeps but cannot be vectorised. Both schemes share the same fixed point, so
    the returned path is unchanged up to the convergence tolerance; Jacobi just needs
    roughly two to three times as many (far cheaper) sweeps to get there, which also means
    ``collision_check_func`` is called correspondingly more often.

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
        # Slices of the interior points and of their two neighbours. For paths shorter
        # than three points these are all empty, so the sweep is a no-op and the loop
        # exits immediately -- matching the scalar `range(1, len(path) - 1)`.
        interior = new_path[1:-1]
        delta = weight_data * (path[1:-1] - interior) + weight_smooth * (
            new_path[:-2] + new_path[2:] - 2.0 * interior
        )
        # Assign rather than `+=` so an integer input path keeps the scalar version's
        # silent truncation instead of raising on the in-place float cast.
        new_path[1:-1] = interior + delta
        change = float(np.abs(delta).sum())

        if collision_check_func:
            if collision_check_func(LineString(new_path)):
                return best_path  # Return best collision-free state found so far
            best_path = np.copy(new_path)

    return new_path
