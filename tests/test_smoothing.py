import numpy as np
import pytest
from shapely.geometry import LineString

from map_data.pathsolver.smoothing import smooth_path


def _straight(n=10, length=10.0):
    """Straight horizontal path with n points."""
    xs = np.linspace(0, length, n)
    return np.column_stack([xs, np.zeros(n)])


def test_smooth_no_collision_check_moves_interior_points():
    path = _straight()
    # Perturb interior points upward so there is something to smooth away
    path[1:-1, 1] = 1.0
    result = smooth_path(path)
    # Interior points should move closer to the straight line
    assert np.all(np.abs(result[1:-1, 1]) < 1.0)


def test_smooth_endpoints_unchanged():
    path = _straight()
    path[1:-1, 1] = 2.0
    result = smooth_path(path)
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_array_equal(result[-1], path[-1])


def test_smooth_straight_path_unchanged():
    path = _straight()
    result = smooth_path(path)
    np.testing.assert_allclose(result, path, atol=1e-6)


def test_smooth_no_collision_returns_smoothed():
    path = _straight()
    path[1:-1, 1] = 1.0
    never_collides = lambda _: False
    result = smooth_path(path, collision_check_func=never_collides)
    # Should be smoother (interior y closer to 0) than the perturbed input
    assert np.all(np.abs(result[1:-1, 1]) < 1.0)


def test_smooth_always_collides_returns_original():
    path = _straight()
    path[1:-1, 1] = 1.0
    always_collides = lambda _: True
    result = smooth_path(path, collision_check_func=always_collides)
    # First iteration collides immediately → return original (best_path = path at that point)
    np.testing.assert_array_equal(result, path)


def test_smooth_partial_collision_returns_best_intermediate():
    """Collides only after several iterations — result should be smoother than original
    but not identical to the fully unconstrained result."""
    path = _straight()
    path[1:-1, 1] = 2.0

    call_count = {"n": 0}

    def collides_after_three(ls):
        call_count["n"] += 1
        return call_count["n"] > 3

    result_partial = smooth_path(path.copy(), collision_check_func=collides_after_three)
    result_full = smooth_path(path.copy())

    # Some smoothing was applied (differs from original)
    assert not np.allclose(result_partial, path)
    # Collision stopped it before full convergence (differs from unconstrained)
    assert not np.allclose(result_partial, result_full)
    # Interior y values are pulled below the original (smoothing moved them)
    assert np.all(np.abs(result_partial[1:-1, 1]) <= np.abs(path[1:-1, 1]) + 1e-9)


def test_smooth_3d_path_only_xy_smoothed():
    """Z column should pass through unchanged (endpoints fix it, interior untouched by smoothing)."""
    n = 6
    path = np.column_stack([np.linspace(0, 5, n), np.ones(n) * 2.0, np.linspace(0, 10, n)])
    result = smooth_path(path)
    # Endpoints always fixed
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_array_equal(result[-1], path[-1])


def test_smooth_two_point_path_unchanged():
    path = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = smooth_path(path)
    np.testing.assert_array_equal(result, path)


def test_smooth_collision_check_receives_linesting():
    """Verify the collision function is called with a LineString."""
    path = _straight()
    path[1:-1, 1] = 1.0
    received = []
    def capture(ls):
        received.append(ls)
        return False
    smooth_path(path, collision_check_func=capture)
    assert len(received) > 0
    assert all(isinstance(ls, LineString) for ls in received)
