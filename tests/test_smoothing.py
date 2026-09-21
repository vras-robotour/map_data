import numpy as np
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

    def never_collides(_):
        return False

    result = smooth_path(path, collision_check_func=never_collides)
    # Should be smoother (interior y closer to 0) than the perturbed input
    assert np.all(np.abs(result[1:-1, 1]) < 1.0)


def test_smooth_always_collides_returns_original():
    path = _straight()
    path[1:-1, 1] = 1.0

    def always_collides(_):
        return True

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
    """Z column passes through unchanged (endpoints fix it, interior untouched by smoothing)."""
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


def _smooth_path_scalar(path, weight_data=0.5, weight_smooth=0.3, tolerance=0.001):
    """Reference scalar Gauss-Seidel implementation (the pre-vectorisation behaviour)."""
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
    return new_path


def test_smooth_matches_scalar_reference():
    """Jacobi and Gauss-Seidel share a fixed point, so both converge to the same path.

    They stop at different sweeps, so agreement is only to the order of the tolerance.
    """
    rng = np.random.default_rng(0)
    for n in (5, 25, 120):
        path = np.column_stack([np.linspace(0, n, n), rng.normal(0.0, 1.0, n)])
        np.testing.assert_allclose(smooth_path(path), _smooth_path_scalar(path), atol=1e-3)


def test_smooth_matches_scalar_reference_3d_and_custom_weights():
    """Same agreement for 3D paths and non-default weights."""
    rng = np.random.default_rng(1)
    path = rng.normal(0.0, 2.0, (40, 3))
    kwargs = {"weight_data": 0.2, "weight_smooth": 0.4, "tolerance": 1e-5}
    np.testing.assert_allclose(
        smooth_path(path, **kwargs), _smooth_path_scalar(path, **kwargs), atol=1e-4
    )


def test_smooth_does_not_mutate_input():
    path = _straight()
    path[1:-1, 1] = 1.0
    original = path.copy()
    smooth_path(path)
    np.testing.assert_array_equal(path, original)


def test_smooth_empty_path():
    path = np.empty((0, 2))
    result = smooth_path(path)
    assert result.shape == (0, 2)


def test_smooth_single_point_path_unchanged():
    path = np.array([[1.0, 2.0]])
    result = smooth_path(path)
    np.testing.assert_array_equal(result, path)


def test_smooth_degenerate_paths_unchanged_with_collision_check():
    """Paths with no interior points converge immediately; the check still runs once."""
    calls = []

    def never_collides(ls):
        calls.append(ls)
        return False

    path = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(smooth_path(path, collision_check_func=never_collides), path)
    assert len(calls) == 1
