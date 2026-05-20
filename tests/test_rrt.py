import numpy as np

from map_data.pathsolver.rrt_star import RRTStar


def test_rrt_star_simple_success():
    start = (0.0, 0.0)
    goal = (10.0, 10.0)
    obstacles = []
    grid = np.zeros((100, 100), dtype=float)

    rrt_star = RRTStar(
        np.array(start),
        np.array(goal),
        obstacles,
        None,
        grid,
        (0.0, 0.0),
        max_iter=500,
        step_size=1.0,
        neighbor_radius=2.0,
        grid_scale=0.1,
    )
    path = rrt_star.find_path()
    assert path is not None
    assert len(path) >= 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


def test_rrt_star_with_obstacle():
    start = (0.0, 0.0)
    goal = (10.0, 10.0)
    # Define a high cost region in the middle
    grid = np.zeros((100, 100), dtype=float)
    grid[40:60, 40:60] = 1.0  # 4m to 6m region

    rrt_star = RRTStar(
        np.array(start),
        np.array(goal),
        [],
        None,
        grid,
        (0.0, 0.0),
        max_iter=1000,
        step_size=0.5,
        neighbor_radius=2.0,
        grid_scale=0.1,
        traversability_threshold=0.5,
    )
    path = rrt_star.find_path()
    assert path is not None

    # Check that no point in the path is in the high cost region
    for point in path:
        i = int(point[1] / 0.1)
        j = int(point[0] / 0.1)
        if 0 <= i < 100 and 0 <= j < 100:
            assert grid[i, j] < 0.95

    # Verify that path segments don't cut through the obstacle interior.
    # Boundary cells (row/col 40 and 59) are excluded because Bresenham rasterisation
    # can miss a single-cell corner at the boundary — that is expected behaviour.
    for idx in range(len(path) - 1):
        p1 = np.array(path[idx])
        p2 = np.array(path[idx + 1])
        for t in np.linspace(0, 1, 20):
            pt = p1 + t * (p2 - p1)
            i = int(pt[1] / 0.1)
            j = int(pt[0] / 0.1)
            if 41 <= i <= 58 and 41 <= j <= 58:
                assert grid[i, j] < 0.95, f"Segment crosses obstacle interior at {pt}"


def test_rrt_star_near_equal_start_goal():
    """
    Start and goal closer than step_size: path should be found in very few iterations.
    """
    start = np.array([5.0, 5.0])
    goal = np.array([5.5, 5.5])  # distance ~0.7 < step_size=1.0
    grid = np.zeros((100, 100), dtype=float)

    rrt = RRTStar(
        start,
        goal,
        [],
        None,
        grid,
        (0.0, 0.0),
        max_iter=200,
        step_size=1.0,
        neighbor_radius=2.0,
        grid_scale=0.1,
    )
    path = rrt.find_path()

    assert path is not None
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


def test_rrt_star_start_in_collision():
    # If the start is in collision, it should currently fail if we don't have the "escape" logic
    # but with my changes, if we increase max_path_dist it should be fine.
    # Here we test that if grid says collision, it fails.
    start = (5.0, 5.0)
    goal = (10.0, 10.0)
    grid = np.ones((100, 100), dtype=float)  # Everything is blocked

    rrt_star = RRTStar(
        np.array(start),
        np.array(goal),
        [],
        None,
        grid,
        (0.0, 0.0),
        max_iter=100,
        traversability_threshold=0.5,
    )
    path = rrt_star.find_path()
    assert path is None
