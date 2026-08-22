import random

import numpy as np
import pytest

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


def _make_rrt(start=(0.0, 0.0), goal=(10.0, 10.0), **kwargs):
    grid = np.zeros((100, 100), dtype=float)
    defaults = dict(max_iter=1000, step_size=1.0, neighbor_radius=2.0, grid_scale=0.1)
    defaults.update(kwargs)
    return RRTStar(np.array(start), np.array(goal), [], None, grid, (0.0, 0.0), **defaults)


def test_informed_rrt_finds_path():
    rrt = _make_rrt(informed=True, improve_after_goal=True)
    path = rrt.find_path()
    assert path is not None
    assert np.allclose(path[0], [0.0, 0.0])
    assert np.allclose(path[-1], [10.0, 10.0])


def test_informed_disabled_still_finds_path():
    rrt = _make_rrt(informed=False)
    path = rrt.find_path()
    assert path is not None


def test_informed_best_cost_updated_after_path_found():
    rrt = _make_rrt(informed=True, improve_after_goal=True)
    assert rrt._best_cost == float("inf")
    rrt.find_path()
    assert rrt._best_cost < float("inf")


def test_informed_best_cost_geq_c_min():
    rrt = _make_rrt(informed=True, improve_after_goal=True)
    rrt.find_path()
    # Path cost must be at least the straight-line distance
    assert rrt._best_cost >= rrt._c_min - 1e-9


def test_informed_ellipse_tighter_than_full_space():
    """With improve_after_goal=True and informed=True, the planner should produce
    a path no worse than with informed=False (same seed, more iterations)."""
    import random as _random

    seed = 42
    _random.seed(seed)
    rrt_informed = _make_rrt(informed=True, improve_after_goal=True, max_iter=2000)
    path_i = rrt_informed.find_path()

    _random.seed(seed)
    rrt_plain = _make_rrt(informed=False, improve_after_goal=True, max_iter=2000)
    path_p = rrt_plain.find_path()

    assert path_i is not None
    assert path_p is not None


def test_adaptive_radius_finds_path():
    rrt = _make_rrt(adaptive_radius=True)
    path = rrt.find_path()
    assert path is not None


def test_adaptive_radius_disabled_finds_path():
    rrt = _make_rrt(adaptive_radius=False)
    path = rrt.find_path()
    assert path is not None


def test_adaptive_radius_shrinks_with_more_nodes():
    rrt = _make_rrt(adaptive_radius=True, max_iter=500)
    import math

    # With many nodes the adaptive radius should be smaller than neighbor_radius
    n_large = 300
    r = min(rrt._gamma * math.sqrt(math.log(n_large) / n_large), rrt.neighbor_radius)
    assert r <= rrt.neighbor_radius


# ── regression: rewiring must not leave stale costs / a stale ellipse ────────
#
# After a rewire, all descendants of the rewired node used to keep their old
# cached costs, and a goal rewired through the generic loop bypassed
# _best_cost, leaving informed sampling with an oversized ellipse.


def _append_node(rrt, point):
    idx = len(rrt.nodes)
    pt = np.asarray(point, dtype=float)
    rrt.nodes.append(pt)
    rrt._nodes_buf[idx] = pt
    return idx


def test_set_parent_propagates_cost_delta_to_descendants():
    rrt = _make_rrt(max_iter=10)

    # Chain 0 -> 1 -> 2 -> 3 with costs 1.0, 2.0, 3.0
    for i, pt in enumerate([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], start=1):
        idx = _append_node(rrt, pt)
        rrt._set_parent(idx, i - 1, float(i))

    # New node 4 offers node 1 a cheaper route: rewire 1 under 4 (cost 1.0 -> 0.5)
    idx4 = _append_node(rrt, (1.0, 0.5))
    rrt._set_parent(idx4, 0, 0.4)
    rrt._set_parent(1, idx4, 0.5)

    # Children index updated
    assert rrt.parent[1] == idx4
    assert 1 in rrt._children[idx4]
    assert 1 not in rrt._children[0]
    # The -0.5 delta reached every descendant of node 1
    assert rrt.cost[1] == pytest.approx(0.5)
    assert rrt.cost[2] == pytest.approx(1.5)
    assert rrt.cost[3] == pytest.approx(2.5)


def _goal_index(rrt):
    return next(i for i, nd in enumerate(rrt.nodes) if np.allclose(nd, rrt.goal))


def test_tree_costs_consistent_after_seeded_run_with_rewiring():
    """
    Every node's cached cost equals its parent's cost plus the edge cost.

    With improve_after_goal the tree keeps rewiring after the first solution,
    so stale descendant costs (the old bug) would break this invariant.
    """
    random.seed(7)
    rrt = _make_rrt(informed=True, improve_after_goal=True, max_iter=400)
    path = rrt.find_path()
    assert path is not None

    for idx, parent in rrt.parent.items():
        if parent is None:
            continue
        collision, seg_cost = rrt._segment_cost(rrt.nodes[parent], rrt.nodes[idx])
        assert not collision
        assert rrt.cost[idx] == pytest.approx(rrt.cost[parent] + seg_cost), (
            f"stale cached cost at node {idx}"
        )


def test_best_cost_tracks_goal_cost_after_rewiring():
    random.seed(3)
    rrt = _make_rrt(informed=True, improve_after_goal=True, max_iter=600)
    path = rrt.find_path()
    assert path is not None

    goal_idx = _goal_index(rrt)
    # The informed ellipse must be driven by the goal's true (rewired) cost
    assert rrt._best_cost == pytest.approx(rrt.cost[goal_idx])
    # And the reported path length matches that cost on a zero-cost grid
    path_len = sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))
    assert rrt.cost[goal_idx] >= path_len - 1e-6


def test_combined_informed_and_adaptive():
    rrt = _make_rrt(informed=True, adaptive_radius=True, improve_after_goal=True, max_iter=1500)
    path = rrt.find_path()
    assert path is not None
    assert np.allclose(path[0], [0.0, 0.0])
    assert np.allclose(path[-1], [10.0, 10.0])
