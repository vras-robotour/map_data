import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from map_data.pathsolver import replan as replan_mod
from map_data.pathsolver.replan import ReplanPath, cancel_replan_backend

ARGS = {
    "low": (0, 0),
    "high": (10, 10),
    "cell_size": 0.5,
    "simplify_path": True,
    "inflate_obstacles": 0.0,
}


@pytest.fixture
def transfer_id():
    """
    A unique transfer id, guaranteed to be removed from the module-level
    cancellation set after the test so state cannot leak between tests.
    """
    tid = "test-replan-cancel-id"
    yield tid
    replan_mod._discard_cancelled(tid)


def _make_replanner(transfer_id=None):
    replanner = ReplanPath([], transfer_id=transfer_id, **ARGS)
    # Pre-warm the grid cache with an all-free grid (same pattern as test_astar)
    replanner.path_grid.grid_2d_cache = np.zeros((20, 20), dtype=float)
    return replanner


# ── cancellation helpers ─────────────────────────────────────────────────────


def test_cancel_helpers_roundtrip(transfer_id):
    assert not replan_mod._is_cancelled(transfer_id)

    cancel_replan_backend(transfer_id)
    assert replan_mod._is_cancelled(transfer_id)

    replan_mod._discard_cancelled(transfer_id)
    assert not replan_mod._is_cancelled(transfer_id)


def test_cancel_helpers_ignore_none():
    # None/empty transfer ids are no-ops: nothing is registered, nothing matches
    cancel_replan_backend(None)
    assert not replan_mod._is_cancelled(None)
    replan_mod._discard_cancelled(None)  # must not raise


# ── replan() cancellation contract ───────────────────────────────────────────


def test_replan_completes_without_cancel(transfer_id):
    replanner = _make_replanner(transfer_id)
    path = np.array([[1.0, 1.0], [9.0, 9.0]])

    result = replanner.replan(path)

    assert result is not None
    assert np.allclose(result[0], [1.0, 1.0])
    assert np.allclose(result[-1], [9.0, 9.0])


def test_replan_stale_cancel_before_start_is_cleared_and_run_completes(transfer_id):
    """
    A cancel left over from an earlier run must not abort a new replan.

    Cancels only apply to the replan that is running when they arrive; an ID
    already present when replan() starts is stale and is discarded on entry.
    """
    replanner = _make_replanner(transfer_id)
    cancel_replan_backend(transfer_id)  # stale — no replan is running yet

    result = replanner.replan(np.array([[1.0, 1.0], [9.0, 9.0]]))

    assert result is not None
    assert not replan_mod._is_cancelled(transfer_id)


def test_replan_cancel_after_completion_does_not_poison_next_run(transfer_id):
    """
    Regression: a cancel arriving after replan() returned used to sit in
    _cancelled_transfers forever, instantly "cancelling" the next replan
    that reused the same transfer_id.
    """
    replanner = _make_replanner(transfer_id)
    path = np.array([[1.0, 1.0], [9.0, 9.0]])

    assert replanner.replan(path) is not None

    # Cancel lands after the run already completed
    cancel_replan_backend(transfer_id)

    # A new replan with the same transfer_id must run normally to completion
    result = replanner.replan(path)
    assert result is not None
    assert not replan_mod._is_cancelled(transfer_id)


def test_replan_discards_cancel_id_even_on_planner_failure(transfer_id):
    """
    The finally-block discard also runs when planning fails mid-way.
    """
    replanner = _make_replanner(transfer_id)

    def cancel_then_fail(start, goal):
        # Simulate a cancel arriving while planning, followed by "no path"
        cancel_replan_backend(transfer_id)
        return None

    replanner._astar = cancel_then_fail
    # Force the planner to be invoked by making the direct segment collide
    replanner._colides = lambda seg: True

    result = replanner.replan(np.array([[1.0, 1.0], [9.0, 9.0]]))

    assert result is None
    assert not replan_mod._is_cancelled(transfer_id)


def test_replan_cancelled_mid_run_stops_and_consumes_cancel(transfer_id):
    replanner = _make_replanner(transfer_id)
    path = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])

    # Inject the cancel deterministically between loop iterations: the
    # collision check runs once per segment, so cancelling from inside it
    # means the cancel lands while segment 0 is being processed.
    calls = []

    def cancelling_colides(path_seg):
        calls.append(path_seg)
        cancel_replan_backend(transfer_id)
        return False

    replanner._colides = cancelling_colides

    result = replanner.replan(path)

    assert result is None
    # Cancelled right after the first segment; the second was never processed
    assert len(calls) == 1
    # The cancel id was consumed (discarded) when the cancellation was honoured
    assert not replan_mod._is_cancelled(transfer_id)


# ── rrt -> astar fallback ────────────────────────────────────────────────────


def test_replan_rrt_failure_falls_back_to_astar(transfer_id):
    replanner = _make_replanner(transfer_id)
    replanner._rrt_star = lambda start, goal: None
    astar_way = np.array([[1.0, 1.0], [1.0, 9.0], [9.0, 9.0]])
    replanner._astar = lambda start, goal: astar_way
    replanner._colides = lambda seg: True  # force a search; also keeps the corner

    result = replanner.replan(np.array([[1.0, 1.0], [9.0, 9.0]]), algorithm="rrt")

    np.testing.assert_array_equal(result, astar_way)


def test_replan_rrt_cancelled_does_not_fall_back(transfer_id):
    replanner = _make_replanner(transfer_id)

    def cancel_then_fail(start, goal):
        cancel_replan_backend(transfer_id)  # RRTStar returns None when cancelled
        return None

    astar_calls = []
    replanner._rrt_star = cancel_then_fail
    replanner._astar = lambda start, goal: astar_calls.append(1)
    replanner._colides = lambda seg: True

    assert replanner.replan(np.array([[1.0, 1.0], [9.0, 9.0]]), algorithm="rrt") is None
    assert astar_calls == []


# ── lazy grid ────────────────────────────────────────────────────────────────


def test_fill_grid_lazily_builds_the_grid_only_for_a_colliding_leg():
    replanner = ReplanPath([], **ARGS)
    fills = []

    def fill(map_data, highway_types=None, max_path_dist=2.0):
        fills.append(map_data)
        replanner.path_grid.grid_2d_cache = np.zeros((20, 20), dtype=float)

    replanner.fill_grid = fill
    replanner.fill_grid_lazily("md")
    path = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])

    assert replanner.replan(path) is not None
    assert fills == []  # no leg collided: grid never built

    replanner._colides = lambda seg: True
    assert replanner.replan(path) is not None
    assert fills == ["md"]  # built once, reused by the second leg


# ── barrier crossing a footway ───────────────────────────────────────────────


def _fence_across_footway(**kwargs):
    """
    A fence spanning the whole 20 x 10 m grid at x = 10, crossed by a buffered
    footway along y = 5: the footway gap is the only way through.
    """
    from map_data.utils.way import Way

    footway = Way(
        id=1,
        nodes=[100, 101],
        tags={"highway": "footway"},
        line=LineString([(0.0, 5.0), (20.0, 5.0)]).buffer(1.5),
    )
    md = SimpleNamespace(
        footways_list=[footway],
        roads_list=[],
        get_points=lambda: {100: np.array([[0.0, 5.0]]), 101: np.array([[20.0, 5.0]])},
    )
    fence = LineString([(10.0, -1.0), (10.0, 11.0)]).buffer(0.2)
    args = {**ARGS, "high": (20, 10), "simplify_path": False, **kwargs}
    replanner = ReplanPath([fence], rng=random.Random(0), **args)
    replanner.fill_grid_lazily(md)
    return replanner


def test_astar_through_a_footway_gap_simplifies_to_a_straight_line():
    replanner = _fence_across_footway(simplify_path=True)
    result = replanner.replan(np.array([[1.0, 5.0], [19.0, 5.0]]))
    np.testing.assert_allclose(result, [[1.0, 5.0], [19.0, 5.0]])


def test_rrt_finds_the_footway_gap():
    replanner = _fence_across_footway()
    replanner._grid()
    assert replanner._rrt_star(np.array([1.0, 5.0]), np.array([19.0, 5.0])) is not None


def test_smoothing_through_a_footway_gap_is_kept():
    replanner = _fence_across_footway(smooth_path=True)
    replanner._grid()
    zigzag = np.array([[1.0, 5.0], [5.0, 6.0], [10.0, 5.0], [15.0, 4.0], [19.0, 5.0]])
    smoothed = replanner._post_process_path(zigzag)
    assert not np.allclose(smoothed, zigzag)


# ── defaults from planner_defaults.yaml ──────────────────────────────────────

YAML_GRID = {
    "cell_size": 0.5,
    "inflate_obstacles": 0.0,
    "simplify_path": False,
    "smooth_path": True,
    "max_path_dist": 3.0,
    "grid_cost_weight": 7.0,
}


@pytest.fixture
def yaml_grid(monkeypatch):
    monkeypatch.setattr(replan_mod, "load_config", lambda name: dict(YAML_GRID))


def test_replan_path_defaults_come_from_planner_defaults(yaml_grid):
    replanner = ReplanPath([], low=(0, 0), high=(10, 10))
    assert replanner.cell_size == 0.5
    assert replanner.simplify_path is False
    assert replanner.smooth_path is True
    assert replanner.max_path_dist == 3.0
    assert replanner.grid_cost_weight == 7.0
    assert replanner.path_grid.cell_size == 0.5


def test_explicit_arguments_override_planner_defaults(yaml_grid):
    replanner = ReplanPath(
        [], low=(0, 0), high=(10, 10), cell_size=1.0, simplify_path=True, grid_cost_weight=1.0
    )
    assert (replanner.cell_size, replanner.simplify_path, replanner.grid_cost_weight) == (
        1.0,
        True,
        1.0,
    )


def test_cli_defaults_come_from_planner_defaults(yaml_grid):
    from map_data.plan_route_cli import build_parser

    args = build_parser().parse_args(["-f", "x.mapdata"])
    assert (args.cell_size, args.inflate_obstacles, args.simplify, args.smooth) == (
        0.5,
        0.0,
        False,
        True,
    )
    args = build_parser().parse_args(["-f", "x.mapdata", "--simplify", "--no-smooth"])
    assert (args.simplify, args.smooth) == (True, False)


# ── blocked and enclosed ends ────────────────────────────────────────────────


def _walled_replanner(obstacles):
    """20 x 20 m grid at 0.5 m with *obstacles* burned in; A* calls are recorded."""
    replanner = ReplanPath(obstacles, **{**ARGS, "high": (20, 20), "simplify_path": False})
    grid = np.zeros((40, 40))
    replanner.path_grid.grid_2d_cache = replanner.path_grid.burn_obstacles(grid, obstacles)
    calls = []
    astar = replanner._astar
    replanner._astar = lambda s, g: calls.append((s, g)) or astar(s, g)
    return replanner, calls


def test_goal_inside_a_thin_wall_is_snapped_and_reached():
    replanner, _ = _walled_replanner([box(9.0, 0.0, 10.0, 15.0)])
    result = replanner.replan(np.array([[5.0, 5.0], [9.5, 5.0]]))
    assert result is not None
    np.testing.assert_allclose(result[-1], [9.5, 5.0])


@pytest.mark.parametrize(
    ("path", "reason"),
    [([[2, 2], [10, 10]], "goal_blocked"), ([[10, 10], [2, 2]], "start_blocked")],
)
def test_end_deep_inside_an_obstacle_fails_fast(path, reason):
    replanner, calls = _walled_replanner([box(5.0, 5.0, 15.0, 15.0)])
    assert replanner.replan(np.array(path, dtype=float)) is None
    assert replanner.failure_reason == reason
    assert calls == []  # no search


def test_enclosed_goal_fails_without_searching():
    ring = box(6.0, 6.0, 14.0, 14.0).difference(box(7.0, 7.0, 13.0, 13.0))
    replanner, calls = _walled_replanner([ring])
    assert replanner.replan(np.array([[2.0, 2.0], [10.0, 10.0]])) is None
    assert replanner.failure_reason == "no_path"
    assert calls == []


def test_endpoint_legs_keep_the_search_ends_when_the_shortcut_collides():
    # A wall forces a search; two posts sit on the straight hops from the requested
    # start/goal to the search's second/second-to-last vertex.
    obstacles = [box(-1, 4.9, 2, 5.1), box(1.4, 0.3, 1.6, 0.7), box(1.4, 9.3, 1.6, 9.7)]
    replanner = ReplanPath(obstacles, **{**ARGS, "low": (-2, -2), "high": (12, 12)})
    replanner.path_grid.grid_2d_cache = np.zeros((28, 28))
    replanner.simplify_path = False
    way = np.array([[0.0, 1.0], [3.0, 1.0], [3.0, 9.0], [0.0, 9.0]])
    replanner._astar = lambda s, g: way

    result = replanner.replan(np.array([[0.0, 0.0], [0.0, 10.0]]))

    np.testing.assert_allclose(result, [[0, 0], *way, [0, 10]])
    assert not any(replanner._colides(LineString(leg)) for leg in itertools.pairwise(result))
