"""Tests for the headless route planner (``map_data.pathsolver.route``)."""

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.annotations import load_mapdata_with_annotations
from map_data.map_data import MapData
from map_data.pathsolver.route import (
    RoutePlanningError,
    densify,
    path_length,
    plan_route,
)
from map_data.utils.gpx import create_gpx_track
from map_data.utils.way import Way


def _latlon(lat0, lon0, dx, dy):
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    return utm.to_latlon(e0 + dx, n0 + dy, zn, zl)


# ── densify / path_length ──────────────────────────────────────────────────


def test_densify_keeps_vertices_and_bounds_spacing():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0]])
    out = densify(path, 3.0)
    assert out[0].tolist() == [0.0, 0.0]
    assert out[-1].tolist() == [10.0, 4.0]
    assert any(np.allclose(p, [10.0, 0.0]) for p in out)
    steps = np.hypot(*np.diff(out, axis=0).T)
    assert steps.max() <= 3.0 + 1e-9
    assert len(out) == 1 + 4 + 2  # 10 m in 4 steps of 2.5, 4 m in 2 steps of 2


def test_densify_noop_cases():
    path = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert densify(path, 0.0) is not None
    assert len(densify(path, 0.0)) == 2
    assert len(densify(path, 50.0)) == 2
    assert len(densify(np.array([[1.0, 1.0]]), 1.0)) == 1


def test_path_length():
    assert path_length(np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])) == pytest.approx(5.0)
    assert path_length(np.array([[0.0, 0.0]])) == 0.0


# ── graph planner ──────────────────────────────────────────────────────────


def test_plan_route_graph_follows_the_network(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    start = _latlon(lat0, lon0, 5.0, 3.0)  # near node 101
    goal = _latlon(lat0, lon0, 100.0, 95.0)  # near node 104, up the T-branch

    res = plan_route(md, [start, goal], algorithm="graph", spacing=3.0)

    # Route goes 101 -> 102 -> 104: through the junction, not diagonally.
    assert res.algorithm == "graph"
    assert res.length_m == pytest.approx(95.0 + 95.0, abs=8.0)
    steps = np.hypot(*np.diff(res.utm, axis=0).T)
    assert steps.max() <= 3.0 + 1e-6
    assert len(res.snap_distances) == 2
    assert max(res.snap_distances) < 5.0
    # Endpoints are the waypoints snapped onto the network, not the raw requests:
    # the start sits 3 m off way 1 and is routed from its projection at (5, 0).
    snapped_start = _latlon(lat0, lon0, 5.0, 0.0)
    assert res.latlon[0] == pytest.approx(snapped_start, abs=1e-6)
    assert res.latlon[-1] == pytest.approx(goal, abs=1e-6)
    assert res.changed


def test_plan_route_graph_keep_start(footway_network_mapdata):
    """
    ``keep_start`` (the action server's ``start_from_robot``) begins the route
    at the robot's own fix rather than at its projection onto the network.
    """
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    start = _latlon(lat0, lon0, 5.0, 3.0)  # 3 m off way 1
    goal = _latlon(lat0, lon0, 100.0, 95.0)

    res = plan_route(md, [start, goal], algorithm="graph", keep_start=True)
    assert res.latlon[0] == pytest.approx(start, abs=1e-6)
    assert res.latlon[1] == pytest.approx(_latlon(lat0, lon0, 5.0, 0.0), abs=1e-6)

    # Default stays snapped
    snapped = plan_route(md, [start, goal], algorithm="graph")
    assert snapped.latlon[0] == pytest.approx(_latlon(lat0, lon0, 5.0, 0.0), abs=1e-6)


def test_plan_route_graph_keep_goal_densifies_the_final_leg(footway_network_mapdata):
    """
    ``keep_goal`` ends the route at the requested coordinate; the off-network
    leg to it is resampled at ``spacing`` like the rest of the route.
    """
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    start = _latlon(lat0, lon0, 5.0, 0.0)
    goal = _latlon(lat0, lon0, 150.0, 20.0)  # 20 m off way 1

    res = plan_route(md, [start, goal], algorithm="graph", spacing=3.0, keep_goal=True)

    assert res.latlon[-1] == pytest.approx(goal, abs=1e-6)
    steps = np.hypot(*np.diff(res.utm, axis=0).T)
    assert steps.max() <= 3.0 + 1e-6
    # The last 20 m are the off-network leg: straight north from the projection.
    assert res.utm[-8][0] == pytest.approx(res.utm[-1][0], abs=1e-6)

    snapped = plan_route(md, [start, goal], algorithm="graph", spacing=3.0)
    assert snapped.latlon[-1] == pytest.approx(_latlon(lat0, lon0, 150.0, 0.0), abs=1e-6)


def test_plan_route_graph_without_spacing_keeps_vertices(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    res = plan_route(md, [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 200.0, 0.0)])
    assert len(res.latlon) <= 5


def test_plan_route_graph_unreachable(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(md, [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 450.0, 0.0)])
    assert e.value.reason == "unreachable"


def test_plan_route_graph_snap_too_far(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    far = _latlon(lat0, lon0, 100.0, -150.0)  # inside the map (y >= -200), 150 m off the ways
    with pytest.raises(RoutePlanningError) as e:
        plan_route(md, [_latlon(lat0, lon0, 0.0, 0.0), far], max_snap_distance=50.0)
    assert e.value.reason == "snap_too_far"
    assert "waypoint 1" in e.value.message


@pytest.mark.parametrize("algorithm", ["graph", "astar"])
def test_plan_route_start_or_goal_outside_map(footway_network_mapdata, algorithm):
    """The map spans -200..700 m east, -200..300 m north of the origin (waypoints + margin)."""
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    inside, outside = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 100.0, -400.0)
    for points, reason in (([inside, outside], "goal"), ([outside, inside], "start")):
        with pytest.raises(RoutePlanningError) as e:
            plan_route(md, points, algorithm=algorithm)  # 200 m out, tolerance 100 m
        assert e.value.reason == f"{reason}_outside_map"
        assert f"the {reason} is 200 m outside" in e.value.message
        try:  # within the tolerance: plans, or fails for another reason (graph: snap_too_far)
            plan_route(md, points, algorithm=algorithm, outside_map_tolerance=250.0)
        except RoutePlanningError as err:
            failure = err.reason
        else:
            failure = ""
        assert not failure.endswith("_outside_map")


def test_plan_route_too_few_points(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(md, [(lat0, lon0)])
    assert e.value.reason == "too_few_points"


# ── grid planner ───────────────────────────────────────────────────────────


def test_plan_route_grid_astar(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    start, goal = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 60.0, 0.0)
    res = plan_route(md, [start, goal], algorithm="astar", cell_size=2.0, spacing=5.0)
    assert res.algorithm == "astar"
    assert res.snap_distances == []
    assert res.length_m == pytest.approx(60.0, abs=15.0)
    assert res.latlon[0] == pytest.approx(start, abs=1e-5)
    assert res.latlon[-1] == pytest.approx(goal, abs=1e-5)


@pytest.mark.parametrize(
    ("algorithm", "sub_algorithm", "expected"),
    [("rrt", "astar", "astar"), ("rrt", None, "rrt"), ("astar", None, "astar")],
)
def test_plan_route_grid_sub_algorithm(
    footway_network_mapdata, monkeypatch, algorithm, sub_algorithm, expected
):
    # The viewer's "All terrain" mode sends algorithm="rrt" with the search it
    # wants in sub_algorithm; an explicit "astar" must not be overridden to RRT*.
    from map_data.pathsolver.replan import ReplanPath

    seen = []

    def spy(self, path, algorithm="astar"):
        seen.append(algorithm)
        return path  # skip the (randomised) search itself

    monkeypatch.setattr(ReplanPath, "replan", spy)
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    start, goal = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 60.0, 0.0)
    plan_route(md, [start, goal], algorithm=algorithm, sub_algorithm=sub_algorithm, cell_size=2.0)
    assert seen == [expected]


def test_plan_route_grid_goal_just_outside_map(footway_network_mapdata):
    """A goal past the map edge but within the tolerance must still lie inside the grid."""
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    e0, n0, _, _ = utm.from_latlon(lat0, lon0)
    # A wall across the straight line, so the grid planner actually has to search.
    wall = LineString([(e0 + 80.0, n0 + 290.0), (e0 + 120.0, n0 + 290.0)])
    md.barriers_list.append(Way(id=9, is_area=False, nodes=[], tags={}, line=wall, in_out=""))
    start, goal = _latlon(lat0, lon0, 100.0, 250.0), _latlon(lat0, lon0, 100.0, 340.0)
    res = plan_route(md, [start, goal], algorithm="astar", cell_size=2.0)
    assert len(res.latlon) > 2  # went around the wall
    assert res.latlon[-1] == pytest.approx(goal, abs=1e-5)


def test_plan_route_grid_too_large(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(
            md,
            [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 60.0, 0.0)],
            algorithm="astar",
            cell_size=0.05,
            max_grid_cells=1000,
        )
    assert e.value.reason == "grid_too_large"


# ── GPX track writer ───────────────────────────────────────────────────────


def test_gpx_track_roundtrip(tmp_path):
    pts = [(50.0, 14.0), (50.0001, 14.0001), (50.0002, 14.0003)]
    xml = create_gpx_track(pts, name="route")
    assert "<trk>" in xml
    assert "<trkpt" in xml
    f = tmp_path / "route.gpx"
    f.write_text(xml)
    md = MapData(str(f), coords_type="file")
    assert len(md.waypoints) == 3
    back = [utm.to_latlon(p[0], p[1], md.zone_number, md.zone_letter) for p in md.waypoints]
    for (lat, lon), (blat, blon) in zip(pts, back, strict=True):
        assert (lat, lon) == pytest.approx((blat, blon), abs=1e-7)


def test_plan_route_reuses_a_prebuilt_planner(footway_network_mapdata):
    from map_data.pathsolver.graph_planner import GraphPlanner

    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    planner = GraphPlanner(md, highway_types=["footway"])
    pts = [_latlon(lat0, lon0, 5.0, 3.0), _latlon(lat0, lon0, 100.0, 95.0)]
    fresh = plan_route(md, pts)
    reused = plan_route(md, pts, planner=planner)
    again = plan_route(md, pts, planner=planner)  # the planner keeps no per-request state
    assert reused.length_m == pytest.approx(fresh.length_m)
    assert again.latlon == reused.latlon
    assert reused.snap_distances == pytest.approx(fresh.snap_distances)


def test_plan_route_cell_size_defaults_to_planner_defaults(footway_network_mapdata, monkeypatch):
    from map_data.pathsolver import replan

    monkeypatch.setattr(replan, "load_config", lambda name: {"cell_size": 0.01})
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(
            md,
            [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 60.0, 0.0)],
            algorithm="astar",
        )
    assert e.value.reason == "grid_too_large"
    assert "cell_size=0.01" in e.value.message


def test_viewer_uses_the_planner_grid_cell_budget():
    from map_data.pathsolver import route
    from map_data.viewer.routes import common

    assert common.MAX_GRID_CELLS is route.MAX_GRID_CELLS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"algorithm": "Graph"},
        {"algorithm": "dijkstra"},
        {"algorithm": "rrt", "sub_algorithm": "bfs"},
        {"algorithm": "astar", "cell_size": 0.0},
        {"algorithm": "astar", "cell_size": -0.5},
        # A negative inflation would shrink the obstacles (buffer(-x)).
        {"algorithm": "astar", "inflate_obstacles": -1.0},
        {"algorithm": "astar", "inflate_obstacles": float("nan")},
        {"algorithm": "astar", "inflate_obstacles": float("inf")},
    ],
)
def test_plan_route_rejects_bad_parameters(footway_network_mapdata, kwargs):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(md, [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 60.0, 0.0)], **kwargs)
    assert e.value.reason == "bad_parameter"


def test_plan_route_reports_a_blocked_goal(footway_network_mapdata):
    from shapely.geometry import box

    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    e0, n0, _, _ = utm.from_latlon(lat0, lon0)
    block = box(e0 + 40.0, n0 + 40.0, e0 + 80.0, n0 + 80.0)
    md.barriers_list.append(Way(id=9, is_area=True, nodes=[], tags={}, line=block, in_out=""))
    start, goal = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 60.0, 60.0)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(md, [start, goal], algorithm="astar", cell_size=1.0)
    assert e.value.reason == "goal_blocked"


# ── PlanRoute goal -> plan_route kwargs ─────────────────────────────────────

NODE_DEFAULTS = {
    "algorithm": "graph",
    "highway_types": ["footway"],
    "spacing": 3.0,
    "max_snap_distance": 100.0,
    "cell_size": 0.25,
    "inflate_obstacles": 0.25,
    "simplify_path": True,
    "smooth_path": True,
}


def _goal(**fields):
    from types import SimpleNamespace

    empty = {
        "algorithm": "",
        "highway_types": [],
        "spacing": 0.0,
        "max_snap_distance": 0.0,
        "cell_size": 0.0,
        "inflate_obstacles": 0.0,
        "simplify_path": False,
        "smooth_path": False,
    }
    return SimpleNamespace(**{**empty, **fields})


def test_empty_goal_fields_take_the_node_defaults_but_bools_are_used_as_sent():
    from map_data.pathsolver.route import goal_plan_kwargs

    kwargs = goal_plan_kwargs(_goal(), NODE_DEFAULTS)
    expected = {**NODE_DEFAULTS, "simplify_path": False, "smooth_path": False}
    assert kwargs == expected


def test_goal_fields_override_the_node_defaults():
    from map_data.pathsolver.route import goal_plan_kwargs

    goal = _goal(
        algorithm="astar",
        highway_types=["road"],
        spacing=-1.0,
        max_snap_distance=5.0,
        cell_size=0.5,
        inflate_obstacles=1.0,
        simplify_path=True,
        smooth_path=True,
    )
    assert goal_plan_kwargs(goal, NODE_DEFAULTS) == {
        "algorithm": "astar",
        "highway_types": ["road"],
        "spacing": 0.0,  # negative = the planner's own vertices
        "max_snap_distance": 5.0,
        "cell_size": 0.5,
        "inflate_obstacles": 1.0,
        "simplify_path": True,
        "smooth_path": True,
    }
