"""Tests for the headless route planner (``map_data.pathsolver.route``)."""

import numpy as np
import pytest
import utm

from map_data.annotations import load_mapdata_with_annotations
from map_data.pathsolver.route import (
    RoutePlanningError,
    densify,
    path_length,
    plan_route,
    route_to_dicts,
)
from map_data.utils.gpx import create_gpx_track, parse_gpx_file


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
    assert densify(path, 0.0) is not None and len(densify(path, 0.0)) == 2
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
    assert len(res.snap_distances) == 2 and max(res.snap_distances) < 5.0
    assert res.latlon[0] == pytest.approx(start, abs=1e-6)
    assert res.latlon[-1] == pytest.approx(goal, abs=1e-6)
    assert res.changed
    dicts = route_to_dicts(res)
    assert dicts[0]["latitude"] == pytest.approx(start[0])


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
    far = _latlon(lat0, lon0, 100.0, -300.0)
    with pytest.raises(RoutePlanningError) as e:
        plan_route(md, [_latlon(lat0, lon0, 0.0, 0.0), far], max_snap_distance=50.0)
    assert e.value.reason == "snap_too_far"
    assert "waypoint 1" in e.value.message


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
    assert "<trk>" in xml and "<trkpt" in xml
    f = tmp_path / "route.gpx"
    f.write_text(xml)
    parsed, zn, zl = parse_gpx_file(str(f))
    assert len(parsed) == 3
    back = [utm.to_latlon(p[0], p[1], zn, zl) for p in parsed]
    for (lat, lon), (blat, blon) in zip(pts, back, strict=True):
        assert (lat, lon) == pytest.approx((blat, blon), abs=1e-7)
