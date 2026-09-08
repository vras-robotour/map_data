import numpy as np
import pytest
from shapely.geometry import LineString

from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.utils.way import Way


def _make_node(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0]).reshape(3, 1)


class MockMapData:
    def __init__(self, footways, nodes_coords, roads=None):
        self.footways_list = footways
        self.roads_list = roads or []
        self._nodes_coords = nodes_coords  # {id: (x, y)}

    def get_points(self):
        return {nid: _make_node(x, y) for nid, (x, y) in self._nodes_coords.items()}


def _make_footway(way_id, node_ids, nodes_coords):
    coords = [nodes_coords[nid] for nid in node_ids]
    return Way(
        id=way_id,
        nodes=list(node_ids),
        tags={"highway": "footway"},
        line=LineString(coords),
    )


def test_graph_planner_simple_path():
    """
    Straight three-node footway: plan from one end to the other.
    """
    nodes_coords = {100: (0.0, 0.0), 101: (10.0, 0.0), 102: (20.0, 0.0)}
    way = _make_footway(1, [100, 101, 102], nodes_coords)
    md = MockMapData([way], nodes_coords)
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 0.0], [20.0, 0.0]]))

    assert not isinstance(result, bool), "Expected a path, not False"
    assert len(result) >= 2
    assert result[0][0] == pytest.approx(0.0, abs=1.5)
    assert result[-1][0] == pytest.approx(20.0, abs=1.5)


def test_graph_planner_disjoint_network():
    """
    Two disconnected footway segments: no path should be found.
    """
    nodes_coords = {
        100: (0.0, 0.0),
        101: (5.0, 0.0),
        200: (50.0, 0.0),
        201: (55.0, 0.0),
    }
    way_a = _make_footway(1, [100, 101], nodes_coords)
    way_b = _make_footway(2, [200, 201], nodes_coords)
    md = MockMapData([way_a, way_b], nodes_coords)
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 0.0], [55.0, 0.0]]))

    assert result is None


def test_graph_planner_same_edge():
    """
    Start and goal both snap to the same edge: direct connection is added.
    """
    nodes_coords = {100: (0.0, 0.0), 101: (10.0, 0.0)}
    way = _make_footway(1, [100, 101], nodes_coords)
    md = MockMapData([way], nodes_coords)
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[2.0, 0.0], [8.0, 0.0]]))

    assert not isinstance(result, bool), "Expected a path, not False"
    assert len(result) >= 2
    # Path stays near y=0 (on the edge)
    for pt in result:
        assert abs(pt[1]) < 1.0


# ── annotation splicing ───────────────────────────────────────────────────────
#
# Annotation ways (negative integer IDs) are spliced into the OSM graph by
# projecting their endpoints onto the nearest foreign edge and inserting a
# synthetic junction node (IDs -2000000 and below) at the projection point.


def _junction_neighbor_positions(planner):
    """
    Map each synthetic junction position to the set of its neighbor positions.

    Returns {(jx, jy): {(nx, ny), ...}} for every junction node the planner
    created while splicing annotations, with coordinates rounded for stable
    set comparison.
    """
    result = {}
    for node_id in planner.graph:
        if not (isinstance(node_id, int) and node_id <= -2000000):
            continue
        pos = tuple(np.round(planner.nodes[node_id].ravel()[:2], 6))
        neighbors = {
            tuple(np.round(planner.nodes[n].ravel()[:2], 6)) for n, _ in planner.graph[node_id]
        }
        result[pos] = neighbors
    return result


def _make_annotation_scenario():
    """
    A four-node straight footway plus an annotation way whose two endpoints
    project onto two different segments (0 and 2) of the footway.
    """
    nodes_coords = {
        10: (0.0, 0.0),
        11: (10.0, 0.0),
        12: (20.0, 0.0),
        13: (30.0, 0.0),
        20: (5.0, 3.0),
        21: (25.0, 3.0),
    }
    main_way = _make_footway(1, [10, 11, 12, 13], nodes_coords)
    annotation = _make_footway(-1, [20, 21], nodes_coords)
    return MockMapData([main_way, annotation], nodes_coords), main_way, annotation


def test_graph_planner_annotation_splits_two_segments_of_same_way():
    """
    Two splits on different segments of one way land between the correct nodes.

    The annotation endpoint at (5, 3) projects onto segment 0 and the one at
    (25, 3) onto segment 2. Each junction must be connected to the endpoints
    of its own segment; applying the segment-0 insert first must not shift the
    segment-2 junction between the wrong nodes.
    """
    md, _, _ = _make_annotation_scenario()
    planner = GraphPlanner(md)

    junctions = _junction_neighbor_positions(planner)

    assert set(junctions) == {(5.0, 0.0), (25.0, 0.0)}
    # Junction at (5, 0) sits inside segment 0 and connects to its annotation endpoint
    assert junctions[(5.0, 0.0)] == {(0.0, 0.0), (10.0, 0.0), (5.0, 3.0)}
    # Junction at (25, 0) sits inside segment 2 and connects to its annotation endpoint
    assert junctions[(25.0, 0.0)] == {(20.0, 0.0), (30.0, 0.0), (25.0, 3.0)}


def test_graph_planner_reconstruction_on_same_map_data():
    """
    Building a second planner on the same MapData works and gives identical results.

    Splicing must not leak synthetic node IDs into the shared Way objects:
    ``get_points()`` is rebuilt fresh on every construction, so a leaked ID
    would raise KeyError the second time around.
    """
    md, _, _ = _make_annotation_scenario()
    waypoints = np.array([[5.0, 3.0], [25.0, 3.0]])

    planner_a = GraphPlanner(md)
    planner_b = GraphPlanner(md)

    result_a = planner_a.plan(waypoints)
    result_b = planner_b.plan(waypoints)

    assert result_a is not None
    assert result_b is not None
    assert np.allclose(result_a, result_b)


def test_graph_planner_does_not_mutate_map_data_ways():
    """
    Planner construction leaves the shared Way.nodes lists untouched.
    """
    md, main_way, annotation = _make_annotation_scenario()

    GraphPlanner(md)

    assert main_way.nodes == [10, 11, 12, 13]
    assert annotation.nodes == [20, 21]


# ── plan() edge cases ────────────────────────────────────────────────────────


def _simple_planner(**kwargs):
    nodes_coords = {100: (0.0, 0.0), 101: (10.0, 0.0)}
    way = _make_footway(1, [100, 101], nodes_coords)
    md = MockMapData([way], nodes_coords)
    return GraphPlanner(md, **kwargs)


def test_graph_planner_fewer_than_two_waypoints_returns_none(caplog):
    """
    plan() honours its documented failure contract: None, not np.array([]).
    """
    planner = _simple_planner()

    with caplog.at_level("WARNING"):
        assert planner.plan(np.array([[5.0, 0.0]])) is None
        assert planner.plan(np.empty((0, 2))) is None

    assert any("two waypoints" in rec.message for rec in caplog.records)


def test_graph_planner_waypoint_beyond_snap_distance_returns_none(caplog):
    """
    A waypoint far from every edge fails the plan instead of snapping to the
    globally nearest edge hundreds of metres away.
    """
    planner = _simple_planner()  # default max_snap_distance = 100 m

    with caplog.at_level("WARNING"):
        result = planner.plan(np.array([[0.0, 0.0], [500.0, 0.0]]))

    assert result is None
    assert any("snap limit" in rec.message for rec in caplog.records)


def test_graph_planner_custom_snap_distance_allows_far_waypoint():
    planner = _simple_planner(max_snap_distance=1000.0)

    result = planner.plan(np.array([[0.0, 0.0], [500.0, 0.0]]))

    assert result is not None
    assert len(result) >= 2
    # The route stays on the network: the far (unsnappable-by-default) waypoint
    # is represented by its projection, the network end at x=10, not repeated
    # verbatim 490 m off the path.
    assert np.allclose(result[0], [0.0, 0.0])
    assert np.allclose(result[-1], [10.0, 0.0])


def test_graph_planner_waypoint_within_snap_distance_still_plans():
    planner = _simple_planner()

    # 50 m off the network is within the default 100 m snap limit
    result = planner.plan(np.array([[0.0, 50.0], [10.0, 0.0]]))

    assert result is not None
    assert len(result) >= 2


def test_graph_planner_l_shaped():
    """
    L-shaped footway: path must navigate around the corner.
    """
    nodes_coords = {100: (0.0, 0.0), 101: (10.0, 0.0), 102: (10.0, 10.0)}
    way = _make_footway(1, [100, 101, 102], nodes_coords)
    md = MockMapData([way], nodes_coords)
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 0.0], [10.0, 10.0]]))

    assert not isinstance(result, bool), "Expected a path, not False"
    assert len(result) >= 2
    xs = [p[0] for p in result]
    ys = [p[1] for p in result]
    # Must reach the far corner
    assert max(xs) == pytest.approx(10.0, abs=1.5)
    assert max(ys) == pytest.approx(10.0, abs=1.5)


def test_graph_planner_via_waypoint_makes_no_spur():
    """
    A via waypoint clicked beside the path used to be re-inserted between its
    own two projections, producing a "projection, click, projection" spur of
    stacked points. The route must pass the waypoint once, on the network.
    """
    nodes_coords = {100 + i: (10.0 * i, 0.0) for i in range(6)}
    way = _make_footway(1, list(nodes_coords), nodes_coords)
    md = MockMapData([way], nodes_coords)
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 0.0], [20.4, 0.3], [50.0, 0.0]]))

    assert result is not None
    # Every vertex lies on the way, and the projection of the via point appears once
    assert np.allclose(result[:, 1], 0.0)
    assert sum(1 for p in result if p[0] == pytest.approx(20.4)) == 1
    _assert_no_stacked_points(result)


def test_graph_planner_via_waypoint_at_junction_makes_no_spur():
    """
    A via waypoint snapping just onto a side branch made the route step off the
    junction onto the branch and immediately back, leaving five points within
    0.3 m of each other.
    """
    nodes_coords = {
        100: (0.0, 0.0),
        101: (10.0, 0.0),
        102: (20.0, 0.0),
        103: (30.0, 0.0),
        104: (40.0, 0.0),
        200: (20.0, 10.0),
    }
    md = MockMapData(
        [
            _make_footway(1, [100, 101, 102, 103, 104], nodes_coords),
            _make_footway(2, [102, 200], nodes_coords),
        ],
        nodes_coords,
    )
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 0.0], [20.1, 0.3], [40.0, 0.0]]))

    assert result is not None
    _assert_no_stacked_points(result)


def test_graph_planner_keeps_genuine_dead_end_detour():
    """
    Spur collapsing must not swallow a real out-and-back: a waypoint on a long
    dead-end branch has to be visited and left the same way.
    """
    nodes_coords = {
        100: (0.0, 0.0),
        101: (20.0, 0.0),
        102: (40.0, 0.0),
        200: (20.0, 30.0),
    }
    md = MockMapData(
        [
            _make_footway(1, [100, 101, 102], nodes_coords),
            _make_footway(2, [101, 200], nodes_coords),
        ],
        nodes_coords,
    )
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 0.0], [20.0, 30.0], [40.0, 0.0]]))

    assert result is not None
    # The dead end is reached and the route comes back through the junction
    assert max(p[1] for p in result) == pytest.approx(30.0)
    assert sum(1 for p in result if np.allclose(p, [20.0, 0.0])) == 2


def _assert_no_stacked_points(path, tol=0.05):
    """
    No vertex coincides with its predecessor, and no vertex returns to where it
    was two steps ago — the two shapes a degenerate spur takes. Short forward
    steps are fine; a projection legitimately lands centimetres past a node.
    """
    duplicates = [i for i in range(1, len(path)) if np.linalg.norm(path[i] - path[i - 1]) <= tol]
    assert not duplicates, f"duplicate vertices at {duplicates}: {path}"
    spurs = [i for i in range(2, len(path)) if np.linalg.norm(path[i] - path[i - 2]) <= tol]
    assert not spurs, f"out-and-back spur ending at {spurs}: {path}"


def test_graph_planner_keep_start_keeps_the_raw_first_waypoint():
    """
    ``keep_start`` (planning from the robot's own pose) begins the route at the
    requested point itself, followed by its projection onto the network.
    """
    planner = _simple_planner()  # straight footway (0,0) -> (10,0)

    result = planner.plan(np.array([[2.0, 4.0], [10.0, 0.0]]), keep_start=True)

    assert result is not None
    assert np.allclose(result[0], [2.0, 4.0])
    assert np.allclose(result[1], [2.0, 0.0])
    # Only the first waypoint is kept verbatim; the goal is still its projection
    assert np.allclose(result[-1], [10.0, 0.0])


def test_graph_planner_keep_start_does_not_duplicate_on_network_start():
    """
    A robot already standing on the path must not produce two stacked points.
    """
    planner = _simple_planner()

    result = planner.plan(np.array([[2.0, 0.0], [10.0, 0.0]]), keep_start=True)

    assert result is not None
    assert np.allclose(result[0], [2.0, 0.0])
    _assert_no_stacked_points(result)


def test_graph_planner_keep_start_leaves_via_points_snapped():
    """
    ``keep_start`` must not resurrect the via-point spurs: only waypoint 0 is
    kept verbatim.
    """
    nodes_coords = {100 + i: (10.0 * i, 0.0) for i in range(6)}
    md = MockMapData([_make_footway(1, list(nodes_coords), nodes_coords)], nodes_coords)
    planner = GraphPlanner(md)

    result = planner.plan(np.array([[0.0, 5.0], [20.4, 0.3], [50.0, 0.0]]), keep_start=True)

    assert result is not None
    assert np.allclose(result[0], [0.0, 5.0])
    # The via point is not repeated off the network
    assert not any(p[1] == pytest.approx(0.3) for p in result)
    _assert_no_stacked_points(result)
