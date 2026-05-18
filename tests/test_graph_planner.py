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
        return {
            nid: _make_node(x, y)
            for nid, (x, y) in self._nodes_coords.items()
        }


def _make_footway(way_id, node_ids, nodes_coords):
    coords = [nodes_coords[nid] for nid in node_ids]
    return Way(
        id=way_id,
        nodes=list(node_ids),
        tags={"highway": "footway"},
        line=LineString(coords),
    )


def test_graph_planner_simple_path():
    """Straight three-node footway: plan from one end to the other."""
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
    """Two disconnected footway segments: no path should be found."""
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
    """Start and goal both snap to the same edge: direct connection is added."""
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


def test_graph_planner_l_shaped():
    """L-shaped footway: path must navigate around the corner."""
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
