"""Tests for osm_cloud pure helper functions and node construction."""

import sys
from unittest.mock import MagicMock, patch

# Mock ROS2 modules before importing osm_cloud
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["rcl_interfaces.msg"] = MagicMock()
sys.modules["ros2_numpy"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["tf2_ros"] = MagicMock()
sys.modules["visualization_msgs.msg"] = MagicMock()

# ── Fake rclpy.node.Node ────────────────────────────────────────────────────
#
# A bare MagicMock cannot be used as a base class (subclassing it silently
# turns OSMCloud itself into a MagicMock instead of a real class), so
# OSMCloud's __init__ never actually runs. To exercise __init__ for real we
# install a tiny, real Python class in place of rclpy.node.Node *before*
# map_data.osm_cloud is imported, implementing just enough of the Node API
# (declare_parameter, add_on_set_parameters_callback, create_publisher,
# create_timer, get_logger, get_clock) for construction to succeed without a
# running ROS graph.

_PENDING_PARAM_OVERRIDES: dict = {}


class _FakeParameter:
    """Stand-in for rclpy.Parameter — only `.value` is used by OSMCloud."""

    def __init__(self, value):
        self.value = value


class _FakeNode:
    """Minimal stand-in for rclpy.node.Node."""

    def __init__(self, node_name):
        self.node_name = node_name
        self.param_overrides = dict(_PENDING_PARAM_OVERRIDES)
        self.created_publishers: list[tuple[str, MagicMock]] = []
        self.created_timers: list[tuple[float, object, MagicMock]] = []
        self.param_callback = None
        self._logger = MagicMock()

    def declare_parameter(self, name, default=None):
        if name in self.param_overrides:
            return _FakeParameter(self.param_overrides[name])
        if isinstance(default, MagicMock):
            # rclpy.Parameter.Type.STRING / DOUBLE_ARRAY sentinels → "unset"
            return _FakeParameter(None)
        return _FakeParameter(default)

    def add_on_set_parameters_callback(self, callback):
        self.param_callback = callback

    def create_publisher(self, msg_type, topic, qos):
        pub = MagicMock(name=f"publisher:{topic}")
        self.created_publishers.append((topic, pub))
        return pub

    def create_timer(self, period, callback):
        timer = MagicMock()
        self.created_timers.append((period, callback, timer))
        return timer

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return MagicMock()


sys.modules["rclpy.node"].Node = _FakeNode

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from map_data.osm_cloud import (  # noqa: E402
    OSMCloud,
    create_grid,
    points_near_ref,
    split_ways_to_points,
    transform_points,
)
from map_data.utils.way import Way  # noqa: E402

# ── create_grid ───────────────────────────────────────────────────────────────


class TestCreateGrid:
    def test_basic_grid_shape(self):
        grid = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=1.0)
        assert grid.ndim == 2
        assert grid.shape[1] == 2
        assert grid.shape[0] == 4  # 2×2 = 4 points

    def test_cell_size_controls_density(self):
        coarse = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=1.0)
        fine = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=0.5)
        assert fine.shape[0] > coarse.shape[0]

    def test_empty_range_returns_empty(self):
        grid = create_grid((5.0, 5.0), (5.0, 5.0), cell_size=1.0)
        assert grid.shape[0] == 0

    def test_grid_bounds_within_range(self):
        grid = create_grid((0.0, 0.0), (3.0, 3.0), cell_size=0.5)
        assert grid[:, 0].min() >= 0.0
        assert grid[:, 0].max() < 3.0
        assert grid[:, 1].min() >= 0.0
        assert grid[:, 1].max() < 3.0


# ── points_near_ref ──────────────────────────────────────────────────────────


class TestPointsNearRef:
    def test_filters_distant_points(self):
        points = np.array([[0.0, 0.0], [10.0, 10.0], [0.5, 0.5]])
        ref = np.array([[0.0, 0.0]])
        result = points_near_ref(points, ref, max_dist=2.0)
        assert result.shape[0] == 2  # (0,0) and (0.5,0.5)
        assert result.shape[1] == 3  # 2 coords + cost

    def test_cost_is_normalized_distance(self):
        points = np.array([[1.0, 0.0]])
        ref = np.array([[0.0, 0.0]])
        result = points_near_ref(points, ref, max_dist=2.0)
        np.testing.assert_almost_equal(result[0, 2], 0.5)  # 1.0 / 2.0

    def test_no_points_near_ref(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        ref = np.array([[100.0, 100.0]])
        result = points_near_ref(points, ref, max_dist=1.0)
        assert result.shape[0] == 0

    def test_accepts_list_inputs(self):
        result = points_near_ref([[0.0, 0.0]], [[0.0, 0.0]], max_dist=1.0)
        assert result.shape[0] == 1


# ── transform_points ────────────────────────────────────────────────────────


class TestTransformPoints:
    def test_identity_preserves_points(self):
        pts = {0: np.array([[1.0], [2.0], [3.0]])}
        result = transform_points(pts, np.eye(4))
        np.testing.assert_array_almost_equal(result[0], pts[0])

    def test_translation(self):
        pts = {0: np.array([[0.0], [0.0], [0.0]])}
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        result = transform_points(pts, T)
        np.testing.assert_array_almost_equal(result[0].ravel(), [1.0, 2.0, 3.0])

    def test_z_override(self):
        pts = {0: np.array([[1.0], [2.0], [3.0]])}
        result = transform_points(pts, np.eye(4), z=0.0)
        assert result[0][2, 0] == 0.0

    def test_type_error_on_non_array(self):
        with pytest.raises(TypeError):
            transform_points({0: [1.0, 2.0, 3.0]}, np.eye(4))

    def test_multiple_points(self):
        pts = {
            0: np.array([[1.0], [0.0], [0.0]]),
            1: np.array([[0.0], [1.0], [0.0]]),
        }
        result = transform_points(pts, np.eye(4))
        assert len(result) == 2


# ── split_ways_to_points ────────────────────────────────────────────────────


class TestSplitWaysToPoints:
    def test_empty_ways_dict(self):
        result = split_ways_to_points({}, {})
        assert result.shape == (0, 2)

    def test_no_footways_key(self):
        result = split_ways_to_points({}, {"roads": []})
        assert result.shape == (0, 2)

    def test_empty_footways_list(self):
        result = split_ways_to_points({}, {"footways": []})
        assert result.shape == (0, 2)


# ── OSMCloud node construction ──────────────────────────────────────────────


class _FakeMapData:
    """Minimal stand-in for map_data.map_data.MapData."""

    min_x, max_x, min_y, max_y = 0.0, 10.0, 0.0, 10.0

    def get_points(self, z: float = 0.0):
        return {
            1: np.array([[0.0], [0.0], [0.0]]),
            2: np.array([[10.0], [10.0], [0.0]]),
        }

    def get_ways(self):
        footway = Way(
            id=1,
            nodes=[1, 2],
            tags={"highway": "footway"},
            line=LineString([(0.0, 0.0), (10.0, 10.0)]),
        )
        return {"footways": [footway]}

    def __str__(self):
        return "FakeMapData"


def _build_osm_cloud(overrides: dict) -> OSMCloud:
    """
    Construct a real OSMCloud instance against the fake Node/MapData.

    ``overrides`` simulates ROS parameter overrides (e.g. from a launch
    file); anything not listed falls back to the parameter's declared
    default.
    """
    _PENDING_PARAM_OVERRIDES.clear()
    _PENDING_PARAM_OVERRIDES.update(overrides)
    with patch("map_data.osm_cloud.md.MapData.load", return_value=_FakeMapData()):
        return OSMCloud()


class TestOSMCloudInit:
    def test_construction_wires_declared_parameters(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "auto_utm": True})

        assert isinstance(node, _FakeNode)
        assert node.node_name == "osm_cloud"
        assert node.utm_frame == "utm"
        assert node.local_frame == "map"
        assert node.mapdata_file == "fake.mapdata"
        assert node.gpx_file is None
        assert node.max_path_dist == pytest.approx(1.0)
        assert node.neighbor_cost == "linear"
        assert node.grid_res == pytest.approx(0.25)
        assert node.grid_topic == "grid"
        assert node.publish_intersections is False

    def test_construction_registers_parameter_callback(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "auto_utm": True})

        assert node.param_callback == node.parameter_callback

    def test_construction_creates_grid_publisher_on_declared_topic(self):
        node = _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "auto_utm": True, "grid_topic": "custom_grid"},
        )

        assert node.grid_topic == "custom_grid"
        topics = [topic for topic, _ in node.created_publishers]
        assert "custom_grid" in topics
        assert node.pub_grid is not None

    def test_construction_only_creates_intersection_publishers_when_enabled(self):
        node = _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "auto_utm": True, "publish_intersections": False},
        )

        assert not hasattr(node, "pub_poses")
        assert not hasattr(node, "pub_markers")

        node_with_intersections = _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "auto_utm": True, "publish_intersections": True},
        )
        topics = [topic for topic, _ in node_with_intersections.created_publishers]
        assert node_with_intersections.intersections_topic in topics
        assert node_with_intersections.intersection_markers_topic in topics

    def test_construction_registers_publish_timer(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "auto_utm": True})

        assert len(node.created_timers) == 1
        period, callback, _ = node.created_timers[0]
        assert period == pytest.approx(10.0)
        assert callback == node.publish_cb

    def test_construction_builds_grid_cloud_from_map_data(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "auto_utm": True})

        assert node.grid_cloud is not None
        assert node.map_data is not None

    def test_construction_exits_without_mapdata_or_gpx_file(self):
        with pytest.raises(SystemExit):
            _build_osm_cloud({})
