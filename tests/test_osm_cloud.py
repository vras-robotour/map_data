"""Tests for osm_cloud pure helper functions and node construction."""

import sys
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

# Mock ROS2 modules while importing osm_cloud; restored below so the real-ROS
# tests (test_launch, test_tracker_node) still see the real rclpy.
_ROS_STUBS = (
    "rclpy",
    "rclpy.executors",
    "rclpy.node",
    "rclpy.qos",
    "geometry_msgs.msg",
    "rcl_interfaces.msg",
    "ros2_numpy",
    "sensor_msgs.msg",
    "tf2_msgs.msg",
    "tf2_ros",
    "visualization_msgs.msg",
)
_REAL_MODULES = {name: sys.modules.get(name) for name in _ROS_STUBS}
sys.modules.update({name: MagicMock() for name in _ROS_STUBS})

# ── Fake rclpy.node.Node ────────────────────────────────────────────────────
#
# Subclassing a bare MagicMock turns OSMCloud itself into a MagicMock, so its
# __init__ never runs. A real class with just enough of the Node API stands in
# for rclpy.node.Node before map_data.osm_cloud is imported.

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
            # rclpy.Parameter.Type.STRING sentinel → "unset"
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

    def destroy_node(self):
        pass

    def get_clock(self):
        return MagicMock()


sys.modules["rclpy.node"].Node = _FakeNode


class _ExternalShutdown(Exception):
    """Stand-in for rclpy.executors.ExternalShutdownException (what spin raises on SIGINT)."""


sys.modules["rclpy.executors"].ExternalShutdownException = _ExternalShutdown

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import utm  # noqa: E402
from conftest import build_footway_network_mapdata  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from map_data import osm_cloud as osm_cloud_module  # noqa: E402
from map_data.annotations import annotation_path_for  # noqa: E402
from map_data.map_data import MapData  # noqa: E402
from map_data.osm_cloud import (  # noqa: E402
    NoMapInputError,
    OSMCloud,
    create_grid,
    points_near_ref,
    split_ways_to_points,
    transform_points,
)
from map_data.utils.way import Way  # noqa: E402

for _name, _module in _REAL_MODULES.items():
    if _module is None:
        del sys.modules[_name]
    else:
        sys.modules[_name] = _module

# ── create_grid ───────────────────────────────────────────────────────────────


class TestCreateGrid:
    def test_basic_grid_shape(self):
        grid = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=1.0)
        assert grid.ndim == 2
        assert grid.shape[1] == 2
        assert grid.shape[0] == 4  # 2x2 = 4 points

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


class TestCreateGridNear:
    """With ``near`` only the cells around the reference points are built."""

    @staticmethod
    def _path(n=400):
        t = np.linspace(0.0, 1.0, n)
        return np.column_stack([-30.0 + 80.0 * t, -20.0 + 60.0 * t + 5.0 * np.sin(8 * t)])

    @pytest.mark.parametrize(("cell", "dist"), [(0.25, 1.0), (0.4, 1.0), (0.5, 2.5), (1.0, 0.7)])
    def test_matches_the_full_grid_once_filtered(self, cell, dist):
        near = self._path()
        low, high = (-40.0, -30.0), (60.0, 50.0)

        full = points_near_ref(create_grid(low, high, cell), near, dist)
        thin = points_near_ref(create_grid(low, high, cell, near=near, max_dist=dist), near, dist)

        np.testing.assert_array_equal(thin, full)

    def test_builds_far_fewer_cells_than_the_bounding_box(self):
        near = self._path()
        full = create_grid((-40.0, -30.0), (60.0, 50.0), 0.25)
        thin = create_grid((-40.0, -30.0), (60.0, 50.0), 0.25, near=near, max_dist=1.0)

        assert len(thin) < len(full) / 5

    def test_points_outside_the_bounds_and_no_points(self):
        assert create_grid((0.0, 0.0), (5.0, 5.0), 1.0, near=np.empty((0, 2))).shape == (0, 2)
        far = np.array([[100.0, 100.0]])
        assert create_grid((0.0, 0.0), (5.0, 5.0), 1.0, near=far, max_dist=1.0).shape == (0, 2)


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

    def test_empty_points_returns_empty_dict(self):
        assert transform_points({}, np.eye(4)) == {}

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

    @staticmethod
    def _footway_and_road():
        points = {
            1: np.array([[0.0], [0.0], [0.0]]),
            2: np.array([[1.0], [0.0], [0.0]]),
            3: np.array([[0.0], [5.0], [0.0]]),
            4: np.array([[1.0], [5.0], [0.0]]),
        }
        ways = {
            "footways": [Way(id=1, nodes=[1, 2], tags={"highway": "footway"})],
            "roads": [Way(id=2, nodes=[3, 4], tags={"highway": "service"})],
        }
        return points, ways

    def test_footways_only_by_default(self):
        points, ways = self._footway_and_road()
        result = split_ways_to_points(points, ways, max_dist=0.5)
        assert set(result[:, 1]) == {0.0}

    def test_roads_only(self):
        points, ways = self._footway_and_road()
        result = split_ways_to_points(points, ways, max_dist=0.5, highway_types=["road"])
        assert set(result[:, 1]) == {5.0}

    def test_footways_and_roads(self):
        points, ways = self._footway_and_road()
        result = split_ways_to_points(points, ways, max_dist=0.5, highway_types=["footway", "road"])
        assert set(result[:, 1]) == {0.0, 5.0}
        assert result.shape == (6, 2)  # 3 samples per 1 m way at 0.5 m steps

    def test_unknown_and_repeated_types_are_ignored(self):
        points, ways = self._footway_and_road()
        result = split_ways_to_points(
            points, ways, max_dist=0.5, highway_types=["footway", "footway", "track"]
        )
        assert result.shape == (3, 2)

    def test_matches_the_reference_implementation(self):
        """The vectorised split must reproduce the plain per-segment loop exactly."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(-50.0, 50.0, size=(40, 2))
        coords[7] = coords[6]  # a duplicated node => a zero-length segment
        points = {i: np.array([[x], [y], [0.0]]) for i, (x, y) in enumerate(coords)}
        ways = {
            "footways": [
                Way(id=1, nodes=[0, 1, 2, 3, 4, 5], tags={"highway": "footway"}),
                Way(id=2, nodes=[6, 7, 8], tags={"highway": "footway"}),
                Way(id=3, nodes=[9], tags={"highway": "footway"}),  # single node: no segment
                Way(id=4, nodes=list(range(10, 40)), tags={"highway": "footway"}),
            ],
        }

        def reference(points, ways, max_dist):
            waypoints = []
            for way in ways["footways"]:
                for i, (id0, id1) in enumerate(zip(way.nodes, way.nodes[1:])):
                    point0 = points[id0].ravel()[:2]
                    point1 = points[id1].ravel()[:2]
                    if i == 0:
                        waypoints.append(point0)
                    dist = float(np.linalg.norm(point1 - point0))
                    if dist <= 1e-3:
                        waypoints.append(point1)
                        continue
                    vec = (point1 - point0) / dist
                    num = int(np.ceil(dist / max_dist))
                    step = dist / num
                    waypoints.extend(point0 + (j + 1) * step * vec for j in range(num))
            return np.array(waypoints)

        for max_dist in (0.25, 3.0, 1000.0):
            result = split_ways_to_points(points, ways, max_dist=max_dist)
            expected = reference(points, ways, max_dist)
            assert result.shape == expected.shape
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)


# ── OSMCloud node construction ──────────────────────────────────────────────


class _FakeMapData:
    """Minimal stand-in for map_data.map_data.MapData."""

    min_x, max_x, min_y, max_y = 0.0, 10.0, 0.0, 10.0
    footways_list: ClassVar[list] = []
    roads_list: ClassVar[list] = []
    crossroads_list: ClassVar[list] = []

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


def _build_osm_cloud(overrides: dict, loader: MagicMock | None = None) -> OSMCloud:
    """
    Construct a real OSMCloud instance against the fake Node/MapData.

    ``overrides`` simulates ROS parameter overrides (e.g. from a launch
    file); anything not listed falls back to the parameter's declared
    default. ``loader`` replaces the annotation-merging map loader, so a test
    can inspect the arguments the node passes to it.
    """
    _PENDING_PARAM_OVERRIDES.clear()
    _PENDING_PARAM_OVERRIDES.update(overrides)
    loader = loader or MagicMock(return_value=(_FakeMapData(), {}))
    with patch("map_data.osm_cloud.load_mapdata_with_annotations", loader):
        return OSMCloud()


class TestOSMCloudInit:
    def test_construction_wires_declared_parameters(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

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
        # "auto" needs no TF at all, so the node subscribes to nothing.
        assert node.tf_sub is None

    def test_construction_registers_parameter_callback(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node.param_callback == node.parameter_callback

    def test_construction_creates_grid_publisher_on_declared_topic(self):
        node = _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "transform_mode": "auto", "grid_topic": "custom_grid"},
        )

        assert node.grid_topic == "custom_grid"
        topics = [topic for topic, _ in node.created_publishers]
        assert "custom_grid" in topics
        assert node.pub_grid is not None

    def test_construction_only_creates_intersection_publishers_when_enabled(self):
        node = _build_osm_cloud(
            {
                "mapdata_file": "fake.mapdata",
                "transform_mode": "auto",
                "publish_intersections": False,
            },
        )

        assert not hasattr(node, "pub_poses")
        assert not hasattr(node, "pub_markers")

        node_with_intersections = _build_osm_cloud(
            {
                "mapdata_file": "fake.mapdata",
                "transform_mode": "auto",
                "publish_intersections": True,
            },
        )
        topics = [topic for topic, _ in node_with_intersections.created_publishers]
        assert node_with_intersections.intersections_topic in topics
        assert node_with_intersections.intersection_markers_topic in topics

    def test_construction_publishes_once_and_registers_no_timer_by_default(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        # Publishers are latched; the cloud is published once at construction.
        grid_pub = dict(node.created_publishers)[node.grid_topic]
        assert grid_pub.publish.call_count == 1
        assert node.created_timers == []

    def test_construction_registers_publish_timer_when_republish_period_set(self):
        node = _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "transform_mode": "auto", "republish_period": 7.5}
        )

        assert len(node.created_timers) == 1
        period, callback, _ = node.created_timers[0]
        assert period == pytest.approx(7.5)
        assert callback == node.publish_cb

    def test_construction_builds_grid_cloud_from_map_data(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node.grid_cloud is not None
        assert node.map_data is not None

    def test_highway_types_default_to_footways(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node.highway_types == ["footway"]

    def test_highway_types_drop_unknown_values(self):
        node = _build_osm_cloud(
            {
                "mapdata_file": "fake.mapdata",
                "transform_mode": "auto",
                "highway_types": ["footway", "road", "track"],
            }
        )

        assert node.highway_types == ["footway", "road"]
        node.get_logger().warning.assert_called()

    def test_highway_types_parameter_change_rebuilds_the_grid(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})
        grid_pub = dict(node.created_publishers)[node.grid_topic]

        param = MagicMock()  # MagicMock(name=...) would name the mock, not set .name
        param.name = "highway_types"
        param.value = ["footway", "road"]
        split = MagicMock(return_value=np.zeros((1, 2)))
        with patch("map_data.osm_cloud.split_ways_to_points", split):
            node.parameter_callback([param])

        assert node.highway_types == ["footway", "road"]
        assert split.call_args.args[3] == ["footway", "road"]
        assert grid_pub.publish.call_count == 2

    def test_construction_refuses_without_mapdata_or_gpx_file(self):
        with pytest.raises(NoMapInputError):
            _build_osm_cloud({})

    def test_main_idles_instead_of_exiting_without_a_map(self):
        """Exiting would only make launch's respawn restart into the same missing parameter."""
        rclpy = osm_cloud_module.rclpy
        rclpy.spin.side_effect = KeyboardInterrupt  # Ctrl+C while idling
        idle = rclpy.create_node.return_value
        with patch("map_data.osm_cloud.OSMCloud", side_effect=NoMapInputError("nothing")):
            osm_cloud_module.main()

        idle.get_logger().error.assert_called_once()
        rclpy.spin.assert_called_once_with(idle)
        idle.destroy_node.assert_called_once()
        rclpy.try_shutdown.assert_called()

    def test_main_shuts_down_cleanly_on_ctrl_c(self):
        rclpy = osm_cloud_module.rclpy
        rclpy.spin.side_effect = _ExternalShutdown  # what rclpy raises on SIGINT
        node = MagicMock()
        with patch("map_data.osm_cloud.OSMCloud", return_value=node):
            osm_cloud_module.main()

        node.destroy_node.assert_called_once()
        rclpy.try_shutdown.assert_called()


class TestOSMCloudMapLoading:
    """The node must load the same map the planner does (annotations + exclusions)."""

    def test_defaults_are_auto_annotations_and_no_stairs(self):
        loader = MagicMock(return_value=(_FakeMapData(), {}))
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"}, loader)

        assert node.annotations == "auto"
        assert node.exclude_highway == ["steps"]
        assert node.traversability_file == ""
        loader.assert_called_once_with(
            "fake.mapdata", None, exclude_highway=["steps"], traversability=None
        )

    def test_annotations_parameter_is_passed_through(self):
        loader = MagicMock(return_value=(_FakeMapData(), {}))
        _build_osm_cloud(
            {
                "mapdata_file": "fake.mapdata",
                "transform_mode": "auto",
                "annotations": "/tmp/store.json",
                "exclude_highway": [],
            },
            loader,
        )

        loader.assert_called_once_with(
            "fake.mapdata", "/tmp/store.json", exclude_highway=[], traversability=None
        )

    def test_annotations_none_loads_the_unedited_map(self):
        loader = MagicMock(return_value=(_FakeMapData(), {}))
        _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "transform_mode": "auto", "annotations": "none"},
            loader,
        )

        loader.assert_called_once_with(
            "fake.mapdata", "none", exclude_highway=["steps"], traversability=None
        )

    def test_traversability_file_is_passed_through(self):
        """A rule file given as a parameter reaches the loader; "" means the package one."""
        loader = MagicMock(return_value=(_FakeMapData(), {}))
        node = _build_osm_cloud(
            {
                "mapdata_file": "fake.mapdata",
                "transform_mode": "auto",
                "traversability_file": "/tmp/rules.yaml",
            },
            loader,
        )

        assert node.traversability_file == "/tmp/rules.yaml"
        loader.assert_called_once_with(
            "fake.mapdata",
            None,
            exclude_highway=["steps"],
            traversability="/tmp/rules.yaml",
        )

    def test_grass_way_is_missing_from_the_loaded_map(self, tmp_path):
        """The default rules reach the real loader: a grass way gets no ring."""
        path = tmp_path / "network.mapdata"
        lat0, lon0 = build_footway_network_mapdata(path)
        md = MapData.load(str(path))
        e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
        md.nodes_cache[107] = dict(
            zip(("lat", "lon"), utm.to_latlon(e0 + 200.0, n0 - 60.0, zn, zl), strict=False)
        ) | {"tags": {}}
        md.footways_list.append(
            Way(
                id=5,
                nodes=[107, 103],
                tags={"highway": "footway", "surface": "grass"},
                line=LineString([(e0 + 200.0, n0 - 60.0), (e0 + 200.0, n0)]).buffer(1.0),
            )
        )
        md.save(str(path))
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        merged = node.load_map_data(str(path))

        assert [w.id for w in merged.footways_list] == [1, 2, 3]

    def test_deleted_way_is_missing_from_the_loaded_map(self, tmp_path):
        """A real (small) map plus a store that deletes way 2: the node sees two ways."""
        path = tmp_path / "network.mapdata"
        build_footway_network_mapdata(path)
        annotation_path_for(path).write_text(
            json.dumps({"version": 1, "annotations": [], "deleted_ways": [{"id": 2}]})
        )
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        plain = MapData.load(str(path))
        merged = node.load_map_data(str(path))

        assert len(plain.footways_list) == 3
        assert [w.id for w in merged.footways_list] == [1, 3]


class TestOSMCloudPlacementTransform:
    """
    The map is placed once at start-up, so a bad or late transform must not stick.

    A Fixposition unit without a fusion fix publishes ``FP_ECEF -> FP_ENU0`` as all
    zeros, which would put the whole map in a metre-wide box and latch an empty cloud.
    """

    def _geodetic_node(self, **overrides) -> OSMCloud:
        """A node built without TF ("auto"), then switched to the geodetic checks."""
        node = _build_osm_cloud(
            {"mapdata_file": "fake.mapdata", "transform_mode": "auto"} | overrides
        )
        node.transform_mode = "geodetic"
        node.earth_frame, node.local_frame = "FP_ECEF", "FP_ENU0"
        return node

    def test_zero_ecef_transform_is_rejected(self):
        node = self._geodetic_node()

        # Identity: FP_ENU0 would sit at the centre of the Earth.
        assert node._placement_error(np.eye(4)) != ""

    def test_transform_to_a_point_on_earth_is_accepted(self):
        node = self._geodetic_node()
        ecef_to_local = np.eye(4)
        ecef_to_local[:3, 3] = [-3969709.07, -1021645.15, -4870468.23]  # Prague

        assert node._placement_error(ecef_to_local) == ""

    def test_utm_modes_do_not_check_the_origin(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node._placement_error(np.eye(4)) == ""

    def test_poll_tf_waits_out_an_invalid_transform(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})
        node.tf = MagicMock()
        good = np.eye(4)
        good[0, 3] = 7.0
        seen: list = []

        def validate(matrix):
            seen.append(matrix)
            return "no fix yet" if len(seen) == 1 else ""

        with (
            patch("map_data.osm_cloud.numpify", side_effect=[np.eye(4), good]),
            patch("map_data.osm_cloud.rclpy.ok", return_value=True),
            patch("map_data.osm_cloud.rclpy.spin_once") as spin,
        ):
            result = node._poll_tf("FP_ENU0", "FP_ECEF", validate)

        assert len(seen) == 2  # the placeholder was waited out, not returned
        assert spin.call_count == 1
        np.testing.assert_array_equal(result, good)

    def test_changed_transform_rebuilds_and_republishes(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})
        grid_pub = dict(node.created_publishers)[node.grid_topic]
        assert grid_pub.publish.call_count == 1
        node.tf = MagicMock()
        moved = np.eye(4)
        moved[0, 3] = 25.0

        with patch("map_data.osm_cloud.numpify", return_value=moved):
            node._replace_placement()

        np.testing.assert_array_equal(node.utm_to_local, moved)
        assert grid_pub.publish.call_count == 2

    def test_repeated_transform_does_not_republish(self):
        """/tf_static is re-broadcast periodically; the same numbers must be a no-op."""
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})
        grid_pub = dict(node.created_publishers)[node.grid_topic]
        node.tf = MagicMock()

        with patch("map_data.osm_cloud.numpify", return_value=node.utm_to_local.copy()):
            node._replace_placement()

        assert grid_pub.publish.call_count == 1

    def test_invalid_transform_does_not_replace_a_good_one(self):
        node = self._geodetic_node()
        node.ecef_to_local = np.eye(4)
        node.ecef_to_local[:3, 3] = [-3969709.07, -1021645.15, -4870468.23]
        good = node.ecef_to_local.copy()
        grid_pub = dict(node.created_publishers)[node.grid_topic]
        node.tf = MagicMock()

        with patch("map_data.osm_cloud.numpify", return_value=np.eye(4)):
            node._replace_placement()

        np.testing.assert_array_equal(node.ecef_to_local, good)
        assert grid_pub.publish.call_count == 1

    def test_tf_static_before_the_first_publish_does_not_rebuild(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})
        node.initialized = False

        with patch.object(node, "_replace_placement") as replace:
            node._tf_static_cb(MagicMock(transforms=[]))

        replace.assert_not_called()

    def test_tf_static_after_the_first_publish_rebuilds(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node.initialized is True
        with patch.object(node, "_replace_placement") as replace:
            node._tf_static_cb(MagicMock(transforms=[]))

        replace.assert_called_once()


class TestOSMCloudDegenerateGrid:
    def test_empty_grid_over_a_map_with_ways_is_not_published(self):
        with patch("map_data.osm_cloud.create_grid", return_value=np.empty((0, 2))):
            node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node.cloud_degenerate is True
        grid_pub = dict(node.created_publishers)[node.grid_topic]
        assert grid_pub.publish.call_count == 0
        node.get_logger().error.assert_called()

    def test_normal_grid_is_published(self):
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        assert node.cloud_degenerate is False
        grid_pub = dict(node.created_publishers)[node.grid_topic]
        assert grid_pub.publish.call_count == 1

    def test_degenerate_grid_holds_back_the_intersections_too(self):
        """The rings come from the same placement, so they are just as misplaced."""
        with patch("map_data.osm_cloud.create_grid", return_value=np.empty((0, 2))):
            node = _build_osm_cloud(
                {
                    "mapdata_file": "fake.mapdata",
                    "transform_mode": "auto",
                    "publish_intersections": True,
                },
            )

        assert node.cloud_degenerate is True
        published = dict(node.created_publishers)
        assert published[node.intersections_topic].publish.call_count == 0
        assert published[node.intersection_markers_topic].publish.call_count == 0


class TestOSMCloudParameterCallback:
    @staticmethod
    def _change(**values):
        params = []
        for name, value in values.items():
            param = MagicMock()
            param.name, param.value = name, value
            params.append(param)
        return params

    @pytest.fixture
    def node(self, monkeypatch):
        # rcl_interfaces is stubbed out in this module: give the result real attributes
        monkeypatch.setattr(
            osm_cloud_module,
            "SetParametersResult",
            lambda successful, reason="": SimpleNamespace(successful=successful, reason=reason),
        )
        return _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

    @pytest.mark.parametrize(
        "change",
        [
            {"grid_res": 0.0},
            {"grid_res": -1.0},
            {"max_path_dist": 0.0},
            {"grid_min": [0.0]},
            {"grid_max": [1.0, 2.0, 3.0]},
        ],
    )
    def test_invalid_values_are_rejected_and_change_nothing(self, node, change):
        before = {
            k: getattr(node, k) for k in ("grid_res", "max_path_dist", "grid_min", "grid_max")
        }
        grid_pub = dict(node.created_publishers)[node.grid_topic]

        result = node.parameter_callback(self._change(**change))

        assert result.successful is False
        assert result.reason
        assert {k: getattr(node, k) for k in before} == before
        assert grid_pub.publish.call_count == 1  # the one from construction

    def test_a_failed_rebuild_restores_the_old_values(self, node):
        old_cloud, old_max, old_auto = node.grid_cloud, node.grid_max, node.grid_bounds_auto
        with patch.object(node, "get_cloud", side_effect=RuntimeError("boom")):
            result = node.parameter_callback(self._change(grid_res=0.5, grid_max=[3.0, 3.0]))

        assert result.successful is False
        assert result.reason == "boom"
        assert node.grid_res == pytest.approx(0.25)
        assert node.grid_max == old_max
        assert node.grid_bounds_auto == old_auto
        assert node.grid_cloud is old_cloud

    def test_a_failed_second_rebuild_keeps_the_old_outputs(self, node):
        node.publish_intersections = True
        old_cloud, old_poses, old_markers = node.grid_cloud, node.poses, node.markers
        with (
            patch.object(node, "get_cloud", return_value=MagicMock(name="new cloud")),
            patch.object(node, "get_intersections", side_effect=RuntimeError("boom")),
        ):
            result = node.parameter_callback(self._change(grid_max=[3.0, 3.0]))

        assert result.successful is False
        assert node.grid_cloud is old_cloud
        assert (node.poses, node.markers) == (old_poses, old_markers)

    def test_a_valid_change_is_applied_and_republished(self, node):
        grid_pub = dict(node.created_publishers)[node.grid_topic]

        result = node.parameter_callback(self._change(grid_res=0.5))

        assert result.successful is True
        assert node.grid_res == pytest.approx(0.5)
        assert grid_pub.publish.call_count == 2


class TestOSMCloudIntersectionMarkers:
    def test_markers_start_with_a_delete_all(self):
        """RViz keeps old spheres until told to drop them, so a rebuild must clear first."""
        marker_cls = MagicMock(side_effect=lambda: MagicMock())
        marker_cls.DELETEALL, marker_cls.ADD, marker_cls.SPHERE = 3, 0, 2
        node = _build_osm_cloud({"mapdata_file": "fake.mapdata", "transform_mode": "auto"})

        with (
            patch("map_data.osm_cloud.Marker", marker_cls),
            patch("map_data.osm_cloud.MarkerArray", lambda markers: MagicMock(markers=markers)),
        ):
            _, markers = node.get_intersections()

        assert markers.markers[0].action == 3


class TestOSMCloudGpxMap:
    """The GPX-only path must build the same map as the .mapdata path."""

    @pytest.fixture
    def gpx(self, tmp_path):
        path = tmp_path / "area.gpx"
        path.write_text(
            '<gpx version="1.1" creator="t"><wpt lat="50.0" lon="14.0"/>'
            '<wpt lat="50.001" lon="14.001"/></gpx>'
        )
        return path

    @staticmethod
    def _download(tmp_path, monkeypatch, *, ok=True):
        """Make MapData's Overpass step "download" the small network (plus a grass way)."""
        source = tmp_path / "network.mapdata"
        lat0, lon0 = build_footway_network_mapdata(source)
        built = MapData.load(str(source))
        e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
        built.nodes_cache[107] = dict(
            zip(("lat", "lon"), utm.to_latlon(e0 + 200.0, n0 - 60.0, zn, zl), strict=False)
        ) | {"tags": {}}
        built.footways_list.append(
            Way(
                id=5,
                nodes=[107, 103],
                tags={"highway": "footway", "surface": "grass"},
                line=LineString([(e0 + 200.0, n0 - 60.0), (e0 + 200.0, n0)]).buffer(1.0),
            )
        )

        def run_queries(self, **_):
            self.osm_data = object() if ok else None

        def run_parse(self):
            self.footways_list, self.roads_list = built.footways_list, built.roads_list
            self.nodes_cache, self.crossroads_list = built.nodes_cache, built.crossroads_list
            return 0

        monkeypatch.setattr(MapData, "run_queries", run_queries)
        monkeypatch.setattr(MapData, "run_parse", run_parse)

    def test_traversability_and_annotations_are_applied(self, gpx, tmp_path, monkeypatch):
        self._download(tmp_path, monkeypatch)
        annotation_path_for(gpx).write_text(
            json.dumps({"version": 1, "annotations": [], "deleted_ways": [{"id": 2}]})
        )

        node = _build_osm_cloud({"gpx_file": str(gpx), "transform_mode": "auto"})

        assert [w.id for w in node.map_data.footways_list] == [1, 3]  # no grass way 5, no way 2

    def test_no_osm_data_fails_instead_of_latching_an_empty_grid(self, gpx, tmp_path, monkeypatch):
        self._download(tmp_path, monkeypatch, ok=False)

        with pytest.raises(RuntimeError, match="Overpass"):
            _build_osm_cloud({"gpx_file": str(gpx), "transform_mode": "auto"})
