"""Tests for osm_cloud pure helper functions."""

import sys
from unittest.mock import MagicMock

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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from map_data.osm_cloud import (  # noqa: E402
    create_grid,
    points_near_ref,
    split_ways_to_points,
    transform_points,
)

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
