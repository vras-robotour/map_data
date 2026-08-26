"""ROS-free parts of the viewer tracker (``map_data.viewer.ros_node``)."""

import math

import numpy as np
import pytest

from map_data.utils.geodesy import ecef_to_latlon, ecef_to_latlon_array, latlon_to_ecef
from map_data.viewer import ros_node
from map_data.viewer.ros_node import (
    heading_from_yaw,
    subsample,
    xyz_to_latlon,
    yaw_from_quaternion,
)

# FP_ENU0 origin of the Stromovka bag (2026-08-21)
ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 50.1048328, 14.4296035, 237.9


def enu_to_ecef_matrix(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, sn, cn = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    rot = np.array([[-sn, -sl * cn, cl * cn], [cn, -sl * sn, cl * sn], [0.0, cl, sl]])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = latlon_to_ecef(lat_deg, lon_deg, alt_m)[0]
    return m


class TestGeodesyArray:
    def test_matches_scalar(self):
        pts = latlon_to_ecef([50.1, 50.2], [14.4, 14.5], [200.0, 300.0])
        arr = ecef_to_latlon_array(pts)
        for row, pt in zip(arr, pts, strict=True):
            assert row == pytest.approx(ecef_to_latlon(*pt), abs=1e-9)

    def test_round_trip(self):
        arr = ecef_to_latlon_array(latlon_to_ecef(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT))
        assert arr[0] == pytest.approx([ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT], abs=1e-6)


class TestXyzToLatlon:
    def test_ecef_points_need_no_matrix(self):
        pts = latlon_to_ecef([ORIGIN_LAT], [ORIGIN_LON], [ORIGIN_ALT])
        out = xyz_to_latlon(pts, None)
        assert out == [{"lat": pytest.approx(ORIGIN_LAT), "lon": pytest.approx(ORIGIN_LON)}]

    def test_local_enu_points_through_matrix(self):
        m = enu_to_ecef_matrix(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
        # 1000 m east -> ~0.01395 deg lon at lat 50.1; 1000 m north -> ~0.00899 deg lat
        out = xyz_to_latlon(np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]]), m)
        assert out[0]["lat"] == pytest.approx(ORIGIN_LAT, abs=1e-7)
        assert out[0]["lon"] == pytest.approx(ORIGIN_LON, abs=1e-7)
        assert out[1]["lon"] - ORIGIN_LON == pytest.approx(0.01395, abs=2e-4)
        assert out[1]["lat"] == pytest.approx(ORIGIN_LAT, abs=1e-5)
        assert out[2]["lat"] - ORIGIN_LAT == pytest.approx(0.00899, abs=2e-4)

    def test_empty(self):
        assert xyz_to_latlon(np.zeros((0, 3)), None) == []


class TestHeading:
    def test_yaw_to_compass(self):
        assert heading_from_yaw(0.0) == 90.0  # east
        assert heading_from_yaw(math.pi / 2) == 0.0  # north
        assert heading_from_yaw(-math.pi / 2) == 180.0  # south
        # Stromovka bag start: yaw -3.085 rad -> heading ~266.8 (west)
        assert heading_from_yaw(-3.0851364553249208) == pytest.approx(266.8, abs=0.1)

    def test_quaternion_yaw(self):
        assert yaw_from_quaternion(0.0, 0.0, math.sin(0.5), math.cos(0.5)) == pytest.approx(1.0)


class TestSubsample:
    def test_keeps_last(self):
        pts = list(range(23))
        out = subsample(pts, 10)
        assert out == [0, 10, 20, 22]

    def test_short_unchanged(self):
        assert subsample([1, 2, 3], 10) == [1, 2, 3]


def test_module_imports_without_ros():
    # The module must stay importable in the plain (non-ROS) test environment.
    assert isinstance(ros_node.ROS_AVAILABLE, bool)
