"""Tests for map_data.utils.geodesy against values recorded on the Helhest robot."""

import numpy as np
import pytest
import utm

from map_data.utils.geodesy import (
    apply_transform,
    ecef_to_latlon,
    latlon_to_ecef,
    utm_to_ecef,
    utm_to_local_via_ecef,
)

# FP_ECEF -> FP_ENU0 static transform from a Stromovka bag (2026-08-21); the local
# frame's pose expressed in ECEF (translation + unit quaternion x, y, z, w).
_ENU_IN_ECEF_T = np.array([3969770.044, 1021450.309, 4870458.658])
_ENU_IN_ECEF_Q = np.array([0.209, 0.27, 0.743, 0.576])


def _quat_to_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@pytest.fixture
def ecef_to_local():
    enu_in_ecef = np.eye(4)
    enu_in_ecef[:3, :3] = _quat_to_matrix(_ENU_IN_ECEF_Q)
    enu_in_ecef[:3, 3] = _ENU_IN_ECEF_T
    return np.linalg.inv(enu_in_ecef)


class TestLatLonEcefRoundTrip:
    def test_round_trip(self):
        lat, lon, alt = 50.1048328, 14.4296035, 237.9
        x, y, z = latlon_to_ecef(lat, lon, alt)[0]
        back = ecef_to_latlon(x, y, z)
        assert back == pytest.approx((lat, lon, alt), abs=1e-6)

    def test_vectorised(self):
        out = latlon_to_ecef(np.array([50.0, 50.1]), np.array([14.0, 14.5]), 0.0)
        assert out.shape == (2, 3)

    def test_origin_of_bag_frame(self):
        lat, lon, alt = ecef_to_latlon(*_ENU_IN_ECEF_T)
        # Stromovka, Prague
        assert lat == pytest.approx(50.10483, abs=1e-4)
        assert lon == pytest.approx(14.42960, abs=1e-4)
        assert 200 < alt < 300


class TestLocalPlacement:
    # Robot sample from the same bag: /fixposition/odometry_llh vs /fixposition/odometry_enu
    LAT, LON, ALT = 50.11027772879592, 14.417394714109097, 227.0433391035097
    ENU = np.array([-873.3, 605.7])

    def test_latlon_sample_lands_on_enu(self, ecef_to_local):
        local = apply_transform(latlon_to_ecef(self.LAT, self.LON, self.ALT), ecef_to_local)
        # the recorded quaternion is rounded to 3 decimals -> ~0.2 m
        assert np.allclose(local[0, :2], self.ENU, atol=0.3)

    def test_utm_sample_lands_on_enu(self, ecef_to_local):
        e, n, zn, zl = utm.from_latlon(self.LAT, self.LON)
        local = utm_to_local_via_ecef(e, n, zn, zl, ecef_to_local)
        assert np.allclose(local[0, :2], self.ENU, atol=0.3)

    def test_plain_utm_translation_is_worse(self, ecef_to_local):
        """Documents why geodetic placement exists: UTM grid convergence at ~1 km."""
        origin = ecef_to_latlon(*_ENU_IN_ECEF_T)
        oe, on, _, _ = utm.from_latlon(origin[0], origin[1])
        e, n, _, _ = utm.from_latlon(self.LAT, self.LON)
        translated = np.array([e - oe, n - on])
        assert np.linalg.norm(translated - self.ENU) > 3.0

    def test_utm_to_ecef_shape(self):
        out = utm_to_ecef(np.array([459210.0, 458000.0]), np.array([5550442.0, 5550000.0]), 33, "U")
        assert out.shape == (2, 3)
