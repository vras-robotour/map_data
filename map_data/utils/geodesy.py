"""
Geodetic conversions between UTM, WGS84 lat/lon and Earth-Centred Earth-Fixed (ECEF).

Map data is stored in UTM. Robots that localise with a GNSS/INS unit (e.g. Fixposition)
expose an Earth frame in ECEF coordinates and a local ENU frame as a static TF child of it.
Converting UTM -> lat/lon -> ECEF and then applying the (inverse) Earth->local TF places map
features in the local frame *exactly*, whereas treating UTM as a translated copy of the
local ENU frame is off by UTM grid convergence and scale (several metres per kilometre).

All functions are vectorised over NumPy arrays and have no ROS dependency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import utm

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2


def latlon_to_ecef(
    lat_deg: npt.ArrayLike,
    lon_deg: npt.ArrayLike,
    alt_m: npt.ArrayLike = 0.0,
) -> np.ndarray:
    """
    Convert WGS84 geodetic coordinates to ECEF.

    Parameters
    ----------
    lat_deg, lon_deg : array-like
        Latitude / longitude in degrees.
    alt_m : array-like
        Height above the WGS84 ellipsoid in metres.

    Returns
    -------
    ecef : np.ndarray, shape (N, 3)
        X, Y, Z in metres.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)


def ecef_to_latlon(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Convert an ECEF point to WGS84 (lat_deg, lon_deg, alt_m) using Bowring's closed form.
    """
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + _EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def ecef_to_latlon_array(xyz: np.ndarray) -> np.ndarray:
    """
    Vectorised ``ecef_to_latlon``: (N, 3) ECEF -> (N, 3) ``[lat_deg, lon_deg, alt_m]``.
    """
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + _EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.stack([np.degrees(lat), np.degrees(lon), alt], axis=-1)


def utm_to_ecef(
    easting: np.ndarray,
    northing: np.ndarray,
    zone_number: int,
    zone_letter: str,
    alt_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    """
    Convert UTM coordinates (one zone) to ECEF, shape (N, 3).
    """
    easting = np.atleast_1d(np.asarray(easting, dtype=float))
    northing = np.atleast_1d(np.asarray(northing, dtype=float))
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter, strict=False)
    return latlon_to_ecef(lat, lon, alt_m)


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to points of shape (N, 3).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ matrix[:3, :3].T + matrix[:3, 3]


def utm_to_local_via_ecef(
    easting: np.ndarray,
    northing: np.ndarray,
    zone_number: int,
    zone_letter: str,
    ecef_to_local: np.ndarray,
    alt_m: np.ndarray | float | None = None,
) -> np.ndarray:
    """
    Place UTM points in a local frame defined by an ECEF->local 4x4 transform.

    ``alt_m`` defaults to the ellipsoidal height of the local frame's origin so that
    the ground plane of the local frame is used (the residual effect of altitude on the
    horizontal ENU coordinates is centimetres over a few kilometres).

    Returns
    -------
    local : np.ndarray, shape (N, 3)
    """
    if alt_m is None:
        origin_ecef = np.linalg.inv(ecef_to_local)[:3, 3]
        alt_m = ecef_to_latlon(*origin_ecef)[2]
    ecef = utm_to_ecef(easting, northing, zone_number, zone_letter, alt_m)
    return apply_transform(ecef, ecef_to_local)
