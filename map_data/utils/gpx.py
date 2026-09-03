import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gpxpy
import numpy as np
import utm
import yaml

logger = logging.getLogger(__name__)


def parse_path(path_file: str) -> tuple[np.ndarray, int, str] | list:
    """
    Parse a path from a GPX or YAML file.
    """
    if not path_file:
        logger.error("No path file provided.")
        return []
    p = Path(path_file)
    if not p.exists():
        logger.error("Path file %s does not exist.", path_file)
        return []

    if p.suffix == ".gpx":
        return parse_gpx_file(str(p))
    if p.suffix == ".yaml":
        return parse_yaml_file(str(p))
    logger.error("Unsupported file format: %s.", path_file)
    return []


def parse_gpx_file(gpx_file: str) -> tuple[np.ndarray, int, str] | list:
    waypoints = []
    zone_num, zone_let = None, None
    try:
        with Path(gpx_file).open() as file:
            gpx = gpxpy.parse(file)

        # Mirror the waypoints -> tracks -> routes fallback from MapData.__init__.
        # Typed as list[Any]: waypoints/tracks/routes yield different gpxpy point
        # classes (GPXWaypoint/GPXTrackPoint/GPXRoutePoint) that are only ever
        # duck-typed below via .latitude/.longitude/.elevation.
        gpx_points: list[Any]
        if gpx.waypoints:
            gpx_points = gpx.waypoints
        elif gpx.tracks:
            gpx_points = [p for track in gpx.tracks for seg in track.segments for p in seg.points]
        elif gpx.routes:
            gpx_points = [p for route in gpx.routes for p in route.points]
        else:
            gpx_points = []

        for waypoint in gpx_points:
            point = {
                "lat": waypoint.latitude,
                "lon": waypoint.longitude,
                "ele": waypoint.elevation or 0,
            }
            if zone_num is None:
                # Anchor the whole path in the first waypoint's UTM zone so a
                # route crossing a zone boundary stays in one coordinate frame.
                zone_num, zone_let = utm.from_latlon(point["lat"], point["lon"])[2:]
            waypoints.append(convert_waypoint(point, zone_num, zone_let))
    except Exception:
        logger.exception("Error parsing GPX file")
        return []
    if not waypoints:
        logger.warning("No waypoints found in GPX file.")
        return []
    else:
        logger.info("Parsed %s waypoints from GPX file.", len(waypoints))

    assert zone_num is not None and zone_let is not None  # set with the first waypoint
    return np.array(waypoints), zone_num, zone_let


def parse_yaml_file(yaml_file: str) -> tuple[np.ndarray, int, str] | list:
    waypoints = []
    zone_num, zone_let = None, None
    try:
        with Path(yaml_file).open() as f:
            data = yaml.safe_load(f)
        file_waypoints = data["waypoints"]
        for waypoint in file_waypoints:
            point = {"lat": waypoint["latitude"], "lon": waypoint["longitude"]}
            if "elevation" in waypoint:
                point["ele"] = waypoint["elevation"]
            else:
                point["ele"] = 0
            if zone_num is None:
                # Anchor the whole path in the first waypoint's UTM zone so a
                # route crossing a zone boundary stays in one coordinate frame.
                zone_num, zone_let = utm.from_latlon(point["lat"], point["lon"])[2:]
            waypoints.append(convert_waypoint(point, zone_num, zone_let))
    except Exception:
        logger.exception("Error parsing YAML file")
        return []
    if not waypoints:
        logger.warning("No waypoints found in YAML file.")
        return []
    else:
        logger.info("Parsed %s waypoints from YAML file.", len(waypoints))

    assert zone_num is not None and zone_let is not None  # set with the first waypoint
    return np.array(waypoints), zone_num, zone_let


def convert_waypoint(
    point: dict[str, float],
    zone_number: int | None = None,
    zone_letter: str | None = None,
) -> tuple[float, float, float]:
    """
    Convert a lat/lon point to UTM, optionally forced into a given zone so
    every point of a path shares the zone reported for the whole array.
    """
    utm_point = utm.from_latlon(
        point["lat"],
        point["lon"],
        force_zone_number=zone_number,
        force_zone_letter=zone_letter,
    )[:2]
    return (*utm_point, point.get("ele", 0))


def utm_path_to_latlon(path: np.ndarray, zone_num: int, zone_let: str) -> list[dict[str, float]]:
    wgs_path = []
    for point in path:
        lat, lon = utm.to_latlon(point[0], point[1], zone_num, zone_let)
        # Ensure point has at least 3 elements for elevation, default to 0 if not
        ele = point[2] if len(point) > 2 else 0
        wgs_path.append({"latitude": lat, "longitude": lon, "elevation": ele})
    return wgs_path


def create_gpx_content(
    waypoints_data: Sequence[Mapping[str, str | float]],
    creator_name: str = "MapData Planner",
) -> str:
    """
    Generates the XML content for a GPX file from a list of waypoint dictionaries.
    """
    gpx_waypoints = []
    for point in waypoints_data:
        try:
            lat = point["latitude"]
            lon = point["longitude"]
            gpx_waypoints.append(f'  <wpt lat="{lat}" lon="{lon}"></wpt>')
        except KeyError as e:
            logger.warning("Skipping a waypoint due to missing key: %s", e)
            continue

    waypoints_xml = "\n".join(gpx_waypoints)

    gpx_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1" creator="{creator_name}">
{waypoints_xml}
</gpx>
    """
    return gpx_template.strip()


def create_gpx_track(
    points: Sequence[Sequence[float]],
    name: str = "route",
    creator_name: str = "MapData Planner",
) -> str:
    """
    Serialize ``[(lat, lon[, ele]), ...]`` as a GPX 1.1 track (``<trk>``).

    Track points keep their order, which is what a waypoint follower needs;
    :func:`create_gpx_content` writes the same points as bare ``<wpt>``
    elements instead. Both are read back by :func:`parse_gpx_file` and by
    ``robot_mission_planner``.
    """
    gpx = gpxpy.gpx.GPX()
    gpx.creator = creator_name
    track = gpxpy.gpx.GPXTrack(name=name)
    segment = gpxpy.gpx.GPXTrackSegment()
    for p in points:
        ele = float(p[2]) if len(p) > 2 else None
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(latitude=float(p[0]), longitude=float(p[1]), elevation=ele)
        )
    track.segments.append(segment)
    gpx.tracks.append(track)
    return gpx.to_xml()
