import logging
import os

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
    if not os.path.exists(path_file):
        logger.error("Path file %s does not exist.", path_file)
        return []

    if path_file.endswith(".gpx"):
        return parse_gpx_file(path_file)
    if path_file.endswith(".yaml"):
        return parse_yaml_file(path_file)
    logger.error("Unsupported file format: %s.", path_file)
    return []


def parse_gpx_file(gpx_file: str) -> tuple[np.ndarray, int, str] | list:
    waypoints = []
    zone_num, zone_let = None, None
    try:
        with open(gpx_file) as file:
            gpx = gpxpy.parse(file)
        for waypoint in gpx.waypoints:
            point = {
                "lat": waypoint.latitude,
                "lon": waypoint.longitude,
                "ele": waypoint.elevation or 0,
            }
            waypoints.append(convert_waypoint(point))
    except Exception as e:
        logger.exception("Error parsing GPX file: %s", e)
        return []
    if not waypoints:
        logger.warning("No waypoints found in GPX file.")
    else:
        logger.info("Parsed %s waypoints from GPX file.", len(waypoints))
        zone_num, zone_let = utm.from_latlon(gpx.waypoints[0].latitude, gpx.waypoints[0].longitude)[
            2:
        ]

    return np.array(waypoints), zone_num, zone_let


def parse_yaml_file(yaml_file: str) -> tuple[np.ndarray, int, str] | list:
    waypoints = []
    zone_num, zone_let = None, None
    try:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        file_waypoints = data["waypoints"]
        for waypoint in file_waypoints:
            point = {"lat": waypoint["latitude"], "lon": waypoint["longitude"]}
            if "elevation" in waypoint:
                point["ele"] = waypoint["elevation"]
            else:
                point["ele"] = 0
            waypoints.append(convert_waypoint(point))
    except Exception as e:
        logger.exception("Error parsing YAML file: %s", e)
        return []
    if not waypoints:
        logger.warning("No waypoints found in YAML file.")
    else:
        logger.info("Parsed %s waypoints from YAML file.", len(waypoints))
        zone_num, zone_let = utm.from_latlon(
            file_waypoints[0]["latitude"], file_waypoints[0]["longitude"],
        )[2:]

    return np.array(waypoints), zone_num, zone_let


def convert_waypoint(point: dict[str, float]) -> tuple[float, float, float]:
    utm_point = utm.from_latlon(point["lat"], point["lon"])[:2]
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
    waypoints_data: list[dict[str, str | float]],
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
