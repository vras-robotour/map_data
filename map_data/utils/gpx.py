import logging
from collections.abc import Mapping, Sequence

import gpxpy

logger = logging.getLogger(__name__)


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
    elements instead.
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
