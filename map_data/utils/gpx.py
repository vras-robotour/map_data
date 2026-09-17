import logging
from collections.abc import Mapping, Sequence

import gpxpy

logger = logging.getLogger(__name__)


def create_gpx_content(
    waypoints_data: Sequence[Mapping[str, str | float]],
    creator_name: str = "MapData Planner",
) -> str:
    """GPX 1.1 file of bare ``<wpt>`` elements from ``{"latitude", "longitude"}`` dicts."""
    gpx = gpxpy.gpx.GPX()
    gpx.creator = creator_name
    for point in waypoints_data:
        try:
            lat, lon = float(point["latitude"]), float(point["longitude"])
        except KeyError as e:
            logger.warning("Skipping a waypoint due to missing key: %s", e)
            continue
        gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon))
    return gpx.to_xml()


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
