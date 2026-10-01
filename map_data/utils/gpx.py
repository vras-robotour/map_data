from collections.abc import Sequence

import gpxpy


def create_gpx_content(points: Sequence[Sequence[float]]) -> str:
    """GPX 1.1 file of bare ``<wpt>`` elements from ``[(lat, lon), ...]``."""
    gpx = gpxpy.gpx.GPX()
    gpx.creator = "MapData Planner"
    for p in points:
        gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=float(p[0]), longitude=float(p[1])))
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
