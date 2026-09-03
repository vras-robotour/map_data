"""
``map_data_plan``: plan a route on a ``.mapdata`` file from the command line.

The offline twin of the viewer's Planner screen and of the ``route_planner``
ROS action server: same map (annotations merged), same planners, same
parameters, no GUI and no ROS.

Examples
--------
Paths-only route between two coordinates, 3 m waypoint spacing, saved as a GPX track::

    map_data_plan -f stromovka.mapdata --start 50.1038,14.4294 --goal 50.1067,14.4193 \\
        --spacing 3 --save route.gpx

All-terrain (grid A*) route through several points, printed as JSON::

    map_data_plan -f stromovka.mapdata -p 50.1038,14.4294 -p 50.1050,14.4250 -p 50.1067,14.4193 \\
        --algorithm astar --cell-size 0.5 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from map_data.annotations import NO_ANNOTATIONS, load_mapdata_with_annotations
from map_data.pathsolver.graph_planner import DEFAULT_MAX_SNAP_DISTANCE
from map_data.pathsolver.route import (
    GRAPH_ALGORITHM,
    GRID_ALGORITHMS,
    RoutePlanningError,
    plan_route,
    route_to_dicts,
)
from map_data.utils.gpx import create_gpx_content, create_gpx_track

logger = logging.getLogger("map_data_plan")


def parse_latlon(text: str) -> tuple[float, float]:
    """``"lat,lon"`` (or ``"geo:lat,lon"`` as printed on a Robotour QR code)."""
    body = text.strip()
    if body.lower().startswith("geo:"):
        body = body[4:]
    parts = body.split(",")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"expected lat,lon, got {text!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected lat,lon, got {text!r}") from e


def annotations_arg(value: str) -> str | None:
    """``--annotations`` -> ``annotations_path`` for :func:`load_mapdata_with_annotations`."""
    if value in ("", "auto"):
        return None
    if value == NO_ANNOTATIONS:
        return NO_ANNOTATIONS
    p = Path(value).expanduser()
    if not p.is_file():
        raise SystemExit(f"annotation store {value!r} not found")
    return str(p)


def resolve_mapdata(name: str) -> Path:
    """A path as given, else the file of that name in the package data directory."""
    p = Path(name).expanduser()
    if p.exists():
        return p.resolve()
    try:
        from ament_index_python.resources import get_resource

        _, package_path = get_resource("packages", "map_data")
        candidate = Path(package_path) / "share" / "map_data" / "data" / name
    except (ImportError, LookupError):
        candidate = (Path(__file__).parent / ".." / "data" / name).resolve()
    if candidate.exists():
        return candidate
    raise SystemExit(f"map data file {name!r} not found (looked at {p} and {candidate})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="map_data_plan",
        description="Plan a route on a .mapdata file (annotations merged) without the viewer.",
    )
    ap.add_argument("-f", "--file", required=True, help=".mapdata file (path or name in data/)")
    ap.add_argument(
        "--annotations",
        default="auto",
        metavar="auto|none|FILE",
        help="annotation store to merge: auto = <file>.annotations.json next to the map "
        "(default), none = the unedited map, or an explicit store file",
    )
    ap.add_argument("--start", type=parse_latlon, help="start lat,lon")
    ap.add_argument("--goal", type=parse_latlon, help="goal lat,lon (or geo:lat,lon)")
    ap.add_argument(
        "-p",
        "--point",
        dest="points",
        action="append",
        type=parse_latlon,
        default=[],
        help="via point lat,lon; repeat for several (used between --start and --goal)",
    )
    ap.add_argument(
        "--algorithm",
        default=GRAPH_ALGORITHM,
        choices=(GRAPH_ALGORITHM, *GRID_ALGORITHMS),
        help="graph = paths only (default); astar/rrt = all-terrain grid planner",
    )
    ap.add_argument(
        "--ways",
        nargs="+",
        default=["footway"],
        metavar="TYPE",
        help="allowed way types: footway and/or road (default: footway)",
    )
    ap.add_argument("--spacing", type=float, default=0.0, help="max m between output waypoints")
    ap.add_argument(
        "--max-snap-distance",
        type=float,
        default=DEFAULT_MAX_SNAP_DISTANCE,
        help="graph: reject waypoints farther than this (m) from every allowed way",
    )
    ap.add_argument("--cell-size", type=float, default=0.25, help="grid: cell size (m)")
    ap.add_argument(
        "--inflate-obstacles", type=float, default=0.25, help="grid: barrier margin (m)"
    )
    ap.add_argument("--no-simplify", action="store_true", help="grid: keep every grid vertex")
    ap.add_argument("--smooth", action="store_true", help="grid: smooth the path")
    ap.add_argument("--save", help="write the route as GPX (track by default)")
    ap.add_argument(
        "--wpt", action="store_true", help="write GPX <wpt> elements instead of a track"
    )
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    points: list[tuple[float, float]] = []
    if args.start:
        points.append(args.start)
    points.extend(args.points)
    if args.goal:
        points.append(args.goal)
    if len(points) < 2:
        logger.error("Give at least two points (--start/--goal or several -p).")
        return 2

    path = resolve_mapdata(args.file)
    md, _ = load_mapdata_with_annotations(path, annotations_arg(args.annotations))
    try:
        result = plan_route(
            md,
            points,
            algorithm=args.algorithm,
            highway_types=args.ways,
            spacing=args.spacing,
            max_snap_distance=args.max_snap_distance,
            cell_size=args.cell_size,
            inflate_obstacles=args.inflate_obstacles,
            simplify_path=not args.no_simplify,
            smooth_path=args.smooth,
        )
    except RoutePlanningError as e:
        logger.error("Planning failed (%s): %s", e.reason, e.message)
        if args.json:
            print(json.dumps({"success": False, "reason": e.reason, "message": e.message}))
        return 1

    if args.save:
        text = (
            create_gpx_content(route_to_dicts(result))
            if args.wpt
            else create_gpx_track(result.latlon, name=Path(args.save).stem)
        )
        Path(args.save).write_text(text)
        logger.info("Saved %d waypoints to %s", len(result.latlon), args.save)

    if args.json:
        print(
            json.dumps(
                {
                    "success": True,
                    "algorithm": result.algorithm,
                    "length_m": round(result.length_m, 1),
                    "waypoints": len(result.latlon),
                    "snap_distances": [round(d, 1) for d in result.snap_distances],
                    "route": [[lat, lon] for lat, lon in result.latlon],
                }
            )
        )
    else:
        snap = ", ".join(f"{d:.1f}" for d in result.snap_distances)
        logger.info(
            "Route: %.0f m, %d waypoints (%s)%s",
            result.length_m,
            len(result.latlon),
            result.algorithm,
            f", snap distances {snap} m" if snap else "",
        )
        if not args.save:
            for lat, lon in result.latlon:
                print(f"{lat:.7f},{lon:.7f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
