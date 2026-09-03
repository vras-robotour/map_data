"""
Plan a route through lat/lon waypoints on a :class:`~map_data.map_data.MapData`.

This is the headless core of the viewer's *Planner* screen
(``POST /api/create_replan``): the same graph ("Paths only") and grid
("All terrain") planners, the same parameters, but callable from a script,
the ``map_data_plan`` CLI or the ``route_planner`` ROS action server. The
viewer route delegates to :func:`plan_route` so both stay in sync.

Failures are reported through :class:`RoutePlanningError` with a machine
readable ``reason`` instead of a bare ``None``:

``too_few_points``
    fewer than two waypoints (after adding the robot position);
``snap_too_far``
    a waypoint is farther than ``max_snap_distance`` from every allowed way
    (graph planner);
``unreachable``
    the waypoints lie on disconnected parts of the network (graph planner);
``no_path``
    the grid planner found no path;
``cancelled``
    the grid planner was cancelled through its ``transfer_id``;
``grid_too_large``
    the requested grid would exceed ``max_grid_cells`` cells.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import utm
from shapely import geometry

from map_data.pathsolver.graph_planner import DEFAULT_MAX_SNAP_DISTANCE, GraphPlanner
from map_data.pathsolver.replan import ReplanPath, parse_args
from map_data.utils.parsing import ways_to_shapely

if TYPE_CHECKING:
    from map_data.map_data import MapData

logger = logging.getLogger(__name__)

#: Grid planners refuse requests needing more cells than this (same budget as the viewer).
MAX_GRID_CELLS = 4_000_000
#: Margin (m) added around the waypoints' bounding box for the grid planners.
GRID_MARGIN_M = 50.0
#: Two paths are "the same" when no vertex moved more than this (m).
SIGNIFICANT_CHANGE_TOLERANCE = 0.1

GRAPH_ALGORITHM = "graph"
GRID_ALGORITHMS = ("astar", "rrt")


class RoutePlanningError(Exception):
    """Planning failed; ``reason`` is one of the module-level reason strings."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass
class RouteResult:
    """A planned route and how it relates to the requested waypoints."""

    #: Route vertices as ``(lat, lon)`` tuples, in order.
    latlon: list[tuple[float, float]]
    #: The same vertices in UTM ``(easting, northing)``.
    utm: np.ndarray
    zone_number: int
    zone_letter: str
    #: Route length in metres.
    length_m: float
    #: Distance (m) from every requested waypoint to the nearest allowed way
    #: (graph planner only; empty for the grid planners).
    snap_distances: list[float] = field(default_factory=list)
    #: ``True`` when the route differs from the polyline through the requested
    #: waypoints by more than :data:`SIGNIFICANT_CHANGE_TOLERANCE` anywhere.
    changed: bool = True
    algorithm: str = GRAPH_ALGORITHM


def path_length(path_utm: np.ndarray) -> float:
    """Polyline length (m) of an ``(N, 2+)`` UTM array."""
    if len(path_utm) < 2:
        return 0.0
    seg = np.diff(np.asarray(path_utm, dtype=float)[:, :2], axis=0)
    return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))


def densify(path_utm: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline so that consecutive vertices are at most ``spacing``
    metres apart. Original vertices are kept; each segment longer than
    ``spacing`` is split into equal parts. A non-positive ``spacing`` returns
    the input unchanged.
    """
    pts = np.asarray(path_utm, dtype=float)
    if spacing <= 0.0 or len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        d = float(np.hypot(*(b[:2] - a[:2])))
        n = max(1, int(np.ceil(d / spacing)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out)


def latlon_to_utm_path(
    points: Sequence[Sequence[float]],
    zone_number: int,
    zone_letter: str,
) -> np.ndarray:
    """``[(lat, lon), ...]`` -> ``(N, 2)`` UTM array forced into the map's zone."""
    out = []
    for lat, lon in points:
        e, n, _, _ = utm.from_latlon(float(lat), float(lon), zone_number, zone_letter)
        out.append([e, n])
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def utm_path_to_latlon_pairs(
    path_utm: np.ndarray,
    zone_number: int,
    zone_letter: str,
) -> list[tuple[float, float]]:
    return [
        tuple(utm.to_latlon(float(p[0]), float(p[1]), zone_number, zone_letter)) for p in path_utm
    ]


def _grid_bbox(
    md: MapData, utm_path: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Planning box: the waypoints' bbox plus :data:`GRID_MARGIN_M`, clipped to the map."""
    p_low = (
        max(md.min_x, float(np.min(utm_path[:, 0])) - GRID_MARGIN_M),
        max(md.min_y, float(np.min(utm_path[:, 1])) - GRID_MARGIN_M),
    )
    p_high = (
        min(md.max_x, float(np.max(utm_path[:, 0])) + GRID_MARGIN_M),
        min(md.max_y, float(np.max(utm_path[:, 1])) + GRID_MARGIN_M),
    )
    return p_low, p_high


def grid_cell_count(area_m2: float, cell_size: float) -> float:
    return area_m2 / (cell_size * cell_size)


def plan_route(
    md: MapData,
    points_latlon: Sequence[Sequence[float]],
    *,
    algorithm: str = GRAPH_ALGORITHM,
    sub_algorithm: str = "astar",
    highway_types: Sequence[str] | None = None,
    max_snap_distance: float = DEFAULT_MAX_SNAP_DISTANCE,
    spacing: float = 0.0,
    cell_size: float = 0.25,
    inflate_obstacles: float = 0.25,
    simplify_path: bool = True,
    smooth_path: bool = False,
    grid_cost_weight: float | None = None,
    highway_costs: dict[str, float] | None = None,
    surface_costs: dict[str, float] | None = None,
    transfer_id: str | None = None,
    max_grid_cells: float = MAX_GRID_CELLS,
) -> RouteResult:
    """
    Plan a route through ``points_latlon`` (``[(lat, lon), ...]``, at least two).

    ``algorithm="graph"`` routes along the allowed ways only (the viewer's
    *Paths only*); ``"astar"``/``"rrt"`` run the grid planner over the cost
    grid built from the map (*All terrain*; ``sub_algorithm`` is derived from
    ``algorithm`` unless given explicitly as the viewer does). Any other value
    also selects the grid planner with ``sub_algorithm``.

    ``spacing`` > 0 resamples the result so waypoints are at most that far
    apart (a waypoint follower wants a few metres). ``changed`` and
    ``snap_distances`` are computed before resampling.

    Raises :class:`RoutePlanningError` on failure.
    """
    highway_types = list(highway_types) if highway_types else ["footway"]
    if len(points_latlon) < 2:
        raise RoutePlanningError("too_few_points", "plan_route() needs at least two waypoints")

    zn, zl = md.zone_number, md.zone_letter
    utm_path = latlon_to_utm_path(points_latlon, zn, zl)

    if algorithm == GRAPH_ALGORITHM:
        planner = GraphPlanner(md, highway_types=highway_types, max_snap_distance=max_snap_distance)
        snap = [planner.snap_distance(p) for p in utm_path]
        too_far = [i for i, d in enumerate(snap) if d > max_snap_distance]
        if too_far:
            i = too_far[0]
            raise RoutePlanningError(
                "snap_too_far",
                f"waypoint {i} is {snap[i]:.1f} m from the nearest "
                f"{'/'.join(highway_types)} (limit {max_snap_distance:.0f} m)",
            )
        res = planner.plan(utm_path)
        if res is None:
            raise RoutePlanningError(
                "unreachable",
                f"no {'/'.join(highway_types)} route connects the waypoints",
            )
    else:
        if algorithm in GRID_ALGORITHMS and sub_algorithm == "astar":
            sub_algorithm = algorithm
        p_low, p_high = _grid_bbox(md, utm_path)
        area_m2 = max(0.0, p_high[0] - p_low[0]) * max(0.0, p_high[1] - p_low[1])
        cells = grid_cell_count(area_m2, cell_size)
        if cells > max_grid_cells:
            raise RoutePlanningError(
                "grid_too_large",
                f"Requested area needs ~{cells / 1e6:.1f} million grid cells at "
                f"cell_size={cell_size} m, exceeding the {max_grid_cells / 1e6:.0f} million "
                "cell limit. Request a smaller area or a larger cell size.",
            )
        args = parse_args([])
        args.simplify_path = simplify_path
        args.smooth_path = smooth_path
        args.cell_size = cell_size
        args.inflate_obstacles = inflate_obstacles
        args.visualize = False
        args.low = p_low
        args.high = p_high

        bbox = geometry.box(p_low[0], p_low[1], p_high[0], p_high[1])
        filtered_barriers = [w for w in md.barriers_list if w.line and w.line.intersects(bbox)]
        replanner = ReplanPath(
            args,
            ways_to_shapely(filtered_barriers),
            transfer_id=transfer_id,
            grid_cost_weight=grid_cost_weight,
            highway_costs=highway_costs,
            surface_costs=surface_costs,
        )
        replanner.fill_grid(md, highway_types=highway_types)
        res = replanner.replan(utm_path, algorithm=sub_algorithm)
        snap = []
        if res is None:
            # ReplanPath returns None both when no path exists and when the run was
            # cancelled through its transfer_id; it does not tell the two apart.
            raise RoutePlanningError("no_path", "the grid planner found no path (or was cancelled)")

    res = np.asarray(res, dtype=float)[:, :2]
    changed = len(res) != len(utm_path) or bool(
        np.any(
            np.linalg.norm(res[: len(utm_path)] - utm_path[: len(res)], axis=1)
            > SIGNIFICANT_CHANGE_TOLERANCE
        )
    )
    if spacing > 0.0:
        res = densify(res, spacing)

    return RouteResult(
        latlon=utm_path_to_latlon_pairs(res, zn, zl),
        utm=res,
        zone_number=zn,
        zone_letter=zl,
        length_m=path_length(res),
        snap_distances=[float(d) for d in snap],
        changed=changed,
        algorithm=algorithm,
    )


def route_to_dicts(result: RouteResult) -> list[dict[str, Any]]:
    """``[{"latitude", "longitude", "elevation"}, ...]`` for the GPX writers."""
    return [{"latitude": lat, "longitude": lon, "elevation": 0.0} for lat, lon in result.latlon]
