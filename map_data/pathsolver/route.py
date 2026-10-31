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
``start_blocked`` / ``goal_blocked``
    a grid leg starts or ends inside an obstacle, with no free cell within
    :data:`~map_data.pathsolver.replan.SNAP_RADIUS_M`;
``cancelled``
    the grid planner was cancelled through its ``transfer_id``;
``start_outside_map`` / ``goal_outside_map``
    the start or goal lies too far outside the map's area (wrong map file?);
``grid_too_large``
    the requested grid would exceed ``max_grid_cells`` cells;
``bad_parameter``
    an unknown ``algorithm`` / ``sub_algorithm``, or ``cell_size`` not above zero.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely as sh
import utm
from shapely import geometry

from map_data.pathsolver.graph_planner import DEFAULT_MAX_SNAP_DISTANCE, GraphPlanner
from map_data.pathsolver.replan import SNAP_RADIUS_M, ReplanPath, grid_defaults
from map_data.traversability import TraversabilityRules
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.way import NON_ROUTABLE_HIGHWAY_VALUES

if TYPE_CHECKING:
    from map_data.map_data import MapData

logger = logging.getLogger(__name__)

#: Grid planners and the viewer's cost grid refuse requests needing more cells than this:
#: a 2 x 2 km box at the overlay's 1 m, or 500 x 500 m at the 0.25 m default cell size.
MAX_GRID_CELLS = 4_000_000
#: Margin (m) added around the waypoints' bounding box for the grid planners.
GRID_MARGIN_M = 50.0
#: Two paths are "the same" when no vertex moved more than this (m).
SIGNIFICANT_CHANGE_TOLERANCE = 0.1

_GRID_FAILURES = {
    "no_path": "the grid planner found no path",
    "start_blocked": f"a start lies in an obstacle, no free cell within {SNAP_RADIUS_M:g} m",
    "goal_blocked": f"a goal lies in an obstacle, no free cell within {SNAP_RADIUS_M:g} m",
    "cancelled": "the grid planner was cancelled",
}

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
    pts = np.asarray(path_utm, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(geometry.LineString(pts[:, :2]).length)


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
    return np.array(sh.segmentize(geometry.LineString(pts[:, :2]), spacing).coords)


def latlon_to_utm_path(
    points: Sequence[Sequence[float]],
    zone_number: int,
    zone_letter: str,
) -> np.ndarray:
    """``[(lat, lon), ...]`` -> ``(N, 2)`` UTM array forced into the map's zone."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    e, n, _, _ = utm.from_latlon(pts[:, 0], pts[:, 1], zone_number, zone_letter)
    return np.stack([e, n], axis=-1).reshape(-1, 2)


def utm_path_to_latlon_pairs(
    path_utm: np.ndarray,
    zone_number: int,
    zone_letter: str,
) -> list[tuple[float, float]]:
    pts = np.asarray(path_utm, dtype=np.float64).reshape(-1, 2)
    lat, lon = utm.to_latlon(pts[:, 0], pts[:, 1], zone_number, zone_letter)
    return list(zip(lat.tolist(), lon.tolist()))


def _grid_bbox(
    md: MapData, utm_path: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Planning box: the waypoints' bbox plus :data:`GRID_MARGIN_M`, clipped to the map.

    A side where a waypoint lies on or past the map edge (:func:`check_in_map` lets start
    and goal be up to ``OUTSIDE_MAP_TOLERANCE_M`` off) keeps its full margin instead, so
    the grid always contains every waypoint (its high edge is exclusive).
    """
    lo = utm_path[:, :2].min(axis=0)
    hi = utm_path[:, :2].max(axis=0)
    map_lo = np.array([md.min_x, md.min_y])
    map_hi = np.array([md.max_x, md.max_y])
    p_low = np.where(lo <= map_lo, lo - GRID_MARGIN_M, np.maximum(map_lo, lo - GRID_MARGIN_M))
    p_high = np.where(hi >= map_hi, hi + GRID_MARGIN_M, np.minimum(map_hi, hi + GRID_MARGIN_M))
    return (float(p_low[0]), float(p_low[1])), (float(p_high[0]), float(p_high[1]))


#: How far (m) past the map's area a start or goal may lie: the robot can start just outside
#: the downloaded box (Stromovka 2026-09-08: 33 m), a wrong map file puts it kilometres off.
OUTSIDE_MAP_TOLERANCE_M = 100.0


def check_in_map(
    md: MapData,
    point_utm: Sequence[float],
    what: str,
    tolerance: float = OUTSIDE_MAP_TOLERANCE_M,
) -> None:
    """
    Raise ``<what>_outside_map`` (``start`` / ``goal``) when the point lies more than
    ``tolerance`` metres outside the map's area (the downloaded box, ``min_x..max_x`` /
    ``min_y..max_y``). A start that far off almost always means the wrong map file is
    loaded; a goal, the wrong file or a code meant for another map.
    """
    x, y = float(point_utm[0]), float(point_utm[1])
    dx = max(md.min_x - x, 0.0, x - md.max_x)
    dy = max(md.min_y - y, 0.0, y - md.max_y)
    distance = float(np.hypot(dx, dy))
    if distance <= tolerance:
        return
    raise RoutePlanningError(
        f"{what}_outside_map",
        f"the {what} is {distance:.0f} m outside the map area "
        f"(limit {tolerance:.0f} m; wrong map file?)",
    )


def goal_plan_kwargs(goal: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """
    :func:`plan_route` keyword arguments for a ``PlanRoute`` action goal.

    An empty or zero ``algorithm``, ``highway_types``, ``spacing``, ``max_snap_distance``,
    ``cell_size`` or ``inflate_obstacles`` takes the value of that name in *defaults* (the
    ``route_planner`` node's parameters); a negative ``spacing`` keeps the planner's own
    vertices. ``simplify_path`` and ``smooth_path`` are used as sent: a bool has no unset.
    """
    return {
        "algorithm": goal.algorithm or defaults["algorithm"],
        "highway_types": list(goal.highway_types) or list(defaults["highway_types"]),
        "spacing": defaults["spacing"] if goal.spacing == 0.0 else max(goal.spacing, 0.0),
        "max_snap_distance": goal.max_snap_distance or defaults["max_snap_distance"],
        "cell_size": goal.cell_size or defaults["cell_size"],
        "inflate_obstacles": goal.inflate_obstacles or defaults["inflate_obstacles"],
        "simplify_path": bool(goal.simplify_path),
        "smooth_path": bool(goal.smooth_path),
    }


def plan_route(
    md: MapData,
    points_latlon: Sequence[Sequence[float]],
    *,
    algorithm: str = GRAPH_ALGORITHM,
    sub_algorithm: str | None = None,
    highway_types: Sequence[str] | None = None,
    max_snap_distance: float = DEFAULT_MAX_SNAP_DISTANCE,
    spacing: float = 0.0,
    cell_size: float | None = None,
    inflate_obstacles: float | None = None,
    simplify_path: bool | None = None,
    smooth_path: bool | None = None,
    grid_cost_weight: float | None = None,
    highway_costs: dict[str, float] | None = None,
    surface_costs: dict[str, float] | None = None,
    transfer_id: str | None = None,
    max_grid_cells: float = MAX_GRID_CELLS,
    planner: GraphPlanner | None = None,
    keep_start: bool = False,
    keep_goal: bool = False,
    exclude_highway: Iterable[str] = NON_ROUTABLE_HIGHWAY_VALUES,
    traversability: TraversabilityRules | str | Path | None = None,
    outside_map_tolerance: float = OUTSIDE_MAP_TOLERANCE_M,
) -> RouteResult:
    """
    Plan a route through ``points_latlon`` (``[(lat, lon), ...]``, at least two).

    ``algorithm="graph"`` routes along the allowed ways only (the viewer's
    *Paths only*); ``"astar"``/``"rrt"`` run the grid planner over the cost
    grid built from the map (*All terrain*). ``sub_algorithm`` picks the grid
    search (``"astar"``/``"rrt"``); ``None`` derives it from ``algorithm``, so
    ``algorithm="rrt"`` alone runs RRT*, while the viewer sends ``"rrt"`` plus an
    explicit ``sub_algorithm``. Any other value fails with ``bad_parameter``.

    ``spacing`` > 0 resamples the result so waypoints are at most that far
    apart (a waypoint follower wants a few metres). ``changed`` and
    ``snap_distances`` are computed before resampling.

    ``planner`` lets a caller reuse a prebuilt :class:`GraphPlanner` for ``md``
    (building the graph costs ~0.3 s, planning on it ~1 ms); it must have been
    built for the same ``highway_types`` and ``max_snap_distance``.

    ``keep_start`` makes the graph route begin at the first waypoint verbatim
    rather than at its projection onto the network — for planning from the
    robot's own position, which is off the network by definition. It has no
    effect on the grid planners, which already start at the requested point.

    ``keep_goal`` does the same at the other end: the route ends at the last
    requested coordinate instead of at its projection, so a goal off the mapped
    network is actually reached. The off-network leg is resampled by ``spacing``
    like the rest of the route.

    ``exclude_highway`` is the ``highway`` tag values the graph planner never
    routes over (stairs by default), and ``traversability`` the tag rules that
    decide the same question from the map's own tags (``None`` = the package's
    ``config/traversability.yaml``); both are ignored when ``planner`` is
    given, which brings its own.

    ``highway_costs``/``surface_costs`` price the ``highway``/``surface`` tags
    for *both* planners: the grid planner's cells and the graph planner's edge
    weights (``length * (1 + cost)``). ``None`` takes
    ``config/planner_defaults.yaml``, as does ``None`` for ``cell_size``,
    ``inflate_obstacles``, ``simplify_path``, ``smooth_path`` and ``grid_cost_weight``
    (see :func:`~map_data.pathsolver.replan.grid_defaults`).

    A first or last point more than ``outside_map_tolerance`` metres outside the map's
    area fails with ``start_outside_map`` / ``goal_outside_map`` (see :func:`check_in_map`).

    Raises :class:`RoutePlanningError` on failure.
    """
    highway_types = list(highway_types) if highway_types else ["footway"]
    if cell_size is None:
        cell_size = float(grid_defaults()["cell_size"])
    if algorithm not in (GRAPH_ALGORITHM, *GRID_ALGORITHMS):
        raise RoutePlanningError("bad_parameter", f"unknown algorithm {algorithm!r}")
    if sub_algorithm is not None and sub_algorithm not in GRID_ALGORITHMS:
        raise RoutePlanningError("bad_parameter", f"unknown sub_algorithm {sub_algorithm!r}")
    if not cell_size > 0:  # also NaN
        raise RoutePlanningError("bad_parameter", f"cell_size must be > 0, got {cell_size}")
    if len(points_latlon) < 2:
        raise RoutePlanningError("too_few_points", "plan_route() needs at least two waypoints")

    zn, zl = md.zone_number, md.zone_letter
    utm_path = latlon_to_utm_path(points_latlon, zn, zl)
    check_in_map(md, utm_path[0], "start", outside_map_tolerance)
    check_in_map(md, utm_path[-1], "goal", outside_map_tolerance)

    if algorithm == GRAPH_ALGORITHM:
        if planner is None:
            planner = GraphPlanner(
                md,
                highway_types=highway_types,
                max_snap_distance=max_snap_distance,
                exclude_highway=exclude_highway,
                traversability=traversability,
                highway_costs=highway_costs,
                surface_costs=surface_costs,
            )
        snap = [planner.snap_distance(p) for p in utm_path]
        too_far = [i for i, d in enumerate(snap) if d > max_snap_distance]
        if too_far:
            i = too_far[0]
            raise RoutePlanningError(
                "snap_too_far",
                f"waypoint {i} is {snap[i]:.1f} m from the nearest "
                f"{'/'.join(highway_types)} (limit {max_snap_distance:.0f} m)",
            )
        res = planner.plan(utm_path, keep_start=keep_start, keep_goal=keep_goal)
        if res is None:
            raise RoutePlanningError(
                "unreachable",
                f"no {'/'.join(highway_types)} route connects the waypoints",
            )
    else:
        if sub_algorithm is None:
            sub_algorithm = algorithm
        p_low, p_high = _grid_bbox(md, utm_path)
        area_m2 = max(0.0, p_high[0] - p_low[0]) * max(0.0, p_high[1] - p_low[1])
        cells = area_m2 / (cell_size * cell_size)
        if cells > max_grid_cells:
            raise RoutePlanningError(
                "grid_too_large",
                f"Requested area needs ~{cells / 1e6:.1f} million grid cells at "
                f"cell_size={cell_size} m, exceeding the {max_grid_cells / 1e6:.0f} million "
                "cell limit. Request a smaller area or a larger cell size.",
            )

        bbox = geometry.box(p_low[0], p_low[1], p_high[0], p_high[1])
        filtered_barriers = [w for w in md.barriers_list if w.line and w.line.intersects(bbox)]
        replanner = ReplanPath(
            ways_to_shapely(filtered_barriers),
            transfer_id=transfer_id,
            grid_cost_weight=grid_cost_weight,
            highway_costs=highway_costs,
            surface_costs=surface_costs,
            low=p_low,
            high=p_high,
            cell_size=cell_size,
            inflate_obstacles=inflate_obstacles,
            simplify_path=simplify_path,
            smooth_path=smooth_path,
        )
        replanner.fill_grid_lazily(md, highway_types=highway_types)
        res = replanner.replan(utm_path, algorithm=sub_algorithm)
        snap = []
        if res is None:
            reason = replanner.failure_reason or "no_path"
            raise RoutePlanningError(reason, _GRID_FAILURES[reason])

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
