#!/usr/bin/env python3

import logging
import random
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely as sh
from scipy import ndimage
from shapely.geometry import LineString

from map_data.pathsolver.grid_astar import (
    GRID_COST_WEIGHT,
    grid_astar,
    nearest_free_cell,
    simplify_path_checked,
)
from map_data.pathsolver.rrt_star import RRTStar
from map_data.pathsolver.way_cost import load_cost_tables
from map_data.utils.config import load_config

from .grid_constructor import PathGrid

# Decoupled components
from .smoothing import smooth_path

if TYPE_CHECKING:
    from map_data.map_data import MapData

logger = logging.getLogger(__name__)


#: Grid-planner settings used where ``config/planner_defaults.yaml`` lacks them.
GRID_DEFAULTS: dict[str, Any] = {
    "cell_size": 0.25,
    "inflate_obstacles": 0.25,
    "simplify_path": True,
    "smooth_path": False,
    "max_path_dist": 2.0,
    "grid_cost_weight": GRID_COST_WEIGHT,
}


def grid_defaults() -> dict[str, Any]:
    """
    The grid planner's default settings: ``config/planner_defaults.yaml`` over
    :data:`GRID_DEFAULTS`, read on every call so file edits apply to the next plan.

    The single source for :class:`ReplanPath`, :func:`~map_data.pathsolver.route.plan_route`,
    ``map_data_plan`` and the ``route_planner`` node's parameter defaults.
    """
    cfg = load_config("planner_defaults.yaml")
    return {key: cfg.get(key, value) for key, value in GRID_DEFAULTS.items()}


#: How far (m) a start or goal on a blocked cell (a robot beside a fence, a goal
#: on a wall) is moved to the nearest free cell before the grid search.
SNAP_RADIUS_M = 2.0

#: Cancels kept for runs that never consume them (already finished, or never started).
MAX_PENDING_CANCELS = 256

_cancel_lock = threading.Lock()
_cancelled_transfers: OrderedDict[str, None] = OrderedDict()


def cancel_replan_backend(transfer_id: str | None) -> None:
    """
    Cancel the replan running, or about to run, under *transfer_id*.

    Transfer ids must be unique per run: a cancel that lands before its run starts
    (while the map loads) still aborts it, so a reused id would abort the next run.
    """
    if transfer_id:
        with _cancel_lock:
            _cancelled_transfers[transfer_id] = None
            while len(_cancelled_transfers) > MAX_PENDING_CANCELS:
                _cancelled_transfers.popitem(last=False)


def _is_cancelled(transfer_id: str | None) -> bool:
    if not transfer_id:
        return False
    with _cancel_lock:
        return transfer_id in _cancelled_transfers


def _discard_cancelled(transfer_id: str | None) -> None:
    if transfer_id:
        with _cancel_lock:
            _cancelled_transfers.pop(transfer_id, None)


class ReplanPath:
    def __init__(
        self,
        obstacles: list[sh.geometry.base.BaseGeometry] | None = None,
        transfer_id: str | None = None,
        grid_cost_weight: float | None = None,
        highway_costs: dict[str, float] | None = None,
        surface_costs: dict[str, float] | None = None,
        rng: random.Random | None = None,
        *,
        low: tuple[float, float] = (0.0, 0.0),
        high: tuple[float, float] = (0.0, 0.0),
        cell_size: float | None = None,
        inflate_obstacles: float | None = None,
        simplify_path: bool | None = None,
        smooth_path: bool | None = None,
    ) -> None:
        # None = the grid_defaults() value.
        grid = grid_defaults()
        cell_size = grid["cell_size"] if cell_size is None else cell_size
        if inflate_obstacles is None:
            inflate_obstacles = grid["inflate_obstacles"]
        self.low = low
        self.cell_size = cell_size
        self.simplify_path = grid["simplify_path"] if simplify_path is None else simplify_path
        self.smooth_path = grid["smooth_path"] if smooth_path is None else smooth_path
        self.max_path_dist = float(grid["max_path_dist"])
        self.transfer_id = transfer_id
        defaults = load_config("planner_defaults.yaml")
        self.grid_cost_weight = (
            grid_cost_weight if grid_cost_weight is not None else grid["grid_cost_weight"]
        )

        # The highway/surface cost tables and their cap are the one place that
        # already knows how to fall back to config/planner_defaults.yaml
        # (map_data.pathsolver.way_cost, shared with the graph planner) — don't
        # re-derive them here.
        self.HIGHWAY_COSTS, self.SURFACE_COSTS, self.PATH_COST_CAP = load_cost_tables(
            highway_costs,
            surface_costs,
        )
        self.DEFAULT_OFF_PATH_COST = float(defaults.get("default_off_path_cost", 0.9))

        # RRT* settings: informed sampling and post-goal improvement are on by
        # default in production (see RRTStar.find_path), capped by improve_iter.
        rrt_defaults = defaults.get("rrt", {})
        self.rrt_informed = bool(rrt_defaults.get("informed", True))
        self.rrt_improve_after_goal = bool(rrt_defaults.get("improve_after_goal", True))
        self.rrt_improve_iter = int(rrt_defaults.get("improve_iter", 200))
        self.rrt_adaptive_radius = bool(rrt_defaults.get("adaptive_radius", True))
        # Random source handed to every RRTStar this replanner builds. Callers
        # that need a reproducible run pass their own; otherwise an `rrt.seed`
        # in planner_defaults.yaml makes runs repeatable without code changes
        # (e.g. to reproduce a planning failure reported from the viewer), and
        # with neither each planner keeps its own unseeded stream.
        rrt_seed = rrt_defaults.get("seed")
        if rng is None and rrt_seed is not None:
            rng = random.Random(rrt_seed)
        self.rrt_rng = rng

        # Use the decoupled PathGrid component
        self.path_grid = PathGrid(
            low=low,
            high=high,
            cell_size=cell_size,
            highway_costs=self.HIGHWAY_COSTS,
            surface_costs=self.SURFACE_COSTS,
            default_off_path_cost=self.DEFAULT_OFF_PATH_COST,
            path_cost_cap=self.PATH_COST_CAP,
        )

        if inflate_obstacles:
            self.obstacles = (
                [obstacle.buffer(inflate_obstacles) for obstacle in obstacles] if obstacles else []
            )
        else:
            self.obstacles = obstacles or []

        self._deferred_fill: tuple[MapData, list[str] | None, float | None] | None = None

        # Spatial index for faster collision checking
        self.obstacles_tree = sh.STRtree(self.obstacles) if self.obstacles else None
        # (grid, its 4-connected free regions), see _free_ends
        self._regions: tuple[np.ndarray, np.ndarray] | None = None
        #: Why the last replan() returned None: "no_path", "start_blocked",
        #: "goal_blocked" or "cancelled".
        self.failure_reason: str | None = None

    def replan(self, path: np.ndarray, algorithm: str = "astar") -> np.ndarray | None:
        # A cancel registered before this run started still aborts it; the id is
        # consumed on exit either way.
        self.failure_reason = None
        try:
            # This is pure-Python, GIL-bound work, so it runs sequentially.
            new_path: list[np.ndarray] = []
            for i in range(len(path) - 1):
                if self._cancelled():
                    return None

                start = path[i][:2]
                goal = path[i + 1][:2]
                segment_path = [start]
                path_seg = LineString([start, goal])
                if self._colides(path_seg):
                    ends = self._free_ends(start, goal)
                    if ends is None:
                        logger.warning("leg %d: %s", i, self.failure_reason)
                        return None
                    if algorithm == "rrt":
                        way = self._rrt_star(*ends)
                        # RRT* gives up after a fixed iteration budget; A* on the
                        # same grid is complete. Not after a cancel, though.
                        if way is None and not self._cancelled():
                            logger.info("rrt found no path for leg %d; falling back to astar.", i)
                            way = self._astar(*ends)
                    else:
                        way = self._astar(*ends)

                    if way is None:
                        if not self._cancelled():
                            self.failure_reason = "no_path"
                        logger.warning("%s failed to find a path.", algorithm)
                        return None
                    # The search runs between cell centres (or snapped ends): keep its
                    # first/last vertex where the hop from the requested point would
                    # otherwise cross an obstacle.
                    mid = list(way[1:-1])
                    if self._colides(LineString([start, mid[0] if mid else way[-1]])):
                        mid.insert(0, way[0])
                    if self._colides(LineString([mid[-1] if mid else start, goal])):
                        mid.append(way[-1])
                    segment_path.extend(mid)

                if self._cancelled():
                    return None
                new_path.extend(segment_path)

            new_path.append(path[-1][:2])
            return self._post_process_path(np.array(new_path))
        finally:
            _discard_cancelled(self.transfer_id)

    def _cancelled(self) -> bool:
        if _is_cancelled(self.transfer_id):
            self.failure_reason = "cancelled"
            return True
        return False

    def _free_ends(
        self, start: np.ndarray, goal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        *start* and *goal* moved off blocked cells (within :data:`SNAP_RADIUS_M`), or
        ``None`` with :attr:`failure_reason` set when either stays blocked or they lie in
        disconnected free regions, which would otherwise flood the whole grid to fail.
        """
        grid = self._grid()
        s = nearest_free_cell(grid, start, self.low, self.cell_size, SNAP_RADIUS_M)
        g = nearest_free_cell(grid, goal, self.low, self.cell_size, SNAP_RADIUS_M)
        if s is None or g is None:
            self.failure_reason = "start_blocked" if s is None else "goal_blocked"
            return None
        if self._regions is None or self._regions[0] is not grid:
            # 4-connected: A* only moves diagonally between two free edge neighbours.
            self._regions = (grid, ndimage.label(np.isfinite(grid))[0])
        regions = self._regions[1]

        def region(p: np.ndarray) -> int:
            ix, iy = np.floor((p - self.low) / self.cell_size).astype(int)  # as nearest_free_cell
            return int(regions[iy, ix])

        if region(s) != region(g):
            self.failure_reason = "no_path"
            return None
        return s, g

    def _post_process_path(self, path: np.ndarray | None) -> np.ndarray | None:
        """
        Simplify and optionally smooth the final path.
        """
        if path is None or len(path) <= 2:
            return path

        # 1. Remove points that are extremely close to each other
        dist_sq = np.sum(np.diff(path, axis=0) ** 2, axis=1)
        mask = np.ones(len(path), dtype=bool)
        mask[1:] = dist_sq > 0.05**2
        path = path[mask]

        if len(path) <= 2:
            return path

        # 2. Smooth path if requested
        if self.smooth_path:
            path = smooth_path(path, collision_check_func=self._colides)

        # 3. Final Douglas-Peucker simplification on the whole path. This is
        #    the *only* simplification pass (grid_astar and RRTStar hand back
        #    their raw paths): running it once here, over the whole assembled
        #    route, does what per-segment simplification did twice already.
        #    Every shortcut the simplification introduces is collision-checked
        #    against the obstacle polygons; colliding shortcuts keep their
        #    original vertices so the path cannot chord into an obstacle.
        if self.simplify_path:
            path = simplify_path_checked(
                path,
                self.cell_size,
                lambda p1, p2: self._colides(LineString([p1, p2])),
            )

        return path

    def _grid(self) -> np.ndarray:
        if self.path_grid.grid_2d_cache is None and self._deferred_fill is not None:
            map_data, highway_types, max_path_dist = self._deferred_fill
            self._deferred_fill = None
            self.fill_grid(map_data, highway_types, max_path_dist)
        grid = self.path_grid.grid_2d_cache
        assert grid is not None, "fill_grid() must run before replan()"
        return grid

    def _rrt_star(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        grid = self._grid()
        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles_tree=self.obstacles_tree,
            grid=grid,
            low=self.low,
            grid_scale=self.cell_size,
            grid_cost_weight=self.grid_cost_weight,
            transfer_id=self.transfer_id,
            informed=self.rrt_informed,
            improve_after_goal=self.rrt_improve_after_goal,
            improve_iter=self.rrt_improve_iter,
            adaptive_radius=self.rrt_adaptive_radius,
            rng=self.rrt_rng,
        )
        return planner.find_path()

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        grid = self._grid()
        return grid_astar(
            grid,
            start,
            goal,
            self.low,
            self.cell_size,
            # The final pass in _post_process_path is the only simplification;
            # doing it again per segment here would just redo that work.
            simplify_path=False,
            grid_cost_weight=self.grid_cost_weight,
        )

    def _colides(self, path_seg: LineString) -> bool:
        if self.obstacles_tree is None:
            return False
        intersecting_indices = self.obstacles_tree.query(path_seg, predicate="intersects")
        return len(intersecting_indices) > 0

    def fill_grid(
        self,
        map_data: "MapData",
        highway_types: list[str] | None = None,
        max_path_dist: float | None = None,
    ) -> None:
        """
        Populate the grid with costs based on map data.

        ``max_path_dist`` ``None`` takes ``max_path_dist`` from ``planner_defaults.yaml``.
        """
        effective = self.path_grid.fill(
            map_data,
            self.obstacles,
            highway_types=highway_types,
            max_path_dist=self.max_path_dist if max_path_dist is None else max_path_dist,
        )
        # The grid lets a path through a barrier it crosses; so must the segment checks.
        self.obstacles_tree = sh.STRtree(effective) if effective else None

    def fill_grid_lazily(
        self,
        map_data: "MapData",
        highway_types: list[str] | None = None,
        max_path_dist: float | None = None,
    ) -> None:
        """
        :meth:`fill_grid`, deferred until the first leg that hits an obstacle needs the
        grid: legs that don't are checked against the obstacle geometries alone, so an
        unobstructed route never pays for building it (~1 s on a large map).
        """
        self._deferred_fill = (map_data, highway_types, max_path_dist)
