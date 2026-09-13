#!/usr/bin/env python3

import argparse
import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely as sh
from shapely.geometry import LineString

from map_data.pathsolver.grid_astar import GRID_COST_WEIGHT, grid_astar, simplify_path_checked
from map_data.pathsolver.rrt_star import RRTStar
from map_data.pathsolver.way_cost import load_cost_tables
from map_data.utils.config import load_config

from .grid_constructor import PathGrid

# Decoupled components
from .smoothing import smooth_path

if TYPE_CHECKING:
    from map_data.map_data import MapData

logger = logging.getLogger(__name__)


_cancel_lock = threading.Lock()
_cancelled_transfers: set = set()


def cancel_replan_backend(transfer_id: str | None) -> None:
    if transfer_id:
        with _cancel_lock:
            _cancelled_transfers.add(transfer_id)


def _is_cancelled(transfer_id: str | None) -> bool:
    if not transfer_id:
        return False
    with _cancel_lock:
        return transfer_id in _cancelled_transfers


def _discard_cancelled(transfer_id: str | None) -> None:
    if transfer_id:
        with _cancel_lock:
            _cancelled_transfers.discard(transfer_id)


def load_planner_defaults() -> dict[str, Any]:
    """
    Load default planner configuration from config/planner_defaults.yaml.
    """
    return load_config("planner_defaults.yaml")


class ReplanPath:
    def __init__(
        self,
        args: argparse.Namespace,
        obstacles: list[sh.geometry.base.BaseGeometry] | None = None,
        transfer_id: str | None = None,
        grid_cost_weight: float | None = None,
        highway_costs: dict[str, float] | None = None,
        surface_costs: dict[str, float] | None = None,
    ) -> None:
        self.args = args
        self.transfer_id = transfer_id
        defaults = load_planner_defaults()
        self.grid_cost_weight = (
            grid_cost_weight if grid_cost_weight is not None else GRID_COST_WEIGHT
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

        # Use the decoupled PathGrid component
        self.path_grid = PathGrid(
            low=args.low,
            high=args.high,
            cell_size=args.cell_size,
            highway_costs=self.HIGHWAY_COSTS,
            surface_costs=self.SURFACE_COSTS,
            default_off_path_cost=self.DEFAULT_OFF_PATH_COST,
            path_cost_cap=self.PATH_COST_CAP,
        )

        if args.inflate_obstacles:
            self.obstacles = (
                [obstacle.buffer(args.inflate_obstacles) for obstacle in obstacles]
                if obstacles
                else []
            )
        else:
            self.obstacles = obstacles or []

        # Spatial index for faster collision checking
        self.obstacles_tree = sh.STRtree(self.obstacles) if self.obstacles else None

    @property
    def grid(self) -> np.ndarray:
        """
        Compatibility property for old access to the raw point grid.
        """
        return self.path_grid.grid

    @grid.setter
    def grid(self, value: np.ndarray) -> None:
        self.path_grid.grid = value

    def _ensure_grid_2d_cache(self) -> np.ndarray:
        """
        Build (once) and return the obstacle-burned 2D cost grid.

        ``replan()``, ``_astar()`` and ``_rrt_star()`` all need this and none
        of them should redundantly rebuild it, so they all call this instead.
        """
        if self.path_grid.grid_2d_cache is None:
            grid_2d = self.path_grid.get_grid_2d()
            self.path_grid.grid_2d_cache = self.path_grid.burn_obstacles(
                grid_2d,
                self.obstacles,
            )
        return self.path_grid.grid_2d_cache

    def replan(self, path: np.ndarray, algorithm: str = "astar") -> np.ndarray | None:
        # A cancel targeting a *previous* replan with the same transfer_id may
        # arrive after that run already returned; discard any such stale ID on
        # entry so it cannot instantly abort this run, and again on exit (via
        # finally) so a cancel arriving after we return cannot poison the next
        # run. Cancels registered while this run is in progress still abort it.
        _discard_cancelled(self.transfer_id)
        try:
            # This is pure-Python, GIL-bound work, so it runs sequentially.
            # Warm the grid cache once up front to avoid every segment lazily
            # (and redundantly) rebuilding it in _astar/_rrt_star.
            self._ensure_grid_2d_cache()

            new_path: list[np.ndarray] = []
            for i in range(len(path) - 1):
                if _is_cancelled(self.transfer_id):
                    return None

                start = path[i]
                goal = path[i + 1]
                segment_path = [start[:2]]
                path_seg = LineString([start[:2], goal[:2]])
                if self._colides(path_seg):
                    if algorithm == "rrt":
                        way = self._rrt_star(start[:2], goal[:2])
                    else:
                        way = self._astar(start[:2], goal[:2])

                    if way is None:
                        logger.warning("%s failed to find a path.", algorithm)
                        return None
                    segment_path.extend(way[1:-1])

                if _is_cancelled(self.transfer_id):
                    return None
                new_path.extend(segment_path)

            new_path.append(path[-1][:2])
            return self._post_process_path(np.array(new_path))
        finally:
            _discard_cancelled(self.transfer_id)

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
        if getattr(self.args, "smooth_path", False):
            path = smooth_path(path, collision_check_func=self._colides)

        # 3. Final Douglas-Peucker simplification on the whole path. This is
        #    the *only* simplification pass (grid_astar and RRTStar hand back
        #    their raw paths): running it once here, over the whole assembled
        #    route, does what per-segment simplification did twice already.
        #    Every shortcut the simplification introduces is collision-checked
        #    against the obstacle polygons; colliding shortcuts keep their
        #    original vertices so the path cannot chord into an obstacle.
        if self.args.simplify_path:
            path = simplify_path_checked(
                path,
                self.args.cell_size,
                lambda p1, p2: self._colides(LineString([p1, p2])),
            )

        return path

    def _rrt_star(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles=self.obstacles,
            obstacles_tree=self.obstacles_tree,
            grid=self._ensure_grid_2d_cache(),
            low=self.args.low,
            grid_scale=self.args.cell_size,
            grid_cost_weight=self.grid_cost_weight,
            transfer_id=self.transfer_id,
            informed=self.rrt_informed,
            improve_after_goal=self.rrt_improve_after_goal,
            improve_iter=self.rrt_improve_iter,
            adaptive_radius=self.rrt_adaptive_radius,
        )
        return planner.find_path()

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        return grid_astar(
            self._ensure_grid_2d_cache(),
            start,
            goal,
            self.args.low,
            self.args.cell_size,
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
        max_path_dist: float = 2.0,
    ) -> None:
        """
        Populate the grid with costs based on map data.
        """
        self.path_grid.fill(
            map_data,
            self.obstacles,
            highway_types=highway_types,
            max_path_dist=max_path_dist,
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Default parameter namespace for :class:`ReplanPath`.

    Not a CLI parser — the standalone ``replan`` script this once served no
    longer exists (see ``map_data_plan`` / :mod:`map_data.plan_route_cli`
    instead). This is just the shared way callers (the viewer, and
    :func:`~map_data.pathsolver.route.plan_route`) build a defaults object
    and override the fields they care about (``low``, ``high``,
    ``cell_size``, ...).
    """
    del args  # nothing left parses real CLI arguments here
    return argparse.Namespace(
        low=(0.0, 0.0),
        high=(0.0, 0.0),
        cell_size=0.25,
        inflate_obstacles=0.25,
        simplify_path=True,
        smooth_path=False,
    )
