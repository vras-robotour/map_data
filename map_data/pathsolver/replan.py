#!/usr/bin/env python3

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import shapely as sh
from joblib import Parallel, delayed
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.pathsolver.grid_astar import grid_astar
from map_data.pathsolver.rrt_star import RRTStar
from map_data.utils.config import load_config
from map_data.utils.gpx import create_gpx_content, parse_path, utm_path_to_latlon
from map_data.utils.parsing import ways_to_shapely

from .grid_constructor import PathGrid

# Decoupled components
from .smoothing import smooth_path
from .visualizer import visualize_replan

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
    # These will be populated from config or fallback to hardcoded defaults if config missing
    _DEFAULTS: dict[str, Any] = load_planner_defaults()
    HIGHWAY_COSTS: dict[str, float] = _DEFAULTS.get(
        "highway_costs",
        {
            "pedestrian": 0.0,
            "footway": 0.0,
            "path": 0.1,
            "living_street": 0.1,
            "track": 0.3,
            "service": 0.3,
            "residential": 0.5,
            "unclassified": 0.5,
            "tertiary": 0.7,
            "secondary": 0.9,
            "primary": 1.0,
        },
    )
    SURFACE_COSTS: dict[str, float] = _DEFAULTS.get(
        "surface_costs",
        {
            "asphalt": 0.0,
            "paving_stones": 0.0,
            "concrete": 0.0,
            "fine_gravel": 0.1,
            "gravel": 0.2,
            "dirt": 0.3,
            "grass": 0.5,
            "sand": 0.7,
        },
    )
    DEFAULT_OFF_PATH_COST: float = _DEFAULTS.get("default_off_path_cost", 0.9)
    PATH_COST_CAP: float = _DEFAULTS.get("path_cost_cap", 0.85)

    def __init__(
        self,
        args: argparse.Namespace,
        obstacles: list[sh.geometry.base.BaseGeometry] | None = None,
        transfer_id: str | None = None,
    ) -> None:
        self.args = args
        self.transfer_id = transfer_id

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
        self.debug = False

    @property
    def grid(self) -> np.ndarray:
        """
        Compatibility property for old access to the raw point grid.
        """
        return self.path_grid.grid

    @grid.setter
    def grid(self, value: np.ndarray) -> None:
        self.path_grid.grid = value

    @property
    def _reshaped_grid_cache(self) -> np.ndarray | None:
        """
        Compatibility property for the 2D cost grid.
        """
        return self.path_grid.grid_2d_cache

    @_reshaped_grid_cache.setter
    def _reshaped_grid_cache(self, value: np.ndarray | None) -> None:
        self.path_grid.grid_2d_cache = value

    def replan(self, path: np.ndarray, algorithm: str = "astar") -> np.ndarray | None:
        def process_segment(
            i: int, path: np.ndarray, args: argparse.Namespace,
        ) -> tuple[list[np.ndarray] | None, int]:
            if _is_cancelled(self.transfer_id):
                return None, i

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
                    return None, i
                segment_path.extend(way[1:-1])
            return segment_path, i

        new_path: list[np.ndarray] = []
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(process_segment)(i, path, self.args) for i in range(len(path) - 1)
        )

        if _is_cancelled(self.transfer_id):
            _discard_cancelled(self.transfer_id)
            return None

        # Sort results by index to maintain path order
        results.sort(key=lambda x: x[1])
        for segment_path, _ in results:
            if segment_path is None:
                logger.warning("%s failed to find a path.", algorithm)
                return None
            new_path.extend(segment_path)

        new_path.append(path[-1][:2])
        return self._post_process_path(np.array(new_path))

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

        # 3. Final Douglas-Peucker simplification on the whole path
        if self.args.simplify_path:
            path = np.array(LineString(path).simplify(self.args.cell_size).coords)

        return path

    def _rrt_star(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        if self.path_grid.grid_2d_cache is None:
            grid_2d = self.path_grid.get_grid_2d()
            self.path_grid.grid_2d_cache = self.path_grid.burn_obstacles(grid_2d, self.obstacles)

        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles=self.obstacles,
            obstacles_tree=self.obstacles_tree,
            grid=self.path_grid.grid_2d_cache,
            low=self.args.low,
            grid_scale=self.args.cell_size,
            transfer_id=self.transfer_id,
            simplify=self.args.simplify_path,
        )
        return planner.find_path()

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        if self.path_grid.grid_2d_cache is None:
            grid_2d = self.path_grid.get_grid_2d()
            self.path_grid.grid_2d_cache = self.path_grid.burn_obstacles(grid_2d, self.obstacles)

        return grid_astar(
            self.path_grid.grid_2d_cache,
            start,
            goal,
            self.args.low,
            self.args.cell_size,
            simplify_path=self.args.simplify_path,
        )

    def _colides(self, path_seg: LineString) -> bool:
        if self.obstacles_tree is None:
            return False
        intersecting_indices = self.obstacles_tree.query(path_seg, predicate="intersects")
        return len(intersecting_indices) > 0

    def _create_grid(
        self,
        low: tuple[float, float],
        high: tuple[float, float],
        cell_size: float = 0.25,
    ) -> np.ndarray:
        """
        Compatibility delegate for _create_grid.
        """
        return self.path_grid._create_empty_grid()

    def _burn_obstacles_into_grid(self, grid_2d: np.ndarray) -> np.ndarray:
        """
        Compatibility delegate for _burn_obstacles_into_grid.
        """
        return self.path_grid.burn_obstacles(grid_2d, self.obstacles)

    def fill_grid(
        self,
        map_data: Any,
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

    def visualize(self, path: np.ndarray | None, old_path: np.ndarray | None = None) -> None:
        """
        Visualize the grid, obstacles, and path using Matplotlib.
        """
        grid_2d = self.path_grid.get_grid_2d()
        visualize_replan(
            path,
            grid_2d,
            self.args.low,
            self.args.high,
            self.obstacles,
            old_path=old_path,
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/coords.gpx", help="Path file")
    parser.add_argument("--file", type=str, default=None, help="Map data file")
    parser.add_argument("--simplify_path", action="store_true", help="Simplify path")
    parser.add_argument("--cell_size", type=float, default=0.25, help="Cell size for the grid")
    parser.add_argument(
        "--inflate_obstacles",
        type=float,
        default=0.25,
        help="Inflate obstacles by this amount",
    )
    parser.add_argument("--smooth_path", action="store_true", help="Smooth path")
    parser.add_argument(
        "--max_path_dist",
        type=float,
        default=2.0,
        help="Maximum distance from path to be traversable",
    )
    parser.add_argument("--save", type=str, default=None, help="Save path to file")
    parser.add_argument("--visualize", action="store_true", help="Visualize the path")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    base_dir = Path(__file__).parent
    path_file = (base_dir / ".." / args.path).resolve()
    path_data = parse_path(str(path_file))

    if args.file is None:
        map_data = MapData(path_data, coords_type="array")
        map_data.run_queries()
        ret = map_data.run_parse()
        if ret:
            sys.exit(1)
    else:
        map_file = (base_dir / ".." / args.file).resolve()
        map_data = MapData.load(str(map_file))

    args.low = (map_data.min_x, map_data.min_y)
    args.high = (map_data.max_x, map_data.max_y)
    obstacles = ways_to_shapely(map_data.barriers_list)

    replanner = ReplanPath(args, obstacles)
    replanner.fill_grid(map_data, max_path_dist=args.max_path_dist)

    new_path = replanner.replan(path_data[0], algorithm="astar")

    if args.save and new_path is not None:
        new_wgs_path = utm_path_to_latlon(new_path, path_data[1], path_data[2])
        gpx_content = create_gpx_content(new_wgs_path, creator_name="A* Replanner")
        with Path(args.save).open("w") as f:
            f.write(gpx_content)

    if args.visualize:
        replanner.visualize(new_path, path_data[0])
