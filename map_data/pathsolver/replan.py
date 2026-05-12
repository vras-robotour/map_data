#!/usr/bin/env python3

import os
import argparse
import threading

import numpy as np
import shapely as sh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from shapely.geometry import LineString
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path

from map_data.pathsolver.grid_astar import grid_astar
from map_data.pathsolver.rrt_star import RRTStar
from map_data.map_data import MapData
from map_data.utils.parsing import (
    ways_to_shapely,
)
from map_data.utils.gpx import (
    parse_path,
    create_gpx_content,
    utm_path_to_latlon,
)


_cancel_lock = threading.Lock()
_cancelled_transfers: set = set()


def cancel_replan_backend(transfer_id):
    if transfer_id:
        with _cancel_lock:
            _cancelled_transfers.add(transfer_id)


def _is_cancelled(transfer_id) -> bool:
    if not transfer_id:
        return False
    with _cancel_lock:
        return transfer_id in _cancelled_transfers


def _discard_cancelled(transfer_id):
    if transfer_id:
        with _cancel_lock:
            _cancelled_transfers.discard(transfer_id)


class ReplanPath:
    def __init__(self, args, obstacles=None, transfer_id=None):
        self.args = args
        self.transfer_id = transfer_id

        self.grid = self._create_grid(args.low, args.high, args.cell_size)
        if args.inflate_obstacles:
            self.obstacles = [
                obstacle.buffer(args.inflate_obstacles) for obstacle in obstacles
            ]
        else:
            self.obstacles = obstacles

        # Spatial index for faster collision checking
        self.obstacles_tree = sh.STRtree(self.obstacles) if self.obstacles else None

        self.debug = False
        self._reshaped_grid_cache = None
        self._converted_obstacles = (
            self._convert_obstacles(self.obstacles) if self.obstacles else []
        )

    def replan(self, path, algorithm="astar"):
        def process_segment(i, path, args):
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

        new_path = []
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
                print(f"{algorithm} failed to find a path.")
                return False
            new_path.extend(segment_path)

        new_path.append(path[-1][:2])
        return self._post_process_path(np.array(new_path))

    def _post_process_path(self, path):
        """
        Simplify the final path by removing redundant points.
        """
        if path is None or len(path) <= 2:
            return path

        # 1. Remove points that are extremely close to each other
        dist_sq = np.sum(np.diff(path, axis=0) ** 2, axis=1)
        mask = np.ones(len(path), dtype=bool)
        # Remove points that are closer than 0.05m to the previous point
        mask[1:] = dist_sq > 0.05**2
        path = path[mask]

        if len(path) <= 2:
            return path

        # 2. Final Douglas-Peucker simplification on the whole path
        if self.args.simplify_path:
            # Use cell_size as tolerance for the whole path
            path = np.array(LineString(path).simplify(self.args.cell_size).coords)

        return path

    def _rrt_star(self, start, goal):
        if self._reshaped_grid_cache is None:
            self._reshaped_grid_cache = self._burn_obstacles_into_grid(
                self._reshape_grid()
            )
        grid = self._reshaped_grid_cache
        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles=self.obstacles,
            obstacles_tree=self.obstacles_tree,
            grid=grid,
            low=self.args.low,
            grid_scale=self.args.cell_size,
            transfer_id=self.transfer_id,
            simplify=self.args.simplify_path,
        )
        return planner.find_path()

    def _reshape_grid(self):
        """
        Reshape the grid to match the shape of the map data.
        """
        low = self.args.low
        high = self.args.high
        cell_size = self.args.cell_size

        num_x = int(np.ceil((high[0] - low[0]) / cell_size))
        num_y = int(np.ceil((high[1] - low[1]) / cell_size))

        grid = np.zeros((num_x, num_y), dtype=np.float32)
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        c = self.grid[:, 3]

        x_indices = np.floor((x - low[0]) / cell_size).astype(int)
        y_indices = np.floor((y - low[1]) / cell_size).astype(int)
        x_indices = np.clip(x_indices, 0, num_x - 1)
        y_indices = np.clip(y_indices, 0, num_y - 1)

        grid[x_indices, y_indices] = c
        return grid.T

    def _burn_obstacles_into_grid(self, grid_2d):
        """
        Burn obstacles into the grid by setting their cost to infinity.
        """
        if not self.obstacles:
            return grid_2d

        # grid_2d is (Y, X)
        ny, nx = grid_2d.shape
        low = self.args.low
        cs = self.args.cell_size

        for obstacle in self.obstacles:
            # Get bounding box of the obstacle
            minx, miny, maxx, maxy = obstacle.bounds

            # Map to grid indices
            ix_min = max(0, int(np.floor((minx - low[0]) / cs)))
            ix_max = min(nx - 1, int(np.ceil((maxx - low[0]) / cs)))
            iy_min = max(0, int(np.floor((miny - low[1]) / cs)))
            iy_max = min(ny - 1, int(np.ceil((maxy - low[1]) / cs)))

            if ix_min > ix_max or iy_min > iy_max:
                continue

            # Create points only for this bounding box
            # We use centers of the cells for better accuracy
            x = np.linspace(
                ix_min * cs + low[0], ix_max * cs + low[0], ix_max - ix_min + 1
            )
            y = np.linspace(
                iy_min * cs + low[1], iy_max * cs + low[1], iy_max - iy_min + 1
            )
            xv, yv = np.meshgrid(x, y)
            points_bbox = np.stack((xv.ravel(), yv.ravel()), axis=-1)

            if obstacle.geom_type == "Polygon":
                poly_path = Path(np.array(obstacle.exterior.coords))
                mask = poly_path.contains_points(points_bbox).reshape(len(y), len(x))
                grid_2d[iy_min : iy_max + 1, ix_min : ix_max + 1][mask] = np.inf
            elif obstacle.geom_type == "MultiPolygon":
                for poly in obstacle.geoms:
                    poly_path = Path(np.array(poly.exterior.coords))
                    # We could also crop to this poly's bounds, but for now just check against bbox of MultiPolygon
                    mask = poly_path.contains_points(points_bbox).reshape(
                        len(y), len(x)
                    )
                    grid_2d[iy_min : iy_max + 1, ix_min : ix_max + 1][mask] = np.inf

        return grid_2d

    def _convert_obstacles(self, obstacles):
        """
        Convert obstacles to a format suitable for RRT*.
        Parameters:
        -----------
        obstacles : list
            List of obstacles as shapely geometries.
        Returns:
        --------
        obst : list
            List of obstacles as shapely polygons.
        """
        obst = []
        for obstacle in obstacles:
            obstacle = sh.transform(obstacle, lambda x: (x - self.args.low))
            obst.append(obstacle)
        return obst

    def _astar(self, start, goal):
        if self._reshaped_grid_cache is None:
            self._reshaped_grid_cache = self._burn_obstacles_into_grid(
                self._reshape_grid()
            )
        grid = self._reshaped_grid_cache
        return grid_astar(
            grid,
            start,
            goal,
            self.args.low,
            self.args.cell_size,
            simplify_path=self.args.simplify_path,
        )

    def _colides(self, path_seg):
        if self.obstacles_tree is None:
            return False
        # STRtree.query returns indices of obstacles that intersect the path_seg bounding box
        # We need to check if any of them actually intersect the segment
        intersecting_indices = self.obstacles_tree.query(
            path_seg, predicate="intersects"
        )
        return len(intersecting_indices) > 0

    def _create_grid(self, low, high, cell_size=0.25):
        """
        Create a grid of points.

        Parameters:
        -----------
        low : tuple
            Lower bounds of the grid.
        high : tuple
            Upper bounds of the grid.
        cell_size : float
            Size of the cell.

        Returns:
        --------
        grid : np.array
            Grid of points.
        """
        xs = np.linspace(
            low[0], high[0], np.ceil((high[0] - low[0]) / cell_size).astype(int)
        )
        ys = np.linspace(
            low[1], high[1], np.ceil((high[1] - low[1]) / cell_size).astype(int)
        )
        grid = np.pad(
            np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2), ((0, 0), (0, 1))
        )
        return grid

    def fill_grid(self, map_data, highway_types=None, max_path_dist=20.0):
        if highway_types is None:
            highway_types = ["footway"]

        points = map_data.get_points()
        path_grid = self.grid

        bbox = sh.box(
            self.args.low[0], self.args.low[1], self.args.high[0], self.args.high[1]
        )
        # Add a bit of margin for ways that might pass nearby
        bbox_buffered = bbox.buffer(max_path_dist)

        allowed_ways = []
        if "footway" in highway_types:
            allowed_ways.extend(
                [
                    w
                    for w in map_data.footways_list
                    if w.line and w.line.intersects(bbox_buffered)
                ]
            )
        if "road" in highway_types:
            allowed_ways.extend(
                [
                    w
                    for w in map_data.roads_list
                    if w.line and w.line.intersects(bbox_buffered)
                ]
            )

        if not allowed_ways:
            path_grid = np.pad(path_grid, ((0, 0), (0, 1)))
            path_grid[:, 3] = 0.5
            self.grid = path_grid
            return

        # PRIORITIZE PATHS OVER OBSTACLES:
        # Subtract path geometries from obstacles so that paths are always traversable.
        path_geoms = [w.line for w in allowed_ways if w.line]
        if path_geoms and self.obstacles:
            unioned_paths = sh.unary_union(path_geoms)
            new_obstacles = []
            for obstacle in self.obstacles:
                # Subtract paths from the obstacle
                diff = obstacle.difference(unioned_paths)
                if not diff.is_empty:
                    # difference might return a MultiPolygon
                    if diff.geom_type == "MultiPolygon":
                        new_obstacles.extend(list(diff.geoms))
                    else:
                        new_obstacles.append(diff)

            self.obstacles = new_obstacles
            # Rebuild STRtree
            self.obstacles_tree = sh.STRtree(self.obstacles) if self.obstacles else None

        paths = np.pad(
            self._split_ways(points, allowed_ways, self.args.cell_size),
            ((0, 0), (0, 1)),
        )
        neighbor_cost = "quadratic"
        tmp, mask = self._points_near_ref(path_grid, paths, max_path_dist)
        # Initialize path_grid cost to 1.0 (un-traversable/high cost)
        path_grid = np.pad(path_grid, ((0, 0), (0, 1)))
        path_grid[:, 3] = 1.0

        if neighbor_cost == "linear":
            pass
        elif neighbor_cost == "quadratic":
            tmp[:, 3] = tmp[:, 3] ** 2
        elif neighbor_cost == "zero":
            tmp[:, 3] = 0
        else:
            print(f"Unknown neighbor cost: {neighbor_cost}")

        # Set points near paths to their calculated cost (0.0 to 1.0)
        path_grid[mask, 3] = tmp[:, 3]

        self.grid = path_grid
        grid_2d = self._reshape_grid()
        self._reshaped_grid_cache = self._burn_obstacles_into_grid(grid_2d)

    def _points_near_ref(self, points, reference, max_dist=1):
        """
        Get points near reference points and set linear distance as cost.

        Parameters:
        -----------
        points : np.array
            Points to check.
        reference : np.array
            Reference points.
        max_dist : float
            Maximum distance to check.

        Returns:
        --------
        points : np.array
            All points with a cost based on distance to reference points.
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not isinstance(reference, np.ndarray):
            reference = np.array(reference)

        tree = cKDTree(reference, compact_nodes=False, balanced_tree=False)
        dists, _ = np.array(tree.query(points, distance_upper_bound=max_dist))
        mask = dists < max_dist
        points = points[mask]
        dists = dists[mask]

        return (np.hstack([points, (dists / max_dist).reshape(-1, 1)]), mask)

    def _split_ways(self, points, ways, max_dist=0.25):
        """
        Equidistantly split ways into points with a maximal step size. Also only use footways from map data,
        as we are not allowed to leave the footways.

        Parameters:
        -----------
        points : dict
            Points to split ways on.
        ways : dict
            Ways to split.
        max_dist : float
            Maximal step size.

        Returns:
        --------
        waypoints : np.array
            Waypoints created from the ways.
        """
        waypoints = []
        for way in ways:
            for i, (n0, n1) in enumerate(zip(way.nodes, way.nodes[1:])):
                point0 = points[n0].ravel()[:2]
                point1 = points[n1].ravel()[:2]
                dist = np.linalg.norm(point1 - point0)

                if i == 0:
                    waypoints.append(point0)
                if dist <= 1e-3:
                    waypoints.append(point1)
                    continue

                vec = (point1 - point0) / dist
                num = int(np.ceil(dist / max_dist))
                step = dist / num
                for j in range(num):
                    waypoints.append(point0 + (j + 1) * step * vec)

        return np.array(waypoints)

    def visualize(self, path, old_path=None):
        """Visualize the grid, obstacles, and path using Matplotlib."""
        _, ax = plt.subplots()

        # Plot grid as a heatmap (0: white, 1: gray)
        grid_display = self._reshape_grid()
        ax.imshow(
            grid_display,
            cmap="Greys",
            origin="lower",
            extent=[
                self.args.low[0],
                self.args.high[0],
                self.args.low[1],
                self.args.high[1],
            ],
        )

        # Plot obstacles
        for obstacle in self.obstacles:
            if obstacle.geom_type == "Polygon":
                x, y = obstacle.exterior.xy
                ax.add_patch(MplPolygon(list(zip(x, y)), color="red", alpha=0.5))

        # Plot old path if provided
        if old_path is not None:
            # old_path = np.array(old_path)
            ax.plot(old_path[:, 0], old_path[:, 1], "c-", linewidth=2, label="Path")

        # Plot path if found
        if path is not None:
            # path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], "m-", linewidth=2, label="Path")
            ax.scatter(path[:, 0], path[:, 1], c="m", s=20, label="Path Points")

            # Plot start and goal
            ax.plot(path[0, 0], path[0, 1], "go", label="Start")
            ax.plot(path[-1, 0], path[-1, 1], "bo", label="Goal")

        # Set plot properties
        ax.set_xlabel("Northing [m]")
        ax.set_ylabel("Easting [m]")
        ax.set_title("Replanned Path")

        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim(self.args.low[0], self.args.high[0])
        ax.set_ylim(self.args.low[1], self.args.high[1])

        plt.savefig("replan.png")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/coords.gpx", help="Path file")
    parser.add_argument("--file", type=str, default=None, help="Map data file")
    parser.add_argument("--simplify_path", action="store_true", help="Simplify path")
    parser.add_argument(
        "--cell_size", type=float, default=0.25, help="Cell size for the grid"
    )
    parser.add_argument(
        "--inflate_obstacles",
        type=float,
        default=0.25,
        help="Inflate obstacles by this amount",
    )
    parser.add_argument(
        "--max_path_dist",
        type=float,
        default=20.0,
        help="Maximum distance from path to be traversable",
    )
    parser.add_argument("--save", type=str, default=None, help="Save path to file")
    parser.add_argument("--visualize", action="store_true", help="Visualize the path")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    path_file = os.path.join(os.path.dirname(__file__), "../", args.path)
    path_data = parse_path(path_file)

    if args.file is None:
        map_data = MapData(path_data, coords_type="array")
        map_data.run_queries()
        ret = map_data.run_parse()
        if ret:
            exit(1)
    else:
        map_data = MapData.load(
            os.path.join(os.path.dirname(__file__), "../", args.file)
        )

    args.low = (map_data.min_x, map_data.min_y)
    args.high = (map_data.max_x, map_data.max_y)
    obstacles = ways_to_shapely(map_data.barriers_list)

    replanner = ReplanPath(args, obstacles)
    replanner.fill_grid(map_data, max_path_dist=args.max_path_dist)

    new_path = replanner.replan(path_data[0], algorithm="astar")

    if args.save:
        new_wgs_path = utm_path_to_latlon(new_path, path_data[1], path_data[2])
        gpx_content = create_gpx_content(new_wgs_path, creator_name="A* Replanner")
        with open(args.save, "w") as f:
            f.write(gpx_content)

    if args.visualize:
        replanner.visualize(new_path, path_data[0])
