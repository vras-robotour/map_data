from typing import TYPE_CHECKING

import numpy as np
import shapely as sh
from matplotlib.path import Path
from scipy.spatial import cKDTree

TOLERANCE = 1e-3

if TYPE_CHECKING:
    from map_data.map_data import MapData


class PathGrid:
    def __init__(
        self,
        low: tuple[float, float],
        high: tuple[float, float],
        cell_size: float,
        highway_costs: dict[str, float],
        surface_costs: dict[str, float],
        default_off_path_cost: float = 0.9,
        path_cost_cap: float = 0.85,
    ) -> None:
        self.low = np.array(low)
        self.high = np.array(high)
        self.cell_size = cell_size
        self.highway_costs = highway_costs
        self.surface_costs = surface_costs
        self.default_off_path_cost = default_off_path_cost
        self.path_cost_cap = path_cost_cap

        self.grid = self.create_empty_grid()
        self.grid_2d_cache: np.ndarray | None = None

    def create_empty_grid(self) -> np.ndarray:
        xs = np.arange(self.low[0], self.high[0], self.cell_size)
        ys = np.arange(self.low[1], self.high[1], self.cell_size)
        # grid is (N, 3) where columns are [x, y, 0]
        return np.pad(np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2), ((0, 0), (0, 1)))

    def fill(
        self,
        map_data: "MapData",
        obstacles: list[sh.geometry.base.BaseGeometry],
        highway_types: list[str] | None = None,
        max_path_dist: float = 2.0,
    ) -> np.ndarray:
        if highway_types is None:
            highway_types = ["footway"]

        points = map_data.get_points()

        # 1. Initialize path_grid with default off-path cost
        path_grid = np.pad(self.grid[:, :2], ((0, 0), (0, 1)))
        path_grid[:, 2] = 0.0
        path_grid = np.pad(path_grid, ((0, 0), (0, 1)))
        path_grid[:, 3] = self.default_off_path_cost

        bbox = sh.box(self.low[0], self.low[1], self.high[0], self.high[1])
        bbox_buffered = bbox.buffer(max_path_dist)

        all_ways = []
        if "footway" in highway_types:
            all_ways.extend(
                [w for w in map_data.footways_list if w.line and w.line.intersects(bbox_buffered)],
            )
        if "road" in highway_types:
            all_ways.extend(
                [w for w in map_data.roads_list if w.line and w.line.intersects(bbox_buffered)],
            )

        # Subtract path geometries from obstacles
        path_geoms = [w.line for w in all_ways]
        effective_obstacles = obstacles
        if path_geoms and obstacles:
            unioned_paths = sh.unary_union(path_geoms)
            new_obstacles = []
            for obstacle in obstacles:
                diff = obstacle.difference(unioned_paths)
                if not diff.is_empty:
                    if diff.geom_type == "MultiPolygon":
                        new_obstacles.extend(list(diff.geoms))
                    else:
                        new_obstacles.append(diff)
            effective_obstacles = new_obstacles

        # 2. Mark hard obstacles in the point grid
        if effective_obstacles:
            for obstacle in effective_obstacles:
                minx, miny, maxx, maxy = obstacle.bounds
                mask_bbox = (
                    (path_grid[:, 0] >= minx)
                    & (path_grid[:, 0] <= maxx)
                    & (path_grid[:, 1] >= miny)
                    & (path_grid[:, 1] <= maxy)
                )
                if not np.any(mask_bbox):
                    continue

                if obstacle.geom_type == "Polygon":
                    poly_path = Path(np.array(obstacle.exterior.coords))
                    mask_inside = poly_path.contains_points(path_grid[mask_bbox, :2])
                    for interior in obstacle.interiors:
                        hole_path = Path(np.array(interior.coords))
                        mask_inside &= ~hole_path.contains_points(path_grid[mask_bbox, :2])
                    path_grid[mask_bbox, 3] = np.where(mask_inside, 1.0, path_grid[mask_bbox, 3])
                elif obstacle.geom_type == "MultiPolygon":
                    for poly in obstacle.geoms:
                        poly_path = Path(np.array(poly.exterior.coords))
                        mask_inside = poly_path.contains_points(path_grid[mask_bbox, :2])
                        for interior in poly.interiors:
                            hole_path = Path(np.array(interior.coords))
                            mask_inside &= ~hole_path.contains_points(path_grid[mask_bbox, :2])
                        path_grid[mask_bbox, 3] = np.where(
                            mask_inside,
                            1.0,
                            path_grid[mask_bbox, 3],
                        )

        # 3. Process ways to set their costs
        path_points = []
        path_point_costs = []

        for way in all_ways:
            hw = way.tags.get("highway", "path")
            surface = way.tags.get("surface", "asphalt")
            base_cost = self.highway_costs.get(hw, 0.5)
            surface_cost = self.surface_costs.get(surface, 0.0)
            way_cost = min(self.path_cost_cap, base_cost + surface_cost)

            for i in range(len(way.nodes) - 1):
                p0 = points[way.nodes[i]].ravel()[:2]
                p1 = points[way.nodes[i + 1]].ravel()[:2]
                dist = np.linalg.norm(p1 - p0)
                if i == 0:
                    path_points.append(p0)
                    path_point_costs.append(way_cost)
                if dist <= TOLERANCE:
                    path_points.append(p1)
                    path_point_costs.append(way_cost)
                    continue
                num = int(np.ceil(dist / self.cell_size))
                step = dist / num
                vec = (p1 - p0) / dist
                for j in range(num):
                    path_points.append(p0 + (j + 1) * step * vec)
                    path_point_costs.append(way_cost)

        if path_points:
            tree = cKDTree(np.array(path_points))
            dists, indices = tree.query(path_grid[:, :2], distance_upper_bound=max_path_dist)
            mask = dists < max_path_dist
            way_costs = np.array(path_point_costs)[indices[mask]]
            final_costs = (
                way_costs
                + (self.default_off_path_cost - way_costs) * (dists[mask] / max_path_dist) ** 2
            )
            path_grid[mask, 3] = np.minimum(path_grid[mask, 3], final_costs)

        self.grid = path_grid
        grid_2d = self.get_grid_2d()
        self.grid_2d_cache = self.burn_obstacles(grid_2d, effective_obstacles)
        return self.grid_2d_cache

    def get_grid_2d(self) -> np.ndarray:
        num_x = int(np.ceil((self.high[0] - self.low[0]) / self.cell_size))
        num_y = int(np.ceil((self.high[1] - self.low[1]) / self.cell_size))
        grid_2d = np.full((num_x, num_y), self.default_off_path_cost, dtype=np.float32)
        x_indices = np.floor((self.grid[:, 0] - self.low[0]) / self.cell_size).astype(int)
        y_indices = np.floor((self.grid[:, 1] - self.low[1]) / self.cell_size).astype(int)
        grid_2d[np.clip(x_indices, 0, num_x - 1), np.clip(y_indices, 0, num_y - 1)] = self.grid[
            :,
            3,
        ]
        return grid_2d.T

    def burn_obstacles(
        self,
        grid_2d: np.ndarray,
        obstacles: list[sh.geometry.base.BaseGeometry],
    ) -> np.ndarray:
        if not obstacles:
            return grid_2d
        ny, nx = grid_2d.shape
        for obstacle in obstacles:
            minx, miny, maxx, maxy = obstacle.bounds
            ix_min = max(0, int(np.floor((minx - self.low[0]) / self.cell_size)))
            ix_max = min(nx - 1, int(np.ceil((maxx - self.low[0]) / self.cell_size)))
            iy_min = max(0, int(np.floor((miny - self.low[1]) / self.cell_size)))
            iy_max = min(ny - 1, int(np.ceil((maxy - self.low[1]) / self.cell_size)))
            if ix_min > ix_max or iy_min > iy_max:
                continue

            x = np.linspace(
                ix_min * self.cell_size + self.low[0],
                ix_max * self.cell_size + self.low[0],
                ix_max - ix_min + 1,
            )
            y = np.linspace(
                iy_min * self.cell_size + self.low[1],
                iy_max * self.cell_size + self.low[1],
                iy_max - iy_min + 1,
            )
            xv, yv = np.meshgrid(x, y)
            points_bbox = np.stack((xv.ravel(), yv.ravel()), axis=-1)

            if obstacle.geom_type == "Polygon":
                mask = (
                    Path(np.array(obstacle.exterior.coords))
                    .contains_points(points_bbox)
                    .reshape(len(y), len(x))
                )
                for interior in obstacle.interiors:
                    hole_mask = (
                        Path(np.array(interior.coords))
                        .contains_points(points_bbox)
                        .reshape(len(y), len(x))
                    )
                    mask &= ~hole_mask
                grid_2d[iy_min : iy_max + 1, ix_min : ix_max + 1][mask] = np.inf
            elif obstacle.geom_type == "MultiPolygon":
                for poly in obstacle.geoms:
                    mask = (
                        Path(np.array(poly.exterior.coords))
                        .contains_points(points_bbox)
                        .reshape(len(y), len(x))
                    )
                    for interior in poly.interiors:
                        hole_mask = (
                            Path(np.array(interior.coords))
                            .contains_points(points_bbox)
                            .reshape(len(y), len(x))
                        )
                        mask &= ~hole_mask
                    grid_2d[iy_min : iy_max + 1, ix_min : ix_max + 1][mask] = np.inf
        return grid_2d
