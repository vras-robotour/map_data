"""
Traversal cost grid for grid-based path planning.

Off-path cells cost ``default_off_path_cost``, cells near a way fall off towards
its :func:`~map_data.pathsolver.way_cost.way_cost`, and obstacle cells are ``inf``.
"""

from typing import TYPE_CHECKING

import numpy as np
import shapely as sh
from scipy.spatial import cKDTree

from map_data.pathsolver.way_cost import way_cost
from map_data.utils.densify import densify_ways

if TYPE_CHECKING:
    from map_data.map_data import MapData


class PathGrid:
    """
    Raster of square ``cell_size`` cells covering ``[low, high)`` in world coordinates.

    Attributes
    ----------
    grid_2d_cache : np.ndarray or None
        ``(num_y, num_x)`` cost raster with obstacles at ``inf``, built by :meth:`fill`.

    """

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
        """
        Create an unfilled grid; :meth:`fill` computes the costs.

        Parameters
        ----------
        highway_costs, surface_costs : dict of {str: float}
            Cost tables passed to :func:`~map_data.pathsolver.way_cost.way_cost`.
        default_off_path_cost : float
            Cost of a cell farther than ``max_path_dist`` from every way.
        path_cost_cap : float
            Upper bound of a way's cost.

        """
        self.low = np.array(low)
        self.high = np.array(high)
        self.cell_size = cell_size
        self.highway_costs = highway_costs
        self.surface_costs = surface_costs
        self.default_off_path_cost = default_off_path_cost
        self.path_cost_cap = path_cost_cap

        # Built by fill(): a planner that never needs the grid never pays for it.
        self.grid_2d_cache: np.ndarray | None = None

    def fill(
        self,
        map_data: "MapData",
        obstacles: list[sh.geometry.base.BaseGeometry],
        highway_types: list[str] | None = None,
        max_path_dist: float = 2.0,
    ) -> list[sh.geometry.base.BaseGeometry]:
        """
        Compute the cell costs and store them in :attr:`grid_2d_cache`.

        A cell within *max_path_dist* of a way costs
        ``c + (default_off_path_cost - c) * (dist / max_path_dist) ** 2`` where ``c``
        is the way's cost. The ways are subtracted from *obstacles* before they are
        burned in, so a barrier polygon overlapping a footway does not block it.

        Parameters
        ----------
        obstacles : list of shapely geometry
            Only ``Polygon``/``MultiPolygon`` geometries block cells.
        highway_types : list of str, optional
            Way categories that lower the cost: ``"footway"`` and/or ``"road"``.
            Defaults to ``["footway"]``.
        max_path_dist : float
            Radius of influence of a way centerline.

        Returns
        -------
        list of shapely geometry
            *obstacles* minus the ways: what the grid actually blocks, so segment
            collision checks must use it too.

        """
        if highway_types is None:
            highway_types = ["footway"]

        points = map_data.get_points()

        # Float arange may yield one cell more or less near the upper edge; ceil is the count.
        num_x = int(np.ceil((self.high[0] - self.low[0]) / self.cell_size))
        num_y = int(np.ceil((self.high[1] - self.low[1]) / self.cell_size))
        xs = self.low[0] + (np.arange(num_x) + 0.5) * self.cell_size
        ys = self.low[1] + (np.arange(num_y) + 0.5) * self.cell_size
        grid_2d = np.full((num_y, num_x), self.default_off_path_cost, dtype=np.float32)

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

        # Same helper as the graph planner's edge weights, so both planners price a way alike.
        costs = [
            way_cost(way.tags, self.highway_costs, self.surface_costs, self.path_cost_cap)
            for way in all_ways
        ]
        path_points, way_index = densify_ways(points, [w.nodes for w in all_ways], self.cell_size)

        if len(path_points):
            tree = cKDTree(path_points)
            centers = np.stack(np.meshgrid(xs, ys), axis=-1)  # (num_y, num_x, 2)
            dists, indices = tree.query(centers, distance_upper_bound=max_path_dist, workers=-1)
            mask = dists < max_path_dist
            way_costs = np.array(costs)[way_index][indices[mask]]
            final_costs = (
                way_costs
                + (self.default_off_path_cost - way_costs) * (dists[mask] / max_path_dist) ** 2
            )
            grid_2d[mask] = np.minimum(grid_2d[mask], final_costs)

        self.grid_2d_cache = self.burn_obstacles(grid_2d, effective_obstacles)
        return effective_obstacles

    def burn_obstacles(
        self,
        grid_2d: np.ndarray,
        obstacles: list[sh.geometry.base.BaseGeometry],
    ) -> np.ndarray:
        """
        Set the cells whose center lies inside an obstacle to ``inf``, in place.

        Only each obstacle's bounding-box cells are tested. Polygon holes are not
        burned; non-polygon geometries are skipped. Returns *grid_2d*.
        """
        if not obstacles:
            return grid_2d
        ny, nx = grid_2d.shape
        for obstacle in obstacles:
            if obstacle.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            minx, miny, maxx, maxy = obstacle.bounds
            ix_min = max(0, int(np.floor((minx - self.low[0]) / self.cell_size)))
            ix_max = min(nx - 1, int(np.ceil((maxx - self.low[0]) / self.cell_size)))
            iy_min = max(0, int(np.floor((miny - self.low[1]) / self.cell_size)))
            iy_max = min(ny - 1, int(np.ceil((maxy - self.low[1]) / self.cell_size)))
            if ix_min > ix_max or iy_min > iy_max:
                continue

            # Cell centers, the same sampling convention as fill and the planners.
            x = np.linspace(
                (ix_min + 0.5) * self.cell_size + self.low[0],
                (ix_max + 0.5) * self.cell_size + self.low[0],
                ix_max - ix_min + 1,
            )
            y = np.linspace(
                (iy_min + 0.5) * self.cell_size + self.low[1],
                (iy_max + 0.5) * self.cell_size + self.low[1],
                iy_max - iy_min + 1,
            )
            xv, yv = np.meshgrid(x, y)
            mask = sh.contains_xy(obstacle, xv, yv)
            grid_2d[iy_min : iy_max + 1, ix_min : ix_max + 1][mask] = np.inf
        return grid_2d
