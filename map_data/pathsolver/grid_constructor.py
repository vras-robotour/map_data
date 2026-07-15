"""
Discretized cost grid construction for grid-based path planning.

This module builds a regular 2D grid of traversal costs over a bounding
box, combining a default off-path cost, per-way costs derived from
``highway``/``surface`` OSM tags with a smooth distance-based falloff, and
hard obstacle burning (infinite cost) for barrier geometries.
"""

from typing import TYPE_CHECKING

import numpy as np
import shapely as sh
from matplotlib.path import Path
from scipy.spatial import cKDTree

# Segments shorter than this (in the same units as grid coordinates, i.e.
# metres) are treated as coincident points rather than subdivided further.
TOLERANCE = 1e-3

if TYPE_CHECKING:
    from map_data.map_data import MapData


class PathGrid:
    """
    Regular grid of traversal costs over a rectangular world-coordinate area.

    The grid is a raster of square cells of side ``cell_size`` covering
    ``[low, high)``. Costs start at ``default_off_path_cost`` everywhere and
    are lowered near footway/road centerlines (weighted by ``highway_costs``
    and ``surface_costs``) and raised to infinity inside obstacle geometries.

    Attributes
    ----------
    low : np.ndarray
        ``(2,)`` array, the ``(x, y)`` world coordinate of the grid's lower
        corner.
    high : np.ndarray
        ``(2,)`` array, the ``(x, y)`` world coordinate of the grid's upper
        corner (exclusive — see :meth:`create_empty_grid`).
    cell_size : float
        Side length of a grid cell, in the same units as *low*/*high*
        (metres for UTM coordinates).
    highway_costs : dict of {str: float}
        Maps an OSM ``highway`` tag value (e.g. ``"footway"``, ``"path"``)
        to a base traversal cost. Unknown tags default to ``0.5`` (see
        :meth:`fill`).
    surface_costs : dict of {str: float}
        Maps an OSM ``surface`` tag value to an additive cost penalty.
        Unknown tags default to ``0.0``.
    default_off_path_cost : float
        Cost assigned to cells that are not within ``max_path_dist`` of any
        way, and the cost that a cell falls back towards at that distance.
    path_cost_cap : float
        Upper bound applied to a way's combined ``highway`` + ``surface``
        cost before distance falloff is applied.
    grid : np.ndarray
        Flat point representation of the grid. ``(N, 3)`` columns
        ``[x, y, 0]`` immediately after construction (see
        :meth:`create_empty_grid`); replaced by an ``(N, 4)`` array with
        columns ``[x, y, 0, cost]`` after :meth:`fill` is called.
    grid_2d_cache : np.ndarray or None
        Most recent rasterized, obstacle-burned cost grid returned by
        :meth:`fill`, or ``None`` before ``fill`` has been called.

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
        Initialize an empty cost grid over a rectangular world-coordinate area.

        Parameters
        ----------
        low : tuple of float
            ``(x, y)`` world coordinate of the grid's lower corner.
        high : tuple of float
            ``(x, y)`` world coordinate of the grid's upper corner.
        cell_size : float
            Side length of a grid cell (same units as *low*/*high*).
        highway_costs : dict of {str: float}
            Base traversal cost per OSM ``highway`` tag value.
        surface_costs : dict of {str: float}
            Additive cost penalty per OSM ``surface`` tag value.
        default_off_path_cost : float
            Cost for cells far from any way (default ``0.9``).
        path_cost_cap : float
            Maximum combined ``highway`` + ``surface`` cost for a way,
            before distance-based falloff (default ``0.85``).

        """
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
        """
        Build the flat point representation of the grid, with no costs yet.

        Cell centers are generated with ``np.arange(low, high, cell_size)``
        along each axis, so the grid covers ``[low, high)`` — the *high*
        corner itself is excluded, and (as with any float ``arange``) the
        exact point count can be off by one cell near the upper edge due to
        floating-point rounding. Points are laid out row-major with *x*
        varying fastest (`numpy.meshgrid` default ``"xy"`` indexing).

        Returns
        -------
        np.ndarray
            ``(N, 3)`` array with columns ``[x, y, 0]``; the third column is
            an unused placeholder (later replaced by a cost column in
            :meth:`fill`).

        """
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
        """
        Populate the grid with path and obstacle costs and rasterize it.

        This is the core cost-assignment routine. It proceeds in several
        layered passes, each of which can only *lower or override* the
        result of the previous one for a given cell — later passes win:

        1. Every grid cell starts at ``default_off_path_cost``.
        2. Obstacle geometries have any path/road geometry subtracted out
           of them first (so a barrier polygon that happens to overlap a
           footway does not block the footway itself), then the flat point
           grid (:attr:`grid`) has cells whose centers fall inside a
           (post-subtraction) obstacle set to cost ``1.0``.
        3. Ways in *highway_types* are sampled into a dense point cloud
           (roughly one point per :attr:`cell_size` of length, via linear
           interpolation), each point tagged with its way's cost
           (``min(path_cost_cap, highway_cost + surface_cost)``). Every
           grid cell within *max_path_dist* of its nearest sampled path
           point gets a cost that blends quadratically from the way cost
           (at distance 0) up to ``default_off_path_cost`` (at
           ``max_path_dist``), via
           ``way_cost + (default_off_path_cost - way_cost) * (dist / max_path_dist) ** 2``.
           This is combined with the previous value using ``min``, so a
           cell that is both near a path *and* inside an obstacle's
           bounding region ends up with the (lower) path cost in
           :attr:`grid` — hard blocking is only guaranteed by step 4 below.
        4. The flat grid is rasterized into a 2D array (:meth:`get_grid_2d`)
           and obstacles are burned into it as hard ``np.inf`` costs
           (:meth:`burn_obstacles`), independently of step 2/3. This is
           what actually prevents a planner from crossing an obstacle; the
           result is cached in :attr:`grid_2d_cache` and returned.

        Only ways whose geometry intersects the grid bounding box buffered
        by *max_path_dist* are considered, so paths just outside the grid
        can still influence the cost of edge cells.

        Parameters
        ----------
        map_data : MapData
            Source of way geometries (:attr:`~map_data.map_data.MapData.footways_list`,
            :attr:`~map_data.map_data.MapData.roads_list`) and node
            coordinates (via :meth:`~map_data.map_data.MapData.get_points`).
        obstacles : list of shapely geometry
            Hard-obstacle polygons (e.g. barriers/buildings) in the same
            world coordinates as the grid. ``Polygon``/``MultiPolygon``
            geometries are honoured (including interior holes); other
            geometry types are silently ignored by the obstacle-marking
            steps.
        highway_types : list of str, optional
            Which way categories to draw path costs from: any of
            ``"footway"``, ``"road"``. Defaults to ``["footway"]``.
        max_path_dist : float
            Radius of influence of a path centerline, in world units
            (default ``2.0``). Cells farther than this from every sampled
            path point keep the (obstacle-adjusted) off-path cost.

        Returns
        -------
        np.ndarray
            The rasterized cost grid, shape ``(num_y, num_x)``, with
            obstacle cells set to ``np.inf``. Also stored in
            :attr:`grid_2d_cache`.

        """
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
        """
        Rasterize the flat point grid (:attr:`grid`) into a 2D cost array.

        Each point's cell index is recovered with
        ``floor((coord - low) / cell_size)`` — the inverse of the
        construction in :meth:`create_empty_grid` — and clipped into range
        to absorb any floating-point spill at the upper edge. Note this
        reads column index 3 (cost) of :attr:`grid`, so it must be called
        after :meth:`fill` has replaced the initial ``(N, 3)`` grid with the
        ``(N, 4)`` cost-augmented one; calling it beforehand raises an
        ``IndexError``. Obstacle burning is *not* applied here — see
        :meth:`burn_obstacles`.

        Returns
        -------
        np.ndarray
            ``(num_y, num_x)`` array of costs, where ``num_x``/``num_y`` are
            ``ceil((high - low) / cell_size)`` along each axis. The array is
            transposed so the first axis indexes *y* and the second *x*
            (row-major raster convention), matching :meth:`burn_obstacles`.

        """
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
        """
        Set cells covered by obstacle geometries to ``np.inf`` in place.

        For each obstacle, only the sub-rectangle of cell indices overlapping
        its bounding box is tested (via `matplotlib.path.Path.contains_points`
        against a mesh of that sub-region's cell centers), so cost is roughly
        linear in obstacle count rather than total grid size. ``Polygon``
        interior rings are treated as holes (points inside a hole are not
        burned). Geometry types other than ``Polygon``/``MultiPolygon`` are
        silently skipped. Overlapping obstacles simply compound — a cell is
        ``inf`` if it falls inside *any* obstacle.

        Parameters
        ----------
        grid_2d : np.ndarray
            ``(num_y, num_x)`` cost grid, as returned by :meth:`get_grid_2d`.
            **Mutated in place** — this is not a copy-on-write operation.
        obstacles : list of shapely geometry
            Obstacle polygons in the same world coordinates as the grid.

        Returns
        -------
        np.ndarray
            The same array passed in *grid_2d* (returned for convenience),
            with obstacle-covered cells set to ``np.inf``. Unchanged (and
            returned as-is) if *obstacles* is empty.

        """
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
