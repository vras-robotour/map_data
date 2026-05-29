"""
RRT* (Rapidly-exploring Random Tree Star) path planning.

This module provides the RRTStar class for finding optimal paths in continuous
space with obstacle avoidance and cost-aware steering.
"""

import logging
import math
import random
from collections.abc import Iterator

import numpy as np
import shapely as sh
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from map_data.utils.config import load_config

logger = logging.getLogger(__name__)

# Rebuild the spatial index after this many new nodes are added since the last build.
# Balances rebuild cost (O(n log n)) against linear-scan cost for the unindexed tail.
_KDTREE_REBUILD_INTERVAL = 50

_DEFAULTS = load_config("planner_defaults.yaml")
GRID_COST_WEIGHT = _DEFAULTS.get("grid_cost_weight", 5.0)
GOAL_SAMPLE_BIAS = 0.1


class RRTStar:
    """
    Rapidly-exploring Random Tree Star (RRT*) path planner.

    Builds a collision-free tree by randomly sampling the free space and
    rewiring edges to minimise path cost. The planner is asymptotically
    optimal: given enough iterations it converges to the shortest feasible
    path.

    Collision checking uses two sources simultaneously:

    - A Shapely STRtree of barrier polygons (hard obstacles).
    - A 2-D cost grid where cells at or above *traversability_threshold*
      are treated as blocked.

    Traversable cells contribute a weighted cost to the edge cost, so the
    planner naturally prefers low-cost corridors (e.g. footways) over open
    terrain.
    """

    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[sh.geometry.base.BaseGeometry],
        obstacles_tree: STRtree | None,
        grid: np.ndarray,
        low: tuple[float, float],
        grid_scale: float = 1.0,
        max_iter: int = 2000,
        step_size: float = 2.0,
        neighbor_radius: float = 5.0,
        traversability_threshold: float = 10.0,  # inf is blocked, high values are expensive
        grid_cost_weight: float = GRID_COST_WEIGHT,
        *,
        simplify: bool = True,
        transfer_id: str | None = None,
        improve_after_goal: bool = False,
        informed: bool = True,
        adaptive_radius: bool = True,
    ) -> None:
        """
        Initialize the RRT* planner.

        Parameters
        ----------
        start : np.ndarray
            Starting position as a 2-element array ``[x, y]`` in world
            (UTM) coordinates.
        goal : np.ndarray
            Goal position as a 2-element array ``[x, y]`` in world
            coordinates.
        obstacles : list
            List of Shapely geometries representing hard barriers. Used
            together with *obstacles_tree* for fast spatial queries.
        obstacles_tree : STRtree or None
            Pre-built Shapely STRtree index over *obstacles*. Pass ``None``
            to skip polygon-based collision checking (grid only).
        grid : np.ndarray
            2-D cost array with shape ``(Y, X)``. A value of ``0.0`` means
            fully free; values at or above *traversability_threshold* are
            blocked. Intermediate values increase edge cost.
        low : tuple of float
            ``(min_x, min_y)`` corner of the grid in world coordinates.
        grid_scale : float
            Metres per grid cell (default ``1.0``).
        max_iter : int
            Maximum number of RRT* iterations (default ``2000``).
        step_size : float
            Maximum distance the tree extends toward a sampled point per
            iteration in metres (default ``2.0``).
        neighbor_radius : float
            Radius in metres within which nearby nodes are considered for
            rewiring (default ``5.0``).
        traversability_threshold : float
            Grid cost at which a cell is considered an obstacle
            (default ``10.0``). ``np.inf`` marks cells as hard obstacles.
        simplify : bool
            If ``True``, post-process the raw node path with a greedy
            line-of-sight simplification before returning (default ``True``).
        transfer_id : str or None
            Optional identifier used to check for external cancellation
            signals during planning. Pass ``None`` to disable.
        improve_after_goal : bool
            If ``True``, continue iterating after the goal is first reached
            to find a lower-cost path. If ``False`` (default), return as
            soon as the goal is reached.

        """
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.obstacles_tree = obstacles_tree
        self.grid = grid  # (Y, X)
        self.grid_shape = grid.shape
        self.low = np.array(low)
        self.grid_scale = grid_scale
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.nodes = [self.start]
        self.parent = {0: None}
        self.cost = {0: 0.0}

        self._nodes_buf = np.empty((max_iter + 2, 2), dtype=np.float64)
        self._nodes_buf[0] = self.start
        self.goal_tolerance = step_size
        self.traversability_threshold = traversability_threshold
        self.grid_cost_weight = grid_cost_weight
        self.simplify = simplify
        self.transfer_id = transfer_id
        self.improve_after_goal = improve_after_goal
        self.informed = informed
        self.adaptive_radius = adaptive_radius
        self._best_cost: float = float("inf")
        self._kdtree: cKDTree | None = None
        self._kdtree_n: int = 0

        # Limit sampling area
        dist = np.linalg.norm(self.goal - self.start)
        margin = max(dist * 0.5, step_size * 10)
        self._sample_min = np.minimum(self.start, self.goal) - margin
        self._sample_max = np.maximum(self.start, self.goal) + margin

        # Clip to grid
        grid_max_x = self.low[0] + self.grid_shape[1] * grid_scale
        grid_max_y = self.low[1] + self.grid_shape[0] * grid_scale
        self._sample_min = np.maximum(self._sample_min, self.low)
        self._sample_max = np.minimum(self._sample_max, [grid_max_x, grid_max_y])

        # Informed RRT*: precompute ellipse geometry
        d = self.goal - self.start
        self._c_min: float = float(np.linalg.norm(d))
        self._ellipse_center: np.ndarray = (self.start + self.goal) / 2.0
        theta = math.atan2(float(d[1]), float(d[0]))
        ct, st = math.cos(theta), math.sin(theta)
        self._C_be: np.ndarray = np.array([[ct, -st], [st, ct]])

        # Adaptive radius: gamma* for 2-D from the asymptotic optimality formula
        sample_area = float(np.prod(self._sample_max - self._sample_min))
        self._gamma: float = 2.449 * math.sqrt(max(sample_area, 1.0) / math.pi)

        # Precompute traversable cells for faster sampling
        _xi_lo = max(0, int((self._sample_min[0] - self.low[0]) / grid_scale))
        _xi_hi = min(
            self.grid_shape[1],
            int(np.ceil((self._sample_max[0] - self.low[0]) / grid_scale)),
        )
        _yi_lo = max(0, int((self._sample_min[1] - self.low[1]) / grid_scale))
        _yi_hi = min(
            self.grid_shape[0],
            int(np.ceil((self._sample_max[1] - self.low[1]) / grid_scale)),
        )

        _sub = self.grid[_yi_lo:_yi_hi, _xi_lo:_xi_hi]
        _ys, _xs = np.where(_sub < self.traversability_threshold)
        if len(_xs) > 0:
            self._trav_xs = (_xs + _xi_lo) * grid_scale + self.low[0]
            self._trav_ys = (_ys + _yi_lo) * grid_scale + self.low[1]
        else:
            self._trav_xs = None
            self._trav_ys = None

    def _is_collision(self, point1: np.ndarray, point2: np.ndarray | None = None) -> bool:
        """
        Return ``True`` if a point or segment intersects an obstacle.

        Checks both the Shapely obstacle polygons (via STRtree) and the cost
        grid (via Bresenham rasterisation). A point is in collision if its
        grid cost meets or exceeds *traversability_threshold*; a segment is
        in collision if any traversed cell does.
        """
        if self.obstacles_tree:
            geom = Point(point1) if point2 is None else LineString([point1, point2])
            if len(self.obstacles_tree.query(geom, predicate="intersects")) > 0:
                return True

        if point2 is None:
            return self._get_grid_cost(point1) >= self.traversability_threshold

        p1_grid = (
            int((point1[0] - self.low[0]) / self.grid_scale),
            int((point1[1] - self.low[1]) / self.grid_scale),
        )
        p2_grid = (
            int((point2[0] - self.low[0]) / self.grid_scale),
            int((point2[1] - self.low[1]) / self.grid_scale),
        )

        for px, py in self._bresenham(p1_grid, p2_grid):
            in_bounds = 0 <= px < self.grid_shape[1] and 0 <= py < self.grid_shape[0]
            if in_bounds and self.grid[py, px] >= self.traversability_threshold:
                return True
        return False

    def _bresenham(
        self, start: tuple[int, int], goal: tuple[int, int],
    ) -> Iterator[tuple[int, int]]:
        """
        Yield integer grid cells along the line from *start* to *goal* (Bresenham).
        """
        x0, y0 = start
        x1, y1 = goal
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                yield (x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                yield (x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        yield (x1, y1)

    def _get_grid_cost(self, point: np.ndarray) -> float:
        """
        Return the grid cost at *point*, clamped to grid bounds.
        """
        ix = int((point[0] - self.low[0]) / self.grid_scale)
        iy = int((point[1] - self.low[1]) / self.grid_scale)
        ix = np.clip(ix, 0, self.grid_shape[1] - 1)
        iy = np.clip(iy, 0, self.grid_shape[0] - 1)
        return float(self.grid[iy, ix])

    def _sample_informed(self) -> np.ndarray:
        """
        Sample uniformly from the informed ellipse defined by the current best cost.

        The ellipse contains all points ``x`` where
        ``d(start, x) + d(x, goal) ≤ c_best``.  Uniform area sampling uses
        the polar parameterisation with ``r = √U``.
        """
        a = self._best_cost / 2.0
        b = math.sqrt(max(self._best_cost ** 2 - self._c_min ** 2, 0.0)) / 2.0
        angle = random.uniform(0.0, 2.0 * math.pi)
        r = math.sqrt(random.random())
        x_e = np.array([a * r * math.cos(angle), b * r * math.sin(angle)])
        point = self._ellipse_center + self._C_be @ x_e
        return np.clip(point, self._sample_min, self._sample_max)

    def _sample_point(self) -> np.ndarray:
        """
        Sample a random point.

        When a solution exists and *informed* is enabled, samples uniformly
        from the informed ellipse.  Otherwise biases toward traversable grid
        cells 90 % of the time.
        """
        if self.informed and self._best_cost < float("inf"):
            return self._sample_informed()
        if self._trav_xs is not None and random.random() > GOAL_SAMPLE_BIAS:
            idx = random.randrange(len(self._trav_xs))
            return np.array(
                [
                    self._trav_xs[idx] + random.uniform(-self.grid_scale / 2, self.grid_scale / 2),
                    self._trav_ys[idx] + random.uniform(-self.grid_scale / 2, self.grid_scale / 2),
                ],
            )
        return np.array(
            [
                random.uniform(self._sample_min[0], self._sample_max[0]),
                random.uniform(self._sample_min[1], self._sample_max[1]),
            ],
        )

    def _nearest_node(self, point: np.ndarray) -> int:
        """
        Return the index of the tree node closest to *point*.

        Uses a lazily rebuilt KD-tree for the bulk of the tree, plus a
        linear scan over nodes added since the last rebuild.
        """
        n = len(self.nodes)
        if self._kdtree is None or n - self._kdtree_n >= _KDTREE_REBUILD_INTERVAL:
            self._kdtree = cKDTree(self._nodes_buf[:n])
            self._kdtree_n = n

        _, best_idx = self._kdtree.query(point)
        best_d2 = float(((self._nodes_buf[best_idx] - point) ** 2).sum())

        # Linear scan over nodes added since the last rebuild
        for i in range(self._kdtree_n, n):
            d2 = float(((self._nodes_buf[i] - point) ** 2).sum())
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        return int(best_idx)

    def _steer(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Return a point at most *step_size* metres from *start* toward *target*.
        """
        direction = target - start
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return target
        return start + (direction / dist) * self.step_size

    def _get_near_nodes(self, new_point: np.ndarray, radius: float | None = None) -> list[int]:
        """
        Return indices of all tree nodes within *radius* of *new_point*.

        *radius* defaults to ``self.neighbor_radius`` when not supplied.
        """
        r = radius if radius is not None else self.neighbor_radius
        n = len(self.nodes)
        new_idx = n - 1  # node just appended by the caller
        r2 = r ** 2

        if self._kdtree is None:
            sq_dists = ((self._nodes_buf[:n] - new_point) ** 2).sum(axis=1)
            return list(np.where((sq_dists < r2) & (sq_dists > 0))[0])

        # KD-tree covers [0, _kdtree_n); _nearest_node always rebuilds first so
        # new_point is never included in the tree.
        result: list[int] = list(self._kdtree.query_ball_point(new_point, r))

        # Linear scan over nodes added since the last rebuild, excluding new_point itself
        for i in range(self._kdtree_n, n):
            if i == new_idx:
                continue
            d2 = float(((self._nodes_buf[i] - new_point) ** 2).sum())
            if d2 < r2:
                result.append(i)

        return result

    def _segment_cost(self, start: np.ndarray, end: np.ndarray) -> tuple[bool, float]:
        """
        Compute the cost of the segment from *start* to *end*.

        Returns
        -------
        tuple of (bool, float)
            ``(collision, cost)`` where *collision* is ``True`` if the
            segment intersects an obstacle or blocked grid cell, and *cost*
            is the weighted traversal cost ``dist * (1 + avg_grid_cost * 5)``.
            Returns ``(True, inf)`` on collision.

        """
        if self.obstacles_tree and len(
            self.obstacles_tree.query(LineString([start, end]), predicate="intersects"),
        ) > 0:
            return True, float("inf")

        p1_grid = (
            int((start[0] - self.low[0]) / self.grid_scale),
            int((start[1] - self.low[1]) / self.grid_scale),
        )
        p2_grid = (
            int((end[0] - self.low[0]) / self.grid_scale),
            int((end[1] - self.low[1]) / self.grid_scale),
        )
        bres_line = self._bresenham(p1_grid, p2_grid)

        total_grid_cost = 0.0
        count = 0
        for x, y in bres_line:
            if 0 <= x < self.grid_shape[1] and 0 <= y < self.grid_shape[0]:
                c = self.grid[y, x]
                if c >= self.traversability_threshold:
                    return True, float("inf")
                total_grid_cost += c
                count += 1

        avg_c = total_grid_cost / count if count > 0 else 0.0
        # Cost = dist * (1 + avg_grid_cost * penalty)  # noqa: ERA001
        # We use grid_cost_weight to match A* logic
        return False, np.linalg.norm(end - start) * (1.0 + avg_c * self.grid_cost_weight)

    def find_path(self) -> np.ndarray | None:
        """
        Run the RRT* algorithm and return the planned path.

        Iterates up to *max_iter* times, growing the tree from ``start``
        toward randomly sampled points and rewiring edges to reduce cost.
        The goal is sampled directly 10 % of the time to encourage
        convergence.

        Returns
        -------
        np.ndarray or None
            Path as an ``(N, 2)`` array of ``[x, y]`` world coordinates,
            or ``None`` if no collision-free path was found within the
            iteration budget or if planning was cancelled via *transfer_id*.

        """
        from .replan import _is_cancelled

        goal_idx = None

        for _ in range(self.max_iter):
            if _is_cancelled(self.transfer_id):
                return None

            rand_point = self.goal if random.random() < GOAL_SAMPLE_BIAS else self._sample_point()
            nearest_idx = self._nearest_node(rand_point)
            new_point = self._steer(self.nodes[nearest_idx], rand_point)

            if self._is_collision(new_point):
                continue

            collision, nearest_seg_cost = self._segment_cost(self.nodes[nearest_idx], new_point)
            if collision:
                continue

            new_idx = len(self.nodes)
            self.nodes.append(new_point)
            self._nodes_buf[new_idx] = new_point
            min_cost = self.cost[nearest_idx] + nearest_seg_cost
            min_parent = nearest_idx

            if self.adaptive_radius:
                n_eff = max(new_idx, int(math.e) + 1)
                r = min(self._gamma * math.sqrt(math.log(n_eff) / n_eff), self.neighbor_radius)
            else:
                r = self.neighbor_radius
            near_indices = self._get_near_nodes(new_point, r)
            for idx in near_indices:
                # Reuse already-computed cost for the nearest node
                if idx == nearest_idx:
                    col, sc = False, nearest_seg_cost
                else:
                    col, sc = self._segment_cost(self.nodes[idx], new_point)
                if not col:
                    c = self.cost[idx] + sc
                    if c < min_cost:
                        min_cost = c
                        min_parent = idx

            self.parent[new_idx] = min_parent
            self.cost[new_idx] = min_cost

            # Rewire
            for idx in near_indices:
                if idx == min_parent:
                    continue
                col, sc = self._segment_cost(new_point, self.nodes[idx])
                if not col:
                    new_c = self.cost[new_idx] + sc
                    if new_c < self.cost[idx]:
                        self.parent[idx] = new_idx
                        self.cost[idx] = new_c

            if np.linalg.norm(new_point - self.goal) < self.goal_tolerance:
                col, sc = self._segment_cost(new_point, self.goal)
                if not col:
                    new_goal_cost = self.cost[new_idx] + sc
                    if goal_idx is None:
                        goal_idx = len(self.nodes)
                        self.nodes.append(self.goal)
                        self._nodes_buf[goal_idx] = self.goal
                    if new_goal_cost < self.cost.get(goal_idx, float("inf")):
                        self.parent[goal_idx] = new_idx
                        self.cost[goal_idx] = new_goal_cost
                        self._best_cost = new_goal_cost
                    if not self.improve_after_goal:
                        path = self._reconstruct_path(goal_idx)
                        return np.array(path)

        if goal_idx is not None:
            path = self._reconstruct_path(goal_idx)
            return np.array(path)
        return None

    def _reconstruct_path(self, goal_idx: int) -> list[np.ndarray]:
        """
        Walk the parent chain from *goal_idx* back to the root and return the path.
        """
        path = []
        curr = goal_idx
        while curr is not None:
            path.append(self.nodes[curr])
            curr = self.parent[curr]
        path.reverse()
        if self.simplify and len(path) > 2:
            return self._simplify_path(path)
        return path

    def _simplify_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """
        Greedily remove intermediate waypoints that have line-of-sight to a later node.
        """
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        curr = 0
        while curr < len(path) - 1:
            next_best = curr + 1
            for i in range(len(path) - 1, curr + 1, -1):
                col, _ = self._segment_cost(path[curr], path[i])
                if not col:
                    next_best = i
                    break
            simplified.append(path[next_best])
            curr = next_best
        return simplified


# Example usage
if __name__ == "__main__":
    from shapely.geometry import Polygon

    # Define start and goal points
    start = (0.0, 0.0)
    goal = (10.0, 10.0)

    # Define obstacles as Shapely polygons
    obstacles = [
        Polygon([(2, 2), (2, 4), (4, 4), (4, 2)]),
        Polygon([(6, 6), (6, 8), (8, 8), (8, 6)]),
        Polygon([(3, 7), (3, 9), (5, 9), (5, 7)]),
    ]

    # Define a 10x10 grid with 0-1 traversability costs
    grid = np.zeros((100, 100), dtype=float)
    grid[40:60, 40:60] = 1  # Non-traversable region

    # Initialize and run RRT*
    rrt_star = RRTStar(
        start,
        goal,
        obstacles,
        obstacles_tree=None,
        grid=grid,
        low=(0.0, 0.0),
        max_iter=2000,
        step_size=0.1,
        neighbor_radius=1.5,
        grid_scale=0.1,
    )
    path = rrt_star.find_path()

    if path is not None:
        logger.info("Path found:")
        for point in path:
            logger.info("(%.2f, %.2f)", point[0], point[1])
    else:
        logger.info("No path found.")
