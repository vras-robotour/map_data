"""
RRT* (Rapidly-exploring Random Tree Star) path planning.

This module provides the RRTStar class for finding optimal paths in continuous
space with obstacle avoidance and cost-aware steering.
"""

import logging
import math
import random

import numpy as np
import shapely as sh
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from map_data.pathsolver.grid_astar import GRID_COST_WEIGHT, _bresenham_cells

logger = logging.getLogger(__name__)

# Rebuild the spatial index after this many new nodes are added since the last build.
# Balances rebuild cost (O(n log n)) against linear-scan cost for the unindexed tail.
_KDTREE_REBUILD_INTERVAL = 50

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
        transfer_id: str | None = None,
        improve_after_goal: bool = False,
        improve_iter: int = 200,
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
            Shapely geometries representing hard barriers. Not read directly
            (collision checks go through *obstacles_tree*); accepted so
            callers can pass the same pair they already have.
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
        transfer_id : str or None
            Optional identifier used to check for external cancellation
            signals during planning. Pass ``None`` to disable.
        improve_after_goal : bool
            If ``True``, continue iterating after the goal is first reached
            to find a lower-cost path via informed sampling and rewiring. If
            ``False`` (default), return as soon as the goal is reached.
        improve_iter : int
            With *improve_after_goal*, the most extra iterations spent
            improving once the goal is first reached (default ``200``);
            *max_iter* still caps the total.
        informed : bool
            If ``True`` (default), sample from the informed ellipse once a
            solution exists (see :meth:`_sample_informed`), shrinking as
            *improve_after_goal* lowers the best cost found.
        adaptive_radius : bool
            If ``True`` (default), shrink the rewiring radius as the tree
            grows per the RRT* asymptotic-optimality formula, instead of
            using a fixed *neighbor_radius*.

        """
        self.start = start
        self.goal = goal
        # obstacles itself is not read (collision checks go through
        # obstacles_tree); kept as a parameter so callers can pass the same
        # (geometries, tree) pair they already have.
        self.obstacles_tree = obstacles_tree
        self.grid = grid  # (Y, X)
        self.grid_shape = grid.shape
        self.low = np.array(low)
        self.grid_scale = grid_scale
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.nodes = [self.start]
        self.parent: dict[int, int | None] = {0: None}
        self.cost = {0: 0.0}
        # Children index (inverse of `parent`), needed to propagate cost
        # changes to descendants when a node is rewired.
        self._children: dict[int, set[int]] = {0: set()}

        self._nodes_buf = np.empty((max_iter + 2, 2), dtype=np.float64)
        self._nodes_buf[0] = self.start
        self.goal_tolerance = step_size
        self.traversability_threshold = traversability_threshold
        self.grid_cost_weight = grid_cost_weight
        self.transfer_id = transfer_id
        self.improve_after_goal = improve_after_goal
        self.improve_iter = improve_iter
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

    def _point_blocked(self, point: np.ndarray) -> bool:
        """
        Return ``True`` if *point* itself sits on an obstacle or a blocked cell.

        Used only to reject a freshly steered point before its segment from
        the nearest node is costed (:meth:`_segment_cost` already re-checks
        every cell the segment crosses, including this endpoint).
        """
        if self.obstacles_tree and len(
            self.obstacles_tree.query(Point(point), predicate="intersects"),
        ):
            return True
        ix = int((point[0] - self.low[0]) / self.grid_scale)
        iy = int((point[1] - self.low[1]) / self.grid_scale)
        ix = np.clip(ix, 0, self.grid_shape[1] - 1)
        iy = np.clip(iy, 0, self.grid_shape[0] - 1)
        return float(self.grid[iy, ix]) >= self.traversability_threshold

    def _set_parent(self, idx: int, parent_idx: int, new_cost: float) -> None:
        """
        Attach node *idx* to *parent_idx* with path cost *new_cost*.

        Keeps the tree bookkeeping consistent: the children index is updated,
        and when *idx* already had a cost (i.e. this is a rewire) the cost
        change is propagated to all of its descendants so their cached costs
        never go stale.
        """
        old_parent = self.parent.get(idx)
        if old_parent is not None:
            self._children[old_parent].discard(idx)
        self.parent[idx] = parent_idx
        self._children.setdefault(parent_idx, set()).add(idx)
        children = self._children.setdefault(idx, set())

        old_cost = self.cost.get(idx)
        self.cost[idx] = new_cost
        if old_cost is not None and children:
            delta = new_cost - old_cost
            if delta != 0.0:
                stack = list(children)
                while stack:
                    d = stack.pop()
                    self.cost[d] += delta
                    stack.extend(self._children.get(d, ()))

    def _sample_informed(self) -> np.ndarray:
        """
        Sample uniformly from the informed ellipse defined by the current best cost.

        The ellipse contains all points ``x`` where
        ``d(start, x) + d(x, goal) ≤ c_best``.  Uniform area sampling uses
        the polar parameterisation with ``r = √U``.
        """
        a = self._best_cost / 2.0
        b = math.sqrt(max(self._best_cost**2 - self._c_min**2, 0.0)) / 2.0
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

    def _get_near_nodes(self, new_point: np.ndarray, radius: float) -> list[int]:
        """
        Return indices of all tree nodes within *radius* of *new_point*.
        """
        n = len(self.nodes)
        new_idx = n - 1  # node just appended by the caller
        r2 = radius**2

        # self._kdtree is never None here: the caller always runs
        # _nearest_node() first this iteration, which builds it.
        # KD-tree covers [0, _kdtree_n); new_point is never included in it.
        result: list[int] = list(self._kdtree.query_ball_point(new_point, radius))  # type: ignore[union-attr]

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
        if (
            self.obstacles_tree
            and len(
                self.obstacles_tree.query(LineString([start, end]), predicate="intersects"),
            )
            > 0
        ):
            return True, float("inf")

        p1_grid = (
            int((start[0] - self.low[0]) / self.grid_scale),
            int((start[1] - self.low[1]) / self.grid_scale),
        )
        p2_grid = (
            int((end[0] - self.low[0]) / self.grid_scale),
            int((end[1] - self.low[1]) / self.grid_scale),
        )
        bres_line = _bresenham_cells(p1_grid, p2_grid)

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
        return False, float(np.linalg.norm(end - start) * (1.0 + avg_c * self.grid_cost_weight))

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
        goal_found_iter = None

        # ponytail: improvement is capped in iterations, not seconds; add a
        # wall-clock budget if per-iteration cost varies too much across maps.
        for i in range(self.max_iter):
            if _is_cancelled(self.transfer_id):
                return None
            if goal_found_iter is not None and i - goal_found_iter > self.improve_iter:
                break

            # Keep the informed-sampling ellipse in sync with the goal's true
            # cost: rewires (direct or propagated) may have improved it since
            # the goal-connection block last ran.
            if goal_idx is not None:
                self._best_cost = self.cost.get(goal_idx, float("inf"))

            rand_point = self.goal if random.random() < GOAL_SAMPLE_BIAS else self._sample_point()
            nearest_idx = self._nearest_node(rand_point)
            new_point = self._steer(self.nodes[nearest_idx], rand_point)

            if self._point_blocked(new_point):
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

            self._set_parent(new_idx, min_parent, min_cost)

            # Rewire
            for idx in near_indices:
                if idx == min_parent:
                    continue
                col, sc = self._segment_cost(new_point, self.nodes[idx])
                if not col:
                    new_c = self.cost[new_idx] + sc
                    if new_c < self.cost[idx]:
                        self._set_parent(idx, new_idx, new_c)

            if np.linalg.norm(new_point - self.goal) < self.goal_tolerance:
                col, sc = self._segment_cost(new_point, self.goal)
                if not col:
                    new_goal_cost = self.cost[new_idx] + sc
                    if goal_idx is None:
                        goal_found_iter = i
                        goal_idx = len(self.nodes)
                        self.nodes.append(self.goal)
                        self._nodes_buf[goal_idx] = self.goal
                    if new_goal_cost < self.cost.get(goal_idx, float("inf")):
                        self._set_parent(goal_idx, new_idx, new_goal_cost)
                        self._best_cost = new_goal_cost
                    if not self.improve_after_goal:
                        path = self._reconstruct_path(goal_idx)
                        return np.array(path)

        if goal_idx is not None:
            # Final sync: rewires in the last iteration may have improved the
            # goal's cost after the loop-top sync last ran.
            self._best_cost = self.cost.get(goal_idx, float("inf"))
            path = self._reconstruct_path(goal_idx)
            return np.array(path)
        return None

    def _reconstruct_path(self, goal_idx: int) -> list[np.ndarray]:
        """
        Walk the parent chain from *goal_idx* back to the root and return the path.

        Not simplified here: :meth:`~map_data.pathsolver.replan.ReplanPath._post_process_path`
        already runs a collision-checked Douglas-Peucker pass over the whole
        assembled route, so simplifying each RRT* segment first would just
        redo the same work at a smaller (and less effective) scale.
        """
        path = []
        curr: int | None = goal_idx
        while curr is not None:
            path.append(self.nodes[curr])
            curr = self.parent[curr]
        path.reverse()
        return path
