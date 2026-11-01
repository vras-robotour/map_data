import array
import heapq
import logging
import math
from collections.abc import Callable, Iterator

import numpy as np
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

#: Fallback for a direct call; ReplanPath passes planner_defaults.yaml's grid_cost_weight.
GRID_COST_WEIGHT = 5.0


def _bresenham_cells(
    start: tuple[int, int],
    goal: tuple[int, int],
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


def nearest_free_cell(
    grid: np.ndarray,
    point: np.ndarray,
    low: tuple[float, float],
    cs: float,
    radius: float,
) -> np.ndarray | None:
    """
    *point* if its cell is free, else the centre of the nearest finite cell within
    *radius* metres of it, or ``None`` when every cell in reach is blocked.
    """
    ny, nx = grid.shape
    ix = int(np.floor((point[0] - low[0]) / cs))
    iy = int(np.floor((point[1] - low[1]) / cs))
    if 0 <= ix < nx and 0 <= iy < ny and np.isfinite(grid[iy, ix]):
        return np.asarray(point, dtype=float)
    r = int(np.ceil(radius / cs))
    x0, x1 = max(ix - r, 0), min(ix + r + 1, nx)
    y0, y1 = max(iy - r, 0), min(iy + r + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.nonzero(np.isfinite(grid[y0:y1, x0:x1]))
    cx = (xs + x0 + 0.5) * cs + low[0]
    cy = (ys + y0 + 0.5) * cs + low[1]
    d2 = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
    if not len(d2) or d2.min() > radius**2:
        return None
    k = int(np.argmin(d2))
    return np.array([cx[k], cy[k]])


def simplify_path_checked(
    path: np.ndarray,
    tolerance: float,
    segment_blocked: Callable[[np.ndarray, np.ndarray], bool],
) -> np.ndarray:
    """
    Douglas-Peucker simplification that never shortcuts through an obstacle.

    Runs ``LineString.simplify(tolerance)`` and then collision-checks every
    shortcut it introduced: for each pair of consecutive kept vertices that
    skips over removed vertices, *segment_blocked* is consulted, and if the
    chord is blocked the original vertices for that span are reinstated.

    Parameters
    ----------
    path : np.ndarray
        ``(N, 2)`` array of path vertices in world coordinates.
    tolerance : float
        Douglas-Peucker tolerance passed to ``LineString.simplify``.
    segment_blocked : callable
        ``segment_blocked(p1, p2) -> bool`` returning ``True`` if the straight
        segment between the two points collides with an obstacle.

    Returns
    -------
    np.ndarray
        The simplified path with any colliding shortcut replaced by the
        original vertices it tried to skip. Always starts and ends with the
        original endpoints and is collision-free wherever *path* was.

    """
    if len(path) <= 2:
        return path
    simplified = np.asarray(LineString(path).simplify(tolerance).coords)
    if len(simplified) >= len(path):
        return path

    # Douglas-Peucker keeps a subset of the original vertices; map each kept
    # vertex back to its index in the original path.
    orig_indices = []
    j = 0
    for vx, vy in simplified:
        while j < len(path) and not (path[j][0] == vx and path[j][1] == vy):
            j += 1
        if j >= len(path):
            # Unexpected vertex mismatch — keep the known-safe unsimplified path.
            return path
        orig_indices.append(j)

    out = [path[orig_indices[0]]]
    for k in range(len(orig_indices) - 1):
        i0, i1 = orig_indices[k], orig_indices[k + 1]
        if i1 - i0 > 1 and segment_blocked(path[i0], path[i1]):
            # The shortcut chords into an obstacle: keep the original vertices.
            out.extend(path[i0 + 1 : i1 + 1])
        else:
            out.append(path[i1])
    return np.array(out)


def grid_astar(
    grid: np.ndarray,
    start_utm: tuple[float, float] | np.ndarray,
    goal_utm: tuple[float, float] | np.ndarray,
    low: tuple[float, float],
    cs: float,
    *,
    grid_cost_weight: float = GRID_COST_WEIGHT,
) -> np.ndarray | None:
    """
    8-connected A* over a cost grid; a step costs ``length * (1 + cost * grid_cost_weight)``.

    Parameters
    ----------
    grid : np.ndarray
        ``(Y, X)`` cell costs, ``inf`` is blocked.
    start_utm : tuple or np.ndarray
        Start coordinates (easting, northing) in UTM.
    goal_utm : tuple or np.ndarray
        Goal coordinates (easting, northing) in UTM.
    low : tuple of float
        ``(min_x, min_y)`` of the grid in UTM.
    cs : float
        Cell size (m).
    grid_cost_weight : float, optional
        Weight multiplier for non-zero cell costs (default 5.0).

    Returns
    -------
    np.ndarray or None
        Path of cell centers in world coordinates, or ``None`` if there is none.

    """
    ny, nx = grid.shape

    def to_idx(p: tuple[float, float] | np.ndarray) -> tuple[int, int]:
        ix = int(np.floor((p[0] - low[0]) / cs))
        iy = int(np.floor((p[1] - low[1]) / cs))
        return ix, iy

    start_ix, start_iy = to_idx(start_utm)
    goal_ix, goal_iy = to_idx(goal_utm)

    if not (0 <= goal_ix < nx and 0 <= goal_iy < ny):
        logger.warning("Goal %s is outside the grid bounds; cannot plan path.", goal_utm)
        return None
    if not (0 <= start_ix < nx and 0 <= start_iy < ny):
        logger.warning("Start %s is outside the grid bounds; cannot plan path.", start_utm)
        return None

    if start_ix == goal_ix and start_iy == goal_iy:
        return np.array([start_utm, goal_utm])

    # Pad with inf so neighbour lookups need no bounds checks. Cells are numbered
    # column-major, (ix + 1) * p_ny + (iy + 1), so ordering by the flat index breaks
    # heap ties by (ix, iy).
    p_nx = nx + 2
    p_ny = ny + 2
    padded_costs = np.full((p_nx, p_ny), np.inf, dtype=np.float32)
    padded_costs[1:-1, 1:-1] = (1.0 + grid * grid_cost_weight).T
    flat_costs = padded_costs.ravel().tolist()
    # A diagonal step's cost, sqrt(2) * cost, as float32 like every g score below.
    diag_costs = array.array("f", (np.float32(math.sqrt(2)) * padded_costs).tobytes())
    # Straight-line heuristic of every cell, in cells.
    hx = (np.arange(p_nx) - 1 - goal_ix) ** 2
    hy = (np.arange(p_ny) - 1 - goal_iy) ** 2
    heuristic = array.array("d", np.sqrt((hx[:, None] + hy[None, :]).astype(float)).tobytes())

    # Plain Python floats: numpy scalar arithmetic would dominate this inner loop.
    # Every g score is still rounded to float32 through _f32, so the expansion
    # order matches float32 arithmetic on the float32 costs.
    _f32 = array.array("f", [0.0])
    g_scores = array.array("f", [math.inf]) * (p_ny * p_nx)
    parents = array.array("i", [-1]) * (p_ny * p_nx)

    start_flat = (start_ix + 1) * p_ny + (start_iy + 1)
    goal_flat = (goal_ix + 1) * p_ny + (goal_iy + 1)
    g_scores[start_flat] = 0.0

    # Priority queue: (f_score, g_score, flat index)
    pq = [(heuristic[start_flat], 0.0, start_flat)]
    inf = math.inf
    heappush, heappop = heapq.heappush, heapq.heappop

    while pq:
        _f, g_pushed, u = heappop(pq)

        g = g_scores[u]
        if g < g_pushed - 1e-4:
            continue

        if u == goal_flat:
            path_indices = []
            curr = u
            while curr != -1:
                c_ix, c_iy = divmod(curr, p_ny)
                path_indices.append((c_ix - 1, c_iy - 1))
                curr = parents[curr]
            path_indices.reverse()

            # Cell centers, matching the grid's sampling convention.
            path = np.array(
                [[(ix + 0.5) * cs + low[0], (iy + 0.5) * cs + low[1]] for ix, iy in path_indices],
            )
            return path

        down, up, left, right = u - 1, u + 1, u - p_ny, u + p_ny
        c_down, c_up = flat_costs[down], flat_costs[up]
        c_left, c_right = flat_costs[left], flat_costs[right]
        # Rounding to float32 is monotonic: a step that is no shorter in float64 cannot
        # be shorter rounded, which skips the rounding for most (already reached) cells.
        # An inf cost fails the first test on its own.
        # Unrolled: a loop over the eight neighbours costs ~10% of the search.
        if g + c_down < g_scores[down]:
            _f32[0] = g + c_down
            new_g = _f32[0]
            if new_g < g_scores[down]:
                g_scores[down] = new_g
                parents[down] = u
                heappush(pq, (new_g + heuristic[down], new_g, down))
        if g + c_up < g_scores[up]:
            _f32[0] = g + c_up
            new_g = _f32[0]
            if new_g < g_scores[up]:
                g_scores[up] = new_g
                parents[up] = u
                heappush(pq, (new_g + heuristic[up], new_g, up))
        if g + c_left < g_scores[left]:
            _f32[0] = g + c_left
            new_g = _f32[0]
            if new_g < g_scores[left]:
                g_scores[left] = new_g
                parents[left] = u
                heappush(pq, (new_g + heuristic[left], new_g, left))
        if g + c_right < g_scores[right]:
            _f32[0] = g + c_right
            new_g = _f32[0]
            if new_g < g_scores[right]:
                g_scores[right] = new_g
                parents[right] = u
                heappush(pq, (new_g + heuristic[right], new_g, right))
        # Diagonal moves must not slip between two corner-touching blocked
        # cells: both edge-adjacent cells have to be free.
        if c_left != inf and c_down != inf:
            v = left - 1
            cost = diag_costs[v]
            if g + cost < g_scores[v]:
                _f32[0] = g + cost
                new_g = _f32[0]
                if new_g < g_scores[v]:
                    g_scores[v] = new_g
                    parents[v] = u
                    heappush(pq, (new_g + heuristic[v], new_g, v))
        if c_left != inf and c_up != inf:
            v = left + 1
            cost = diag_costs[v]
            if g + cost < g_scores[v]:
                _f32[0] = g + cost
                new_g = _f32[0]
                if new_g < g_scores[v]:
                    g_scores[v] = new_g
                    parents[v] = u
                    heappush(pq, (new_g + heuristic[v], new_g, v))
        if c_right != inf and c_down != inf:
            v = right - 1
            cost = diag_costs[v]
            if g + cost < g_scores[v]:
                _f32[0] = g + cost
                new_g = _f32[0]
                if new_g < g_scores[v]:
                    g_scores[v] = new_g
                    parents[v] = u
                    heappush(pq, (new_g + heuristic[v], new_g, v))
        if c_right != inf and c_up != inf:
            v = right + 1
            cost = diag_costs[v]
            if g + cost < g_scores[v]:
                _f32[0] = g + cost
                new_g = _f32[0]
                if new_g < g_scores[v]:
                    g_scores[v] = new_g
                    parents[v] = u
                    heappush(pq, (new_g + heuristic[v], new_g, v))

    return None
