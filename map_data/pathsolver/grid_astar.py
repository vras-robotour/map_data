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
    low : tuple
        ``(min_x, min_y)`` of the grid.
    cs : float
        Cell size.

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

    # Pad with inf so neighbour lookups need no bounds checks.
    costs = 1.0 + grid * grid_cost_weight
    padded_costs = np.full((ny + 2, nx + 2), np.inf, dtype=np.float32)
    padded_costs[1:-1, 1:-1] = costs

    # Flattened grid size with padding
    p_nx = nx + 2
    p_ny = ny + 2

    # Plain Python floats: numpy scalar arithmetic would dominate this inner loop.
    # Every g score is still rounded to float32 through _f32, so the expansion
    # order matches float32 arithmetic on the float32 costs.
    _f32 = array.array("f", [0.0])
    g_scores = array.array("f", [math.inf]) * (p_ny * p_nx)
    parents = np.full(p_ny * p_nx, -1, dtype=np.int32)

    start_flat = (start_iy + 1) * p_nx + (start_ix + 1)
    goal_flat = (goal_iy + 1) * p_nx + (goal_ix + 1)
    g_scores[start_flat] = 0.0

    # Priority queue: (f_score, g_score, ix, iy)
    h0 = math.sqrt((start_ix - goal_ix) ** 2 + (start_iy - goal_iy) ** 2)
    pq = [(h0, 0.0, start_ix, start_iy)]

    # Neighbor offsets in flattened padded grid (dy * p_nx + dx, dist, ortho_offsets).
    # For diagonal moves, ortho_offsets are the two edge-adjacent cells the move
    # passes between; both must be traversable so the path cannot cut through
    # the corner where two blocked cells touch diagonally.
    neighbors_data = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            ortho_offsets = (dy * p_nx, dx) if dx != 0 and dy != 0 else ()
            dist = float(np.float32(math.sqrt(dx**2 + dy**2)))
            neighbors_data.append((dy * p_nx + dx, dist, ortho_offsets))

    flat_costs = padded_costs.ravel().tolist()

    while pq:
        _f, g_pushed, ix, iy = heapq.heappop(pq)

        u_flat = (iy + 1) * p_nx + (ix + 1)
        if g_scores[u_flat] < g_pushed - 1e-4:
            continue

        if u_flat == goal_flat:
            # Path found, reconstruct
            path_indices = []
            curr = u_flat
            while curr != -1:
                c_iy, c_ix = divmod(curr, p_nx)
                path_indices.append((c_ix - 1, c_iy - 1))
                curr = parents[curr]
            path_indices.reverse()

            # Cell centers, matching the grid's sampling convention.
            path = np.array(
                [[(ix + 0.5) * cs + low[0], (iy + 0.5) * cs + low[1]] for ix, iy in path_indices],
            )
            return path

        current_g = g_scores[u_flat]

        for offset, dist, ortho_offsets in neighbors_data:
            v_flat = u_flat + offset
            cost_val = flat_costs[v_flat]

            if cost_val == math.inf:
                continue

            # Diagonal moves must not slip between two corner-touching
            # blocked cells: both edge-adjacent cells have to be free.
            if ortho_offsets:
                o1, o2 = ortho_offsets
                if flat_costs[u_flat + o1] == math.inf or flat_costs[u_flat + o2] == math.inf:
                    continue

            _f32[0] = dist * cost_val
            _f32[0] = current_g + _f32[0]
            new_g = _f32[0]
            if new_g < g_scores[v_flat]:
                g_scores[v_flat] = new_g
                parents[v_flat] = u_flat
                v_iy, v_ix = divmod(v_flat, p_nx)
                h = math.sqrt((v_ix - 1 - goal_ix) ** 2 + (v_iy - 1 - goal_iy) ** 2)
                heapq.heappush(pq, (new_g + h, new_g, v_ix - 1, v_iy - 1))

    return None
