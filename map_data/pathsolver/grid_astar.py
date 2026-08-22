import heapq
import logging
from collections.abc import Callable, Iterator

import numpy as np
from shapely.geometry import LineString

from map_data.utils.config import load_config

logger = logging.getLogger(__name__)

_DEFAULTS = load_config("planner_defaults.yaml")
GRID_COST_WEIGHT = _DEFAULTS.get("grid_cost_weight", 5.0)


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


def grid_segment_blocked(
    grid: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    low: tuple[float, float],
    cs: float,
) -> bool:
    """
    Return ``True`` if the world-coordinate segment *p1*→*p2* crosses a blocked cell.

    The segment is rasterized onto *grid* (cell size *cs*, origin *low*) with
    Bresenham; any traversed cell with an ``inf`` cost blocks the segment.
    """
    ny, nx = grid.shape
    c1 = (int(np.floor((p1[0] - low[0]) / cs)), int(np.floor((p1[1] - low[1]) / cs)))
    c2 = (int(np.floor((p2[0] - low[0]) / cs)), int(np.floor((p2[1] - low[1]) / cs)))
    for px, py in _bresenham_cells(c1, c2):
        if 0 <= px < nx and 0 <= py < ny and np.isinf(grid[py, px]):
            return True
    return False


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
    simplify_path: bool = True,
    grid_cost_weight: float = GRID_COST_WEIGHT,
) -> np.ndarray | None:
    """
    Optimized A* search on a 2D grid.

    Parameters
    ----------
    grid : np.ndarray
        2D grid of costs (Y, X). ``inf`` means blocked.
    start_utm : tuple or np.ndarray
        Starting point in UTM coordinates.
    goal_utm : tuple or np.ndarray
        Goal point in UTM coordinates.
    low : tuple
        ``(min_x, min_y)`` of the grid in UTM metres.
    cs : float
        Cell size of the grid in metres.
    simplify_path : bool
        Whether to simplify the resulting path with Douglas-Peucker.

    Returns
    -------
    np.ndarray or None
        Found path as UTM coordinate array, or ``None`` if no path exists.

    """
    ny, nx = grid.shape

    # Convert UTM to grid indices
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

    # Pre-calculate costs and pad with infinity to avoid boundary checks
    # grid is assumed to be 0.0 near paths, 1.0 away from paths.
    # Base traversal cost is 1.0 + grid_value * grid_cost_weight
    costs = 1.0 + grid * grid_cost_weight
    padded_costs = np.full((ny + 2, nx + 2), np.inf, dtype=np.float32)
    padded_costs[1:-1, 1:-1] = costs

    # Flattened grid size with padding
    p_nx = nx + 2
    p_ny = ny + 2

    g_scores = np.full(p_ny * p_nx, np.inf, dtype=np.float32)
    parents = np.full(p_ny * p_nx, -1, dtype=np.int32)

    start_flat = (start_iy + 1) * p_nx + (start_ix + 1)
    goal_flat = (goal_iy + 1) * p_nx + (goal_ix + 1)
    g_scores[start_flat] = 0.0

    # Priority queue: (f_score, g_score, ix, iy)
    h0 = np.sqrt((start_ix - goal_ix) ** 2 + (start_iy - goal_iy) ** 2)
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
            neighbors_data.append((dy * p_nx + dx, float(np.sqrt(dx**2 + dy**2)), ortho_offsets))

    flat_costs = padded_costs.ravel()

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

            # Convert back to UTM (cell centers, matching the grid's
            # cell-center sampling convention)
            path = np.array(
                [[(ix + 0.5) * cs + low[0], (iy + 0.5) * cs + low[1]] for ix, iy in path_indices],
            )

            # Simplify path, collision-checking every shortcut against the grid
            if simplify_path and len(path) > 2:
                path = simplify_path_checked(
                    path,
                    cs / 2.0,
                    lambda p1, p2: grid_segment_blocked(grid, p1, p2, low, cs),
                )
            return path

        current_g = g_scores[u_flat]

        for offset, dist, ortho_offsets in neighbors_data:
            v_flat = u_flat + offset
            cost_val = flat_costs[v_flat]

            if np.isinf(cost_val):
                continue

            # Diagonal moves must not slip between two corner-touching
            # blocked cells: both edge-adjacent cells have to be free.
            if ortho_offsets and any(np.isinf(flat_costs[u_flat + o]) for o in ortho_offsets):
                continue

            new_g = current_g + dist * cost_val
            if new_g < g_scores[v_flat]:
                g_scores[v_flat] = new_g
                parents[v_flat] = u_flat
                v_iy, v_ix = divmod(v_flat, p_nx)
                h = np.sqrt((v_ix - 1 - goal_ix) ** 2 + (v_iy - 1 - goal_iy) ** 2)
                heapq.heappush(pq, (new_g + h, new_g, v_ix - 1, v_iy - 1))

    return None
