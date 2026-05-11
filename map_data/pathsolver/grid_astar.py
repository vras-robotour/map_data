import heapq
import numpy as np
from shapely.geometry import LineString


def grid_astar(grid, start_utm, goal_utm, low, cs, simplify_path=True):
    """
    Optimized A* search on a 2D grid.

    Parameters:
    -----------
    grid : np.array
        2D grid of costs (Y, X). infinity means blocked.
    start_utm : tuple/array
        Starting point in UTM coordinates.
    goal_utm : tuple/array
        Goal point in UTM coordinates.
    low : tuple
        (min_x, min_y) of the grid in UTM.
    cs : float
        Cell size of the grid.
    simplify_path : bool
        Whether to simplify the resulting path.

    Returns:
    --------
    path : np.array or None
        Found path in UTM coordinates, or None.
    """
    ny, nx = grid.shape

    # Convert UTM to grid indices
    def to_idx(p):
        ix = int(np.floor((p[0] - low[0]) / cs))
        iy = int(np.floor((p[1] - low[1]) / cs))
        return np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)

    start_ix, start_iy = to_idx(start_utm)
    goal_ix, goal_iy = to_idx(goal_utm)

    if start_ix == goal_ix and start_iy == goal_iy:
        return np.array([start_utm, goal_utm])

    # Pre-calculate costs and pad with infinity to avoid boundary checks
    # grid is assumed to be 0.0 near paths, 1.0 away from paths.
    # Base traversal cost is 1.0 + grid_value * 5.0
    costs = 1.0 + grid * 5.0
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

    # Neighbor offsets in flattened padded grid (dy * p_nx + dx, dist)
    neighbors_data = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbors_data.append((dy * p_nx + dx, float(np.sqrt(dx**2 + dy**2))))

    flat_costs = padded_costs.ravel()

    while pq:
        f, g_pushed, ix, iy = heapq.heappop(pq)

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

            # Convert back to UTM
            path = np.array(
                [[ix * cs + low[0], iy * cs + low[1]] for ix, iy in path_indices]
            )

            # Simplify path
            if simplify_path and len(path) > 2:
                path = np.array(LineString(path).simplify(cs / 2.0).coords)
            return path

        current_g = g_scores[u_flat]

        for offset, dist in neighbors_data:
            v_flat = u_flat + offset
            cost_val = flat_costs[v_flat]

            if np.isinf(cost_val):
                continue

            new_g = current_g + dist * cost_val
            if new_g < g_scores[v_flat]:
                g_scores[v_flat] = new_g
                parents[v_flat] = u_flat
                v_iy, v_ix = divmod(v_flat, p_nx)
                h = np.sqrt((v_ix - 1 - goal_ix) ** 2 + (v_iy - 1 - goal_iy) ** 2)
                heapq.heappush(pq, (new_g + h, new_g, v_ix - 1, v_iy - 1))

    return None
