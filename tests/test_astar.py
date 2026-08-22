import numpy as np
import shapely.geometry as sh

from map_data.pathsolver.grid_astar import (
    grid_astar,
    grid_segment_blocked,
    simplify_path_checked,
)
from map_data.pathsolver.replan import ReplanPath


class Args:
    def __init__(self):
        self.low = (0, 0)
        self.high = (10, 10)
        self.cell_size = 0.5
        self.simplify_path = True
        self.inflate_obstacles = 0.0


def test_astar_grid_simple_success():
    args = Args()
    obstacles = []
    replanner = ReplanPath(args, obstacles)
    # Mock a grid where everything is passable (cost 0)
    replanner._reshaped_grid_cache = np.zeros((20, 20), dtype=float)

    start = (1.0, 1.0)
    goal = (9.0, 9.0)
    path = replanner._astar(start, goal)

    assert path is not None
    assert len(path) >= 2
    # Check start and goal are close to requested points
    assert np.linalg.norm(path[0] - start) < 1.0
    assert np.linalg.norm(path[-1] - goal) < 1.0


def test_astar_grid_with_obstacle():
    args = Args()
    # Create a building in the middle
    building = sh.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    obstacles = [building]

    replanner = ReplanPath(args, obstacles)
    # Fill grid with base cost 0.5 (random terrain)
    grid = np.full((20, 20), 0.5, dtype=float)
    replanner._reshaped_grid_cache = replanner._burn_obstacles_into_grid(grid)

    start = (2.0, 2.0)
    goal = (8.0, 8.0)
    path = replanner._astar(start, goal)

    assert path is not None

    # Check that no point in the path is inside the obstacle
    for pt in path:
        point = sh.Point(pt)
        assert not building.contains(point)


def test_astar_grid_no_path():
    args = Args()
    # Create a wall blocking the way
    wall = sh.Polygon([(0, 4), (10, 4), (10, 6), (0, 6)])
    obstacles = [wall]

    replanner = ReplanPath(args, obstacles)
    grid = np.zeros((20, 20), dtype=float)
    replanner._reshaped_grid_cache = replanner._burn_obstacles_into_grid(grid)

    start = (1.0, 1.0)
    goal = (9.0, 9.0)
    path = replanner._astar(start, goal)

    assert path is None


class MockMapData:
    def __init__(self, footways):
        self.footways_list = footways
        self.roads_list = []

    def get_points(self):
        # ReplanPath._split_ways uses points[node_id]
        points = {}
        for way in self.footways_list:
            for node_id, coord in zip(
                way.nodes,
                way.line.exterior.coords if hasattr(way.line, "exterior") else way.line.coords,
                strict=True,
            ):
                points[node_id] = np.array(coord).reshape(1, -1)
        return points


def test_astar_grid_with_obstacle_and_path():
    args = Args()
    # Create a wall blocking the way
    wall = sh.Polygon([(0, 4), (10, 4), (10, 6), (0, 6)])
    obstacles = [wall]

    # Create a path that goes through the wall
    # Way expects a LineString or Polygon in its 'line' attribute
    # We'll use a simple LineString for the nodes part and buffer it for the 'line' part
    path_coords = [(5.0, 0.0), (5.0, 10.0)]
    path_line = sh.LineString(path_coords).buffer(1.5)
    from map_data.utils.way import Way

    # Give it some nodes so _split_ways doesn't fail
    footway = Way(id=1, nodes=[100, 101], tags={"highway": "footway"}, line=path_line)

    replanner = ReplanPath(args, obstacles)
    # ReplanPath.grid needs to be initialized for fill_grid
    # ReplanPath.grid is expected to have shape (N, 3) before padding to (N, 4) in fill_grid
    replanner.grid = replanner._create_grid(args.low, args.high, args.cell_size)

    # Mock points for the nodes
    points = {
        100: np.array([5.0, 0.0]).reshape(1, 2),
        101: np.array([5.0, 10.0]).reshape(1, 2),
    }

    md = MockMapData(footways=[footway])
    md.get_points = lambda: points

    # fill_grid should subtract the path from the wall
    replanner.fill_grid(md, highway_types=["footway"])

    start = (5.0, 1.0)
    goal = (5.0, 9.0)
    path = replanner._astar(start, goal)

    assert path is not None

    # Verify the path actually crosses the obstacle area (y from 4 to 6)
    # Since it's a straight line and might be simplified, we check if it spans across the obstacle
    min_y = min(pt[1] for pt in path)
    max_y = max(pt[1] for pt in path)
    assert min_y <= 4.0 and max_y >= 6.0, "Path should span across the obstacle area"


def test_post_process_path_simplification():
    args = Args()
    args.simplify_path = True
    args.cell_size = 0.5
    replanner = ReplanPath(args, [])

    # Create a path with many points very close to each other along a line
    # (5, 0), (5, 0.01), (5, 0.02), ..., (5, 1), then (5, 10)
    noisy_segment = [[5.0, y] for y in np.arange(0, 1.01, 0.01)]
    path = np.array([*noisy_segment, [5.0, 10.0]])

    processed_path = replanner._post_process_path(path)

    # The whole (5, 0) to (5, 10) segment is a straight line, so it should be simplified to 2 points
    assert len(processed_path) == 2
    assert np.allclose(processed_path[0], [5.0, 0.0])
    assert np.allclose(processed_path[-1], [5.0, 10.0])


def test_astar_grid_goal_outside_boundary():
    """
    Goal UTM outside the grid is caught — returns None.
    """
    args = Args()
    replanner = ReplanPath(args, [])
    replanner._reshaped_grid_cache = np.zeros((20, 20), dtype=float)

    start = (1.0, 1.0)
    goal = (15.0, 15.0)  # beyond 10x10 grid
    path = replanner._astar(start, goal)

    assert path is None


def test_astar_grid_start_equals_goal_same_cell():
    """
    Points that map to the same grid cell return a 2-point trivial path.
    """
    args = Args()
    replanner = ReplanPath(args, [])
    replanner._reshaped_grid_cache = np.zeros((20, 20), dtype=float)

    start = (5.0, 5.0)
    goal = (5.05, 5.05)  # floor(5.05/0.5)=10 == floor(5.0/0.5)=10
    path = replanner._astar(start, goal)

    assert path is not None
    assert len(path) == 2


def _diagonal_wall_grid(n=10):
    """
    Free n x n grid with the main diagonal blocked (cells touch only at corners).
    """
    grid = np.zeros((n, n), dtype=float)
    for i in range(n):
        grid[i, i] = np.inf
    return grid


def test_grid_astar_does_not_cut_through_diagonal_corners():
    """
    A diagonal wall of blocked cells fully separates start and goal.

    The blocked cells touch only at their corners; a diagonal move slipping
    between two corner-touching blocked cells would cut through the wall,
    so no path may be found.
    """
    grid = _diagonal_wall_grid()

    path = grid_astar(grid, (8.5, 1.5), (1.5, 8.5), (0.0, 0.0), 1.0)

    assert path is None


def test_grid_astar_diagonal_wall_with_gap_is_passable():
    """
    Opening one cell in the diagonal wall makes the goal reachable again.
    """
    grid = _diagonal_wall_grid()
    grid[5, 5] = 0.0

    path = grid_astar(grid, (8.5, 1.5), (1.5, 8.5), (0.0, 0.0), 1.0)

    assert path is not None
    assert len(path) >= 2


def _two_corridor_grid():
    """
    3 x 11 grid with two corridors between the endpoints.

    Row 2 (where start and goal sit) is a short direct corridor whose interior
    cells carry a high traversal cost (grid value 1.0). Row 0 is a longer
    zero-cost detour, reachable only through the connector cells in column 0
    and column 10 — everything else in row 1 is blocked, which also keeps
    diagonal moves from cutting between the corridors (both edge-adjacent
    cells of such a diagonal are blocked).
    """
    grid = np.full((3, 11), np.inf, dtype=float)
    grid[0, :] = 0.0  # low-cost detour corridor
    grid[2, 1:10] = 1.0  # high-cost interior of the direct corridor
    grid[2, 0] = grid[2, 10] = 0.0  # start / goal cells
    grid[1, 0] = grid[1, 10] = 0.0  # connectors between the corridors
    return grid


def test_grid_astar_weighted_prefers_low_cost_corridor():
    """
    With cost weighting on, the 9-cell high-cost direct corridor costs
    9 * (1 + 5.0) + 1 = 55 while the 14-step zero-cost detour costs 14,
    so the planner must route through the detour (row 0, y == 0).
    """
    path = grid_astar(
        _two_corridor_grid(),
        (0.5, 2.5),
        (10.5, 2.5),
        (0.0, 0.0),
        1.0,
        simplify_path=False,
        grid_cost_weight=5.0,
    )

    assert path is not None
    # Path coordinates are cell centers: row 0 centers sit at y == 0.5
    assert min(pt[1] for pt in path) == 0.5, "path should use the low-cost row-0 corridor"
    # It still starts and ends in the direct corridor's row (center y == 2.5)
    assert path[0][1] == 2.5
    assert path[-1][1] == 2.5


def test_grid_astar_negligible_weight_takes_short_corridor():
    """
    With a negligible cost weight every free cell costs (almost) the same, so
    the shorter direct corridor (10 steps vs 14) wins and the path stays in
    row 2. (Exactly 0.0 would turn the inf blocked cells into 0 * inf = NaN,
    so a tiny epsilon stands in for "weighting off".)
    """
    path = grid_astar(
        _two_corridor_grid(),
        (0.5, 2.5),
        (10.5, 2.5),
        (0.0, 0.0),
        1.0,
        simplify_path=False,
        grid_cost_weight=1e-9,
    )

    assert path is not None
    # Path coordinates are cell centers: row 2 centers sit at y == 2.5
    assert all(pt[1] == 2.5 for pt in path), "uniform costs should keep the path in the direct row"


# ── regression: simplification must not chord through obstacles ──────────────
#
# Path simplification used to run *after* collision checking without being
# collision-checked itself, so a Douglas-Peucker shortcut could cut straight
# through an obstacle the planner had carefully routed around.


def _staircase_scenario():
    """
    Path hugging a blocked cell where naive DP simplification chords through it.

    The staircase of cell centers deviates < cs/2 from the straight chord
    between its endpoints, so ``LineString.simplify(cs / 2)`` collapses it to
    the chord — which crosses the blocked cell (3, 1) that every original
    segment avoids.
    """
    grid = np.zeros((3, 5), dtype=float)
    grid[1, 3] = np.inf
    path = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 1.5], [3.5, 2.5], [4.5, 2.5]])
    return grid, path


def test_simplify_path_checked_keeps_vertices_when_chord_is_blocked():
    grid, path = _staircase_scenario()

    # Sanity: the naive chord really is blocked, the original segments are not
    assert grid_segment_blocked(grid, path[0], path[-1], (0.0, 0.0), 1.0)
    for i in range(len(path) - 1):
        assert not grid_segment_blocked(grid, path[i], path[i + 1], (0.0, 0.0), 1.0)

    result = simplify_path_checked(
        path,
        0.5,
        lambda p1, p2: grid_segment_blocked(grid, p1, p2, (0.0, 0.0), 1.0),
    )

    # The colliding shortcut was rejected: the original vertices survive and
    # no resulting segment crosses a blocked cell.
    assert np.allclose(result, path)
    for i in range(len(result) - 1):
        assert not grid_segment_blocked(grid, result[i], result[i + 1], (0.0, 0.0), 1.0)


def test_simplify_path_checked_still_simplifies_free_space():
    _, path = _staircase_scenario()
    free_grid = np.zeros((3, 5), dtype=float)

    result = simplify_path_checked(
        path,
        0.5,
        lambda p1, p2: grid_segment_blocked(free_grid, p1, p2, (0.0, 0.0), 1.0),
    )

    # Without the obstacle the same path collapses to its two endpoints
    assert len(result) == 2
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-1], path[-1])


def test_grid_astar_simplified_path_stays_off_blocked_cells():
    """
    End-to-end: the path through the diagonal-wall gap survives simplification.

    The near-diagonal path through the gap is almost collinear, so unchecked
    DP simplification would chord straight through the wall.
    """
    grid = _diagonal_wall_grid()
    grid[5, 5] = 0.0

    path = grid_astar(grid, (8.5, 1.5), (1.5, 8.5), (0.0, 0.0), 1.0, simplify_path=True)

    assert path is not None
    for i in range(len(path) - 1):
        assert not grid_segment_blocked(grid, path[i], path[i + 1], (0.0, 0.0), 1.0), (
            f"simplified segment {path[i]} -> {path[i + 1]} crosses the wall"
        )


def test_post_process_path_simplification_does_not_chord_into_obstacle():
    """
    Replan-level: the final DP simplification is collision-checked.

    The midpoint deviates 0.4 m (< cell_size = 0.5 m) from the straight
    chord, so unchecked simplification would drop it — and the chord passes
    straight through the obstacle the original path skirts around.
    """
    args = Args()
    obstacle = sh.Polygon([(4.0, -0.1), (6.0, -0.1), (6.0, 0.1), (4.0, 0.1)])
    replanner = ReplanPath(args, [obstacle])

    path = np.array([[0.0, 0.0], [5.0, 0.4], [10.0, 0.0]])
    # Sanity: the original path avoids the obstacle, the naive chord does not
    assert not sh.LineString(path).intersects(obstacle)
    assert sh.LineString([path[0], path[-1]]).intersects(obstacle)

    processed = replanner._post_process_path(path)

    assert processed is not None
    assert not sh.LineString(processed).intersects(obstacle)
    assert np.allclose(processed[0], path[0])
    assert np.allclose(processed[-1], path[-1])


def test_post_process_path_very_close_points():
    args = Args()
    args.simplify_path = False  # Disable DP simplification to test only distance-based removal
    replanner = ReplanPath(args, [])

    # Two points extremely close to each other
    path = np.array([[0.0, 0.0], [0.0, 0.01], [1.0, 1.0]])

    processed_path = replanner._post_process_path(path)

    # [0.0, 0.01] should be removed because it's within 0.05m of [0.0, 0.0]
    assert len(processed_path) == 2
    assert np.allclose(processed_path[0], [0.0, 0.0])
    assert np.allclose(processed_path[1], [1.0, 1.0])
