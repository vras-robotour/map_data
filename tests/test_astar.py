import numpy as np
import shapely.geometry as sh

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
    path = np.array(noisy_segment + [[5.0, 10.0]])

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
    goal = (15.0, 15.0)  # beyond 10×10 grid
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
