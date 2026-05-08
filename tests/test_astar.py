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
