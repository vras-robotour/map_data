import numpy as np
import pytest
import shapely.geometry as geom
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath
from map_data.utils.way import Way

_ZN, _ZL = 33, "U"
_E0, _N0 = 458000.0, 5550500.0


def _make_md_with_footway():
    """MapData with a 16m footway running along the grid midline."""
    waypoints = np.array([[_E0, _N0], [_E0 + 20, _N0 + 20]])
    md = MapData([waypoints, _ZN, _ZL], coords_type="array")

    lat1, lon1 = utm.to_latlon(_E0 + 2, _N0 + 10, _ZN, _ZL)
    lat2, lon2 = utm.to_latlon(_E0 + 18, _N0 + 10, _ZN, _ZL)
    md.nodes_cache = {
        1: {"lat": lat1, "lon": lon1, "tags": {}},
        2: {"lat": lat2, "lon": lon2, "tags": {}},
    }
    footway = Way(
        id=1,
        is_area=False,
        nodes=[1, 2],
        tags={"highway": "footway"},
        line=LineString([(_E0 + 2, _N0 + 10), (_E0 + 18, _N0 + 10)]),
        in_out="",
    )
    md.footways_list.append(footway)
    return md


def _make_args(low, high, cell_size=0.5):
    return {"low": low, "high": high, "cell_size": cell_size, "inflate_obstacles": 0.0}


# ── footway cost ──────────────────────────────────────────────────────────────


def test_fill_grid_cells_on_footway_have_low_cost():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath([], **args)
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    assert grid_2d is not None

    # Footway is at n0+10 → y_idx ≈ floor(10 / 0.5) = 20 out of 40
    # Footway x midpoint e0+10 → x_idx ≈ 20
    y_idx = int(10 / args["cell_size"])
    x_idx = int(10 / args["cell_size"])
    # cap at grid bounds
    y_idx = min(y_idx, grid_2d.shape[0] - 1)
    x_idx = min(x_idx, grid_2d.shape[1] - 1)

    assert grid_2d[y_idx, x_idx] < rp.default_off_path_cost


def test_fill_grid_cells_far_from_footway_have_high_cost():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath([], **args)
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    # Far from footway: n0+0 → y_idx=0, far from any path
    corner_cost = grid_2d[0, 0]
    assert not np.isinf(corner_cost)
    assert corner_cost >= rp.default_off_path_cost - 0.05  # allow tiny float tolerance


# ── barrier obstacles ─────────────────────────────────────────────────────────


def test_fill_grid_cells_inside_barrier_are_inf():
    md = _make_md_with_footway()
    # Barrier 2m x 2m square at (e0+5, n0+2) - away from footway
    barrier = geom.box(_E0 + 5, _N0 + 2, _E0 + 7, _N0 + 4)

    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath([barrier], **args)
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    # Centroid of barrier: (e0+6, n0+3) → x_idx=12, y_idx=6
    x_idx = int(6 / args["cell_size"])
    y_idx = int(3 / args["cell_size"])
    x_idx = min(x_idx, grid_2d.shape[1] - 1)
    y_idx = min(y_idx, grid_2d.shape[0] - 1)

    assert np.isinf(grid_2d[y_idx, x_idx])


def test_fill_grid_cells_outside_barrier_not_inf():
    md = _make_md_with_footway()
    barrier = geom.box(_E0 + 5, _N0 + 2, _E0 + 7, _N0 + 4)

    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath([barrier], **args)
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    # Point clearly outside the barrier: (e0+1, n0+1) → x_idx=2, y_idx=2
    assert not np.isinf(grid_2d[2, 2])


# ── grid shape ────────────────────────────────────────────────────────────────


def test_fill_grid_produces_2d_cache():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath([], **args)
    assert rp.path_grid.grid_2d_cache is None  # not filled yet
    rp.fill_grid(md)
    assert rp.path_grid.grid_2d_cache is not None
    assert rp.path_grid.grid_2d_cache.ndim == 2


# ── no stray zero-cost cells off any way ────────────────────────────────────
#
# fill() must write every grid_2d cell, or a planner reads the unwritten ones as
# free. Walks every cell, not just a few coordinates.


def test_fill_grid_no_zero_cost_cells_off_path():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath([], **args)
    max_path_dist = 2.0
    rp.fill_grid(md, max_path_dist=max_path_dist)

    grid_2d = rp.path_grid.grid_2d_cache
    low = rp.path_grid.low
    cell_size = rp.path_grid.cell_size
    footway_line = md.footways_list[0].line

    num_y, num_x = grid_2d.shape
    # PathGrid.fill samples the way every cell_size, so pad the off-path
    # threshold by one cell to never count a near-path cell as off-path.
    off_path_threshold = max_path_dist + cell_size

    off_path_cells = 0
    for iy in range(num_y):
        for ix in range(num_x):
            # Cells are sampled at their centers
            x = low[0] + (ix + 0.5) * cell_size
            y = low[1] + (iy + 0.5) * cell_size
            dist_to_way = footway_line.distance(geom.Point(x, y))
            if dist_to_way > off_path_threshold:
                off_path_cells += 1
                cost = grid_2d[iy, ix]
                assert not np.isclose(cost, 0.0), (
                    f"cell ({ix},{iy}) at ({x},{y}), {dist_to_way:.2f}m from the "
                    f"footway, has stray zero cost"
                )
                assert cost == pytest.approx(rp.default_off_path_cost)

    # Sanity check that the test actually exercised off-path cells.
    assert off_path_cells > 0


# ── cells are sampled at their centers, not the lower-left corner ───────────


def _make_path_grid(cell_size=1.0, low=(0.0, 0.0), high=(10.0, 10.0)):
    from map_data.pathsolver.grid_constructor import PathGrid

    return PathGrid(
        low=low,
        high=high,
        cell_size=cell_size,
        highway_costs={},
        surface_costs={},
    )


def test_fill_samples_cell_centers_and_keeps_no_flat_grid():
    from types import SimpleNamespace

    pg = _make_path_grid()
    assert pg.grid_2d_cache is None
    md = SimpleNamespace(get_points=lambda: {}, footways_list=[], roads_list=[])
    pg.fill(md, [geom.box(0.1, 0.1, 0.9, 0.9)])
    # Obstacle in cell (x=0, y=0) only: the cell centre (0.5, 0.5) is inside it
    assert pg.grid_2d_cache.shape == (10, 10)
    assert np.isinf(pg.grid_2d_cache[0, 0])
    assert np.isinf(pg.grid_2d_cache).sum() == 1
    assert not hasattr(pg, "grid")


def test_burn_obstacles_sub_cell_obstacle_centered_in_cell_burns_it():
    pg = _make_path_grid()
    # Sub-cell obstacle at the center of cell (x=4, y=4), clear of its
    # lower-left corner (4.0, 4.0).
    obstacle = geom.box(4.3, 4.3, 4.7, 4.7)

    grid_2d = np.full((10, 10), 0.5, dtype=np.float32)
    pg.burn_obstacles(grid_2d, [obstacle])

    assert np.isinf(grid_2d[4, 4])
    # Neighboring cells (centers 1m away) are untouched
    assert not np.isinf(grid_2d[4, 3])
    assert not np.isinf(grid_2d[3, 4])
    assert not np.isinf(grid_2d[4, 5])
    assert not np.isinf(grid_2d[5, 4])


def test_burn_obstacles_region_not_shifted_toward_origin():
    pg = _make_path_grid()
    # Obstacle exactly covering cells x in {2, 3}, y in {2, 3}; cells whose
    # lower-left corner lies on its 4.0 boundary stay unburned.
    obstacle = geom.box(2.0, 2.0, 4.0, 4.0)

    grid_2d = np.full((10, 10), 0.5, dtype=np.float32)
    pg.burn_obstacles(grid_2d, [obstacle])

    burned = np.argwhere(np.isinf(grid_2d))
    assert {tuple(c) for c in burned} == {(2, 2), (2, 3), (3, 2), (3, 3)}
