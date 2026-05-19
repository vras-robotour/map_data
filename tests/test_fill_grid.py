import numpy as np
import shapely.geometry as geom
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath, parse_args
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
    args = parse_args([])
    args.low = low
    args.high = high
    args.cell_size = cell_size
    args.inflate_obstacles = 0.0
    return args


# ── footway cost ──────────────────────────────────────────────────────────────


def test_fill_grid_cells_on_footway_have_low_cost():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath(args, [])
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    assert grid_2d is not None

    # Footway is at n0+10 → y_idx ≈ floor(10 / 0.5) = 20 out of 40
    # Footway x midpoint n0+10 → x_idx ≈ 20
    y_idx = int(10 / args.cell_size)
    x_idx = int(10 / args.cell_size)
    # cap at grid bounds
    y_idx = min(y_idx, grid_2d.shape[0] - 1)
    x_idx = min(x_idx, grid_2d.shape[1] - 1)

    assert grid_2d[y_idx, x_idx] < rp.DEFAULT_OFF_PATH_COST


def test_fill_grid_cells_far_from_footway_have_high_cost():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath(args, [])
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    # Far from footway: n0+0 → y_idx=0, far from any path
    corner_cost = grid_2d[0, 0]
    assert not np.isinf(corner_cost)
    assert corner_cost >= rp.DEFAULT_OFF_PATH_COST - 0.05  # allow tiny float tolerance


# ── barrier obstacles ─────────────────────────────────────────────────────────


def test_fill_grid_cells_inside_barrier_are_inf():
    md = _make_md_with_footway()
    # Barrier 2m x 2m square at (e0+5, n0+2) – away from footway
    barrier = geom.box(_E0 + 5, _N0 + 2, _E0 + 7, _N0 + 4)

    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath(args, [barrier])
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    # Centroid of barrier: (e0+6, n0+3) → x_idx=12, y_idx=6
    x_idx = int(6 / args.cell_size)
    y_idx = int(3 / args.cell_size)
    x_idx = min(x_idx, grid_2d.shape[1] - 1)
    y_idx = min(y_idx, grid_2d.shape[0] - 1)

    assert np.isinf(grid_2d[y_idx, x_idx])


def test_fill_grid_cells_outside_barrier_not_inf():
    md = _make_md_with_footway()
    barrier = geom.box(_E0 + 5, _N0 + 2, _E0 + 7, _N0 + 4)

    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath(args, [barrier])
    rp.fill_grid(md)

    grid_2d = rp.path_grid.grid_2d_cache
    # Point clearly outside the barrier: (e0+1, n0+1) → x_idx=2, y_idx=2
    assert not np.isinf(grid_2d[2, 2])


# ── grid shape ────────────────────────────────────────────────────────────────


def test_fill_grid_produces_2d_cache():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath(args, [])
    assert rp.path_grid.grid_2d_cache is None  # not filled yet
    rp.fill_grid(md)
    assert rp.path_grid.grid_2d_cache is not None
    assert rp.path_grid.grid_2d_cache.ndim == 2
