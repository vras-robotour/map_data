import numpy as np
import pytest
import shapely.geometry as geom
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath, parse_args
from map_data.utils.way import Way

_ZN, _ZL = 33, "U"
_E0, _N0 = 458000.0, 5550500.0


def _make_md_with_footway():
    """
    MapData with a 16m footway running along the grid midline.
    """
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
    # Barrier 2m x 2m square at (e0+5, n0+2) - away from footway
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


# ── regression: no stray zero-cost cells off any way ────────────────────────
#
# create_empty_grid (np.arange) and get_grid_2d (np.ceil-based dims + floor
# indexing) must agree on how many cells the grid has and where each point
# lands, otherwise some grid_2d cells never get written to. PathGrid.get_grid_2d
# initializes with default_off_path_cost precisely so a mismatch like that
# would silently leave cost 0.0 (from stale/never-hit entries) instead of the
# off-path cost — which a planner would read as "free to drive through",
# including deep inside solid obstacles or far off any path. This test walks
# every cell explicitly (rather than sampling one or two coordinates like the
# tests above) so a discretization regression anywhere in the grid gets caught.


def test_fill_grid_no_zero_cost_cells_off_path():
    md = _make_md_with_footway()
    args = _make_args((_E0, _N0), (_E0 + 20, _N0 + 20))
    rp = ReplanPath(args, [])
    max_path_dist = 2.0
    rp.fill_grid(md, max_path_dist=max_path_dist)

    grid_2d = rp.path_grid.grid_2d_cache
    low = rp.path_grid.low
    cell_size = rp.path_grid.cell_size
    footway_line = md.footways_list[0].line

    num_y, num_x = grid_2d.shape
    # PathGrid.fill samples the way every `cell_size`, so a point on the line
    # can be up to ~cell_size away from the nearest sampled path point that
    # was actually fed into the cKDTree query. Pad the "definitely off-path"
    # threshold so we never mistake a near-path cell for an off-path one.
    off_path_threshold = max_path_dist + cell_size

    off_path_cells = 0
    for iy in range(num_y):
        for ix in range(num_x):
            x = low[0] + ix * cell_size
            y = low[1] + iy * cell_size
            dist_to_way = footway_line.distance(geom.Point(x, y))
            if dist_to_way > off_path_threshold:
                off_path_cells += 1
                cost = grid_2d[iy, ix]
                assert not np.isclose(cost, 0.0), (
                    f"cell ({ix},{iy}) at ({x},{y}), {dist_to_way:.2f}m from the "
                    f"footway, has stray zero cost"
                )
                assert cost == pytest.approx(rp.DEFAULT_OFF_PATH_COST)

    # Sanity check that the test actually exercised off-path cells.
    assert off_path_cells > 0
