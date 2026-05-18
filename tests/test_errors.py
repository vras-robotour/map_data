import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
import utm

from map_data.map_data import MapData
from map_data.pathsolver.rrt_star import RRTStar
from map_data.utils.overpass import OverpassClient


def test_mapdata_nonexistent_gpx_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        MapData(str(tmp_path / "no_such.gpx"), coords_type="file")


def test_mapdata_invalid_gpx_xml_raises(tmp_path):
    gpx_path = str(tmp_path / "bad.gpx")
    with open(gpx_path, "w") as f:
        f.write("not xml content at all")
    with pytest.raises(Exception):
        MapData(gpx_path, coords_type="file")


def test_load_mapdata_invalid_json(tmp_path):
    path = str(tmp_path / "bad.mapdata")
    with open(path, "w") as f:
        f.write("not json content {{{")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        MapData.load(path)


def test_load_mapdata_missing_metadata_key(tmp_path):
    path = str(tmp_path / "empty.mapdata")
    with open(path, "w") as f:
        json.dump({}, f)
    with pytest.raises((KeyError, TypeError)):
        MapData.load(path)


def test_load_mapdata_legacy_pickle_raises(tmp_path):
    path = str(tmp_path / "legacy.mapdata")
    with open(path, "wb") as f:
        f.write(b"\x80\x04\x95some pickle data")
    with pytest.raises(ValueError):
        MapData.load(path)


def test_overpass_timeout_returns_none():
    client = OverpassClient()
    with patch.object(client.session, "get", return_value=MagicMock(status_code=200, text="")), \
         patch.object(client.session, "post", side_effect=requests.Timeout("timeout")), \
         patch("map_data.utils.overpass.time.sleep"):
        result = client.query_raw("test query", retries=3)
    assert result is None


def test_run_parse_without_queries_returns_error():
    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    md = MapData([np.array([[e, n]]), int(zn), zl], coords_type="array")
    # osm_*_data are all None at construction — run_parse should signal failure
    result = md.run_parse()
    assert result == 1


def test_rrt_star_goal_in_isolated_obstacle():
    """When start is free but the rest of the grid is blocked, path is None."""
    grid = np.ones((100, 100), dtype=float)
    grid[0:5, 0:5] = 0.0  # tiny free patch around start at (0,0)

    rrt_star = RRTStar(
        np.array([0.0, 0.0]),
        np.array([5.0, 5.0]),  # goal inside the fully blocked region
        [],
        None,
        grid,
        (0.0, 0.0),
        max_iter=300,
        step_size=1.0,
        neighbor_radius=2.0,
        grid_scale=0.1,
        traversability_threshold=0.5,
    )
    path = rrt_star.find_path()
    assert path is None
