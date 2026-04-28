#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "overpy",
#   "shapely",
#   "utm",
#   "gpxpy",
#   "tqdm",
#   "flask",
# ]
# ///
"""
Quick functionality check. Run with:
    uv run check.py
"""

import logging
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

REPO = Path(__file__).parent
GPX_FILE = REPO / "data" / "stromovka.gpx"

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"


def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        return True
    except Exception:
        print(f"{FAIL}  {label}")
        traceback.print_exc()
        return False


# ── 1. Way class ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(REPO))
from map_data.way import Way  # noqa: E402

def test_mutable_default():
    w1, w2 = Way(), Way()
    w1.nodes.append("x")
    assert w2.nodes == [], "nodes mutable default not fixed"

def test_classification():
    assert Way(tags={"highway": "footway"}).is_footway()
    assert not Way(tags={"highway": "footway"}).is_road()
    assert Way(tags={"highway": "primary"}).is_road()
    assert not Way(tags={"highway": "primary"}).is_footway()
    assert not Way(tags={}).is_road()
    assert not Way(tags={}).is_footway()

def test_is_barrier():
    yes = {"barrier": ["wall"]}
    anti = {"barrier": ["wall"]}
    assert Way(tags={"barrier": "wall"}).is_barrier(yes, {}, {}) is True
    assert Way(tags={"barrier": "wall"}).is_barrier(yes, {}, anti) is False
    assert Way(tags={}).is_barrier(yes, {}, {}) is False

# ── 2. CoordsData ─────────────────────────────────────────────────────────────

from map_data.map_data import CoordsData  # noqa: E402

def test_coords_data_margin():
    cd = CoordsData(14.0, 14.5, 50.0, 50.5)
    assert cd.x_margin == cd.y_margin, "margins should always be equal"
    assert cd.x_margin > 0

# ── 3. Full parse with real GPX ───────────────────────────────────────────────

from map_data.map_data import MapData  # noqa: E402

map_data = None

def test_parse_gpx():
    global map_data
    assert GPX_FILE.exists(), f"GPX file not found: {GPX_FILE}"
    map_data = MapData(str(GPX_FILE))
    map_data.run_queries()

    if any(d is None for d in (map_data.osm_ways_data, map_data.osm_rels_data, map_data.osm_nodes_data)):
        raise RuntimeError(
            "All Overpass endpoints returned errors (406 / timeout). "
            "The server may be rate-limiting — wait a minute and try again, "
            "or check https://overpass-api.de/api/status for service status."
        )

    ret = map_data.run_parse()
    assert ret == 0, "run_parse() returned failure"
    map_data.save_to_pickle()

    assert len(map_data.roads_list) > 0,    "no roads parsed"
    assert len(map_data.footways_list) > 0, "no footways parsed"
    assert len(map_data.barriers_list) > 0, "no barriers parsed"

    out = GPX_FILE.with_suffix(".mapdata")
    assert out.exists(), f".mapdata file not written to {out}"

    print(f"\n       roads={len(map_data.roads_list)}"
          f"  footways={len(map_data.footways_list)}"
          f"  barriers={len(map_data.barriers_list)}"
          f"  → {out.name}")

# ── 4. Viewer GeoJSON conversion ──────────────────────────────────────────────

def test_viewer_geojson():
    assert map_data is not None, "run test_parse_gpx first"
    from map_data.viewer.app import _mapdata_to_geojson
    fc = _mapdata_to_geojson(map_data)
    assert fc["type"] == "FeatureCollection"
    categories = {f["properties"]["category"] for f in fc["features"]}
    assert "road"     in categories, "no roads in GeoJSON output"
    assert "footway"  in categories, "no footways in GeoJSON output"
    assert "barrier"  in categories, "no barriers in GeoJSON output"
    assert "waypoint" in categories, "no waypoints in GeoJSON output"
    print(f"\n       {len(fc['features'])} GeoJSON features across {sorted(categories)}")

# ── 5. Viewer Flask app imports ───────────────────────────────────────────────

def test_viewer_imports():
    from map_data.viewer.app import app  # noqa: F401
    assert app is not None

# ── Run all ───────────────────────────────────────────────────────────────────

print("\n── Way class ────────────────────────────────")
results = [
    check("mutable default nodes=[] fixed",        test_mutable_default),
    check("is_road / is_footway classification",   test_classification),
    check("is_barrier with yes/anti tags",         test_is_barrier),
]

print("\n── CoordsData ───────────────────────────────")
results += [
    check("margin is symmetric and positive",      test_coords_data_margin),
]

print("\n── Full OSM parse (network required) ────────")
results += [
    check(f"parse {GPX_FILE.name} end-to-end",    test_parse_gpx),
]

print("\n── Viewer ───────────────────────────────────")
results += [
    check("GeoJSON conversion covers all layers", test_viewer_geojson),
    check("Flask app imports cleanly",             test_viewer_imports),
]

passed = sum(results)
total  = len(results)
color  = "\033[32m" if passed == total else "\033[31m"
print(f"\n{color}{'─'*44}")
print(f"  {passed}/{total} checks passed\033[0m\n")

sys.exit(0 if passed == total else 1)
