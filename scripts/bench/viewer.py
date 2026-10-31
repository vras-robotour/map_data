"""
Benchmark of the viewer's HTTP API through the Flask test client.

For each map it times (p50/p95 over N requests, cold and warm caches): the map GET
(``/api/mapdata``, with its size and gzip-ability), annotations, the segments and
nodes of the map's longest way, an edit cycle (move, tags, delete, then the
``/api/mapdata`` GET the frontend issues after it), ``/api/create_replan`` (graph,
rrt, astar on a ~500 m route), ``/api/cost_grid`` and ``/api/export``.

Maps are copied to a scratch directory, so edits never touch the originals.

    uv run python scripts/bench/viewer.py --quick          # tracked demo map, < 60 s
    uv run python scripts/bench/viewer.py --data-dir DIR   # every DIR/*.mapdata
    uv run python scripts/bench/viewer.py --profile mapdata --data-dir DIR

Cold = every viewer cache cleared before each request (first request after a
start or an edit); warm = repeated request on filled caches.
"""

import argparse
import cProfile
import gzip
import io
import json
import logging
import pstats
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import utm
from shapely.geometry import LineString

from map_data.edits import get_deleted_way_ids, load_annotations
from map_data.map_data import MapData
from map_data.viewer import cache, helpers
from map_data.viewer.app import create_app
from map_data.viewer.routes import planning

logging.getLogger("map_data").setLevel(logging.ERROR)  # expected 'Replan failed' while probing

DEMO = Path(__file__).resolve().parents[2] / "demo"


def clear_caches() -> None:
    cache._load.cache_clear()
    cache._geojson_cache.clear()
    cache._merged_cache.clear()
    planning._graph_planner.cache_clear()
    clear = getattr(
        helpers.geom_to_geojson, "cache_clear", None
    )  # absent before the geometry cache
    if clear:
        clear()


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def pct(samples: list[float], q: float) -> float:
    s = sorted(samples)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


class Bench:
    def __init__(self, data_dir: Path, name: str, reps: int, cold_reps: int) -> None:
        self.dir, self.name, self.reps, self.cold_reps = data_dir, name, reps, cold_reps
        self.file = f"{name}.mapdata"
        self.client = create_app(data_dir=str(data_dir)).test_client()
        self.rows: list[dict[str, Any]] = []
        md = MapData.load(str(data_dir / self.file))
        self.zone = (md.zone_number, md.zone_letter)
        # Fixed fixtures: the footway/road with most nodes (positive id) and a ~500 m
        # route along the centreline of the footway with most nodes (ways are polygons).
        deleted = get_deleted_way_ids(load_annotations(str(data_dir / f"{name}.annotations.json")))
        ways = [
            w
            for w in md.footways_list + md.roads_list
            if isinstance(w.id, int) and w.id > 0 and w.nodes and w.id not in deleted
        ]
        self.long_way = max(ways, key=lambda w: len(w.nodes))
        feet = sorted(
            (w for w in ways if w in md.footways_list),
            key=lambda w: -len(w.nodes),
        )
        for foot in feet:
            xy = [
                utm.from_latlon(md.nodes_cache[n]["lat"], md.nodes_cache[n]["lon"], *self.zone)[:2]
                for n in foot.nodes
            ]
            center = LineString(xy)
            length = min(500.0, center.length)
            pts = [center.interpolate(length * i / 4) for i in range(5)]
            self.route = [[float(c) for c in utm.to_latlon(p.x, p.y, *self.zone)] for p in pts]
            body = {"file": self.file, "points": self.route, "algorithm": "graph"}
            if self.client.post("/api/create_replan", json=body).get_json()["retrieveNum"] != 1:
                break  # the first footway the graph planner can route along
        else:
            raise SystemExit(f"{name}: no footway the graph planner can route along")
        self.bbox = md_bbox(self.route, self.zone, margin=60.0)
        self.ann = data_dir / f"{name}.annotations.json"
        self.ann_bytes = self.ann.read_bytes() if self.ann.exists() else None

    def reset_edits(self) -> None:
        if self.ann_bytes is None:
            self.ann.unlink(missing_ok=True)
        else:
            self.ann.write_bytes(self.ann_bytes)

    def record(self, label: str, cold: list[float], warm: list[float], resp: Any = None) -> None:
        size = len(resp.data) if resp is not None else 0
        row = {
            "map": self.name,
            "endpoint": label,
            "cold_p50": pct(cold, 0.5) if cold else None,
            "cold_p95": pct(cold, 0.95) if cold else None,
            "warm_p50": pct(warm, 0.5) if warm else None,
            "warm_p95": pct(warm, 0.95) if warm else None,
            "bytes": size,
            "gzip_bytes": len(gzip.compress(resp.data, 6)) if resp is not None else 0,
            "content_encoding": resp.headers.get("Content-Encoding") if resp is not None else None,
        }
        self.rows.append(row)

    def read(self, label: str, call: Callable[[], Any], *, slow: bool = False) -> None:
        resp = None
        cold_ms = []
        # A multi-second request (grid planner) gets fewer repetitions.
        reps, cold_reps = (min(self.reps, 3), 1) if slow else (self.reps, self.cold_reps)
        for _ in range(cold_reps):
            clear_caches()
            ms, resp = timed(call)
            assert resp.status_code == 200, (label, resp.status_code, resp.data[:200])
            assert b'"retrieveNum":1' not in resp.data.replace(b" ", b""), (label, resp.data[:200])
            cold_ms.append(ms)
        call()
        warm_ms = []
        for _ in range(reps):
            ms, resp = timed(call)
            assert resp.status_code == 200, (label, resp.status_code, resp.data[:200])
            warm_ms.append(ms)
        self.record(label, cold_ms, warm_ms, resp)

    def run_reads(self) -> None:
        f, c = self.file, self.client
        self.reset_edits()
        gz = {"Accept-Encoding": "gzip"}
        self.read("GET /api/mapdata", lambda: c.get("/api/mapdata", query_string={"file": f}))
        self.read(
            "GET /api/mapdata (gzip accepted)",
            lambda: c.get("/api/mapdata", query_string={"file": f}, headers=gz),
        )
        # The browser's revalidation: warm hit, 304.
        etag = c.get("/api/mapdata", query_string={"file": f}).headers["ETag"]
        resp304 = c.get("/api/mapdata", query_string={"file": f}, headers={"If-None-Match": etag})
        assert resp304.status_code == 304
        warm = [
            timed(
                lambda: c.get(
                    "/api/mapdata", query_string={"file": f}, headers={"If-None-Match": etag}
                )
            )[0]
            for _ in range(self.reps)
        ]
        self.record("GET /api/mapdata (If-None-Match, 304)", [], warm)
        self.read(
            "GET /api/annotations", lambda: c.get("/api/annotations", query_string={"file": f})
        )
        wid = self.long_way.id
        self.read(
            f"GET /api/ways/{wid}/segments",
            lambda: c.get(f"/api/ways/{wid}/segments", query_string={"file": f}),
        )
        self.read(
            f"GET /api/way_nodes ({len(self.long_way.nodes)} nodes)",
            lambda: c.get("/api/way_nodes", query_string={"file": f, "way_id": wid}),
        )
        for algo, sub in (("graph", None), ("rrt", "astar"), ("astar", "astar")):
            body = {"file": f, "points": self.route, "algorithm": algo}
            if sub:
                body["sub_algorithm"] = sub
            self.read(
                f"POST /api/create_replan {algo}",
                lambda body=body: c.post("/api/create_replan", json=body),
                slow=algo != "graph",
            )
        s, w, n, e = self.bbox
        q = {"file": f, "min_lat": s, "min_lon": w, "max_lat": n, "max_lon": e}
        self.read("GET /api/cost_grid", lambda: c.get("/api/cost_grid", query_string=q))
        self.read("GET /api/export", lambda: c.get("/api/export", query_string={"file": f}))
        self.read(
            "GET /api/export/geojson",
            lambda: c.get("/api/export/geojson", query_string={"file": f}),
        )

    def run_edit_cycle(self) -> None:
        """Move a node, set tags, delete a way, then the map GET after each (like the UI)."""
        f, c, wid = self.file, self.client, self.long_way.id
        q = {"file": f}
        steps: dict[str, list[float]] = {
            k: []
            for k in ("PUT way_nodes/move", "PUT tags", "DELETE way", "GET /api/mapdata after")
        }
        for _ in range(self.cold_reps + 2):
            self.reset_edits()
            c.get("/api/mapdata", query_string=q)
            node = c.get("/api/way_nodes", query_string={"file": f, "way_id": wid}).get_json()[
                "nodes"
            ][0]
            move = {"nodes": [{"id": node["id"], "lat": node["lat"] + 1e-5, "lon": node["lon"]}]}
            ms, r = timed(
                lambda move=move: c.put(
                    "/api/way_nodes/move", query_string={**q, "way_id": wid}, json=move
                )
            )
            assert r.status_code == 204, r.data
            steps["PUT way_nodes/move"].append(ms)
            ms, r = timed(lambda: c.get("/api/mapdata", query_string=q))
            steps["GET /api/mapdata after"].append(ms)
            ms, r = timed(
                lambda: c.put(
                    f"/api/ways/{wid}/tags", query_string=q, json={"tags": {"highway": "path"}}
                )
            )
            assert r.status_code == 204, r.data
            steps["PUT tags"].append(ms)
            ms, r = timed(lambda: c.get("/api/mapdata", query_string=q))
            steps["GET /api/mapdata after"].append(ms)
            ms, r = timed(lambda: c.delete(f"/api/ways/{wid}", query_string=q, json={}))
            assert r.status_code == 204, r.data
            steps["DELETE way"].append(ms)
            ms, r = timed(lambda: c.get("/api/mapdata", query_string=q))
            steps["GET /api/mapdata after"].append(ms)
        self.reset_edits()
        for label, samples in steps.items():
            self.record(f"edit: {label}", samples, [])


def md_bbox(
    latlon: list[list[float]], zone: tuple[int, str], margin: float
) -> tuple[float, float, float, float]:
    xy = [utm.from_latlon(la, lo, *zone)[:2] for la, lo in latlon]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    s, w = utm.to_latlon(min(xs) - margin, min(ys) - margin, *zone)
    n, e = utm.to_latlon(max(xs) + margin, max(ys) + margin, *zone)
    return s, w, n, e


def print_table(rows: list[dict[str, Any]]) -> None:
    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.1f}"

    print(f"{'endpoint':<44}{'cold p50':>10}{'cold p95':>10}{'warm p50':>10}{'warm p95':>10}  size")
    current = None
    for r in rows:
        if r["map"] != current:
            current = r["map"]
            print(f"--- {current}")
        size = ""
        if r["bytes"]:
            size = f"{r['bytes'] / 1024:.0f} KiB (gzip {r['gzip_bytes'] / 1024:.0f})"
            if r["content_encoding"]:
                size += f" [{r['content_encoding']}]"
        print(
            f"{r['endpoint']:<44}{fmt(r['cold_p50']):>10}{fmt(r['cold_p95']):>10}"
            f"{fmt(r['warm_p50']):>10}{fmt(r['warm_p95']):>10}  {size}"
        )


def profile(bench: Bench, what: str, top: int) -> None:
    f, c = bench.file, bench.client
    calls = {
        "mapdata": lambda: c.get("/api/mapdata", query_string={"file": f}),
        "annotations": lambda: c.get("/api/annotations", query_string={"file": f}),
        "export": lambda: c.get("/api/export", query_string={"file": f}),
        "way_nodes": lambda: c.get(
            "/api/way_nodes", query_string={"file": f, "way_id": bench.long_way.id}
        ),
        "segments": lambda: c.get(
            f"/api/ways/{bench.long_way.id}/segments", query_string={"file": f}
        ),
        "cost_grid": lambda: c.get(
            "/api/cost_grid",
            query_string=dict(zip(("min_lat", "min_lon", "max_lat", "max_lon"), bench.bbox))
            | {"file": f},
        ),
    }
    for algo in ("graph", "rrt", "astar"):
        calls[f"replan_{algo}"] = lambda a=algo: c.post(
            "/api/create_replan", json={"file": f, "points": bench.route, "algorithm": a}
        )
    for warm in (False, True):
        if not warm:
            clear_caches()
        else:
            calls[what]()
        prof = cProfile.Profile()
        prof.enable()
        calls[what]()
        prof.disable()
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(top)
        print(f"===== {what} ({'warm' if warm else 'cold'})\n{out.getvalue()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="directory of .mapdata files (default: demo/)")
    ap.add_argument("--quick", action="store_true", help="demo map only, few reps")
    ap.add_argument("--reps", type=int, default=20, help="warm requests per endpoint")
    ap.add_argument("--cold-reps", type=int, default=3, help="cold requests per endpoint")
    ap.add_argument("--maps", nargs="*", help="map stems to run (default: all in the dir)")
    ap.add_argument("--profile", help="cProfile one endpoint (mapdata, export, way_nodes, ...)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--only", choices=("reads", "edit"), help="skip the other group")
    ap.add_argument("--json", type=Path, help="also write the rows here")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.cold_reps, args.data_dir, args.maps = 5, 2, DEMO, ["buchlovice"]
    src = args.data_dir or DEMO
    stems = args.maps or sorted(p.stem for p in src.glob("*.mapdata"))

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="viewer-bench-") as tmp:
        for stem in stems:
            for p in src.glob(f"{stem}.*"):
                if p.suffix in (".mapdata", ".json"):
                    shutil.copy(p, tmp)
            bench = Bench(Path(tmp), stem, args.reps, args.cold_reps)
            if args.profile:
                profile(bench, args.profile, args.top)
                continue
            if args.only != "edit":
                bench.run_reads()
            if args.only != "reads":
                bench.run_edit_cycle()
            rows += bench.rows
    if rows:
        print_table(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
