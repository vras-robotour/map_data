"""
Per-goal latency of ``plan_route`` as ``route_planner`` and the viewer call it.

The map (with its annotation store and the default traversability rules) is loaded
once per map and not timed. For ``graph`` the cold time builds the
:class:`GraphPlanner` and plans the first goal; the warm time plans on a cached
planner, as both callers do. The grid planners (``astar``, ``rrt``) cache nothing
between goals, so every goal is cold.

Goals are footway-node pairs drawn with a fixed seed, 300 m to 2 km apart (grid
pairs also within ``MAX_GRID_CELLS`` at the default cell size). RRT is seeded too.
``--json`` saves timings and a fingerprint of every route; ``--compare`` diffs the
fingerprints against a saved run.

    uv run python scripts/bench/planner.py --quick
    uv run python scripts/bench/planner.py --json /tmp/base.json
    uv run python scripts/bench/planner.py --compare /tmp/base.json
"""

import argparse
import hashlib
import json
import logging
import random
import statistics
import time
from pathlib import Path

import numpy as np

import map_data.pathsolver.replan as replan
from map_data.annotations import load_mapdata_with_annotations
from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.pathsolver.route import (
    GRID_MARGIN_M,
    MAX_GRID_CELLS,
    RoutePlanningError,
    plan_route,
    utm_path_to_latlon_pairs,
)

ROOT = Path(__file__).resolve().parents[2]
TRACKED = [ROOT / "tests/data/stromovka.mapdata", ROOT / "demo/buchlovice.mapdata"]
# Untracked real maps looked up in --data-dir; missing ones are skipped.
LOCAL = ("strom.mapdata", "stromovka_full.mapdata", "kralovska_obora.mapdata", "KN.mapdata")
ALGORITHMS = ("graph", "astar", "rrt")
SEED = 7
MIN_DIST, MAX_DIST = 300.0, 2000.0

_load_config = replan.load_config


def _seeded_config(name: str) -> dict:
    """planner_defaults.yaml with ``rrt.seed`` set, so RRT routes are reproducible."""
    cfg = _load_config(name)
    if name == "planner_defaults.yaml":
        cfg = {**cfg, "rrt": {**cfg.get("rrt", {}), "seed": SEED}}
    return cfg


replan.load_config = _seeded_config


def sample_pairs(md, planner: GraphPlanner, n: int, grid: bool, rng: random.Random) -> list:
    """
    *n* footway-node pairs MIN_DIST..MAX_DIST apart: for *grid*, within MAX_GRID_CELLS;
    otherwise ones the graph planner connects.
    """
    xy = {k: planner.nodes[k].ravel()[:2] for k in planner.graph if isinstance(k, int) and k > 0}
    ids = sorted(xy)
    cell = float(replan.grid_defaults()["cell_size"])
    pairs = []
    for _ in range(20000):
        if len(pairs) == n or len(ids) < 2:
            break
        a, b = rng.sample(ids, 2)
        d = float(np.linalg.norm(xy[a] - xy[b]))
        if not MIN_DIST <= d <= MAX_DIST:
            continue
        if grid:
            side = np.abs(xy[a] - xy[b]) + 2 * GRID_MARGIN_M
            if side[0] * side[1] / cell**2 > MAX_GRID_CELLS:
                continue
        elif planner.plan(np.array([xy[a], xy[b]])) is None:
            continue
        pairs.append(
            utm_path_to_latlon_pairs(np.array([xy[a], xy[b]]), md.zone_number, md.zone_letter)
        )
    return pairs


def fingerprint(result) -> dict:
    digest = hashlib.sha1(np.round(result.utm, 3).tobytes()).hexdigest()[:12]
    return {"length_m": round(result.length_m, 3), "points": len(result.utm), "sha1": digest}


def run_goal(md, pts, algorithm, planner=None) -> tuple[float, dict]:
    t0 = time.perf_counter()
    try:
        result = plan_route(md, pts, algorithm=algorithm, planner=planner)
        fp = fingerprint(result)
    except RoutePlanningError as e:
        fp = {"error": e.reason}
    return time.perf_counter() - t0, fp


def bench_map(path: Path, goals: int, reps: int, algorithms) -> list[dict]:
    md, _ = load_mapdata_with_annotations(path)
    probe = GraphPlanner(md)
    cases = []
    for algorithm in algorithms:
        # A fresh stream per algorithm: --algorithms does not change the goals, and
        # astar and rrt plan the same ones.
        rng = random.Random(SEED)
        pairs = sample_pairs(md, probe, goals, algorithm != "graph", rng)
        if not pairs:
            print(f"  {path.name} {algorithm}: no pairs")
            continue
        cold, warm, fps = [], [], []
        if algorithm == "graph":
            for _ in range(reps):
                t0 = time.perf_counter()
                planner = GraphPlanner(md)
                run_goal(md, pairs[0], algorithm, planner)
                cold.append(time.perf_counter() - t0)
            planner = GraphPlanner(md)
            fps = [run_goal(md, p, algorithm, planner)[1] for p in pairs]  # warm the planner
        for _ in range(reps):
            times = []
            for i, p in enumerate(pairs):
                dt, fp = run_goal(md, p, algorithm, planner if algorithm == "graph" else None)
                times.append(dt)
                if len(fps) < len(pairs):
                    fps.append(fp)
                elif fps[i] != fp:
                    fps[i] = {"error": "nondeterministic"}
            warm.append(statistics.fmean(times))
        cases.append(
            {
                "map": path.name,
                "algorithm": algorithm,
                "goals": len(pairs),
                "cold": cold,
                "warm": warm,
                "routes": fps,
            }
        )
        print("  " + _row(cases[-1]), flush=True)
    return cases


def _ms(xs: list[float]) -> str:
    if not xs:
        return "-"
    med = statistics.median(xs) * 1e3
    spread = (max(xs) - min(xs)) / 2 * 1e3
    return f"{med:9.1f} ± {spread:6.1f}"


def _row(c: dict) -> str:
    errors = sum("error" in r for r in c["routes"])
    return (
        f"{c['map']:<26} {c['algorithm']:<6} {c['goals']:>3} goals  "
        f"cold {_ms(c['cold'])} ms  per-goal {_ms(c['warm'])} ms  errors {errors}"
    )


def compare(cases: list[dict], baseline_path: Path) -> int:
    base = {(c["map"], c["algorithm"]): c for c in json.loads(baseline_path.read_text())}
    changed = 0
    for c in cases:
        b = base.get((c["map"], c["algorithm"]))
        if b is None:
            continue
        for i, (r, br) in enumerate(zip(c["routes"], b["routes"], strict=False)):
            if r != br:
                changed += 1
                print(f"  ROUTE CHANGED {c['map']} {c['algorithm']} goal {i}: {br} -> {r}")
        print(
            f"  {c['map']:<26} {c['algorithm']:<6} per-goal {_ms(b['warm'])} -> {_ms(c['warm'])} ms"
        )
        if c["cold"] and b["cold"]:
            print(f"  {'':<26} {'':<6} cold     {_ms(b['cold'])} -> {_ms(c['cold'])} ms")
    print(f"{changed} route(s) changed")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tracked maps, fewer goals (< 60 s)")
    ap.add_argument("--goals", type=int, help="goals per map and algorithm")
    ap.add_argument("--reps", type=int, help="repetitions of every goal set")
    ap.add_argument("--maps", nargs="*", type=Path, help="mapdata files (default: all found)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data", help="where the LOCAL maps are")
    ap.add_argument("--algorithms", nargs="*", default=ALGORITHMS, choices=ALGORITHMS)
    ap.add_argument("--json", type=Path, help="save timings and route fingerprints")
    ap.add_argument("--compare", type=Path, help="diff route fingerprints against a --json run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    goals = args.goals or (2 if args.quick else 5)
    reps = args.reps or (2 if args.quick else 3)
    maps = args.maps or (
        TRACKED
        if args.quick
        else TRACKED + [args.data_dir / n for n in LOCAL if (args.data_dir / n).is_file()]
    )
    t0 = time.perf_counter()
    cases = []
    for path in maps:
        cases.extend(bench_map(path, goals, reps, args.algorithms))
    print(f"total {time.perf_counter() - t0:.1f} s")
    if args.json:
        args.json.write_text(json.dumps(cases, indent=1))
    if args.compare:
        raise SystemExit(1 if compare(cases, args.compare) else 0)


if __name__ == "__main__":
    main()
