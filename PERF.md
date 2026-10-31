# Performance log

## Planner

Metric: per-goal latency of `plan_route` as `route_planner` and the viewer call it (map,
annotations and traversability loaded once; the graph planner cached, the grid planners
rebuilt per goal as both callers do). Benchmark: `scripts/bench/planner.py`, 5 footway-node
goal pairs 300 m to 2 km apart per map and algorithm (grid pairs also within
`MAX_GRID_CELLS` at the default 0.25 m cell), seed 7, RRT seeded, 3 repetitions; median ±
half the range of the per-repetition mean. `--compare` checked every route (length,
vertex count, SHA-1 of the vertices) against the baseline after each change: **no route
changed**. Machine: 16 cores, shared with other jobs (load ~5).

    uv run python scripts/bench/planner.py --quick        # tracked maps, ~25 s
    uv run python scripts/bench/planner.py --data-dir <dir with strom.mapdata ...> --json out.json [--compare base.json]

Where the time went (cProfile, baseline): the grid planners were 99% of it. `grid_astar`'s
pure-Python loop took 60–87% of an `astar` goal (~1 M expansions a leg, the search
covers 30–75% of the grid because the Euclidean heuristic is weak against the 5.5x
off-path step cost); `PathGrid.fill` most of the rest (a cKDTree query for every cell,
then `contains_xy` in `burn_obstacles`); RRT's own loop ~0.7 s a leg, and a failed RRT
leg falls back to A*. The graph planner was already 3–4 ms per goal warm and ≤ 82 ms
cold.

| Idea | Baseline -> Result (median ± spread) | Verdict | Why |
|---|---|---|---|
| `grid_astar`: column-major flat index as the heap tie-break (same order as `(ix, iy)`), precomputed heuristic and float32 diagonal-cost tables, float64 pre-check before the float32 rounding (rounding is monotonic) | astar stromovka 4682 ± 105 -> 3321 ± 77 ms; strom 3347 ± 76 -> 2437 ± 214; KN 800 ± 14 -> 557 ± 22 (all 6 maps 1.3–1.6x) | kept | Same expansions, less Python work each; routes bit-identical |
| `grid_astar`: plain lists instead of `array` for g scores / parents | captured strom grid 4.4 s -> 5.1 s (micro A/B) | reverted | Slower, and ~130 MB more per 4 M cells |
| `PathGrid.fill`: KD-query only cells within `max_path_dist` + 1 cell of a way point (`maximum_filter` mask; points off the grid clipped to its edge) | rrt stromovka 793 ± 17 -> 623 ± 3 ms; astar 3321 ± 77 -> 3107 ± 69; 11 of 12 grid cases faster beyond spread | kept | Most cells are far from every way; grid bit-identical on 30 real boxes |
| `burn_obstacles`: `shapely.prepare` each obstacle before `contains_xy` | fill 0.159 -> 0.156 s per goal (30 boxes) | reverted | No gain: `contains_xy` already prepares internally |
| RRT: vectorize the linear scans over nodes added since the last KD-tree rebuild (`_nearest_node`, `_get_near_nodes`), same first-minimum tie-break | rrt stromovka 623 ± 3 -> 540 ± 15 ms; strom 1620 ± 24 -> 1471 ± 6; all 6 maps −6…−14% | kept | Up to 49 numpy scalar ops per call became one |
| `grid_astar`: unroll the 8-neighbour loop | astar stromovka 3107 ± 69 -> 2754 ± 13 ms; buchlovice 1561 ± 22 -> 1334 ± 65; all 6 maps −6…−15% | kept | Tuple building/unpacking per neighbour was ~10% of the search |
| RRT `_segment_cost`: numpy over the Bresenham cells (`cumsum` keeps the float32 running total) | rrt stromovka 540 ± 15 -> 628 ± 11 ms; stromovka_full 598 ± 11 -> 696 ± 13 | reverted | ~10 cells per segment: array overhead beats the scalar loop |
| `GraphPlanner`: build all edge `LineString`s with one `shapely.linestrings` call | graph cold strom 82 ± 2 -> 54 ± 13 ms; kralovska_obora 65 ± 1 -> 41 ± 1; stromovka 23 ± 1 -> 16 ± 1 | kept | 20 k single-geometry constructions were ~40% of a cold build |

End to end (baseline -> all kept changes, same run conditions as above):

| Map | graph cold | graph warm | astar | rrt |
|---|---|---|---|---|
| stromovka (tracked) | 23.1 ± 0.5 -> 15.3 ± 0.4 ms | 3.1 -> 2.8 ms | 4683 ± 105 -> 2870 ± 58 ms | 805 ± 10 -> 529 ± 4 ms |
| buchlovice (tracked) | – (no connected footway pair ≥ 300 m) | – | 2751 ± 107 -> 1448 ± 40 ms | 2347 ± 11 -> 1392 ± 99 ms |
| strom | 81.7 ± 2.1 -> 56.6 ± 1.8 ms | 3.9 -> 4.0 ms | 3347 ± 76 -> 2216 ± 41 ms | 2226 ± 65 -> 1509 ± 35 ms |
| stromovka_full | 71.4 ± 1.3 -> 59.0 ± 1.2 ms | 3.7 -> 3.9 ms | 2693 ± 48 -> 1736 ± 26 ms | 819 ± 16 -> 601 ± 21 ms |
| kralovska_obora | 65.3 ± 1.3 -> 44.9 ± 1.6 ms | 3.6 -> 3.4 ms | 2960 ± 50 -> 1815 ± 41 ms | 3099 ± 51 -> 1833 ± 15 ms |
| KN | 24.0 ± 9.2 -> 16.2 ± 0.3 ms | 2.9 -> 2.8 ms | 800 ± 14 -> 456 ± 2 ms | 841 ± 40 -> 520 ± 5 ms |

Graph warm is unchanged by design (±0.2 ms is run-to-run drift between sessions).

Left alone: `grid_astar` is now near its pure-Python floor (heap operations alone are
~25%); a bigger win needs a different search (fewer expansions changes which of several
equal-cost routes is returned) or native code, both out of scope here. `MapData.get_points`
(~30 ms per grid goal) belongs to map loading.
