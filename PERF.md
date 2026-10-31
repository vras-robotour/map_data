# Performance notes

## Map loading and osm_cloud

Benchmark: `uv run python scripts/bench/mapload.py --repeat 5 --inputs <maps> --osm-cache <cache> --gpx <gpx>`. Values are median ± median absolute deviation. The real-map run used `strom`, `stromovka_full`, `kralovska_obora`, `KN`, and `robotour`, plus a cached Buchlovice Overpass response.

| Idea | Baseline -> Result (median ± spread) | Verdict | Why |
|---|---|---|---|
| Replace repeated `np.unique` sorting in `create_grid` with a boolean cell mask | `strom` grid 1.752 ± 0.014 s -> 1.137 ± 0.012 s; `stromovka_full` 1.568 ± 0.004 s -> 1.128 ± 0.035 s; peak RSS 333.0 -> 302.5 MB | Kept | cProfile put ~1.36 s of `get_cloud` in `create_grid`, mostly sorting expanded cells. The resulting arrays are identical on `strom` and `KN`. |
| Drop the initial `np.unique` on reference cells and expand all input cells directly | `strom` grid 1.137 ± 0.012 s -> 1.048 ± 0.002 s, but peak RSS 302.5 -> 773.1 MB | Reverted | Faster, but the memory increase breaks the peak-memory objective. |
| Batch all OSM way nodes into one `utm.from_latlon` call in `parse_osm_ways` | Focused cached-response parse 0.0673 ± 0.0015 s -> 0.0425 ± 0.0001 s; full benchmark 0.0894 ± 0.0019 s -> 0.0569 ± 0.0006 s | Kept | cProfile showed repeated per-way UTM calls dominating `run_parse`. Parsed way count and output lists are unchanged. |
| Additional map-load changes | `strom` load remained ~0.44–0.46 s across baseline and final runs, with run-to-run spread up to ~0.023 s | Reverted / not attempted | No repeated-work candidate beat the measured spread, so speculative changes were not kept. |

Final real-map medians: loads 0.080–0.446 s; grids 0.140–1.109 s; cached-response parse 0.0569 ± 0.0006 s; peak RSS 302.3 MB. `uv run python scripts/bench/mapload.py --quick` completed in about 1.2 s on tracked maps.
