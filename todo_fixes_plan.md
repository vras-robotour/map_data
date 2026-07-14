# Implementation Plan: Todo Fixes Batch

Address a curated batch of 11 self-contained, high-impact items from the [backlog](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/docs/dev/todo_items.json) across bugs, testing, CI, and documentation — all on a new `todo-fixes` branch.

## Branch Strategy

```bash
git checkout -b todo-fixes master
```

All changes are committed to this branch. No changes to `master` until PR review.

---

## Proposed Changes

### Bug Fixes — `map_data/utils/`

---

#### 1. [MODIFY] `way.py` — `to_pcd_points` ignores `density` for linestrings

**Lines:** [163–174](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/utils/way.py#L163-L174)

The linestring branch hardcodes `get_point_line(p1, p2, 0.5)`. Since `get_point_line` takes a **step size** and `density` is **points per metre**, the correct value is `1 / density`.

```diff
         else:
             # For LineString or unfilled Polygon
             geom = self.line
             coords = list(geom.exterior.coords if hasattr(geom, "exterior") else geom.coords)

             pcd_points = np.empty((0, 2))
+            step = 1.0 / density
             for i in range(len(coords) - 1):
                 p1 = Point(coords[i])
                 p2 = Point(coords[i + 1])
-                _, line, _ = get_point_line(p1, p2, 0.5)
+                _, line, _ = get_point_line(p1, p2, step)
                 pcd_points = np.concatenate([pcd_points, line])
             self.pcd_points = pcd_points
```

---

#### 2. [MODIFY] `way.py` — `to_pcd_points` cache ignores arguments

**Lines:** [138–139](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/utils/way.py#L138-L139)

The first call caches `pcd_points`; subsequent calls with different `density`/`filled` return stale data. Fix: key the cache on the arguments.

```diff
-        if self.pcd_points is not None:
-            return self.pcd_points
+        cache_key = (density, filled)
+        if self._pcd_cache_key == cache_key and self.pcd_points is not None:
+            return self.pcd_points
```

At the end of both branches, after `self.pcd_points = ...`:
```diff
+        self._pcd_cache_key = cache_key
```

Also add `_pcd_cache_key: tuple | None = None` to the `Way` dataclass init field with `field(default=None, repr=False)`.

---

#### 3a. [MODIFY] `gpx.py` — `parse_gpx_file` returns inconsistent shape on empty input

**Lines:** [48–56](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/utils/gpx.py#L48-L56)

When no waypoints are found, the function returns `(np.array([]), None, None)` — a 1-D empty array with `None` zone values. Fix: return `[]` early (matching the documented error contract in `parse_path`).

```diff
     if not waypoints:
         logger.warning("No waypoints found in GPX file.")
-    else:
+        return []
+    else:
         logger.info("Parsed %s waypoints from GPX file.", len(waypoints))
         zone_num, zone_let = utm.from_latlon(gpx.waypoints[0].latitude, gpx.waypoints[0].longitude)[
             2:
         ]

-    return np.array(waypoints), zone_num, zone_let
+    return np.array(waypoints), zone_num, zone_let
```

Same pattern for `parse_yaml_file` (lines 76–85) — if `not waypoints`, return `[]` early.

---

#### 3b. [MODIFY] `gpx.py` — `parse_gpx_file` only reads waypoints, not tracks or routes

**Lines:** [38–44](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/utils/gpx.py#L38-L44)

`MapData.__init__` uses the fallback `waypoints → tracks → routes` ([map_data.py:126-134](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/map_data.py#L126-L134)), but `parse_gpx_file` only iterates `gpx.waypoints`. A track-only GPX works for map download but not as replan CLI path input.

Fix: add the same fallback chain. GPX track/route points expose the same `.latitude`, `.longitude`, `.elevation` attributes as waypoints, so `convert_waypoint` works unchanged.

```diff
     try:
         with Path(gpx_file).open() as file:
             gpx = gpxpy.parse(file)
-        for waypoint in gpx.waypoints:
+
+        # Mirror the waypoints → tracks → routes fallback from MapData.__init__.
+        if gpx.waypoints:
+            gpx_points = gpx.waypoints
+        elif gpx.tracks:
+            gpx_points = [
+                p for track in gpx.tracks for seg in track.segments for p in seg.points
+            ]
+        elif gpx.routes:
+            gpx_points = [p for route in gpx.routes for p in route.points]
+        else:
+            gpx_points = []
+
+        for waypoint in gpx_points:
             point = {
                 "lat": waypoint.latitude,
                 "lon": waypoint.longitude,
                 "ele": waypoint.elevation or 0,
             }
             waypoints.append(convert_waypoint(point))
```

And the zone extraction must use the first point from the resolved list, not `gpx.waypoints[0]`:

```diff
-        zone_num, zone_let = utm.from_latlon(gpx.waypoints[0].latitude, gpx.waypoints[0].longitude)[
+        zone_num, zone_let = utm.from_latlon(gpx_points[0].latitude, gpx_points[0].longitude)[
             2:
         ]
```

---

### Bug Fixes — `map_data/pathsolver/`

---

#### 4. [MODIFY] `rrt_star.py` — `__main__` demo is broken

**Lines:** [556–557](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/pathsolver/rrt_star.py#L556-L557)

Start/goal are plain tuples; `self.goal - self.start` in `__init__` raises `TypeError`.

```diff
-    start = (0.0, 0.0)
-    goal = (10.0, 10.0)
+    start = np.array([0.0, 0.0])
+    goal = np.array([10.0, 10.0])
```

---

#### 5. [MODIFY] `replan.py` — Hardcoded fallback `sand` cost drifts from YAML

**Line:** [90](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/map_data/pathsolver/replan.py#L90)

Fallback says `sand: 0.7` but [planner_defaults.yaml](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/config/planner_defaults.yaml#L21) says `sand: 0.4`.

```diff
-            "sand": 0.7,
+            "sand": 0.4,
```

---

### Testing

---

#### 6. [NEW] `tests/test_osm_cloud.py` — Unit tests for pure functions

The four pure functions `create_grid`, `points_near_ref`, `split_ways_to_points`, and `transform_points` are completely untested (no rclpy required). Tests stay pure-function only — no node-init mock.

```python
"""Tests for osm_cloud pure helper functions."""

import numpy as np
import pytest

from map_data.osm_cloud import (
    create_grid,
    points_near_ref,
    split_ways_to_points,
    transform_points,
)


# ── create_grid ───────────────────────────────────────────────────────────────


class TestCreateGrid:
    def test_basic_grid_shape(self):
        grid = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=1.0)
        assert grid.ndim == 2
        assert grid.shape[1] == 2
        assert grid.shape[0] == 4  # 2×2 = 4 points

    def test_cell_size_controls_density(self):
        coarse = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=1.0)
        fine = create_grid((0.0, 0.0), (2.0, 2.0), cell_size=0.5)
        assert fine.shape[0] > coarse.shape[0]

    def test_empty_range_returns_empty(self):
        grid = create_grid((5.0, 5.0), (5.0, 5.0), cell_size=1.0)
        assert grid.shape[0] == 0

    def test_grid_bounds_within_range(self):
        grid = create_grid((0.0, 0.0), (3.0, 3.0), cell_size=0.5)
        assert grid[:, 0].min() >= 0.0
        assert grid[:, 0].max() < 3.0
        assert grid[:, 1].min() >= 0.0
        assert grid[:, 1].max() < 3.0


# ── points_near_ref ──────────────────────────────────────────────────────────


class TestPointsNearRef:
    def test_filters_distant_points(self):
        points = np.array([[0.0, 0.0], [10.0, 10.0], [0.5, 0.5]])
        ref = np.array([[0.0, 0.0]])
        result = points_near_ref(points, ref, max_dist=2.0)
        assert result.shape[0] == 2  # (0,0) and (0.5,0.5)
        assert result.shape[1] == 3  # 2 coords + cost

    def test_cost_is_normalized_distance(self):
        points = np.array([[1.0, 0.0]])
        ref = np.array([[0.0, 0.0]])
        result = points_near_ref(points, ref, max_dist=2.0)
        np.testing.assert_almost_equal(result[0, 2], 0.5)  # 1.0 / 2.0

    def test_no_points_near_ref(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        ref = np.array([[100.0, 100.0]])
        result = points_near_ref(points, ref, max_dist=1.0)
        assert result.shape[0] == 0

    def test_accepts_list_inputs(self):
        result = points_near_ref([[0.0, 0.0]], [[0.0, 0.0]], max_dist=1.0)
        assert result.shape[0] == 1


# ── transform_points ────────────────────────────────────────────────────────


class TestTransformPoints:
    def test_identity_preserves_points(self):
        pts = {0: np.array([[1.0], [2.0], [3.0]])}
        result = transform_points(pts, np.eye(4))
        np.testing.assert_array_almost_equal(result[0], pts[0])

    def test_translation(self):
        pts = {0: np.array([[0.0], [0.0], [0.0]])}
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        result = transform_points(pts, T)
        np.testing.assert_array_almost_equal(result[0].ravel(), [1.0, 2.0, 3.0])

    def test_z_override(self):
        pts = {0: np.array([[1.0], [2.0], [3.0]])}
        result = transform_points(pts, np.eye(4), z=0.0)
        assert result[0][2, 0] == 0.0

    def test_type_error_on_non_array(self):
        with pytest.raises(TypeError):
            transform_points({0: [1.0, 2.0, 3.0]}, np.eye(4))

    def test_multiple_points(self):
        pts = {
            0: np.array([[1.0], [0.0], [0.0]]),
            1: np.array([[0.0], [1.0], [0.0]]),
        }
        result = transform_points(pts, np.eye(4))
        assert len(result) == 2


# ── split_ways_to_points ────────────────────────────────────────────────────


class TestSplitWaysToPoints:
    def test_empty_ways_dict(self):
        result = split_ways_to_points({}, {})
        assert result.shape == (0, 2)

    def test_no_footways_key(self):
        result = split_ways_to_points({}, {"roads": []})
        assert result.shape == (0, 2)

    def test_empty_footways_list(self):
        result = split_ways_to_points({}, {"footways": []})
        assert result.shape == (0, 2)
```

---

#### 7. [MODIFY] `ci.yml` + `pyproject.toml` — `pytest-cov` coverage in CI

##### [MODIFY] [pyproject.toml](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/pyproject.toml#L30-L36)
```diff
 dev = [
     "pytest",
+    "pytest-cov",
     "ruff",
     "mkdocs",
     "mkdocs-material",
     "mkdocstrings[python]",
 ]
```

##### [MODIFY] [ci.yml](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/.github/workflows/ci.yml#L50-L51)
```diff
       - name: Run tests
-        run: pytest tests/
+        run: pytest tests/ --cov=map_data --cov-report=term-missing
```

No threshold enforced — informational only for now.

---

### CI / DX

---

#### 8. [NEW] `.pre-commit-config.yaml`

Mirrors what CI already enforces via ruff. Runs locally before commit.

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.13
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

##### [MODIFY] [pyproject.toml](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/pyproject.toml#L30-L36) — add `pre-commit` to dev deps

```diff
 dev = [
     "pytest",
     "pytest-cov",
+    "pre-commit",
     "ruff",
```

---

### Documentation

---

#### 9. [MODIFY] `changelog.md` — Add Unreleased section

**Lines:** [1–3](file:///Users/vlkjan/Documents/projekty/CVUT/PhD/map_data/docs/dev/changelog.md#L1-L3)

```diff
 # Changelog

+## [Unreleased]
+
+### Fixed
+
+- Fixed `Way.to_pcd_points` ignoring the `density` parameter for linestring geometries
+- Fixed `Way.to_pcd_points` cache returning stale results when called with different arguments
+- Fixed `parse_gpx_file` / `parse_yaml_file` returning inconsistent shapes on empty input
+- Fixed `parse_gpx_file` only reading waypoints — now falls back to tracks and routes
+- Fixed RRT* `__main__` demo crashing due to tuple start/goal (requires `np.array`)
+- Synced hardcoded fallback `sand` surface cost (`0.7` → `0.4`) with `planner_defaults.yaml`
+
+### Added
+
+- Unit tests for `osm_cloud` pure helper functions (`create_grid`, `points_near_ref`, `transform_points`, `split_ways_to_points`)
+- `pytest-cov` coverage reporting in CI
+- `.pre-commit-config.yaml` for local ruff lint/format checks
+
 ## [1.2.0] — 2026-07-14
```

---

#### 10. [MODIFY] `todo_items.json` — Mark resolved items

Remove the entries resolved by this batch:
- "Way.to_pcd_points ignores its density parameter for linestrings"
- "Way.to_pcd_points cache ignores its arguments"
- "parse_gpx_file returns an inconsistent shape on empty input"
- "parse_gpx_file only reads waypoints, not tracks or routes"
- "RRTStar __main__ demo is broken"
- "Hardcoded fallback costs drift from planner_defaults.yaml"
- "Measure coverage in CI" (resolved)
- "Changelog has no entries for unreleased work" (resolved)

Partially resolved (update `desc` to note what's done):
- "Unit-test osm_cloud's pure functions and node init" → pure functions done, rclpy node-init test deferred

---

## Summary Table

| # | Category | Item | Severity | File(s) |
|---|----------|------|----------|---------|
| 1 | Bug | `to_pcd_points` density ignored | Minor | `way.py` |
| 2 | Bug | `to_pcd_points` stale cache | Minor | `way.py` |
| 3a | Bug | `parse_gpx_file` empty return | Minor | `gpx.py` |
| 3b | Improvement | `parse_gpx_file` tracks/routes fallback | Minor | `gpx.py` |
| 4 | Bug | RRT* demo broken | Nice-to-have | `rrt_star.py` |
| 5 | Improvement | Fallback `sand` cost drift | Nice-to-have | `replan.py` |
| 6 | Testing | osm_cloud pure-function tests | Important | `test_osm_cloud.py` **[NEW]** |
| 7 | Testing | Coverage in CI | Nice-to-have | `ci.yml`, `pyproject.toml` |
| 8 | CI/DX | Pre-commit config | Minor | `.pre-commit-config.yaml` **[NEW]** |
| 9 | Docs | Changelog unreleased section | Nice-to-have | `changelog.md` |
| 10 | Docs | Update todo backlog | — | `todo_items.json` |

---

## User Review Required

> [!IMPORTANT]
> **Pre-commit hook version**: Pinned to `ruff-pre-commit` `v0.11.13`. Let me know if you prefer a different version.

> [!IMPORTANT]
> **Coverage threshold**: Added as informational only (`--cov-report=term-missing`, no `--cov-fail-under`). We can add a minimum threshold in a follow-up.

---

## Verification Plan

### Automated Tests

```bash
# Run the full test suite (including new osm_cloud tests) with coverage
pytest tests/ -v --cov=map_data --cov-report=term-missing

# Verify ruff passes (matching CI)
ruff check .
ruff format --check .
```

### Manual Verification

1. **Density fix**: Inspect that `get_point_line` is called with `1/density` in the linestring branch
2. **Cache fix**: Call `to_pcd_points` twice with different `density`/`filled` → verify different results
3. **Empty GPX**: `parse_gpx_file` on an empty GPX returns `[]`, not `(array, None, None)`
4. **Tracks/routes**: `parse_gpx_file` on a track-only GPX returns UTM waypoints with valid zone info
5. **RRT* demo**: `python -m map_data.pathsolver.rrt_star` runs without `TypeError`
6. **Sand cost**: Fallback in `replan.py` matches `planner_defaults.yaml` (both `0.4`)
