# Tag Configuration

The `parameters/` directory contains six CSV files that control how OSM tags are mapped to the three semantic categories used internally: **barrier**, **obstacle**, and **traversable way**. Modifying these files lets you tune which real-world features are treated as physical obstacles without changing any Python code.

---

## Overview

OSM features carry arbitrary `key=value` tag pairs. The classification pipeline reads all tags on a feature and applies three ranked rule sets for each category. The rules are loaded by `MapData._load_tag_configs()` at construction time and are available as instance attributes (`BARRIER_TAGS`, `NOT_BARRIER_TAGS`, etc.).

---

## Barrier classification (ways and polygons)

Barriers are OSM *ways* (linear or area features) — walls, buildings, water bodies, fences, etc. — that the planner treats as impassable polygons.

### `barrier_tags.csv` — inclusion list

A way is a **candidate barrier** if any of its tags matches a row in this file.

Representative rows:

```
barrier,*
waterway,*
man_made,*
building,*
natural,water
natural,cliff
leisure,playground
```

The wildcard `*` in the value column means *any value for this key*.

### `not_barrier_tags.csv` — exclusion list

If a way matches `barrier_tags.csv` but also matches a row in `not_barrier_tags.csv`, it is **removed from barrier consideration**. This handles cases where a broad wildcard rule would otherwise capture features that are clearly traversable.

Representative rows:

```
man_made,bridge
man_made,pier
amenity,parking
amenity,university
sport,orienteering
```

For example, `man_made,*` in `barrier_tags.csv` would classify a bridge as a barrier. The `man_made,bridge` row in `not_barrier_tags.csv` prevents that.

### `anti_barrier_tags.csv` — override list (highest priority)

If a way matches any row in `anti_barrier_tags.csv`, it is **unconditionally removed from the barrier list**, regardless of what `barrier_tags.csv` or `not_barrier_tags.csv` say. This is the highest-priority rule.

All rows (the file is short):

```
location,underground
location,underwater
location,overhead
```

These cover tunnels, underwater infrastructure, and overhead cables that share OSM keys with surface barriers but are not physical obstacles at ground level.

---

## Obstacle classification (standalone nodes)

Obstacles are OSM *nodes* (point features) that are not part of any way — individual bollards, trees, gateposts, information terminals, etc. They are buffered to a 2 m radius circle and appended to `barriers_list`.

### `obstacle_tags.csv` — inclusion list

A standalone node is a **candidate obstacle** if any of its tags matches a row in this file.

Representative rows:

```
barrier,block
barrier,bollard
barrier,gate
waterway,*
natural,tree
information,board
```

### `not_obstacle_tags.csv` — exclusion list

Nodes that match `obstacle_tags.csv` but also match a row here are excluded.

```
barrier,entrance
barrier,coupure
man_made,surveillance
historic,archaeological_site
```

### `anti_obstacle_tags.csv` — override list

Nodes matching any row here are unconditionally excluded from obstacle classification.

```
barrier,entrance
barrier,coupure
barrier,sally_port
historic,city_gate
```

---

## CSV format

Each file uses two columns with no header row:

```
key,value
```

- `key` — the OSM tag key (e.g. `barrier`, `natural`, `man_made`)
- `value` — the OSM tag value, or `*` to match any value for that key

The files are read with `numpy.genfromtxt(..., dtype=str, delimiter=",")`. Rows with duplicate keys are merged into a list: `{key: [value1, value2, ...]}`.

---

## Evaluation order

For a given OSM feature, the classification logic in `map_data/utils/parsing.py` (`separate_ways`, `parse_osm_nodes`) applies the three rule sets in the following priority order:

```
anti_* (highest)  >  not_*  >  *_tags (lowest)
```

A match in `anti_*` always wins. A match in `not_*` overrides a match in `*_tags`. Only if neither exclusion list matches does the inclusion list apply.

---

## How to add a new barrier type

To classify `amenity,playground` as a barrier (it is already in `barrier_tags.csv` as the broad `amenity,*` rule, but this shows the procedure for a specific value):

1. Open `parameters/barrier_tags.csv`.
2. Add a line:
   ```
   amenity,playground
   ```
3. Ensure `amenity,playground` is **not** present in `not_barrier_tags.csv`. If it is, remove that row.
4. Re-run `create_mapdata` (or call `map_data_instance.run_parse()`) to regenerate the classification with the updated rules.

---

## How to make a barrier traversable

To prevent a feature from being classified as a barrier — for example, to allow passage through `barrier,gate` nodes at ground level — add a row to the appropriate exclusion file:

- Add `barrier,gate` to `not_obstacle_tags.csv` to exclude gate nodes from obstacle classification.
- Add `man_made,bridge` to `not_barrier_tags.csv` (it is already there) to exclude bridge ways from barrier classification.

After editing any CSV file, re-run `create_mapdata` or call `MapData.run_parse()` for the change to take effect. The tag configuration is loaded once at `MapData` construction and is not re-read during the viewer session.
