# Tag configuration

The six CSV files in `parameters/` decide which OSM features become barriers. `MapData._load_tag_configs()` reads them at construction into the instance attributes `BARRIER_TAGS`, `NOT_BARRIER_TAGS`, `ANTI_BARRIER_TAGS`, `OBSTACLE_TAGS`, `NOT_OBSTACLE_TAGS` and `ANTI_OBSTACLE_TAGS`.

| Feature | Include | Wildcard exceptions | Override |
|---|---|---|---|
| Ways and multipolygon relations | `barrier_tags.csv` | `not_barrier_tags.csv` | `anti_barrier_tags.csv` |
| Standalone nodes | `obstacle_tags.csv` | `not_obstacle_tags.csv` | `anti_obstacle_tags.csv` |

---

## CSV format

Two columns, no header:

```
key,value
```

- `key`: OSM tag key (`barrier`, `natural`, `man_made`).
- `value`: OSM tag value, or `*` for any value of that key.

Rows with the same key merge into `{key: [value1, value2, ...]}`.

---

## Precedence

`Way.is_barrier(yes_tags, not_tags, anti_tags)` in `map_data/utils/way.py` decides both cases:

1. **Override.** A feature with any tag listed in the `anti_*` file is never a barrier.
2. **Exact match.** A tag whose exact `key,value` is in the include file makes the feature a barrier. The `not_*` file has no effect on exact matches.
3. **Wildcard match.** A tag whose key has `*` in the include file makes the feature a barrier, unless that `key,value` is in the `not_*` file.

So `not_*` only carves exceptions out of wildcard rules: `man_made,*` plus `man_made,bridge` in `not_barrier_tags.csv` keeps bridges traversable. To suppress an exact match, remove it from the include file or add it to the `anti_*` file.

Each tag is checked independently: a feature with one excepted tag and another matching tag is still a barrier.

---

## Barrier ways

`separate_ways` in `map_data/utils/parsing.py` sorts every parsed way:

- any `highway` value in `FOOTWAY_VALUES` → footway, buffered to `buffer_widths.footway` (default 3 m full width);
- any other `highway` value → road, buffered to `buffer_widths.road` (default 7 m);
- otherwise, if `is_barrier` with the barrier tables → barrier. Closed ways and multipolygons stay polygons; open lines are buffered to `buffer_widths.barrier` (default 2 m).

The `highway` check runs first, so a way tagged both `highway=*` and `barrier=*` is a road or footway.

`anti_barrier_tags.csv` holds `location,underground`, `location,underwater` and `location,overhead`: tunnels and overhead lines that share keys with surface barriers.

## Obstacle nodes

`parse_osm_nodes` checks nodes that are not part of any parsed way. A node that passes `is_barrier` with the obstacle tables becomes a circle of radius `obstacle_radius` (default 2 m) and is added to `barriers_list`.

`anti_obstacle_tags.csv` holds passable openings in walls (`barrier,entrance`, `barrier,coupure`, `barrier,sally_port`, `historic,city_gate`).

Buffer widths and the obstacle radius come from `config/planner_defaults.yaml`; see [Planner configuration](planner_config.md).

---

## Common edits

Make a specific value a barrier:

1. Add `key,value` to `barrier_tags.csv` (ways) or `obstacle_tags.csv` (nodes).
2. If the key already has a `*` rule, make sure the value is not in the matching `not_*` file.

Make a feature passable:

- covered by a wildcard rule: add `key,value` to the `not_*` file;
- an exact match, or a tag that should cancel every other match (`location,underground`): add it to the `anti_*` file.

Tag tables are read when `MapData` is constructed. After editing a CSV, regenerate the `.mapdata` file with `create_mapdata` or the viewer's GPX upload; existing `.mapdata` files keep their old classification.
