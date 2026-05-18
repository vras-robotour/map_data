# Planner Configuration

All default planning parameters are stored in `config/planner_defaults.yaml`. The file is loaded once at import time by `map_data/map_data.py` (which reads `osm_margin` and `reserve_margin`) and by `ReplanPath` (which reads all cost and grid parameters). Values can be overridden at runtime through the viewer's Highway Costs modal or via CLI flags such as `--cell_size` and `--inflate_obstacles`.

---

## Full default file

```yaml
highway_costs:
  pedestrian: 0.0
  footway: 0.0
  path: 0.1
  living_street: 0.1
  track: 0.3
  service: 0.3
  residential: 0.5
  unclassified: 0.5
  tertiary: 0.7
  secondary: 0.9
  primary: 1.0
surface_costs:
  asphalt: 0.0
  paving_stones: 0.0
  concrete: 0.0
  fine_gravel: 0.1
  gravel: 0.2
  dirt: 0.3
  grass: 0.5
  sand: 0.4
default_off_path_cost: 0.9
max_path_dist: 2.0
cell_size: 0.25
inflate_obstacles: 0.25
simplify_path: true
smooth_path: false
osm_margin: 100
reserve_margin: 50
path_cost_cap: 0.85
```

---

## `highway_costs`

Assigns a base traversal cost to each OSM `highway=*` value. The cost scale runs from `0.0` (freely preferred) to `1.0` (equivalent to an obstacle). Costs are capped at `path_cost_cap` so that even the most expensive way type remains cheaper than off-path terrain.

| OSM highway type | Default cost | Interpretation |
|-----------------|-------------|----------------|
| `pedestrian` | 0.0 | Pedestrian-only street or square |
| `footway` | 0.0 | Dedicated footpath |
| `path` | 0.1 | Informal path |
| `living_street` | 0.1 | Shared pedestrian/vehicle area |
| `track` | 0.3 | Unpaved agricultural or forestry track |
| `service` | 0.3 | Private or access road |
| `residential` | 0.5 | Residential street |
| `unclassified` | 0.5 | Minor road, no specific classification |
| `tertiary` | 0.7 | Local connecting road |
| `secondary` | 0.9 | Regional road |
| `primary` | 1.0 | Major road (capped to `path_cost_cap` = 0.85 in practice) |

Highway types not listed in the YAML receive `default_off_path_cost`.

---

## `surface_costs`

An additive penalty applied on top of the highway cost when the way carries a `surface=*` tag. If no `surface` tag is present, the penalty is 0.0.

| OSM surface value | Default penalty | Notes |
|------------------|----------------|-------|
| `asphalt` | 0.0 | Hard, smooth |
| `paving_stones` | 0.0 | Hard, smooth |
| `concrete` | 0.0 | Hard, smooth |
| `fine_gravel` | 0.1 | Compact gravel |
| `gravel` | 0.2 | Loose gravel |
| `dirt` | 0.3 | Earthen track |
| `sand` | 0.4 | Soft, slow |
| `grass` | 0.5 | Natural grass |

Surface values not listed receive a penalty of 0.0.

---

## Top-level parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_off_path_cost` | float | 0.9 | Cost assigned to grid cells that are not within `max_path_dist` metres of any way. Represents hard off-road terrain. |
| `max_path_dist` | float (m) | 2.0 | Radius around each way centerline within which cells receive the way's highway/surface cost. Cells beyond this radius use `default_off_path_cost`. |
| `cell_size` | float (m) | 0.25 | Side length of each grid cell used by Grid A* and RRT*. Smaller values give finer paths but increase memory and computation time. |
| `inflate_obstacles` | float (m) | 0.25 | Safety buffer added around all barrier polygons before rasterisation. Increases the clearance between the planned path and physical obstacles. |
| `simplify_path` | bool | `true` | Apply Douglas-Peucker simplification to the output path after planning. Reduces the number of waypoints while preserving the overall shape. |
| `smooth_path` | bool | `false` | Apply gradient-descent smoothing after planning (and after simplification if enabled). Produces rounder curves but may shift the path slightly away from the original grid solution. |
| `osm_margin` | int (m) | 100 | Metres added to each side of the waypoint bounding box when constructing the Overpass API query. Ensures features near the route boundary are included. |
| `reserve_margin` | int (m) | 50 | Additional metres added on top of `osm_margin` for the internal UTM bounding box (`min_x/max_x/min_y/max_y`). Used to clip the planning grid with a small safety margin. |
| `path_cost_cap` | float | 0.85 | Maximum cost a way cell can receive after adding highway and surface penalties. Ensures that all recognised way types remain cheaper than `default_off_path_cost` (0.9), so the planner always prefers a way over open terrain. |

---

## Runtime overrides

### Viewer Highway Costs modal

The viewer exposes a modal panel where `highway_costs` values can be edited per session. Changes take effect immediately for the next path planning request without reloading the `.mapdata` file.

### CLI flags

The `ReplanPath` constructor accepts `--cell_size` and `--inflate_obstacles` as command-line arguments. These override the YAML defaults for that invocation only; the YAML file is not modified.

### Programmatic override

```python
from map_data.pathsolver.replan import ReplanPath

planner = ReplanPath(
    map_data=md,
    cell_size=0.5,          # coarser grid for faster planning
    inflate_obstacles=0.5,  # wider obstacle clearance
)
```

Any keyword argument accepted by `ReplanPath.__init__` takes precedence over the YAML defaults.
