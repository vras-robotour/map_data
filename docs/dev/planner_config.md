# Planner Configuration

All default planning parameters are stored in `config/planner_defaults.yaml`. The file is
loaded once at import time by four modules, each reading a different subset:

| Module | Keys it reads |
|--------|---------------|
| `map_data/map_data.py` | `grid_margin` |
| `map_data/utils/parsing.py` | `obstacle_radius`, `buffer_widths` |
| `map_data/pathsolver/replan.py` (`ReplanPath`) | `highway_costs`, `surface_costs`, `default_off_path_cost`, `path_cost_cap`, `grid_cost_weight` |
| `map_data/pathsolver/grid_astar.py`, `rrt_star.py` | `grid_cost_weight` |

Every key is a *default* — each is overridable per call, either through a keyword argument
(see [Runtime overrides](#runtime-overrides)) or through the viewer, which reads the whole
file over `GET /api/planner_defaults` and posts explicit values back with each request.

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
grid_margin: 150
path_cost_cap: 0.85
grid_cost_weight: 5.0
obstacle_radius: 2.0
buffer_widths:
  road: 7.0
  footway: 3.0
  barrier: 2.0
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
| `grid_margin` | float (m) | 150 | Metres added to each side of the waypoint bounding box, in UTM. This single margin sets both the internal planning bounds (`min_x/max_x/min_y/max_y`) *and*, after conversion back to WGS84, the Overpass query bounding box — see [Bounding box and margins](../usage.md#bounding-box-and-margins). Increase it for routes near dense urban areas with large building footprints; decrease it to cut query time and file size on simple open terrain. |
| `path_cost_cap` | float | 0.85 | Maximum cost a way cell can receive after adding highway and surface penalties. Ensures that all recognised way types remain cheaper than `default_off_path_cost` (0.9), so the planner always prefers a way over open terrain. |
| `grid_cost_weight` | float | 5.0 | How strongly terrain cost is weighted against distance in the search. Both Grid A\* and RRT\* score a step as `length × (1 + cell_cost × grid_cost_weight)`, so at the default a metre of off-path terrain (`cell_cost` 0.9) costs the same as 5.5 m of free footway. Raise it to hug good surfaces at the price of longer detours; lower it toward `0` to approach a shortest-distance planner. |
| `obstacle_radius` | float (m) | 2.0 | Radius of the circular footprint generated for *point* obstacles — OSM nodes tagged as barriers (bollards, gates, trees), which have no geometry of their own. Way and area barriers use `buffer_widths` instead. |

!!! warning "`grid_margin` is applied at parse time"
    `grid_margin` is baked into a `.mapdata` file when it is created, because it determines
    the area that was downloaded. Changing the value does not retroactively widen an
    existing file — re-download with `create_mapdata -d` (or the viewer's fetch panel) for
    the new margin to take effect.

---

## `buffer_widths`

Linear OSM features have no width, so each parsed way is buffered into a polygon before it
reaches the planner. `buffer_widths` sets the buffer distance per category, in metres.

| Key | Default | Applies to |
|-----|---------|-----------|
| `road` | 7.0 | Ways classified as roads (`highway=*` outside the footway value set) |
| `footway` | 3.0 | Ways classified as footways |
| `barrier` | 2.0 | Non-area barrier ways (walls, fences); barriers already mapped as closed areas keep their own geometry |

Each value is the **total** corridor width: `buffer_line` buffers the centerline by
`width / 2` on both sides, so `road: 7.0` produces a 7 m-wide polygon. Roads are
deliberately wider than footways so the planner keeps clear of the carriageway.

Closed ways that are not tagged `area=yes` (roundabouts, loop paths) are converted back to
a `LineString` before buffering, so they become an annular ring rather than a filled disc.

---

## Runtime overrides

### Viewer Highway Costs modal

The viewer exposes a modal panel where `highway_costs` values can be edited per session. Changes take effect immediately for the next path planning request without reloading the `.mapdata` file.

### CLI flags

Running `replan.py` as a script exposes its own flags via `parse_args()`. They override the
YAML defaults for that invocation only; the YAML file is never modified.

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | `data/coords.gpx` | Waypoint file to replan |
| `--file` | `None` | `.mapdata` file to load. If omitted, OSM data is downloaded for the waypoints first. |
| `--cell_size` | `0.25` | Grid resolution in metres |
| `--inflate_obstacles` | `0.25` | Buffer added around obstacle geometries in metres |
| `--max_path_dist` | `2.0` | Radius around a way within which cells inherit its cost |
| `--simplify_path` | off | Apply Douglas–Peucker simplification (flag, no value) |
| `--smooth_path` | off | Apply gradient-descent smoothing (flag, no value) |
| `--save` | `None` | Write the replanned path to this GPX file |
| `--visualize` | off | Save a matplotlib debug plot as `replan.png` |

```bash
python -m map_data.pathsolver.replan --file coords.mapdata --cell_size 0.5 --visualize
```

### Programmatic override

`ReplanPath` takes an `argparse.Namespace` as its first positional argument rather than
individual keyword arguments. Build one with `parse_args([])` to get the defaults above,
then set the fields you want — `low` and `high` are required and have no default:

```python
from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath, parse_args
from map_data.utils.parsing import ways_to_shapely

md = MapData.load("coords.mapdata")

args = parse_args([])
args.low = (md.min_x, md.min_y)  # required: planning bounds in UTM metres
args.high = (md.max_x, md.max_y)
args.cell_size = 0.5  # coarser grid for faster planning
args.inflate_obstacles = 0.5  # wider obstacle clearance

planner = ReplanPath(
    args,
    ways_to_shapely(md.barriers_list),
    grid_cost_weight=2.0,  # weight terrain cost less against distance
)
```

Besides `args`, the constructor accepts `obstacles`, `transfer_id`, `grid_cost_weight`,
`highway_costs`, and `surface_costs`; the last three override the YAML values for that
instance. See the [Pathsolvers API reference](../api/pathsolver.md#replanpath) for the full
signature.

The parse-time keys are overridden separately, as arguments to `MapData`:

```python
md = MapData(
    "coords.gpx",
    grid_margin=300,  # download a wider area
    obstacle_radius=1.0,  # smaller footprint for point barriers
    buffer_widths={"road": 10.0, "footway": 3.0, "barrier": 2.0},
)
```
