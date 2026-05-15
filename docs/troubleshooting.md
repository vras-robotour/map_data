# Troubleshooting

## Overpass API errors

### Rate limiting (HTTP 429 / 404)

The Overpass API enforces per-IP rate limits. If a request is rejected, `map_data` automatically backs off and retries on an alternate endpoint. If you see repeated failures in the log:

- Wait a few minutes before retrying.
- Avoid running multiple `create_mapdata` calls in parallel.
- Consider hosting a local [Overpass API instance](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation) for high-volume use.

### Query timeout (HTTP 504)

Overpass imposes a default query timeout. Very large bounding boxes (many kilometres across) may exceed it. Reduce the query area by using a shorter GPX route or decreasing `osm_margin` in `config/planner_defaults.yaml`.

### No features returned

If `md.footways_list` or `md.roads_list` is empty after `run_all()`, the bounding box may be in an area with sparse OSM coverage. Verify the coverage on [openstreetmap.org](https://www.openstreetmap.org) before debugging the code.

---

## GPX / YAML parsing failures

### `parse_path` raises an exception

- Confirm the file uses the `.gpx` or `.yaml` extension — format detection is extension-based.
- GPX files must contain `<wpt>` elements. Files with only `<trkpt>` or `<rtept>` are supported by `MapData` for area calculation but **not** by `parse_gpx_file` for path planning. Convert track points to waypoints in your GPX editor first.
- YAML files must have a top-level `waypoints:` key. See [YAML waypoint format](dev/data_formats.md#yaml-waypoint-format) for the expected schema.

### Only one waypoint found

`parse_gpx_file` reads `<wpt>` elements in document order. If your GPX editor exports waypoints as a single track instead of individual `<wpt>` tags, re-export using the "waypoints" export option.

---

## Path planning returns `None` or `False`

### `GraphPlanner.plan()` returns `None`

The start or goal point could not be snapped to the road/footway graph. This happens when:

- The input waypoints are outside the loaded `.mapdata` bounding box.
- The requested `highway_types` (e.g. `["footway"]`) do not include ways that reach both endpoints.
- The two endpoints are in disconnected graph components.

Try widening the `highway_types` list to include `"road"` or check the map in the viewer to confirm that ways connect the two points.

### `ReplanPath.replan()` returns `False`

No collision-free path exists between two consecutive waypoints given the current obstacle inflation. Try:

- Reducing `inflate_obstacles` (e.g. from `0.25` to `0.1`).
- Increasing `cell_size` to coarsen the grid and allow the planner to find routes through narrow gaps.
- Opening the viewer and verifying that the barriers do not completely block the corridor.

### `ReplanPath.replan()` returns `None`

The planning request was cancelled via `cancel_replan_backend()`. This is expected behaviour when the viewer starts a new plan before the previous one finishes.

---

## Viewer issues

### Flask server fails to start

- Check that port `5000` is not already in use: `lsof -i :5000`.
- If you see `ImportError: cannot import name 'rclpy'`, you are running the viewer from within a ROS2 workspace where `rclpy` was expected. The viewer itself does not require ROS2, but the `visualize_mapdata` ROS2 node wraps it. Run the viewer directly instead:

```bash
map_data_viewer --file coords.mapdata
```

### Viewer shows a blank map

The Leaflet tile layer requires internet access to load the OpenStreetMap basemap. The vector overlays (roads, footways, barriers) are served locally and will still appear. If the background is blank but features are visible, the tile server is unreachable from your network.

### Annotations not saving

Annotation files are written to the same directory as the `.mapdata` file. If the directory is read-only (e.g. inside a ROS2 install prefix), saving will silently fail. Run the viewer from a directory where you have write permission, or copy the `.mapdata` file to a writable location first.

---

## ROS2 node errors

### `create_mapdata` / `visualize_mapdata` not found

These are ROS2 nodes. They are available only after building the package with `colcon` and sourcing the workspace:

```bash
colcon build --packages-select map_data
source install/setup.bash
```

After that, use `ros2 run map_data create_mapdata ...` rather than calling the script directly.

### `osm_cloud` does not publish

1. Check that a `.mapdata` file path is set via the `mapdata_file` ROS2 parameter.
2. Verify the UTM frame is correct — the node transforms the point cloud from UTM to `local_frame`. If no static transform between `utm` and `local_frame` is available in TF, the cloud will not be published.
3. Run `ros2 topic list` and confirm the `grid` topic (or whatever `grid_topic` is set to) is present.

---

## `.mapdata` file issues

### `MapData.load()` raises `JSONDecodeError`

The file may be truncated (interrupted write) or a legacy pickle file that failed migration. Try:

```python
with open("coords.mapdata", "rb") as f:
    header = f.read(2)
print(header)  # b'\x80\x05' → pickle; b'{\n' → JSON
```

If it is a valid pickle file, `MapData.load()` should handle it transparently. If neither format is detected, the file is corrupt — re-run `create_mapdata` to regenerate it.

### UTM zone boundary warning

If the log prints `WARNING: ways span multiple UTM zones`, the `.mapdata` bounding box crosses a zone boundary. Path planning will still work, but geometry accuracy degrades near the boundary because all features are projected into a single UTM zone. For affected areas, consider splitting the route into segments that stay within one zone.
