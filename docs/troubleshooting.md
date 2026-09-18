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

## GPX parsing failures

### `MapData(coords="...", coords_type="file")` raises "No points (waypoints, tracks or routes) found"

`MapData` reads `<wpt>` elements if the file has any, otherwise falls back to `<trkpt>` (tracks) and then `<rtept>` (routes) — see [GPX waypoint format](dev/data_formats.md#gpx-waypoint-format). Confirm the file contains at least one of these element types.

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
- If you see `ImportError: cannot import name 'rclpy'`, you are running the viewer from within a ROS2 workspace where `rclpy` was expected. The viewer itself does not require ROS2. Run the viewer directly:

```bash
map_data_viewer --file coords.mapdata
```

### Viewer shows a blank map

The Leaflet tile layer requires internet access to load the OpenStreetMap basemap. The vector overlays (roads, footways, barriers) are served locally and will still appear. If the background is blank but features are visible, the tile server is unreachable from your network.

### Annotations not saving

Annotation files are written to the same directory as the `.mapdata` file. If the directory is read-only (e.g. inside a ROS2 install prefix), saving will silently fail. Run the viewer from a directory where you have write permission, or copy the `.mapdata` file to a writable location first.

---

## ROS2 node errors

### `create_mapdata` not found

The command is installed with the package. Standalone, `pip install -e .` puts it on
your `PATH` directly. In a ROS2 workspace, build with `colcon` and source the workspace:

```bash
colcon build --packages-select map_data
source install/setup.bash
```

After that, use `ros2 run map_data create_mapdata ...`.

### `osm_cloud` does not publish

1. Check that a `.mapdata` file path is set via the `mapdata_file` ROS2 parameter.
2. Verify the placement frames are correct — the node transforms the point cloud into `local_frame`. Until that transform is available in TF (`utm → local_frame` for `transform_mode` `tf`, `earth_frame → local_frame` for `geodetic`), the node logs `Failed to get ... transform` every 10 s and publishes nothing. It picks the transform up as soon as it arrives; no restart is needed.
3. Run `ros2 topic list` and confirm the `grid` topic (or whatever `grid_topic` is set to) is present.

### `osm_cloud` publishes an empty cloud

The log says `Grid is empty although the map has N way points` or, in older versions, printed a `(0, 4)` shape and an `FP_ENU0 origin at lat=180.0000000 lon=0.0000000 alt=-6378137.0`.

That origin is the centre of the Earth: the `earth_frame → local_frame` transform was all zeros. A Fixposition unit publishes `FP_ECEF → FP_ENU0` as zeros until it has a fusion fix, and `transform_mode: geodetic` then works at an altitude of -6378 km, which collapses the whole map into a metre-wide box — no grid cell ends up within `max_path_dist` of a way.

The node now rejects such a transform and waits for a real one, and never publishes an empty cloud over a map that has ways. If you see it on an older build, restart `osm_cloud` once the GNSS/INS unit has a fix (`ros2 topic echo /tf_static` should show a `FP_ECEF → FP_ENU0` translation of several million metres, not zeros).

---

## `.mapdata` file issues

### `MapData.load()` raises `ValueError` or `JSONDecodeError`

The file may be truncated (interrupted write) or a legacy pickle file (detected by a `0x80` header byte). Support for the legacy pickle format has been removed for security reasons.

If you have a legacy file, you must re-run `create_mapdata` to regenerate it in the JSON format or parse the GPX/YAML file through the web viewer.

### UTM zone boundary warning

If the log prints `WARNING: ways span multiple UTM zones`, the `.mapdata` bounding box crosses a zone boundary. Path planning will still work, but geometry accuracy degrades near the boundary because all features are projected into a single UTM zone. For affected areas, consider splitting the route into segments that stay within one zone.
