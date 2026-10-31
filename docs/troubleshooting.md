# Troubleshooting

## Overpass API errors

### Rate limiting (HTTP 429 / 404)

The Overpass API enforces per-IP rate limits. If a request is rejected, `map_data` automatically backs off and retries on an alternate endpoint. If you see repeated failures in the log:

- Wait a few minutes before retrying.
- Avoid running multiple `create_mapdata` calls in parallel.
- Consider hosting a local [Overpass API instance](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation) for high-volume use.

### Query timeout (HTTP 504)

Overpass imposes a default query timeout. Very large bounding boxes (many kilometers across) may exceed it. Reduce the query area by using a shorter GPX route or decreasing `grid_margin` in `config/planner_defaults.yaml`.

### No features returned

If `md.footways_list` or `md.roads_list` is empty after `run_all()`, the bounding box may be in an area with sparse OSM coverage. Verify the coverage on [openstreetmap.org](https://www.openstreetmap.org) before debugging the code.

---

## GPX parsing failures

### `MapData(coords="...", coords_type="file")` raises "No points (waypoints, tracks or routes) found"

`MapData` reads `<wpt>` elements if the file has any, otherwise falls back to `<trkpt>` (tracks) and then `<rtept>` (routes). Confirm the file contains at least one of these element types.

---

## Path planning returns `None`

### `GraphPlanner.plan()` returns `None`

The start or goal point could not be snapped to the road/footway graph. This happens when:

- The input waypoints are outside the loaded `.mapdata` bounding box.
- The requested `highway_types` (such as `["footway"]`) do not include ways that reach both endpoints.
- The two endpoints are in disconnected graph components.

Try widening the `highway_types` list to include `"road"` or check the map in the viewer to confirm that ways connect the two points.

### `ReplanPath.replan()` returns `None`

`ReplanPath.replan()` returns `None` when:

- **No path exists:** No collision-free path exists between waypoints given the current obstacle inflation. Try reducing `inflate_obstacles` (for example from `0.25` to `0.1`), adjusting `cell_size`, or verifying in the viewer that barrier polygons do not completely block the route.
- **Request was cancelled:** The planning request was cancelled via `cancel_replan_backend()` or an action goal cancellation.

---

## Viewer issues

### Flask server fails to start

- Check that port `5000` is not already in use: `lsof -i :5000`.
- If you see `ImportError: cannot import name 'rclpy'`, you are running the viewer from within a Python environment where ROS 2 packages are not sourced. The viewer itself does not require ROS 2 for Viewer and Planner modes. Run the viewer directly:

```bash
map_data_viewer --data-dir /path/to/data
```

### Viewer shows a blank map

The Leaflet tile layer requires internet access to load the OpenStreetMap basemap. The vector overlays (roads, footways, barriers) are served locally and will still appear. If the background is blank but features are visible, the tile server is unreachable from your network.

### Annotations not saving

Annotation files are written to `<mapdata_filename>.annotations.json` next to the `.mapdata` file. If that directory is read-only (such as inside a ROS 2 install prefix), saving fails. Run the viewer with `--data-dir` set to a writable directory, or copy the `.mapdata` file to a writable location first.

---

## ROS 2 node errors

### `create_mapdata` not found

The command is installed with the package. Standalone, `uv sync` or `pip install -e .` puts it on your `PATH`. In a ROS 2 workspace, create the workspace symlink for `map_data_interfaces` and build with `colcon`:

```bash
bash scripts/setup_ws.sh
colcon build --packages-up-to map_data
source install/setup.bash
```

After sourcing, run `ros2 run map_data create_mapdata ...`.

### `osm_cloud` does not publish

1. Check that a `.mapdata` file path is provided via the `mapdata_file` parameter or launch argument. If no input file is given, `osm_cloud` logs an error and idles.
2. Verify the placement frames are available in `/tf_static`. The node transforms the point cloud into `local_frame`. Until that transform is available (`utm -> local_frame` for `transform_mode: tf`, or `earth_frame -> local_frame` for `geodetic`), the node logs `Failed to get ... transform` and publishes nothing. It picks up the transform as soon as it arrives on `/tf_static`.
3. Confirm the grid topic (`osm_grid` or `grid`) is present using `ros2 topic list`.

### `osm_cloud` publishes an empty cloud

A Fixposition unit publishes `FP_ECEF -> FP_ENU0` as all zeros until it has a fusion fix. In `transform_mode: geodetic`, an all-zero transform puts the origin at the center of the Earth (-6378 km altitude), collapsing the map into a meter-wide box.

The node rejects all-zero transforms, waits for a valid transform on `/tf_static`, and logs an error without publishing if the grid is empty while the map contains ways. If you see transform warnings, ensure the GNSS/INS unit has achieved a valid RTK/fusion fix.

---

## .mapdata file issues

### `MapData.load()` raises `ValueError` or `JSONDecodeError`

The file may be truncated (interrupted write) or a legacy pickle file (detected by a `0x80` header byte). Support for the legacy pickle format was removed for security reasons.

If you have a legacy file, regenerate it with `create_mapdata -d -f <file>.gpx`.

### UTM zone boundary warning

If the log prints `WARNING: ways span multiple UTM zones`, the `.mapdata` bounding box crosses a UTM zone boundary. Path planning will still work, but geometry accuracy degrades near the boundary because all features are projected into a single UTM zone. For affected areas, consider splitting the route into segments that stay within one zone.
