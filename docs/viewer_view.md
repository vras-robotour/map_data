# Viewer

The **Viewer** mode is the default mode of the interactive viewer. It lets you inspect
and edit parsed map features, manage layer visibility, and draw manual annotations.

## Toolbar

| Tool | Shortcut | Action |
|------|:--------:|--------|
| **VIEW** | `v` | Inspect map features. Click any object to view its OSM tags and properties in the sidebar |
| **EDIT** | `e` | Select and edit existing ways, nodes, and custom annotations |
| **+ OBS** | `a` | Draw rectangles, polygons, or circles to add manual obstacle annotations |
| **+ PATH** | `p` | Draw custom paths or footways as line annotations. Intersections with other ways form crossroads |
| **DEL** | `d` | Remove objects and custom annotations by clicking on them |
| **FETCH** | `f` | Draw a bounding box to download and parse a new OSM area (up to 25 km²) |
| **GPX** | `g` | Upload a `.gpx` file to create a new map from a GPS track |
| **CLEAR** | | Clear currently loaded map data |

### Keyboard shortcuts

Keyboard shortcuts operate when not typing into input fields or modals (and mutating actions are disabled in the read-only demo):

| Key | Action |
|-----|--------|
| `v` | Switch to View mode |
| `e` | Switch to Edit mode |
| `a` | Add obstacle annotation mode |
| `p` | Add path annotation mode |
| `d` | Delete mode |
| `f` | Fetch OSM area mode |
| `g` | Open GPX file picker |
| `n` | Toggle display of way vertices (in View mode) |
| `h` | Hide or unhide currently selected way (in View mode) |
| `Escape` | Deselect current selection, or return to View mode |
| `Delete` / `Backspace` | Delete selected annotation (Edit mode), or selected node/way (View mode) |
| `Ctrl+Z` / `Cmd+Z` | Undo last change |
| `Ctrl+Shift+Z` / `Ctrl+Y` / `Cmd+Shift+Z` | Redo last undone change |

## Layers panel

The sidebar **Layers** panel toggles visibility for each feature category:

| Layer | Color | Description |
|-------|-------|-------------|
| Roads | Dark grey | Vehicle roads from OSM |
| Footways | Yellow | Pedestrian paths |
| Barriers | Red | Untraversable features (walls, buildings, fences, water) |
| Crossroads | Purple | Detected footway intersection points |
| Annotations | Orange | Manually drawn obstacles and paths |
| Waypoints | Blue | GPX waypoints from the loaded track |
| Robot | Green | Live robot position (visible when ROS 2 is available) |

A base-layer switcher in the bottom-left corner of the map toggles between standard
**OpenStreetMap** tiles and **Satellite** imagery (Esri World Imagery). The choice is
preserved across sessions.

## Properties panel

Clicking any feature in **VIEW** mode opens its properties in the sidebar, displaying
its OSM tags, category, and geometry. When inspecting individual nodes, copy buttons next
to latitude and longitude copy coordinates to the clipboard.

## Annotation panels

Three collapsible panels track manual edits:

- **Annotations** — list of drawn obstacles and paths. Revert buttons restore individual geometries.
- **Changes** — features modified since load (tag edits, node moves, splits, additions, deletions). Each entry supports revert; `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo edits sequentially.
- **Hidden** — features hidden from the map via the context menu or `h` key.

### Exporting edited map data

- **Export button:** Downloads the fully merged map (with all annotations, node moves, splits, and deletions applied) as a downloadable `<stem>.exported.mapdata` file. It does not overwrite the source `.mapdata` file on disk.
- **GeoJSON button:** Downloads the merged map as a `.geojson` file for inspection in GIS tools (such as QGIS).

Incremental edits are automatically saved to `<stem>.annotations.json` in the data directory.

---

## Programmatic annotations

`map_data.edits.annotation_store` opens a store for editing and writes it back atomically (see [Data formats](dev/data_formats.md#annotationsjson-sidecar-format) for the schema). Its lock is per process, so do not edit a map the viewer is editing at the same time.

```python
import uuid

from map_data.annotations import annotation_path_for
from map_data.edits import annotation_store

with annotation_store(str(annotation_path_for("coords.mapdata"))) as store:
    store.setdefault("annotations", []).append(
        {
            "id": str(uuid.uuid4()),
            "type": "obstacle",
            # WGS-84 [lon, lat]
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[14.5678, 50.1234], [14.5680, 50.1234], [14.5680, 50.1236], [14.5678, 50.1234]]
                ],
            },
            "properties": {},
        }
    )
```

The viewer, `map_data_plan`, `route_planner` and `osm_cloud` merge the store when they load the map.
