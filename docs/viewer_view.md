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

Sidecar `.annotations.json` files can be inspected or generated programmatically:

```python
import json
import pathlib
import time
import uuid

mapdata_path = pathlib.Path("coords.mapdata")
sidecar_path = mapdata_path.with_suffix(".annotations.json")

if sidecar_path.exists():
    with open(sidecar_path) as f:
        sidecar = json.load(f)
else:
    sidecar = {
        "version": 1,
        "annotations": [],
        "deleted_ways": [],
        "hidden_ways": [],
        "tag_overrides": {},
        "split_ways": {},
        "deleted_nodes": [],
        "added_nodes": [],
        "node_position_overrides": {},
        "change_log": [],
    }

# Annotation IDs are UUID4 strings; coordinates are WGS-84 [lon, lat] pairs
sidecar["annotations"].append(
    {
        "id": str(uuid.uuid4()),
        "type": "obstacle",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [14.5678, 50.1234],
                    [14.5680, 50.1234],
                    [14.5680, 50.1236],
                    [14.5678, 50.1234],
                ]
            ],
        },
        "properties": {},
    }
)

# Change-log entries record a floating-point timestamp
sidecar["change_log"].append(
    {
        "action": "add_annotation",
        "ann_id": sidecar["annotations"][-1]["id"],
        "ts": time.time(),
    }
)

with open(sidecar_path, "w") as f:
    json.dump(sidecar, f, indent=2)
```

The viewer merges sidecar edits on load, applying them to the displayed map.
