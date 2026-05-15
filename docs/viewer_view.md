# Viewer

The **Viewer** mode is the default mode of the interactive viewer. It lets you inspect
and edit parsed map features, manage layer visibility, and draw manual annotations.

## Toolbar

The toolbar at the top of the page provides the following tools:

| Tool | Shortcut | Action |
|------|:--------:|--------|
| **VIEW** | `v` | Inspect map features. Click any object to see its OSM tags and properties in the sidebar. |
| **EDIT** | `e` | Select and reshape existing objects and custom annotations. |
| **+ OBS** | `a` | Draw rectangles, polygons, or circles to add manual obstacle annotations. |
| **+ PATH** | `p` | Draw custom paths or footways as line annotations. |
| **DEL** | `d` | Remove objects and custom annotations by clicking on them. |
| **FETCH** | `f` | Draw a bounding box to download and parse a new OSM area on the fly. |
| **GPX** | `g` | Upload a `.gpx` file to create a new map from a GPS track. |
| **CLEAR** |  | Clear currently loaded data. |

## Layers Panel

The sidebar shows a **Layers** panel with toggles for each feature category:

| Layer | Colour | Description |
|-------|--------|-------------|
| Roads | Dark grey | Vehicle-intended ways from OSM |
| Footways | Yellow | Pedestrian paths |
| Barriers | Red | Untraversable features (walls, buildings, fences, water…) |
| Crossroads | Purple | Detected footway intersection points |
| Annotations | Orange | Manually drawn obstacles and paths |
| Waypoints | Blue | GPX waypoints from the loaded file |
| Robot | Green | Live robot position (visible when ROS2 is available) |

First three categories also have an expandable sub-filter to toggle individual types.
Use the **Search** box at the top of the Layers panel to filter features by OSM ID or name.

## Properties Panel

Clicking any map feature in **VIEW** mode opens its properties in the **Properties** panel.
The panel shows the feature's OSM tags, category, and geometry type.
In **VIEW** mode you can also open a feature's properties for editing via the context menu.

## Annotation Panels

Three collapsible panels at the bottom of the sidebar track the state of manual edits:

- **Annotations** — list of all drawn obstacles and paths.
- **Changes** — features whose properties have been modified since the file was loaded.
- **Hidden** — features that have been hidden from the map via the context menu.

Use the **Export** button in the toolbar to save all annotations and changes back to the `.mapdata` file.
Deleted features will be removed from the file, while edits and annotations are added to it, hidden
features are preserved.

---

## Programmatic annotations

Annotations can also be created or inspected without the viewer UI by editing the sidecar
`.annotations.json` file directly. This is useful for scripting bulk edits or seeding
annotations from an external data source.

**Add an obstacle polygon programmatically:**

```python
import json, time, pathlib

mapdata_path = pathlib.Path("coords.mapdata")
sidecar_path = mapdata_path.with_suffix(".annotations.json")

# Load existing sidecar or start fresh
if sidecar_path.exists():
    with open(sidecar_path) as f:
        sidecar = json.load(f)
else:
    sidecar = {"version": 1, "annotations": [], "deleted_ways": [],
                "hidden_ways": [], "tag_overrides": {}, "split_ways": {},
                "deleted_nodes": {}, "node_position_overrides": {},
                "change_log": [], "change_log_migration": "1"}

# Polygon coordinates are WGS-84 (lon, lat) pairs, closed ring
sidecar["annotations"].append({
    "id": f"ann_{int(time.time() * 1000)}",
    "type": "obstacle",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [14.5678, 50.1234],
            [14.5680, 50.1234],
            [14.5680, 50.1236],
            [14.5678, 50.1234],
        ]]
    },
    "properties": {}
})

with open(sidecar_path, "w") as f:
    json.dump(sidecar, f, indent=2)
```

The viewer merges the sidecar at load time, so reloading the file in the browser will show the
new annotation immediately. See [Data Formats — `.annotations.json`](dev/data_formats.md#annotationsjson-sidecar-format)
for the full sidecar schema.
