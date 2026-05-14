# Viewer

The **Viewer** mode is the default mode of the interactive viewer. It lets you inspect
parsed map features, manage layer visibility, and draw manual annotations.

## Toolbar

The toolbar at the top of the page provides the following tools:

| Tool | Shortcut | Action |
|------|:--------:|--------|
| **VIEW** | `v` | Inspect map features. Click any object to see its OSM tags and properties in the sidebar. |
| **EDIT** | `e` | Select and reshape existing custom annotations. |
| **+ OBS** | `a` | Draw rectangles, polygons, or circles to add manual obstacle annotations. |
| **+ PATH** | `p` | Draw custom paths or footways as line annotations. |
| **DEL** | `d` | Remove custom annotations by clicking on them. |
| **FETCH** | `f` | Draw a bounding box to download and parse a new OSM area on the fly. |
| **GPX** | `g` | Upload a `.gpx` file to create a new map from a GPS track. |

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

Each category also has an expandable sub-filter to toggle individual highway or surface subtypes.
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
