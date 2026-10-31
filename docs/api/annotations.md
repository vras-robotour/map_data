# Annotations and edits

Load a map the way the viewer shows it: with its annotation store merged and the traversability rules applied.

```python
from map_data.annotations import load_mapdata_with_annotations

md, store = load_mapdata_with_annotations("my_area.mapdata")
```

## Merging a store

::: map_data.annotations
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - load_mapdata_with_annotations
        - edit_mapdata
        - apply_store
        - annotation_path_for
        - load_summary

## Store and way edits

::: map_data.edits
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - AnnotationStoreError
        - load_annotations
        - annotation_store
        - quarantine_annotation_store
        - split_way
        - apply_node_position_overrides
        - apply_added_nodes
        - rebuild_way_without_nodes
