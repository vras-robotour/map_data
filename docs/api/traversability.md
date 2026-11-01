# Traversability

Which ways the robot may drive on, from `config/traversability.yaml`. See [Traversability rules](../planning.md#traversability-rules) for the file format.

::: map_data.traversability
    options:
      show_root_heading: false
      heading_level: 2
      members:
        - load_traversability
        - resolve_traversability_path
        - TraversabilityRules
        - Verdict
        - TraversabilityError
