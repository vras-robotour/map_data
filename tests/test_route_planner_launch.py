"""route_planner.launch.py: the config file and the arguments layered on top of it."""

import re
from pathlib import Path

import yaml

from map_data.utils.launch import flag, resolve_config_file, way_types

PKG = Path(__file__).resolve().parents[1]
CONFIG = PKG / "config" / "route_planner.yaml"


def _declared_parameters() -> set[str]:
    """Parameter names the node declares, read out of its source."""
    src = (PKG / "map_data" / "route_planner.py").read_text()
    return set(re.findall(r'\bp\(\s*"([a-z_]+)"', src))


def test_config_only_sets_parameters_the_node_declares():
    params = yaml.safe_load(CONFIG.read_text())["route_planner"]["ros__parameters"]
    unknown = set(params) - _declared_parameters()
    assert not unknown, f"route_planner.yaml sets parameters the node ignores: {sorted(unknown)}"


def test_config_defaults_match_the_node():
    params = yaml.safe_load(CONFIG.read_text())["route_planner"]["ros__parameters"]
    assert params["algorithm"] == "graph"
    assert params["highway_types"] == ["footway"]
    assert params["annotations"] == "auto"
    assert params["spacing"] == 3.0
    assert params["exclude_highway"] == ["steps"]  # stairs are not routable
    assert params["traversability_file"] == ""  # "" = the package's traversability.yaml
    assert params["max_snap_distance"] == 100.0  # the start (the robot's own fix)
    assert params["goal_max_snap_distance"] == 30.0  # the goal, failing with snap_too_far
    assert params["keep_goal"] is True


def test_way_types_accepts_commas_and_spaces():
    assert way_types("footway") == ["footway"]
    assert way_types("footway,road") == ["footway", "road"]
    assert way_types(" footway road ") == ["footway", "road"]
    assert way_types("") == []


def testflag():
    assert flag("true") and flag("True") and flag("1") and flag("yes")
    assert not flag("false") and not flag("") and not flag("0")


def test_params_file_lookup():
    assert resolve_config_file("route_planner.yaml").endswith("config/route_planner.yaml")
    assert resolve_config_file("/tmp/other.yaml") == "/tmp/other.yaml"


def test_node_declares_the_same_defaults_as_the_config():
    """The yaml must not silently disagree with the node's own defaults."""
    src = (PKG / "map_data" / "route_planner.py").read_text()
    assert 'p("keep_goal", True)' in src
    assert 'p("goal_max_snap_distance", 30.0)' in src
    assert 'p("exclude_highway", sorted(NON_ROUTABLE_HIGHWAY_VALUES))' in src
    assert 'p("traversability_file", "")' in src


# ── traversability ─────────────────────────────────────────────────────────

LAUNCH_FILE = PKG / "launch" / "route_planner.launch.py"


def test_launch_maps_traversability_onto_the_node_parameter():
    """``traversability:=<file>`` is the argument; the node parameter is the file name."""
    src = LAUNCH_FILE.read_text()
    assert '"traversability",\n            default_value="",' in src
    assert 'given("traversability", "traversability_file")' in src


def test_traversability_file_reaches_the_map_load_and_the_planner():
    src = (PKG / "map_data" / "route_planner.py").read_text()
    assert "traversability=self.traversability_file or None" in src
    # once for load_mapdata_with_annotations, once for the GraphPlanner
    assert src.count("traversability=self.traversability_file or None") == 2


def test_traversability_file_is_part_of_both_cache_keys():
    """
    Editing the rule file and restarting must not hand back the map or the graph
    built with the old rules.
    """
    src = (PKG / "map_data" / "route_planner.py").read_text()
    map_key = src.split("key = (str(path), str(ann)")[1][:40]
    assert "self.traversability_file" in map_key
    planner_key = src.split("        key = (\n            cache[0],")[1][:300]
    assert "self.traversability_file" in planner_key
    assert "trav_path.stat().st_mtime" in src  # the file's own mtime, for the map cache
