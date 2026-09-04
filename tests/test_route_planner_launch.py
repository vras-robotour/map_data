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
