"""Tracker config files (``map_data.viewer.tracker_config``): load, validate, save."""

from pathlib import Path

import pytest
import yaml

from map_data.viewer.tracker_config import (
    DEFAULTS,
    SETTING_DEFAULTS,
    TrackerConfigError,
    _split_comment,
    load_tracker_config,
    node_parameters,
    save_tracker_config,
    topic_type,
    validate_settings,
)

PKG = Path(__file__).resolve().parents[1]

HELHEST_LIKE = """\
# Robot stack
map_data_tracker:
  ros__parameters:
    earth_frame: "FP_ECEF"
    gps_fix_topic: "/fixposition/odometry_llh"       # NavSatFix, status 2 = RTK fixed
    joy_topic: "/joy"                                # any stick/button -> TELEOP ACTIVE
    battery_low_voltage: 46.0

osm_cloud:
  ros__parameters:
    grid_topic: "grid"
"""


def _write(tmp_path, text, name="tracker.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


class TestShippedConfigs:
    def test_tracker_yaml_holds_the_defaults(self):
        assert load_tracker_config(PKG / "config" / "tracker.yaml") == DEFAULTS

    def test_helhest_jr_yaml_sets_every_parameter(self):
        config = load_tracker_config(PKG / "config" / "helhest_jr.yaml")
        assert set(config) == set(DEFAULTS)
        assert node_parameters(config)["heading_type"] == "yaw_vector3"


class TestLoad:
    def test_missing_file_sets_nothing(self, tmp_path):
        assert load_tracker_config(tmp_path / "missing.yaml") == {}

    def test_node_section_wins_over_wildcard(self, tmp_path):
        path = _write(
            tmp_path,
            "/**:\n  ros__parameters:\n    use_sim_time: true\n    path_topic: /a\n"
            "    goal_topic: /g\n"
            "/map_data_tracker:\n  ros__parameters:\n    path_topic: /b\n",
        )
        config = load_tracker_config(path)
        assert config["path_topic"] == "/b"
        assert config["goal_topic"] == "/g"
        assert "use_sim_time" not in node_parameters(config)

    def test_file_for_other_nodes_sets_nothing(self, tmp_path):
        path = _write(tmp_path, "osm_cloud:\n  ros__parameters:\n    grid_topic: grid\n")
        assert load_tracker_config(path) == {}

    @pytest.mark.parametrize(
        "text",
        [
            "[1, 2]\n",
            "map_data_tracker: 3\n",
            "map_data_tracker:\n  ros__parameters: [1]\n",
            "map_data_tracker: [\n",
        ],
    )
    def test_malformed(self, tmp_path, text):
        with pytest.raises(TrackerConfigError):
            load_tracker_config(_write(tmp_path, text))


class TestNodeParameters:
    def test_defaults(self):
        assert node_parameters({}) == DEFAULTS

    def test_null_topic_disables(self):
        assert node_parameters({"path_topic": None})["path_topic"] == ""

    def test_int_is_accepted_for_float(self):
        value = node_parameters({"stale_after": 5})["stale_after"]
        assert value == 5.0
        assert isinstance(value, float)

    @pytest.mark.parametrize(
        "config",
        [
            {"trail_length": 2.5},
            {"trail_length": True},
            {"stale_after": "3"},
            {"path_topic": 5},
            {"heading_type": "compass"},
        ],
    )
    def test_rejects(self, config):
        with pytest.raises(TrackerConfigError):
            node_parameters(config)


class TestValidateSettings:
    def test_strips(self):
        assert validate_settings({"path_topic": " /p "}) == {"path_topic": "/p"}

    @pytest.mark.parametrize(
        "values",
        [
            None,
            ["path_topic"],
            {"nope_topic": "/x"},
            {"battery_low_voltage": "22"},  # file-only
            {"path_topic": 3},
            {"path_topic": "/a b"},
            {"heading_type": "gps"},
        ],
    )
    def test_rejects(self, values):
        with pytest.raises(TrackerConfigError):
            validate_settings(values)


def test_heading_topic_type_follows_heading_type():
    assert topic_type("heading_topic", {"heading_type": "odometry"}) == "nav_msgs/msg/Odometry"
    assert topic_type("gps_fix_topic", {}) == "sensor_msgs/msg/NavSatFix"


def test_split_comment():
    assert _split_comment('"/x"   # note') == ('"/x"', "# note")
    assert _split_comment('"a #b"  # c') == ('"a #b"', "# c")
    assert _split_comment("/a#b") == ("/a#b", "")


class TestSave:
    def test_changes_only_the_value_and_keeps_the_comment_column(self, tmp_path):
        path = _write(tmp_path, HELHEST_LIKE)
        settings = {
            **SETTING_DEFAULTS,
            "earth_frame": "FP_ECEF",
            "gps_fix_topic": "/fp/llh",
            "joy_topic": "/joy",
        }
        save_tracker_config(path, settings)

        old, new = HELHEST_LIKE.splitlines(), path.read_text().splitlines()
        assert [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b] == [4]
        assert new[4].startswith('    gps_fix_topic: "/fp/llh" ')
        assert new[4].index("#") == old[4].index("#")
        assert node_parameters(load_tracker_config(path)) == {
            **DEFAULTS,
            **settings,
            "battery_low_voltage": 46.0,
        }

    def test_long_value_pushes_the_comment_right(self, tmp_path):
        path = _write(tmp_path, HELHEST_LIKE)
        topic = "/a/very/long/topic/name/that/does/not/fit/before/the/comment"
        save_tracker_config(path, {"joy_topic": topic})
        line = next(ln for ln in path.read_text().splitlines() if "joy_topic" in ln)
        assert line == f'    joy_topic: "{topic}" # any stick/button -> TELEOP ACTIVE'

    def test_adds_missing_keys_to_the_tracker_block(self, tmp_path):
        path = _write(tmp_path, HELHEST_LIKE)
        save_tracker_config(path, {"goal_topic": "/goal"})
        lines = path.read_text().splitlines()
        assert lines[7] == '    goal_topic: "/goal"'
        doc = yaml.safe_load(path.read_text())
        assert doc["osm_cloud"] == {"ros__parameters": {"grid_topic": "grid"}}

    def test_adds_a_tracker_block_to_another_nodes_file(self, tmp_path):
        path = _write(tmp_path, "osm_cloud:\n  ros__parameters:\n    grid_topic: grid\n")
        save_tracker_config(path, {"path_topic": "/p"})
        assert load_tracker_config(path) == {"path_topic": "/p"}
        assert yaml.safe_load(path.read_text())["osm_cloud"]["ros__parameters"] == {
            "grid_topic": "grid"
        }

    def test_creates_the_file(self, tmp_path):
        path = tmp_path / "new" / "tracker.yaml"
        save_tracker_config(path, {"path_topic": "/p", "goal_topic": ""})
        assert load_tracker_config(path) == {"path_topic": "/p"}

    def test_nothing_changed_leaves_the_file_alone(self, tmp_path):
        text = "map_data_tracker:\n  ros__parameters:\n    path_topic: ''   # off\n"
        path = _write(tmp_path, text)
        save_tracker_config(path, {"path_topic": "", "goal_topic": ""})
        assert path.read_text() == text

    def test_writes_through_a_symlink(self, tmp_path):
        source = _write(tmp_path, HELHEST_LIKE, "source.yaml")
        link = tmp_path / "share" / "tracker.yaml"
        link.parent.mkdir()
        link.symlink_to(source)
        assert save_tracker_config(link, {"joy_topic": ""}) == source
        assert link.is_symlink()
        assert load_tracker_config(source)["joy_topic"] == ""

    def test_refuses_what_it_cannot_edit(self, tmp_path):
        text = "map_data_tracker:\n  ros__parameters: {path_topic: /a}\n"
        path = _write(tmp_path, text)
        with pytest.raises(TrackerConfigError):
            save_tracker_config(path, {"path_topic": "/b"})
        assert path.read_text() == text
