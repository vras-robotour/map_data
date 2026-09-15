"""Viewer endpoints for the tracker's topic settings, against a stand-in for TrackerNode."""

import pytest

from map_data.viewer.app import create_app
from map_data.viewer.tracker_config import (
    SETTING_DEFAULTS,
    SETTINGS,
    load_tracker_config,
    validate_settings,
)
from map_data.viewer.tracker_routes import TRACKER_EXTENSION

CONFIG = 'map_data_tracker:\n  ros__parameters:\n    gps_fix_topic: "/saved_fix"  # keep me\n'


class FakeTracker:
    """The part of TrackerNode the endpoints use."""

    def __init__(self):
        self.values = dict(SETTING_DEFAULTS)

    def settings(self):
        return dict(self.values)

    def apply_settings(self, values):
        new = validate_settings(values)
        changed = any(self.values[k] != v for k, v in new.items())
        self.values.update(new)
        return changed

    def available_topics(self):
        return {"/fix": ["sensor_msgs/msg/NavSatFix"]}


@pytest.fixture
def client(tmp_path):
    config = tmp_path / "tracker.yaml"
    config.write_text(CONFIG)
    app = create_app(data_dir=str(tmp_path), tracker_config=config)
    app.config["TESTING"] = True
    tracker = FakeTracker()
    app.extensions[TRACKER_EXTENSION] = tracker
    with app.test_client() as c:
        yield c, app, tracker, config


def test_get_lists_live_and_saved_values(client):
    c, _, _, config = client
    data = c.get("/api/tracker/settings").get_json()
    assert data["path"] == str(config)
    assert data["exists"] is True
    assert data["file_error"] is None
    assert data["topics"] == {"/fix": ["sensor_msgs/msg/NavSatFix"]}
    assert [s["name"] for s in data["settings"]] == [s.name for s in SETTINGS]
    gps = next(s for s in data["settings"] if s["name"] == "gps_fix_topic")
    assert gps["value"] == "/gps/fix"
    assert gps["saved"] == "/saved_fix"
    assert gps["msg_type"] == "sensor_msgs/msg/NavSatFix"


def test_get_reports_an_unreadable_file(client):
    c, _, _, config = client
    config.write_text("map_data_tracker: [\n")
    data = c.get("/api/tracker/settings").get_json()
    assert data["file_error"]
    assert all(s["saved"] == s["default"] for s in data["settings"])


def test_put_applies_without_saving(client):
    c, _, tracker, config = client
    resp = c.put("/api/tracker/settings", json={"settings": {"path_topic": "/p"}})
    assert resp.status_code == 200, resp.data
    assert resp.get_json() == {"changed": True, "path": None}
    assert tracker.values["path_topic"] == "/p"
    assert config.read_text() == CONFIG


def test_put_save_writes_the_live_settings(client):
    c, _, _, config = client
    body = {"settings": {"path_topic": "/p", "gps_fix_topic": "/saved_fix"}, "save": True}
    resp = c.put("/api/tracker/settings", json=body)
    assert resp.status_code == 200, resp.data
    assert resp.get_json()["path"] == str(config.resolve())
    assert "# keep me" in config.read_text()
    assert load_tracker_config(config) == {"gps_fix_topic": "/saved_fix", "path_topic": "/p"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"settings": "x"},
        {"settings": {"nope_topic": "/x"}},
        {"settings": {"path_topic": "/a b"}},
    ],
)
def test_put_rejects_bad_settings(client, body):
    c, _, tracker, config = client
    assert c.put("/api/tracker/settings", json={**body, "save": True}).status_code == 400
    assert tracker.values == SETTING_DEFAULTS
    assert config.read_text() == CONFIG


def test_put_reports_a_failed_save(client, tmp_path):
    c, app, tracker, _ = client
    app.config["TRACKER_CONFIG"] = str(tmp_path)  # a directory cannot be written as a file
    resp = c.put("/api/tracker/settings", json={"settings": {"path_topic": "/p"}, "save": True})
    assert resp.status_code == 500
    assert b"Applied, but not saved" in resp.data
    assert tracker.values["path_topic"] == "/p"


def test_without_a_tracker(client):
    c, app, _, _ = client
    app.extensions.pop(TRACKER_EXTENSION)
    assert c.get("/api/tracker/settings").status_code == 503
    assert c.put("/api/tracker/settings", json={"settings": {}}).status_code == 503
