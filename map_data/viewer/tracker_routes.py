"""
Viewer API for the Tracker's topic settings: read them, switch them live, save them.

The settings live on the running :class:`~map_data.viewer.ros_node.TrackerNode`
(``app.extensions["map_data_tracker"]``) and in its config file
(``app.config["TRACKER_CONFIG"]``); see :mod:`map_data.viewer.tracker_config`.
"""

from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, current_app, jsonify, request

from .tracker_config import (
    DEFAULTS,
    HEADING_TYPES,
    TrackerConfigError,
    load_tracker_config,
    node_parameters,
    save_tracker_config,
    settings_metadata,
)

TRACKER_EXTENSION = "map_data_tracker"

bp = Blueprint("tracker", __name__)


def _tracker_node() -> Any:
    node = current_app.extensions.get(TRACKER_EXTENSION)
    if node is None:
        abort(503, "The tracker is not running (no ROS 2 context)")
    return node


def _config_path() -> Path:
    return Path(current_app.config["TRACKER_CONFIG"])


@bp.route("/api/tracker/settings")
def get_tracker_settings() -> Response:
    """
    Return the tracker's settings, what its config file holds and the topics on the graph.

    ``settings`` lists ``{name, section, description, msg_type, choices, default, value,
    saved}`` in display order, ``value`` being live and ``saved`` the file's value (the
    default when the file does not set it). ``topics`` maps each topic on the ROS graph to
    its message types.
    """
    node = _tracker_node()
    path = _config_path()
    file_error = None
    try:
        saved = node_parameters(load_tracker_config(path))
    except TrackerConfigError as e:
        saved, file_error = dict(DEFAULTS), str(e)
    live = node.settings()
    return jsonify(
        {
            "path": str(path),
            "exists": path.is_file(),
            "file_error": file_error,
            "heading_types": HEADING_TYPES,
            "topics": node.available_topics(),
            "settings": [
                {**meta, "value": live[meta["name"]], "saved": saved[meta["name"]]}
                for meta in settings_metadata()
            ],
        }
    )


@bp.route("/api/tracker/settings", methods=["PUT"])
def put_tracker_settings() -> Response:
    """
    Apply the body's ``settings`` (``{name: value}``, a subset is fine) to the running tracker.

    With ``"save": true`` the tracker's settings are also written to its config file,
    comments kept. Returns ``{"changed": bool, "path": str | null}``.
    """
    body = request.get_json(force=True) or {}
    node = _tracker_node()
    try:
        changed = node.apply_settings(body.get("settings"))
    except TrackerConfigError as e:
        abort(400, str(e))
    path = None
    if body.get("save"):
        try:
            path = str(save_tracker_config(_config_path(), node.settings()))
        except (TrackerConfigError, OSError) as e:
            abort(500, f"Applied, but not saved: {e}")
    return jsonify({"changed": changed, "path": path})
