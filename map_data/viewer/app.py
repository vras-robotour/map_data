import argparse
import hmac
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, request
from flask_socketio import SocketIO
from werkzeug.routing import IntegerConverter

from ..utils.config import setup_logging
from .ros_node import ROS_AVAILABLE, TrackerNode
from .routes import bp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS: default to same-origin only. Set MAP_DATA_CORS_ORIGINS to a
# comma-separated list of allowed origins, or to "*" to explicitly allow any
# origin (e.g. when serving a separately-hosted frontend). Same-origin is
# safe for the normal case of opening the viewer at http://<host>:<port>/,
# regardless of what host/port is chosen, since it is computed per-request
# from the request's own Host header rather than hardcoded.
# ---------------------------------------------------------------------------


def _resolve_cors_origins() -> str | list[str] | None:
    raw = os.environ.get("MAP_DATA_CORS_ORIGINS", "").strip()
    if not raw:
        return None  # engineio default: only the request's own origin is allowed
    if raw == "*":
        return "*"
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


socketio = SocketIO()
tracker_node = None

# ---------------------------------------------------------------------------
# Optional access token, off by default. Set MAP_DATA_ACCESS_TOKEN to require
# it on every request (HTTP and SocketIO). Read fresh on every check (rather
# than cached at startup) so it behaves consistently under test monkeypatching
# and so toggling it doesn't require restarting a long-lived process object.
# ---------------------------------------------------------------------------
ACCESS_TOKEN_HEADER = "X-Access-Token"
ACCESS_TOKEN_QUERY_PARAM = "access_token"
ACCESS_TOKEN_COOKIE = "map_data_access_token"


def _configured_access_token() -> str | None:
    return os.environ.get("MAP_DATA_ACCESS_TOKEN") or None


def _access_token_valid(req: Any) -> bool:
    expected = _configured_access_token()
    if not expected:
        return True
    supplied = (
        req.headers.get(ACCESS_TOKEN_HEADER)
        or req.args.get(ACCESS_TOKEN_QUERY_PARAM)
        or req.cookies.get(ACCESS_TOKEN_COOKIE)
    )
    return bool(supplied) and hmac.compare_digest(supplied, expected)


@socketio.on("connect")
def _authenticate_socketio_connection() -> bool | None:
    """
    Reject SocketIO connections when MAP_DATA_ACCESS_TOKEN is set and the
    connecting client didn't supply a matching token (header, query param, or
    the cookie set by a prior authenticated HTTP request). No-op when the
    token is unset.
    """
    if not _access_token_valid(request):
        logger.warning("Rejected SocketIO connection: missing or invalid access token")
        return False
    return None


class SignedIntConverter(IntegerConverter):
    regex = r"-?\d+"


def telemetry_broadcaster(interval: float) -> None:
    """
    Background thread to broadcast ROS2 telemetry via WebSockets.
    """
    global tracker_node
    while True:
        if tracker_node:
            try:
                data = tracker_node.get_telemetry()
                if data:
                    socketio.emit("telemetry", data)
            except Exception:
                logger.exception("Error in telemetry broadcaster")
        time.sleep(interval)


def create_app(data_dir: str | None = None, telemetry_hz: float = 2.0) -> Flask:
    # Explicitly set paths relative to this file
    base_dir = Path(__file__).parent
    template_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.url_map.converters["signed_int"] = SignedIntConverter

    if data_dir:
        app.config["DATA_DIR"] = data_dir

    app.register_blueprint(bp)
    socketio.init_app(app, cors_allowed_origins=_resolve_cors_origins())

    # Optional access-token gate (opt-in via MAP_DATA_ACCESS_TOKEN, see above).
    # Checked fresh on every request so it's a no-op when the env var is unset.
    @app.before_request
    def _enforce_access_token() -> None:
        if not _access_token_valid(request):
            abort(401, "Missing or invalid access token")

    @app.after_request
    def _persist_access_token_cookie(response: Response) -> Response:
        expected = _configured_access_token()
        if expected and request.args.get(ACCESS_TOKEN_QUERY_PARAM) == expected:
            # Lets the browser UI authenticate once via a URL query param
            # (e.g. http://host:5000/?access_token=...) and have the cookie
            # carry that authentication for subsequent same-origin static
            # asset / API / SocketIO requests made by the page's own JS.
            response.set_cookie(
                ACCESS_TOKEN_COOKIE,
                expected,
                httponly=True,
                samesite="Lax",
            )
        return response

    # Context processor to expose ROS status to templates
    @app.context_processor
    def inject_vars() -> dict[str, bool]:
        return {"ros_available": ROS_AVAILABLE}

    global tracker_node
    if ROS_AVAILABLE:
        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
            tracker_node = TrackerNode()

            # Start ROS2 spin in a separate thread
            def ros_spin() -> None:
                rclpy.spin(tracker_node)

            spin_thread = threading.Thread(target=ros_spin, daemon=True)
            spin_thread.start()

            # Start telemetry broadcaster
            broadcaster_thread = threading.Thread(
                target=telemetry_broadcaster, args=(1.0 / telemetry_hz,), daemon=True
            )
            broadcaster_thread.start()

            logger.info("ROS2 TrackerNode initialized and spinning.")
        except Exception:
            logger.exception("Failed to initialize ROS2")
            tracker_node = None

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive map data viewer")
    parser.add_argument("--data-dir", help="Directory containing .mapdata files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--telemetry-rate",
        type=float,
        default=2.0,
        help="Tracker telemetry broadcast rate in Hz (default: 2)",
    )

    # Filter out ROS-specific arguments before parsing
    ros_args = []
    try:
        from rclpy.utilities import remove_ros_args

        ros_args = remove_ros_args(args=sys.argv[1:])
    except ImportError:
        ros_args = sys.argv[1:]

    args, _ = parser.parse_known_args(args=ros_args)
    if args.telemetry_rate <= 0:
        parser.error("--telemetry-rate must be positive")

    data_dir = None

    if args.data_dir:
        data_dir = str(Path(args.data_dir).resolve())

    app = create_app(data_dir=data_dir, telemetry_hz=args.telemetry_rate)

    setup_logging()
    # Using socketio.run instead of app.run
    # Disable debug mode to prevent the Flask reloader from initializing the ROS node twice
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
