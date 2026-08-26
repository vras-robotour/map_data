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

# Upper bound on any request body (uploads included). Keeps a single oversized
# POST from exhausting disk/memory; comfortably above any realistic .mapdata
# or .gpx file. Flask rejects larger requests with 413 before the view runs.
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

# ---------------------------------------------------------------------------
# Optional access token, off by default. Set MAP_DATA_ACCESS_TOKEN to require
# it on every request (HTTP and SocketIO). Read fresh on every check (rather
# than cached at startup) so it behaves consistently under test monkeypatching
# and so toggling it doesn't require restarting a long-lived process object.
# ---------------------------------------------------------------------------
ACCESS_TOKEN_HEADER = "X-Access-Token"
ACCESS_TOKEN_QUERY_PARAM = "access_token"
ACCESS_TOKEN_COOKIE = "map_data_access_token"

# CSRF protection for cookie-based auth: state-changing requests that
# authenticate via the cookie alone must also carry this custom header (any
# value). Browsers refuse to attach custom headers to cross-site requests
# without a successful CORS preflight -- which this server never grants for
# its HTTP API -- so a cross-site page can't forge such a request, while the
# viewer's own JS (see static/js/utils.js) sets the header on every fetch.
CSRF_CUSTOM_HEADER = "X-Requested-With"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _configured_access_token() -> str | None:
    return os.environ.get("MAP_DATA_ACCESS_TOKEN") or None


def _access_token_valid(req: Any) -> bool:
    expected = _configured_access_token()
    if not expected:
        return True
    # Header and query param are immune to CSRF (a cross-site page can't set
    # either on a request that carries them to a same-origin-only API), so
    # they authenticate any method.
    supplied = req.headers.get(ACCESS_TOKEN_HEADER) or req.args.get(ACCESS_TOKEN_QUERY_PARAM)
    if supplied:
        return hmac.compare_digest(supplied, expected)
    cookie = req.cookies.get(ACCESS_TOKEN_COOKIE)
    if not cookie or not hmac.compare_digest(cookie, expected):
        return False
    # Cookie-only auth: sufficient for safe methods (page loads, static
    # assets, API reads), but state-changing requests additionally need the
    # custom CSRF header -- the browser attaches the cookie automatically, so
    # the cookie alone doesn't prove the request came from the viewer's own
    # page. SocketIO traffic is exempt: engineio already enforces its own
    # same-origin Origin check on polling requests (MAP_DATA_CORS_ORIGINS).
    if getattr(req, "method", "GET") in _SAFE_METHODS:
        return True
    if req.path.startswith("/socket.io"):
        return True
    return CSRF_CUSTOM_HEADER in req.headers


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
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

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
            from rclpy.signals import SignalHandlerOptions

            if not rclpy.ok():
                # Keep Python's default SIGINT/SIGTERM handling: rclpy's handlers only shut
                # the ROS context down and would leave the web server running (port 5000
                # stays busy after Ctrl-C / kill).
                rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
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
