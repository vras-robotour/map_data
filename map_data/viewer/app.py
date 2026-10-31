import argparse
import hmac
import importlib.util
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from flask import Flask, Response, abort, redirect, request
from flask.typing import ResponseReturnValue
from flask_socketio import SocketIO, emit

from ..utils.config import config_path, setup_logging
from .ros_node import ROS_AVAILABLE, TrackerNode
from .routes import bp
from .tracker_config import (
    DEFAULT_CONFIG_NAME,
    TrackerConfigError,
    load_tracker_config,
    node_parameters,
)
from .tracker_routes import TRACKER_EXTENSION
from .tracker_routes import bp as tracker_bp

logger = logging.getLogger(__name__)

# CORS: same-origin only by default (checked per request against its own Host header).
# MAP_DATA_CORS_ORIGINS takes a comma-separated list of origins, or "*" for any.


def _resolve_cors_origins() -> str | list[str] | None:
    raw = os.environ.get("MAP_DATA_CORS_ORIGINS", "").strip()
    if not raw:
        return None  # engineio default: only the request's own origin is allowed
    if raw == "*":
        return "*"
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# manage_session=False: the Socket.IO handlers never use the Flask session, and a managed
# one breaks connecting on Flask >= 3.0.2 with Flask-SocketIO 5.3 (it assigns the
# read-only RequestContext.session).
socketio = SocketIO(manage_session=False)
tracker_node = None
_ros_thread: threading.Thread | None = None
# Serializes the one-time ROS init in create_app: the node and its spin/telemetry
# threads are process-wide, so later create_app calls reuse them.
_ros_init_lock = threading.Lock()

# Request body limit (uploads included), well above any realistic .mapdata or .gpx;
# Flask answers larger requests with 413 before the view runs.
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

# Optional access token: when MAP_DATA_ACCESS_TOKEN is set, every HTTP and SocketIO
# request must carry it. Read on every check, not cached, so tests can monkeypatch it.
ACCESS_TOKEN_HEADER = "X-Access-Token"
ACCESS_TOKEN_QUERY_PARAM = "access_token"
ACCESS_TOKEN_COOKIE = "map_data_access_token"

# CSRF guard for cookie auth: a state-changing request authenticated by the cookie alone
# must also carry this header (any value). Browsers send custom headers cross-site only
# after a CORS preflight, which the HTTP API never grants; static/js/utils.js sets it.
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
    # Bytes, as compare_digest raises TypeError on a non-ASCII str.
    if supplied:
        return hmac.compare_digest(supplied.encode(), expected.encode())
    cookie = req.cookies.get(ACCESS_TOKEN_COOKIE)
    if not cookie or not hmac.compare_digest(cookie.encode(), expected.encode()):
        return False
    # Cookie alone suffices for safe methods; anything else also needs the CSRF header,
    # since the browser attaches the cookie to cross-site requests too. SocketIO is
    # exempt: engineio checks the Origin itself (MAP_DATA_CORS_ORIGINS).
    if getattr(req, "method", "GET") in _SAFE_METHODS:
        return True
    if req.path.startswith("/socket.io"):
        return True
    return CSRF_CUSTOM_HEADER in req.headers


@socketio.on("connect")
def _authenticate_socketio_connection() -> bool | None:
    """Reject a SocketIO connection without a valid access token (when one is set)."""
    if not _access_token_valid(request):
        logger.warning("Rejected SocketIO connection: missing or invalid access token")
        return False
    # The broadcast carries only what changed, so a new browser gets the full state first
    if tracker_node is not None:
        emit("telemetry", tracker_node.snapshot())
    return None


def telemetry_broadcaster(interval: float) -> None:
    """Thread body: emit the tracker's telemetry over SocketIO every *interval* seconds."""
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


def stop_ros() -> None:
    """Stop the executor thread and destroy the tracker node (no-op without ROS)."""
    global tracker_node
    if tracker_node is None:
        return
    import rclpy

    rclpy.try_shutdown()  # makes the executor thread return from spin()
    if _ros_thread is not None:
        _ros_thread.join(timeout=5.0)
    tracker_node.destroy_node()
    tracker_node = None


def create_app(
    data_dir: str | None = None,
    telemetry_hz: float = 2.0,
    tracker_config: str | Path | None = None,
) -> Flask:
    base_dir = Path(__file__).parent
    template_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    if data_dir:
        app.config["DATA_DIR"] = data_dir

    # Tracker topics come from this ROS 2 parameter file (the package's config/tracker.yaml
    # unless --config names another); the web app saves topic changes back to it.
    app.config["TRACKER_CONFIG"] = str(tracker_config or config_path(DEFAULT_CONFIG_NAME))

    app.register_blueprint(bp)
    app.register_blueprint(tracker_bp)
    socketio.init_app(app, cors_allowed_origins=_resolve_cors_origins())

    # Access-token gate; a no-op while MAP_DATA_ACCESS_TOKEN is unset.
    @app.before_request
    def _enforce_access_token() -> ResponseReturnValue | None:
        if not _access_token_valid(request):
            abort(401, "Missing or invalid access token")
        expected = _configured_access_token()
        if (
            expected
            and request.method == "GET"
            and not request.path.startswith("/socket.io")
            and request.args.get(ACCESS_TOKEN_QUERY_PARAM) == expected
        ):
            # Redirect to the same URL without the token (the cookie set in
            # _persist_access_token_cookie carries it from here on), so it stays out of
            # access logs, the history and the Referer.
            rest = [
                (k, v) for k, v in request.args.items(multi=True) if k != ACCESS_TOKEN_QUERY_PARAM
            ]
            return redirect(request.path + (f"?{urlencode(rest)}" if rest else ""))
        return None

    @app.after_request
    def _persist_access_token_cookie(response: Response) -> Response:
        expected = _configured_access_token()
        if expected and request.args.get(ACCESS_TOKEN_QUERY_PARAM) == expected:
            # Authenticate once via ?access_token=...; the cookie carries it for the
            # page's later asset, API and SocketIO requests.
            response.set_cookie(
                ACCESS_TOKEN_COOKIE,
                expected,
                httponly=True,
                samesite="Lax",
            )
        return response

    @app.context_processor
    def inject_vars() -> dict[str, bool]:
        return {"ros_available": ROS_AVAILABLE}

    global tracker_node, _ros_thread
    if ROS_AVAILABLE:
        # Outside the try below: a broken config file must stop the viewer, not silently
        # start a tracker on default topics.
        tracker_params = node_parameters(load_tracker_config(app.config["TRACKER_CONFIG"]))
        with _ros_init_lock:
            if tracker_node is None:
                try:
                    import rclpy
                    from rclpy.signals import SignalHandlerOptions

                    if not rclpy.ok():
                        # Keep Python's default SIGINT/SIGTERM handling: rclpy's handlers only shut
                        # the ROS context down and would leave the web server running (port 5000
                        # stays busy after Ctrl-C / kill).
                        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
                    tracker_node = TrackerNode(tracker_params)

                    def ros_spin() -> None:
                        from rclpy.executors import ExternalShutdownException

                        # The C++ events executor needs a fraction of rclpy.spin's CPU (6 % vs
                        # 60 % at 250 Hz); not every distro ships it.
                        try:
                            try:
                                from rclpy.experimental.events_executor import EventsExecutor
                            except ImportError:
                                rclpy.spin(tracker_node)
                                return
                            executor = EventsExecutor()
                            executor.add_node(tracker_node)
                            executor.spin()
                        except ExternalShutdownException:
                            pass  # stop_ros() shut the context down

                    _ros_thread = threading.Thread(target=ros_spin, daemon=True)
                    _ros_thread.start()

                    threading.Thread(
                        target=telemetry_broadcaster, args=(1.0 / telemetry_hz,), daemon=True
                    ).start()

                    logger.info(
                        "ROS2 TrackerNode initialized and spinning (config %s).",
                        app.config["TRACKER_CONFIG"],
                    )
                except Exception:
                    logger.exception("Failed to initialize ROS2")
                    tracker_node = None
        if tracker_node is not None:
            app.extensions[TRACKER_EXTENSION] = tracker_node

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
    parser.add_argument(
        "--config",
        help="Tracker topics: a ROS 2 parameter file with a map_data_tracker section, e.g. "
        "config/helhest_jr.yaml. The web app saves topic changes to it "
        f"(default: the package's config/{DEFAULT_CONFIG_NAME})",
    )

    # Filter out ROS-specific arguments before parsing
    try:
        from rclpy.utilities import remove_ros_args

        ros_args = remove_ros_args(args=sys.argv[1:])
    except ImportError:
        ros_args = sys.argv[1:]

    args, _ = parser.parse_known_args(args=ros_args)
    if args.telemetry_rate <= 0:
        parser.error("--telemetry-rate must be positive")

    data_dir = str(Path(args.data_dir).resolve()) if args.data_dir else None

    setup_logging()
    tracker_config = Path(args.config).resolve() if args.config else None
    if tracker_config and not tracker_config.is_file():
        logger.warning("%s does not exist yet: default topics, Save creates it", tracker_config)

    try:
        app = create_app(
            data_dir=data_dir, telemetry_hz=args.telemetry_rate, tracker_config=tracker_config
        )
    except TrackerConfigError as e:
        parser.error(str(e))

    # Explicit handlers: a SIGINT inherited as "ignored" (a background job of a
    # non-interactive shell) would otherwise stay ignored, as rclpy's own handler is off.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if importlib.util.find_spec("simple_websocket") is None:
        logger.warning(
            "simple-websocket is not installed: browsers fall back to HTTP long-polling, "
            "one request per telemetry message (pip install simple-websocket)"
        )
    # debug=False: the Flask reloader would initialize the ROS node twice.
    try:
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ros()


if __name__ == "__main__":
    main()
