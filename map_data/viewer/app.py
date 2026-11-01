import argparse
import hmac
import importlib.util
import ipaddress
import logging
import os
import re
import secrets
import signal
import socket
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

# CSRF guard: a state-changing request must carry this header (any value) or the token
# header, with or without a token configured. Browsers send custom headers cross-site only
# after a CORS preflight, which the HTTP API never grants; static/js/utils.js sets it.
CSRF_CUSTOM_HEADER = "X-Requested-With"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'",
}

_TOKEN_IN_URL = re.compile(rf"({ACCESS_TOKEN_QUERY_PARAM}=)[^&\s]*")


def _redact_token(record: logging.LogRecord) -> bool:
    """Logging filter: blank ``access_token`` values; werkzeug puts the request line in args."""
    if isinstance(record.args, tuple):
        record.args = tuple(
            _TOKEN_IN_URL.sub(r"\1***", a) if isinstance(a, str) else a for a in record.args
        )
    return True


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
    return bool(cookie) and hmac.compare_digest(cookie.encode(), expected.encode())


def _csrf_header_present(req: Any) -> bool:
    """Whether *req* is safe, SocketIO (engineio checks the Origin) or carries a custom header."""
    return (
        req.method in _SAFE_METHODS
        or req.path.startswith("/socket.io")
        or CSRF_CUSTOM_HEADER in req.headers
        or ACCESS_TOKEN_HEADER in req.headers
    )


def _token_file() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser()
    return config_home / "map_data" / "access_token"


def _ensure_access_token(host: str, port: int) -> None:
    """
    Give a viewer that listens beyond loopback an access token.

    MAP_DATA_ACCESS_TOKEN wins; otherwise the token in :func:`_token_file` is used, created
    (mode 0600) on first use, so links and cookies survive a restart.
    """
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if loopback or _configured_access_token():
        return
    path = _token_file()
    try:
        token = path.read_text().strip()
    except FileNotFoundError:
        token = ""
    if not token:
        token = secrets.token_urlsafe(24)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
    os.environ["MAP_DATA_ACCESS_TOKEN"] = token
    logger.warning(
        "Listening on %s without MAP_DATA_ACCESS_TOKEN: using the token in %s. Open "
        "http://%s:%d/?%s=%s",
        host,
        path,
        socket.gethostname(),
        port,
        ACCESS_TOKEN_QUERY_PARAM,
        token,
    )


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
    """Thread body: emit tracker telemetry deltas over SocketIO every *interval* seconds."""
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
        if not _csrf_header_present(request):
            abort(403, f"Missing {CSRF_CUSTOM_HEADER} header")
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
            # One leading "/": "/\host" or "//host" would be a scheme-relative redirect off-site.
            path = "/" + request.path.lstrip("/\\")
            return redirect(path + (f"?{urlencode(rest)}" if rest else ""))
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

    @app.after_request
    def _security_headers(response: Response) -> Response:
        response.headers.update(SECURITY_HEADERS)
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

    # argparse does not recognize ROS arguments (--ros-args, -r, etc.)
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
    logging.getLogger("werkzeug").addFilter(_redact_token)
    _ensure_access_token(args.host, args.port)
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
