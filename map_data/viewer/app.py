import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO
from werkzeug.routing import IntegerConverter

from ..utils.config import setup_logging
from .ros_node import ROS_AVAILABLE, TrackerNode
from .routes import bp

logger = logging.getLogger(__name__)
socketio = SocketIO(cors_allowed_origins="*")
tracker_node = None


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
    socketio.init_app(app)

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
