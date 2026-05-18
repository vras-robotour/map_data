import argparse
import logging
import os
import threading
import time

from flask import Flask
from flask_socketio import SocketIO
from werkzeug.routing import IntegerConverter

from .routes import bp
from .ros_node import TrackerNode, ROS_AVAILABLE

socketio = SocketIO(cors_allowed_origins="*")
tracker_node = None


class SignedIntConverter(IntegerConverter):
    regex = r"-?\d+"


def telemetry_broadcaster() -> None:
    """Background thread to broadcast ROS2 telemetry via WebSockets."""
    global tracker_node
    while True:
        if tracker_node:
            try:
                data = tracker_node.get_telemetry()
                if data:
                    socketio.emit("telemetry", data)
            except Exception as e:
                logging.error(f"Error in telemetry broadcaster: {e}")
        time.sleep(0.5)  # 2 Hz update rate


def create_app(data_dir: Optional[str] = None) -> Flask:
    # Explicitly set paths relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(base_dir, "templates")
    static_dir = os.path.join(base_dir, "static")

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.url_map.converters["signed_int"] = SignedIntConverter

    if data_dir:
        app.config["DATA_DIR"] = data_dir

    app.register_blueprint(bp)
    socketio.init_app(app)

    # Context processor to expose ROS status to templates
    @app.context_processor
    def inject_vars():
        return dict(ros_available=ROS_AVAILABLE)

    global tracker_node
    if ROS_AVAILABLE:
        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
            tracker_node = TrackerNode()

            # Start ROS2 spin in a separate thread
            def ros_spin():
                rclpy.spin(tracker_node)

            spin_thread = threading.Thread(target=ros_spin, daemon=True)
            spin_thread.start()

            # Start telemetry broadcaster
            broadcaster_thread = threading.Thread(
                target=telemetry_broadcaster, daemon=True
            )
            broadcaster_thread.start()

            logging.info("ROS2 TrackerNode initialized and spinning.")
        except Exception as e:
            logging.error(f"Failed to initialize ROS2: {e}")
            tracker_node = None

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive map data viewer")
    parser.add_argument("--data-dir", help="Directory containing .mapdata files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)

    # Filter out ROS-specific arguments before parsing
    import sys

    ros_args = []
    try:
        from rclpy.utilities import remove_ros_args

        ros_args = remove_ros_args(args=sys.argv[1:])
    except ImportError:
        ros_args = sys.argv[1:]

    args, unknown = parser.parse_known_args(args=ros_args)

    data_dir = None

    if args.data_dir:
        data_dir = os.path.realpath(args.data_dir)

    app = create_app(data_dir=data_dir)

    logging.basicConfig(level=logging.INFO)
    # Using socketio.run instead of app.run
    # Disable debug mode to prevent the Flask reloader from initializing the ROS node twice
    socketio.run(
        app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True
    )


if __name__ == "__main__":
    main()
