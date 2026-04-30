import argparse
import os

import logging
from flask import Flask
from .routes import bp


def create_app(data_dir=None):
    app = Flask(__name__)

    if data_dir:
        app.config["DATA_DIR"] = data_dir

    app.register_blueprint(bp)

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive map data viewer")
    parser.add_argument("--data-dir", help="Directory containing .mapdata files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    data_dir = None
    if args.data_dir:
        data_dir = os.path.realpath(args.data_dir)

    app = create_app(data_dir=data_dir)

    logging.basicConfig(level=logging.INFO)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
