#!/usr/bin/env python3

import os
import argparse
import pickle
import logging
from ament_index_python.resources import get_resource
import map_data.map_data as md

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("create_mapdata")


def process_map_data(file_name, download):
    """Process map data based on command-line arguments."""
    try:
        _, package_path = get_resource("packages", "map_data")
        data_path = os.path.join(package_path, "share", "map_data", "data")
    except LookupError:
        logger.error("Package 'map_data' not found")
        raise SystemExit(1)

    try:
        if download:
            map_data = md.MapData(os.path.join(data_path, file_name))
            map_data.run_queries()
            if map_data.run_parse():
                logger.error("Failed to parse map data")
                raise SystemExit(1)
            map_data.save_to_pickle()
        else:
            with open(os.path.join(data_path, file_name), "rb") as fh:
                map_data = pickle.load(fh)
            if map_data.run_parse():
                logger.error("Failed to parse map data")
                raise SystemExit(1)
            map_data.save_to_pickle()
        logger.info(f"Successfully processed map data for {file_name}")
    except FileNotFoundError:
        logger.error(f"File {file_name} not found in {data_path}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Error processing map data: {str(e)}")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create map data from GPX or download from OSM."
    )
    parser.add_argument(
        "-f",
        "--file",
        default="buchlovice.gpx",
        help="GPX file name (default: buchlovice.gpx)",
    )
    parser.add_argument(
        "-d", "--download", action="store_true", help="Download data from OSM"
    )
    args = parser.parse_args()

    process_map_data(args.file, args.download)


if __name__ == "__main__":
    main()
