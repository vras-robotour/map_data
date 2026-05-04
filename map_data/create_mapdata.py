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
    # Try to use file_name as a direct path first
    if os.path.exists(file_name):
        full_path = os.path.abspath(file_name)
    else:
        # Fallback to package share directory
        try:
            _, package_path = get_resource("packages", "map_data")
            data_path = os.path.join(package_path, "share", "map_data", "data")
            full_path = os.path.join(data_path, file_name)
        except LookupError:
            logger.error(f"File '{file_name}' not found and package 'map_data' not found")
            raise SystemExit(1)

    if not os.path.exists(full_path) and not download:
        logger.error(f"File {full_path} not found")
        raise SystemExit(1)

    try:
        if download:
            # If downloading and using fallback path, ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            map_data = md.MapData(full_path)
            map_data.run_queries()
            if map_data.run_parse():
                logger.error("Failed to parse map data")
                raise SystemExit(1)
            map_data.save_to_pickle()
        else:
            with open(full_path, "rb") as fh:
                map_data = pickle.load(fh)
            if map_data.run_parse():
                logger.error("Failed to parse map data")
                raise SystemExit(1)
            map_data.save_to_pickle()
        logger.info(f"Successfully processed map data for {file_name}")
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
