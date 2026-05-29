#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from ament_index_python.resources import get_resource

import map_data.map_data as md
from map_data.utils.config import setup_logging

logger = logging.getLogger(__name__)


def process_map_data(file_name: str, *, download: bool) -> None:
    """
    Process map data based on command-line arguments.
    """
    # Try to use file_name as a direct path first
    p = Path(file_name)
    if p.exists():
        full_path = p.resolve()
    else:
        # Fallback to package share directory
        try:
            _, package_path = get_resource("packages", "map_data")
            full_path = Path(package_path) / "share" / "map_data" / "data" / file_name
        except LookupError as e:
            logger.exception("File '%s' not found and package 'map_data' not found", file_name)
            raise SystemExit(1) from e

    if not full_path.exists() and not download:
        logger.error("File %s not found", full_path)
        raise SystemExit(1)

    try:
        if download:
            map_data = md.MapData(str(full_path))
            map_data.run_queries()
            if map_data.run_parse() != 0:
                logger.error("Failed to parse map data")
                raise SystemExit(1)
            map_data.save()
        else:
            mapdata_path = str(full_path.with_suffix(".mapdata"))
            map_data = md.MapData.load(mapdata_path)
            map_data.save()
        logger.info("Successfully processed map data for %s", file_name)
    except Exception as e:
        logger.exception("Error processing map data")
        raise SystemExit(1) from e


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Create map data from GPX or download from OSM.")
    parser.add_argument(
        "-f",
        "--file",
        default="buchlovice.gpx",
        help="GPX file name (default: buchlovice.gpx)",
    )
    parser.add_argument("-d", "--download", action="store_true", help="Download data from OSM")
    args = parser.parse_args()

    process_map_data(args.file, download=args.download)


if __name__ == "__main__":
    main()
