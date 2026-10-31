#!/usr/bin/env python3

import argparse
import logging

import map_data.map_data as md
from map_data.utils.config import resolve_data_file, setup_logging

logger = logging.getLogger(__name__)


def process_map_data(file_name: str, *, download: bool, use_cache: bool = False) -> None:
    """
    Download and parse (*download*) or re-save the ``.mapdata`` of *file_name*.

    *file_name* is a path, or a name in the package's ``data/`` directory or the
    bundled ``demo/`` (a source checkout; an install copies the demo into ``data/``).
    """
    full_path = resolve_data_file(file_name)
    if full_path is None:
        logger.error("File '%s' not found", file_name)
        raise SystemExit(1)

    try:
        if download:
            map_data = md.MapData(str(full_path))
            map_data.run_queries(use_cache=use_cache)
            if map_data.run_parse() != 0:
                logger.error("Failed to parse map data")
                raise SystemExit(1)
            map_data.save()
        else:
            mapdata_path = str(full_path.with_suffix(".mapdata"))
            map_data = md.MapData.load(mapdata_path)
            map_data.save(mapdata_path)
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
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use the local OSM response cache (30 days; ignored by --no-cache)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Refetch OSM data")
    args = parser.parse_args()

    process_map_data(args.file, download=args.download, use_cache=args.cache and not args.no_cache)


if __name__ == "__main__":
    main()
