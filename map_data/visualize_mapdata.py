#!/usr/bin/env python3

import os
import argparse
import pickle
import logging
import matplotlib.pyplot as plt
from ament_index_python.resources import get_resource
from map_data.vis_utils import plot_map, save_map, plot_footways_plan

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("visualize_mapdata")


def visualize_map_data(file_name, save_map_img, image_file, save_bgd, bgd_file):
    """Visualize map data based on command-line arguments."""
    try:
        _, package_path = get_resource("packages", "map_data")
        data_path = os.path.join(package_path, "share", "map_data", "data")
    except LookupError:
        logger.error("Package 'map_data' not found")
        raise SystemExit(1)

    path = os.getcwd()

    try:
        with open(os.path.join(data_path, file_name), "rb") as fh:
            map_data = pickle.load(fh)
    except FileNotFoundError:
        logger.error(f"Map data file {file_name} not found in {data_path}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Error loading map data: {str(e)}")
        raise SystemExit(1)

    try:
        logger.info(f"Plotting map data from {file_name}")

        final_bgd_file = bgd_file if save_bgd else None
        plot_map(map_data, final_bgd_file)

        if save_map_img:
            out_img_path = os.path.join(path, "data", image_file)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            save_map(out_img_path)
            logger.info(f"Saved map to {out_img_path}")

        plt.show()

        logger.info("Plotting footways plan")
        plot_footways_plan(map_data)
        plt.show()
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Visualize map data.")
    parser.add_argument(
        "-f", "--file", default="buchlovice.mapdata", help="Map data file name"
    )
    parser.add_argument(
        "-sm",
        "--save-map",
        action="store_true",
        help="Save the plotted map as an image",
    )
    parser.add_argument(
        "-if",
        "--image-file",
        default="map.png",
        help="Output image file name (if -sm is set)",
    )
    parser.add_argument(
        "-sb", "--save-bgd", action="store_true", help="Save the background map"
    )
    parser.add_argument(
        "-bf",
        "--bgd-file",
        default="bgd_map.png",
        help="Background file name (if -sb is set)",
    )

    args = parser.parse_args()

    visualize_map_data(
        args.file, args.save_map, args.image_file, args.save_bgd, args.bgd_file
    )


if __name__ == "__main__":
    main()
