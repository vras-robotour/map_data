#!/usr/bin/env python3

import os
import sys
import rclpy
import pickle
from rclpy.node import Node
import matplotlib.pyplot as plt
from ament_index_python.resources import get_resource
from map_data.vis_utils import plot_map, save_map, plot_footways_plan


class VisualizeMapData(Node):
    def __init__(self):
        super().__init__("visualize_mapdata")
        self.path = sys.path[0]
        self.visualize_map_data()

    def visualize_map_data(self):
        """Visualize map data based on command-line arguments."""
        _, package_path = get_resource("packages", "map_data")
        self.data_path = os.path.join(package_path, "share", "map_data", "data")
        file_name = "buchlovice.mapdata"
        image_file = None
        bgd_file = None
        if len(sys.argv) > 1:
            if "-f" in sys.argv:
                try:
                    file_name = sys.argv[sys.argv.index("-f") + 1]
                except IndexError:
                    self.get_logger().error("No file name provided after -f flag")
                    raise SystemExit(1)
            if "-sm" in sys.argv:
                image_file = "map.png"
                if "-if" in sys.argv:
                    try:
                        image_file = sys.argv[sys.argv.index("-if") + 1]
                    except IndexError:
                        self.get_logger().error(
                            "No image file name provided after -if flag"
                        )
                        raise SystemExit(1)
            if "-sb" in sys.argv:
                bgd_file = "bgd_map.png"
                if "-bf" in sys.argv:
                    try:
                        bgd_file = sys.argv[sys.argv.index("-bf") + 1]
                    except IndexError:
                        self.get_logger().error(
                            "No background file name provided after -bf flag"
                        )
                        raise SystemExit(1)

        try:
            with open(os.path.join(self.data_path, file_name), "rb") as fh:
                map_data = pickle.load(fh)
        except FileNotFoundError:
            self.get_logger().error(
                f"Map data file {file_name} not found in {self.data_path}"
            )
            raise SystemExit(1)
        # except Exception as e:
        #     self.get_logger().error(f"Error loading map data: {str(e)}")
        #     raise SystemExit(1)

        try:
            self.get_logger().info(f"Plotting map data from {file_name}")
            plot_map(map_data, bgd_file)
            if image_file is not None:
                save_map(self.path + "/data/" + image_file)
                self.get_logger().info(f"Saved map to {image_file}")
            plt.show()

            self.get_logger().info("Plotting footways plan")
            plot_footways_plan(map_data)
            plt.show()
        except Exception as e:
            self.get_logger().error(f"Error during visualization: {str(e)}")
            raise SystemExit(1)


def main():
    rclpy.init()
    node = VisualizeMapData()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
