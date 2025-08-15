#!/usr/bin/env python3

import os
import sys
import rclpy
import pickle
from rclpy.node import Node
from ament_index_python.resources import get_resource

import map_data.map_data as md


class CreateMapData(Node):
    def __init__(self):
        super().__init__("create_mapdata")
        try:
            _, package_path = get_resource("packages", "map_data")
            self.data_path = os.path.join(package_path, "share", "map_data", "data")
        except LookupError:
            self.get_logger().error("Package 'your_package' not found")
            raise SystemExit(1)
        self.process_map_data()

    def process_map_data(self):
        """Process map data based on command-line arguments."""
        download = False
        file_name = "buchlovice.gpx"
        if len(sys.argv) > 1:
            if "-d" in sys.argv:
                download = True
            if "-f" in sys.argv:
                try:
                    file_name = sys.argv[sys.argv.index("-f") + 1]
                except IndexError:
                    self.get_logger().error("No file name provided after -f flag")
                    raise SystemExit(1)

        try:
            if download:
                map_data = md.MapData(os.path.join(self.data_path, file_name))
                map_data.run_queries()
                if map_data.run_parse():
                    self.get_logger().error("Failed to parse map data")
                    raise SystemExit(1)
                map_data.save_to_pickle()
            else:
                with open(os.path.join(self.data_path, file_name), "rb") as fh:
                    map_data = pickle.load(fh)
                if map_data.run_parse():
                    self.get_logger().error("Failed to parse map data")
                    raise SystemExit(1)
                map_data.save_to_pickle()
            self.get_logger().info(f"Successfully processed map data for {file_name}")
        except FileNotFoundError:
            self.get_logger().error(f"File {file_name} not found in {self.data_path}")
            raise SystemExit(1)
        except Exception as e:
            self.get_logger().error(f"Error processing map data: {str(e)}")
            raise SystemExit(1)


def main():
    rclpy.init()
    node = CreateMapData()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
