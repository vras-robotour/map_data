import argparse
import os
import logging
from map_data.map_data import MapData

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("map_data_info")


def get_stats(path: str):
    if not os.path.isfile(path):
        logger.error(f"File not found: {path}")
        return

    try:
        md = MapData.load(path)
    except Exception as e:
        logger.error(f"Failed to load map data: {e}")
        return

    print("=" * 40)
    print(f"MAP DATA STATISTICS: {os.path.basename(path)}")
    print("=" * 40)

    source = f"File: {md.coords_file}" if md.coords_file else "Array"
    print(f"Source:      {source}")
    print(f"UTM Zone:    {md.zone_number}{md.zone_letter}")
    print(f"Waypoints:   {len(md.waypoints)}")

    area = (md.max_x - md.min_x) * (md.max_y - md.min_y)
    print(f"Bounds X:    [{md.min_x:.1f}, {md.max_x:.1f}]")
    print(f"Bounds Y:    [{md.min_y:.1f}, {md.max_y:.1f}]")
    print(f"Total Area:  {area:,.0f} m²")

    print("-" * 40)
    print(f"Roads:       {len(md.roads_list)}")
    print(f"Footways:    {len(md.footways_list)}")
    print(f"Barriers:    {len(md.barriers_list)}")

    total_footway_len = sum(w.line.length for w in md.footways_list if w.line)
    print(f"Total Footway Distance: {total_footway_len:.1f} m")

    # Check for annotations sidecar
    ann_path = path.rsplit(".", 1)[0] + ".annotations.json"
    if os.path.isfile(ann_path):
        import json

        with open(ann_path) as f:
            ann_data = json.load(f)
            anns = ann_data.get("annotations", [])
            print(f"Annotations: {len(anns)} (manual edits)")

    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Display statistics for a .mapdata file"
    )
    parser.add_argument("file", help="Path to the .mapdata file")
    args = parser.parse_args()

    get_stats(args.file)


if __name__ == "__main__":
    main()
