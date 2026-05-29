import argparse
import logging
from pathlib import Path

from map_data.map_data import MapData
from map_data.utils.config import setup_logging

logger = logging.getLogger(__name__)


def get_stats(path: str) -> None:
    p = Path(path)
    if not p.is_file():
        logger.error("File not found: %s", path)
        return

    try:
        md = MapData.load(path)
    except Exception:
        logger.exception("Failed to load map data")
        return

    print("=" * 40)
    print(f"MAP DATA STATISTICS: {p.name}")
    print("=" * 40)

    source = f"File: {md.coords_file}" if md.coords_file else "Array"
    print(f"Source:      {source}")
    print(f"UTM Zone:    {md.zone_number}{md.zone_letter}")

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
    ann_path = p.with_suffix(".annotations.json")
    if ann_path.is_file():
        import json

        with ann_path.open() as f:
            ann_data = json.load(f)
            anns = ann_data.get("annotations", [])
            print(f"Annotations: {len(anns)} (manual edits)")

    print("=" * 40)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Display statistics for a .mapdata file")
    parser.add_argument("file", help="Path to the .mapdata file")
    args = parser.parse_args()

    get_stats(args.file)


if __name__ == "__main__":
    main()
