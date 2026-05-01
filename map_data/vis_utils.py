from typing import List, Optional, Any
import utm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from map_data.background_map import get_background_image
from map_data.way import Way


def plot_background_map(ax: Axes, image: Image.Image, coords_data: Any):
    """
    Plot background map.
    """
    min_utm = utm.from_latlon(
        coords_data.min_lat - coords_data.y_margin,
        coords_data.min_long - coords_data.x_margin,
    )
    max_utm = utm.from_latlon(
        coords_data.max_lat + coords_data.y_margin,
        coords_data.max_long + coords_data.x_margin,
    )
    ax.imshow(
        image,
        extent=[min_utm[0], max_utm[0], min_utm[1], max_utm[1]],
        alpha=1,
        zorder=0,
    )

    ax.set_ylim([min_utm[1], max_utm[1]])
    ax.set_xlim([min_utm[0], max_utm[0]])
    print("Background map plotted")


def plot_path(ax: Axes, path: np.ndarray):
    """
    Plot path.
    """
    ax.scatter(
        path[:, 0],
        path[:, 1],
        color="#000000",
        alpha=0.8,
        s=3,
        marker="o",
        zorder=18000,
    )
    ax.scatter(
        path[:, 0],
        path[:, 1],
        color="#50C2F6",
        alpha=0.8,
        s=2,
        marker="o",
        zorder=20000,
    )
    print("Path plotted")


def plot_barrier_areas(ax: Axes, barrier_areas: List[Way]):
    """
    Plot barriers in map.
    """
    for area in barrier_areas:
        if not area.line:
            continue
        x, y = area.line.exterior.xy
        ax.plot(x, y, c="#BF0009", linewidth=1, zorder=7)

        if area.in_out != "inner":
            ax.fill(x, y, c="#BF0009", alpha=0.4, zorder=5)
    print("Barrier areas plotted")


def plot_footways(ax: Axes, footways: List[Way]):
    """
    Plot footways in map.
    """
    for footway in footways:
        if not footway.line:
            continue
        x, y = footway.line.exterior.xy
        ax.plot(x, y, c="#FFD700", linewidth=0.5, zorder=6)
        ax.fill(x, y, c="#FFD700", alpha=0.4, zorder=4)
    print("Footways plotted")


def plot_roads(ax: Axes, roads: List[Way]):
    """
    Plot roads in map.
    """
    for road in roads:
        if not road.line:
            continue
        x, y = road.line.exterior.xy
        ax.plot(x, y, c="#000000", linewidth=1, zorder=6)
        ax.fill(x, y, c="#000000", alpha=0.8, zorder=5)
    print("Roads plotted")


def save_map(file_name: str):
    """
    Save map to file.
    """
    plt.savefig(file_name)
    print(f"Map saved to {file_name}")


def save_bgd_map(bgd_map: Image.Image, bgd_file: Optional[str] = None):
    """
    Save the background image to file.
    """
    if bgd_file is not None:
        # Use a more robust way to find the data directory
        try:
            from ament_index_python.resources import get_resource

            _, package_path = get_resource("packages", "map_data")
            data_path = os.path.join(package_path, "share", "map_data", "data")
        except Exception:
            data_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data")
            )

        file_name = os.path.join(data_path, bgd_file)
        bgd_map.save(file_name)
        print(f"Background image saved to {file_name}")


def plot_map(map_data: Any, bgd_file: Optional[str] = None):
    """
    Plot map from map_data.
    """
    _, ax = plt.subplots(figsize=(12, 12), dpi=400)

    coords_data = map_data.coords_data
    bgd_map = get_background_image(
        coords_data.min_long,
        coords_data.max_long,
        coords_data.min_lat,
        coords_data.max_lat,
        coords_data.x_margin,
        coords_data.y_margin,
    )

    plot_background_map(ax, bgd_map, coords_data)

    save_bgd_map(bgd_map, bgd_file)

    plot_barrier_areas(ax, map_data.barriers_list)
    plot_footways(ax, map_data.footways_list)
    plot_roads(ax, map_data.roads_list)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


def plot_footways_plan(map_data: Any, bgd_file: Optional[str] = None):
    """
    Plot only footways from map_data.
    """
    _, ax = plt.subplots(figsize=(12, 12), dpi=400)

    coords_data = map_data.coords_data
    bgd_map = get_background_image(
        coords_data.min_long,
        coords_data.max_long,
        coords_data.min_lat,
        coords_data.max_lat,
        coords_data.x_margin,
        coords_data.y_margin,
    )

    plot_background_map(ax, bgd_map, coords_data)

    save_bgd_map(bgd_map, bgd_file)

    plot_footways(ax, map_data.footways_list)
