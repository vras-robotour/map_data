import matplotlib.pyplot as plt
import numpy as np
import shapely as sh
from matplotlib.patches import Polygon as MplPolygon


def visualize_replan(
    path: np.ndarray | None,
    grid_2d: np.ndarray,
    low: tuple[float, float],
    high: tuple[float, float],
    obstacles: list[sh.geometry.base.BaseGeometry],
    old_path: np.ndarray | None = None,
    save_path: str = "replan.png",
) -> None:
    """Visualize the grid, obstacles, and path using Matplotlib."""
    _, ax = plt.subplots()

    # Plot grid as a heatmap (0: white, 1: gray)
    ax.imshow(
        grid_2d,
        cmap="Greys",
        origin="lower",
        extent=[
            low[0],
            high[0],
            low[1],
            high[1],
        ],
    )

    # Plot obstacles
    for obstacle in obstacles:
        if obstacle.geom_type == "Polygon":
            x, y = obstacle.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), color="red", alpha=0.5))
        elif obstacle.geom_type == "MultiPolygon":
            for poly in obstacle.geoms:
                x, y = poly.exterior.xy
                ax.add_patch(MplPolygon(list(zip(x, y)), color="red", alpha=0.5))

    # Plot old path if provided
    if old_path is not None:
        ax.plot(old_path[:, 0], old_path[:, 1], "c-", linewidth=2, label="Old Path")

    # Plot path if found
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "m-", linewidth=2, label="New Path")
        ax.scatter(path[:, 0], path[:, 1], c="m", s=20, label="Path Points")

        # Plot start and goal
        ax.plot(path[0, 0], path[0, 1], "go", label="Start")
        ax.plot(path[-1, 0], path[-1, 1], "bo", label="Goal")

    # Set plot properties
    ax.set_xlabel("Northing [m]")
    ax.set_ylabel("Easting [m]")
    ax.set_title("Replanned Path")

    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    plt.savefig(save_path)
    plt.close()
