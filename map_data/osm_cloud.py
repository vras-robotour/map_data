#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros2_numpy import msgify, numpify
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.spatial import cKDTree
import pickle

import map_data.map_data as md


class OSMCloud(Node):
    def __init__(self):
        super().__init__("osm_cloud")
        self.utm_frame = self.declare_parameter("utm_frame", "utm").value
        self.local_frame = self.declare_parameter("local_frame", "map").value
        self.utm_to_local = self.declare_parameter("utm_to_local", None).value
        self.mapdata_file = self.declare_parameter("mapdata_file", None).value
        self.gpx_file = self.declare_parameter("gpx_file", None).value
        self.save_mapdata = self.declare_parameter("save_mapdata", False).value
        self.max_path_dist = self.declare_parameter("max_path_dist", 1.0).value
        self.neighbor_cost = self.declare_parameter("neighbor_cost", "linear").value
        self.grid_res = self.declare_parameter("grid_res", 0.25).value
        self.grid_max = self.declare_parameter("grid_max", [250.0, 250.0]).value
        self.grid_min = self.declare_parameter("grid_min", [-250.0, -250.0]).value

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_grid = self.create_publisher(PointCloud2, "grid", qos)

        self.tf = Buffer()
        self.tf_sub = TransformListener(self.tf, self)

        if self.mapdata_file is not None:
            with open(self.mapdata_file, "rb") as fh:
                self.map_data = pickle.load(fh)
        elif self.gpx_file is not None:
            self.map_data = md.MapData(self.gpx_file)
            self.map_data.run_all(self.save_mapdata)
        else:
            self.get_logger().error("No map data or gpx file provided")
            exit(1)

        if not self.utm_to_local:
            self.get_utm_to_local()
        else:
            self.utm_to_local = np.array(self.utm_to_local)
        self.get_logger().info(f"Using UTM to local transform: {self.utm_to_local}")

        self.grid_cloud = self.get_cloud()
        self.create_timer(10.0, self.publish_cb)
        self.get_logger().info("Initialized OSM cloud")

    def publish_cb(self):
        """
        Timer callback to publish the grid cloud.
        """
        self.grid_cloud.header.stamp = self.get_clock().now().to_msg()
        self.pub_grid.publish(self.grid_cloud)
        self.get_logger().info("Published grid cloud")

    def get_utm_to_local(self):
        """
        While rospy is not shutdown, try to get the UTM to local transform every second or until successful.
        """
        while rclpy.ok():
            try:
                utm_to_local = self.tf.lookup_transform(
                    self.local_frame,
                    self.utm_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=15.0),
                )
                self.utm_to_local = numpify(utm_to_local.transform)
                self.get_logger().info(
                    f"Got UTM to local transform: {self.utm_to_local}"
                )
                break
            except Exception as e:
                self.get_logger().warn(f"Failed to get UTM to local transform: {e}")
                rclpy.spin_once(self, timeout_sec=1.0)

    def get_cloud(self):
        """
        Return a point cloud from the map data.

        Returns:
        --------
        cloud : sensor_msgs.PointCloud2
            Created point cloud.
        """
        points = transform_points(self.map_data.get_points(), self.utm_to_local, 0)
        grid = np.pad(
            create_grid(self.grid_min, self.grid_max, self.grid_res), ((0, 0), (0, 1))
        )
        waypoints = np.pad(
            split_ways(points, self.map_data.get_ways(), self.grid_res),
            ((0, 0), (0, 1)),
        )

        grid = points_near_ref(grid, waypoints, self.max_path_dist)
        if self.neighbor_cost == "linear":
            pass
        elif self.neighbor_cost == "quadratic":
            grid[:, 3] = grid[:, 3] ** 2
        elif self.neighbor_cost == "zero":
            grid[:, 3] = 0
        else:
            self.logger.warn(f"Unknown neighbor cost: {self.neighbor_cost}")

        grid[:, 3] /= self.max_path_dist**2 if self.neighbor_cost == "quadratic" else 1
        cloud = create_cloud(grid)
        cloud.header.frame_id = self.local_frame
        cloud.header.stamp = rclpy.time.Time().to_msg()

        return cloud


def create_grid(low, high, cell_size=0.25):
    """
    Create a grid of points.

    Parameters:
    -----------
    low : tuple
        Lower bounds of the grid.
    high : tuple
        Upper bounds of the grid.
    cell_size : float
        Size of the cell.

    Returns:
    --------
    grid : np.array
        Grid of points.
    """
    low = np.round(low)
    high = np.round(high)
    xs = np.linspace(
        int(low[0]), int(high[0]), int(np.ceil((high[0] - low[0]) / cell_size))
    )
    ys = np.linspace(
        int(low[1]), int(high[1]), int(np.ceil((high[1] - low[1]) / cell_size))
    )
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid


def create_cloud(points):
    """
    Create a point cloud from points.

    Parameters:
    -----------
    points : np.array
        Points in a grid to create the cloud from.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    assert points.ndim == 2
    assert points.shape[1] == 4

    points = points.astype(np.float32)
    cloud = msgify(
        PointCloud2, unstructured_to_structured(points, names=["x", "y", "z", "cost"])
    )
    return cloud


def points_near_ref(points, reference, max_dist=1):
    """
    Get points near reference points and set linear distance as cost.

    Parameters:
    -----------
    points : np.array
        Points to check.
    reference : np.array
        Reference points.
    max_dist : float
        Maximum distance to check.

    Returns:
    --------
    points : np.array
        All points with a cost based on distance to reference points.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if not isinstance(reference, np.ndarray):
        reference = np.array(reference)

    tree = cKDTree(reference, compact_nodes=False, balanced_tree=False)
    dists, _ = np.array(tree.query(points, distance_upper_bound=max_dist))
    points = points[dists < max_dist]
    dists = dists[dists < max_dist]
    points = np.hstack([points, (dists / max_dist).reshape(-1, 1)])
    return points


def transform_points(points, transform, z=None):
    """
    Transform points.

    Parameters:
    -----------
    points : dict
        Points to transform.
    transform : np.array
        Transformation matrix.
    z : float
        Z value to set.

    Returns:
    --------
    transformed : dict
        Dictionary id: transformed point.
    """

    def transform_point(point, transform):
        """
        Transform a point with a transformation matrix.

        Parameters:
        -----------
        point : np.array
            Point to transform.
        transform : np.array
            Transformation matrix.

        Returns:
        --------
        point : np.array
            Transformed point.
        """
        assert isinstance(point, np.ndarray)
        assert isinstance(transform, np.ndarray)

        point = np.dot(transform[:3, :3], point) + transform[:3, 3:]
        return point

    transformed = {}
    for id, point in points.items():
        transformed[id] = transform_point(point, transform)
        if z is not None:
            transformed[id][2] = z
    return transformed


def split_ways(points, ways, max_dist=0.25):
    """
    Equidistantly split ways into points with a maximal step size. Also only use footways from map data,
    as we are not allowed to leave the footways.

    Parameters:
    -----------
    points : dict
        Points to split ways on.
    ways : dict
        Ways to split.
    max_dist : float
        Maximal step size.

    Returns:
    --------
    waypoints : np.array
        Waypoints created from the ways.
    """
    waypoints = []
    for way in ways["footways"]:
        for i, (n0, n1) in enumerate(zip(way.nodes, way.nodes[1:])):
            point0 = points[n0.id].ravel()[:2]
            point1 = points[n1.id].ravel()[:2]

            if i == 0:
                waypoints.append(point0)

            dist = np.linalg.norm(point1 - point0)

            if dist <= 1e-3:
                waypoints.append(point1)
                continue

            vec = (point1 - point0) / dist
            num = int(np.ceil(dist / max_dist))
            step = dist / num
            for j in range(num):
                waypoints.append(point0 + (j + 1) * step * vec)

    return np.array(waypoints)


def main():
    rclpy.init()
    osm_cloud = OSMCloud()
    rclpy.spin(osm_cloud)
    osm_cloud.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
