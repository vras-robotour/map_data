#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros2_numpy import numpify
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any

import map_data.map_data as md


class OSMIntersections(Node):
    def __init__(self) -> None:
        super().__init__("osm_intersections")
        self.utm_frame: str = self.declare_parameter("utm_frame", "utm").value
        self.local_frame: str = self.declare_parameter("local_frame", "map").value
        self.utm_to_local_param = self.declare_parameter(
            "utm_to_local", rclpy.Parameter.Type.DOUBLE_ARRAY
        ).value
        self.mapdata_file = self.declare_parameter(
            "mapdata_file", ""
        ).value
        self.gpx_file = self.declare_parameter(
            "gpx_file", ""
        ).value

        # Handle empty strings as None for consistency
        if self.mapdata_file == "": self.mapdata_file = None
        if self.gpx_file == "": self.gpx_file = None
        self.save_mapdata: bool = self.declare_parameter("save_mapdata", False).value
        self.grid_max: List[float] = self.declare_parameter(
            "grid_max", [250.0, 250.0]
        ).value
        self.grid_min: List[float] = self.declare_parameter(
            "grid_min", [-250.0, -250.0]
        ).value

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_poses = self.create_publisher(PoseArray, "intersections", qos)
        self.pub_markers = self.create_publisher(MarkerArray, "intersection_markers", qos)

        self.tf = Buffer()
        self.tf_sub = TransformListener(self.tf, self)

        self.utm_to_local: Optional[np.ndarray] = None

        if self.mapdata_file is not None:
            with open(self.mapdata_file, "rb") as fh:
                self.map_data: md.MapData = pickle.load(fh)
        elif self.gpx_file is not None:
            self.map_data = md.MapData(self.gpx_file)
            self.map_data.run_all(self.save_mapdata)
        else:
            self.get_logger().error("No map data or gpx file provided")
            exit(1)

        if self.utm_to_local_param is None:
            self.get_utm_to_local()
        else:
            self.utm_to_local = np.array(self.utm_to_local_param)
        self.get_logger().info(f"Using UTM to local transform: {self.utm_to_local}")

        self.poses, self.markers = self.get_intersection_msgs()
        self.create_timer(10.0, self.publish_cb)
        self.get_logger().info("Initialized OSM intersections")

    def publish_cb(self) -> None:
        """
        Timer callback to publish the intersections.
        """
        now = self.get_clock().now().to_msg()
        self.poses.header.stamp = now
        for marker in self.markers.markers:
            marker.header.stamp = now

        self.pub_poses.publish(self.poses)
        self.pub_markers.publish(self.markers)
        self.get_logger().info("Published intersections")

    def get_utm_to_local(self) -> None:
        """
        While rclpy is not shutdown, try to get the UTM to local transform every second or until successful.
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
                self.get_logger().info(f"Got UTM to local transform: {self.utm_to_local}")
                break
            except Exception as e:
                self.get_logger().warn(f"Failed to get UTM to local transform: {e}")
                rclpy.spin_once(self, timeout_sec=1.0)

    def get_intersection_msgs(self) -> Tuple[PoseArray, MarkerArray]:
        """
        Create PoseArray and MarkerArray from intersections.
        """
        ways = self.map_data.get_ways()
        crossroads = ways.get("crossroads", [])

        points_to_transform = {}
        for way in crossroads:
            # way.line is a buffered Point (Polygon)
            centroid = way.line.centroid
            points_to_transform[way.id] = np.array([centroid.x, centroid.y, 0.0]).reshape(3, 1)

        transformed_points = transform_points(points_to_transform, self.utm_to_local, 0.0)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.local_frame

        marker_array = MarkerArray()

        marker_id = 0
        for pid, point in transformed_points.items():
            p = point.ravel()

            # Spatial filtering based on local frame coordinates
            if not (
                self.grid_min[0] <= p[0] <= self.grid_max[0]
                and self.grid_min[1] <= p[1] <= self.grid_max[1]
            ):
                continue

            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = 0.0
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header.frame_id = self.local_frame
            marker.ns = "intersections"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(p[0])
            marker.pose.position.y = float(p[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 4.0
            marker.scale.y = 4.0
            marker.scale.z = 4.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        return pose_array, marker_array


def transform_points(
    points: Dict[int, np.ndarray], transform: np.ndarray, z: Optional[float] = None
) -> Dict[int, np.ndarray]:
    """
    Transform points using a transformation matrix.
    """

    def transform_point(point: np.ndarray, transform_mat: np.ndarray) -> np.ndarray:
        res = np.dot(transform_mat[:3, :3], point) + transform_mat[:3, 3:]
        return res

    transformed = {}
    for pid, point in points.items():
        transformed[pid] = transform_point(point, transform)
        if z is not None:
            transformed[pid][2] = z
    return transformed


def main() -> None:
    rclpy.init()
    node = OSMIntersections()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
