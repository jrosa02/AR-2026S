"""ROS2 node that publishes a growing nav_msgs/Path trail and waypoint markers for RViz."""

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray


_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class PathVisualizerNode(Node):
    """Subscribes to odometry and publishes a Path trail + waypoint MarkerArray for RViz."""

    def __init__(self):
        """Initialize publishers, subscriber, and optional waypoint markers."""
        super().__init__('path_visualizer')

        self.declare_parameter('odom_topic', '/crazyflie/odom')
        self.declare_parameter('path_file', '')

        odom_topic = self.get_parameter('odom_topic').value
        path_file = self.get_parameter('path_file').value

        self._path = Path()
        self._path_pub = self.create_publisher(Path, '/crazyflie/path', _LATCHED_QOS)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/crazyflie/waypoints_viz', _LATCHED_QOS
        )

        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        if path_file:
            self._publish_waypoint_markers(path_file)

        self.get_logger().info(
            f'Path visualizer started | odom={odom_topic}  path_file={path_file or "(none)"}'
        )

    def _odom_cb(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        # Use 'world' — Gazebo ground-truth odometry positions are in the world frame,
        # but the message header says 'crazyflie/odom' which has no TF entry.
        self._path.header.frame_id = 'world'
        self._path.header.stamp = msg.header.stamp
        self._path.poses.append(pose)
        self._path_pub.publish(self._path)

    def _publish_waypoint_markers(self, path_file: str):
        try:
            with open(path_file) as f:
                data = yaml.safe_load(f)
        except OSError as e:
            self.get_logger().warn(f'Could not load waypoints from {path_file}: {e}')
            return

        waypoints = data.get('waypoints', [])
        markers = MarkerArray()

        for idx, wp in enumerate(waypoints):
            x = float(wp['x'])
            y = float(wp['y'])
            z = float(wp['z'])
            tolerance = float(wp.get('tolerance', 0.3))

            # Tolerance sphere (semi-transparent)
            sphere = Marker()
            sphere.header.frame_id = 'odom'
            sphere.ns = 'waypoints'
            sphere.id = idx * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = tolerance * 2.0
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 0.2

            # Solid center dot
            dot = Marker()
            dot.header.frame_id = 'odom'
            dot.ns = 'waypoints'
            dot.id = idx * 2 + 1
            dot.type = Marker.SPHERE
            dot.action = Marker.ADD
            dot.pose.position.x = x
            dot.pose.position.y = y
            dot.pose.position.z = z
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = dot.scale.z = 0.1
            dot.color.r = 1.0
            dot.color.g = 0.2
            dot.color.b = 0.0
            dot.color.a = 1.0

            markers.markers.append(sphere)
            markers.markers.append(dot)

        self._marker_pub.publish(markers)
        self.get_logger().info(f'Published {len(waypoints)} waypoint markers')


def main(args=None):
    """Spin the PathVisualizerNode."""
    rclpy.init(args=args)
    node = PathVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
