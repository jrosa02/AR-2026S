# Copyright 2026 developer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROS 2 node: reads a YAML waypoint file and sends a FollowWaypoints action goal."""

import yaml
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cf_control_msgs.action import FollowWaypoints
from cf_control_msgs.msg import Waypoint


def _parse_waypoints(data: dict) -> list[Waypoint]:
    """Parse list of waypoint dicts from YAML into Waypoint messages."""
    raw = data.get('waypoints')
    if not raw:
        raise ValueError("YAML must contain a non-empty 'waypoints' list")

    waypoints = []
    for i, entry in enumerate(raw):
        wp = Waypoint()
        for field in ('x', 'y', 'z'):
            if field not in entry:
                raise ValueError(f"Waypoint {i}: missing required field '{field}'")
            setattr(wp, field, float(entry[field]))
        setattr(wp, 'yaw', float(entry.get('yaw', 0.0)))
        if 'tolerance' not in entry:
            raise ValueError(f"Waypoint {i}: missing required field 'tolerance'")
        wp.tolerance = float(entry['tolerance'])
        wp.vx = float(entry.get('vx', 0.0))
        wp.vy = float(entry.get('vy', 0.0))
        wp.vz = float(entry.get('vz', 0.0))
        waypoints.append(wp)
    return waypoints


class PathPublisherNode(Node):
    """Sends a FollowWaypoints goal loaded from a YAML file."""

    def __init__(self):
        """Declare parameters, load YAML, wait for action server, send goal."""
        super().__init__('path_publisher')

        self.declare_parameter('path_file', '')
        self.declare_parameter('action_server', 'follow_waypoints')

        path_file = self.get_parameter('path_file').get_parameter_value().string_value
        action_server = (
            self.get_parameter('action_server').get_parameter_value().string_value
        )

        if not path_file:
            self.get_logger().fatal('path_file parameter is required')
            raise RuntimeError('path_file parameter is required')

        with open(path_file, 'r') as f:
            data = yaml.safe_load(f)

        self._waypoints = _parse_waypoints(data)
        self.get_logger().info(
            f'Loaded {len(self._waypoints)} waypoints from {path_file}'
        )

        self._client = ActionClient(self, FollowWaypoints, action_server)
        self.get_logger().info(f'Waiting for action server "{action_server}"...')
        self._client.wait_for_server()
        self.get_logger().info('Action server ready — sending goal')

        goal = FollowWaypoints.Goal()
        goal.waypoints = self._waypoints
        send_future = self._client.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        """Handle goal acceptance; attach result callback."""
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejected')
            rclpy.shutdown()
            return
        self.get_logger().info('Goal accepted')
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg) -> None:
        """Log progress."""
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'Waypoint {fb.current_waypoint_index + 1}/{fb.total_waypoints} '
            f'— distance: {fb.distance_to_current:.3f} m'
        )

    def _result_cb(self, future) -> None:
        """Log result; keep node alive so the drone can be observed after path completion."""
        result = future.result().result
        self.get_logger().info(
            f'Done: {result.message} '
            f'({result.waypoints_reached}/{len(self._waypoints)} reached)'
        )


def main(args=None):
    """Entry point for the path publisher node."""
    rclpy.init(args=args)
    node = PathPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
