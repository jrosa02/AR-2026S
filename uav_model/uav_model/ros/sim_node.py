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
"""ROS 2 node that runs the UAV rigid-body dynamics model."""

import numpy as np
import rclpy
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

from uav_model.config_loader.sdf_adapter import SDFAdapter
from uav_model.model.uav_model import UAVModel


class SimNode(Node):
    """ROS 2 simulation node for UAV rigid-body dynamics."""

    def __init__(self):
        """Declare parameters, load SDF, create model and ROS interfaces."""
        super().__init__('uav_model_sim')

        self.declare_parameter('sdf_path', '')
        self.declare_parameter('sim_rate_hz', 500.0)
        self.declare_parameter(
            'control_topic', '/cf_control/control_command'
        )
        self.declare_parameter('state_topic', '/uav_model/odom')
        self.declare_parameter(
            'initial_state',
            [0.0] * 6 + [1.0] + [0.0] * 6,
        )

        sdf_path = (
            self.get_parameter('sdf_path').get_parameter_value().string_value
        )
        if not sdf_path:
            self.get_logger().fatal('sdf_path parameter is required')
            raise RuntimeError('sdf_path parameter is required')

        sim_rate = (
            self.get_parameter('sim_rate_hz')
            .get_parameter_value().double_value
        )
        control_topic = (
            self.get_parameter('control_topic')
            .get_parameter_value().string_value
        )
        state_topic = (
            self.get_parameter('state_topic')
            .get_parameter_value().string_value
        )
        initial_state = (
            self.get_parameter('initial_state')
            .get_parameter_value().double_array_value
        )

        adapter = SDFAdapter(sdf_path)
        params = adapter.extract()
        self._model = UAVModel(params)

        if len(initial_state) == 13:
            self._model.reset(np.array(initial_state, dtype=np.float64))

        self._dt = 1.0 / sim_rate
        self._u = np.zeros(4, dtype=np.float64)

        self._sub = self.create_subscription(
            ThrustAndTorque, control_topic,
            self._control_cb, 10,
        )
        self._pub = self.create_publisher(Odometry, state_topic, 10)
        self._timer = self.create_timer(self._dt, self._step_cb)

        self.get_logger().info(
            f'UAV model node started: rate={sim_rate} Hz, '
            f'mass={params.mass} kg, rotors={params.num_rotors}'
        )

    def _control_cb(self, msg):
        """Store latest control command."""
        self._u[0] = msg.collective_thrust
        self._u[1] = msg.torque.x
        self._u[2] = msg.torque.y
        self._u[3] = msg.torque.z

    def _step_cb(self):
        """Run one model step and publish state as Odometry."""
        self._model.step(self._u, self._dt)
        state = self._model.state

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'uav_model'

        odom.pose.pose.position = Point(
            x=state[0], y=state[1], z=state[2],
        )
        odom.pose.pose.orientation = Quaternion(
            w=state[6], x=state[7], y=state[8], z=state[9],
        )
        odom.twist.twist = Twist(
            linear=Vector3(x=state[3], y=state[4], z=state[5]),
            angular=Vector3(x=state[10], y=state[11], z=state[12]),
        )

        self._pub.publish(odom)


def main(args=None):
    """Entry point for the UAV model simulation node."""
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
