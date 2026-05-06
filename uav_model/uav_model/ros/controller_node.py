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
"""ROS 2 node that runs the Mellinger trajectory-tracking controller."""

import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque
from uav_model.config_loader.sdf_adapter import SDFAdapter
from uav_model.model.mellinger import MellingerController, MellingerGains
from uav_model.model.uav_model import UAVFlatState, UAVState

_DEFAULT_SDF = os.path.join(
    get_package_share_directory('ros_gz_crazyflie_bringup'),
    'gazebo_files',
    'gazebo',
    'crazyflie',
    'model.sdf',
)


class ControllerNode(Node):
    """ROS 2 node wrapping MellingerController for fixed-point hover."""

    def __init__(self):
        """Declare parameters, load UAVParams, create controller and ROS interfaces."""
        super().__init__('mellinger_controller')

        self.declare_parameter('odom_topic', '/uav_model/odom')
        self.declare_parameter('control_topic', '/cf_control/control_command')
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('hover_x', 0.0)
        self.declare_parameter('hover_y', 0.0)
        self.declare_parameter('hover_z', 1.0)
        self.declare_parameter('hover_yaw', 0.0)
        self.declare_parameter('kx', [6.0, 6.0, 6.0])
        self.declare_parameter('kv', [4.0, 4.0, 4.0])
        self.declare_parameter('kR', [8.0e-3, 8.0e-3, 2.0e-3])
        self.declare_parameter('kOmega', [1.5e-3, 1.5e-3, 5.0e-4])

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        control_topic = self.get_parameter('control_topic').get_parameter_value().string_value
        control_rate = self.get_parameter('control_rate_hz').get_parameter_value().double_value

        hover_x = self.get_parameter('hover_x').get_parameter_value().double_value
        hover_y = self.get_parameter('hover_y').get_parameter_value().double_value
        hover_z = self.get_parameter('hover_z').get_parameter_value().double_value
        hover_yaw = self.get_parameter('hover_yaw').get_parameter_value().double_value

        kx = np.array(self.get_parameter('kx').get_parameter_value().double_array_value)
        kv = np.array(self.get_parameter('kv').get_parameter_value().double_array_value)
        kR = np.array(self.get_parameter('kR').get_parameter_value().double_array_value)
        kOmega = np.array(self.get_parameter('kOmega').get_parameter_value().double_array_value)

        params = SDFAdapter(_DEFAULT_SDF).extract()
        gains = MellingerGains(kx=kx, kv=kv, kR=kR, kOmega=kOmega)
        self._controller = MellingerController(params, gains)

        # Hover setpoint: all feedforward terms zero; only position and yaw set.
        # Index layout of UAVFlatState: [pos(3), vel(3), acc(3), jerk(3), snap(3), yaw, yaw_vel, yaw_acc]
        hover_data = np.zeros(18)
        hover_data[0:3] = [hover_x, hover_y, hover_z]
        hover_data[15] = hover_yaw
        self._hover = UAVFlatState(hover_data)

        self._state: UAVState | None = None

        self._sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self._pub = self.create_publisher(ThrustAndTorque, control_topic, 10)
        self.create_timer(1.0 / control_rate, self._control_cb)

        self.get_logger().info(
            f'Mellinger controller started: hover=({hover_x}, {hover_y}, {hover_z}), '
            f'yaw={hover_yaw}, rate={control_rate} Hz'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        """Cache latest odometry as UAVState."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        self._state = UAVState(
            np.array(
                [
                    p.x,
                    p.y,
                    p.z,
                    lv.x,
                    lv.y,
                    lv.z,
                    q.w,
                    q.x,
                    q.y,
                    q.z,
                    av.x,
                    av.y,
                    av.z,
                ]
            )
        )

    def _control_cb(self) -> None:
        """Compute and publish control command at fixed rate."""
        if self._state is None:
            return

        u = self._controller.compute(self._hover, self._state)

        cmd = ThrustAndTorque()
        cmd.collective_thrust = float(u[0])
        cmd.torque = Vector3(x=float(u[1]), y=float(u[2]), z=float(u[3]))
        self._pub.publish(cmd)


def main(args=None):
    """Entry point for the Mellinger controller node."""
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
