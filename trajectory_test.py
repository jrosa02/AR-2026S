#!/usr/bin/env python3
"""Test trajectory publisher: flies the CrazyFlie on a horizontal circle.

Runs a full Mellinger controller internally — subscribe to odometry, publish
ThrustAndTorque commands. Use instead of controller_node when you want the
drone to follow a trajectory rather than hover at a fixed point.

Usage (after sourcing install/setup.bash):
    python3 src/trajectory_test.py

Parameters (edit constants below):
    RADIUS   - circle radius in metres
    OMEGA    - angular velocity in rad/s  (period = 2π/OMEGA)
    HEIGHT   - constant flight height in metres
    YAW      - fixed heading in radians (0 = facing +x)
    ODOM_TOPIC - odometry source (/uav_model/odom or /crazyflie/odom)
"""

import os
import sys

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node

# Extend path so we can import uav_model regardless of install layout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'install', 'lib',
                                'python3.10', 'site-packages'))

from cf_control_msgs.msg import ThrustAndTorque  # noqa: E402
from uav_model.config_loader.sdf_adapter import SDFAdapter  # noqa: E402
from uav_model.model.mellinger import MellingerController, MellingerGains  # noqa: E402
from uav_model.model.uav_model import UAVFlatState, UAVState  # noqa: E402

# ---------------------------------------------------------------------------
# Trajectory parameters — edit here
# ---------------------------------------------------------------------------
RADIUS = 0.5        # metres
OMEGA = 0.5         # rad/s  →  period ≈ 12.6 s
HEIGHT = 1.0        # metres above takeoff origin
YAW = 0.0           # fixed heading (rad)
CONTROL_RATE = 100  # Hz

ODOM_TOPIC = '/uav_model/odom'   # change to /crazyflie/odom for Gazebo sim
CONTROL_TOPIC = '/cf_control/control_command'

_DEFAULT_SDF = os.path.join(
    get_package_share_directory('ros_gz_crazyflie_bringup'),
    'gazebo_files', 'gazebo', 'crazyflie', 'model.sdf',
)

# ---------------------------------------------------------------------------
# Flat-output trajectory: circle in XY at constant height
#
# x(t)  =  R·cos(ω·t)          y(t)  =  R·sin(ω·t)
# ẋ     = -R·ω·sin(ω·t)        ẏ     =  R·ω·cos(ω·t)
# ẍ     = -R·ω²·cos(ω·t)       ÿ     = -R·ω²·sin(ω·t)
# x⃛     =  R·ω³·sin(ω·t)       y⃛     = -R·ω³·cos(ω·t)
# x⁽⁴⁾  =  R·ω⁴·cos(ω·t)       y⁽⁴⁾  =  R·ω⁴·sin(ω·t)
# ---------------------------------------------------------------------------

def circle_flat_state(t: float, radius: float, omega: float,
                      height: float, yaw: float) -> UAVFlatState:
    """Return UAVFlatState for a circular trajectory at time t."""
    c, s = np.cos(omega * t), np.sin(omega * t)
    data = np.zeros(18)

    # position [0:3]
    data[0] = radius * c
    data[1] = radius * s
    data[2] = height

    # velocity [3:6]
    data[3] = -radius * omega * s
    data[4] =  radius * omega * c
    data[5] = 0.0

    # acceleration [6:9]
    data[6] = -radius * omega**2 * c
    data[7] = -radius * omega**2 * s
    data[8] = 0.0

    # jerk [9:12]
    data[9]  =  radius * omega**3 * s
    data[10] = -radius * omega**3 * c
    data[11] = 0.0

    # snap [12:15]
    data[12] =  radius * omega**4 * c
    data[13] =  radius * omega**4 * s
    data[14] = 0.0

    # yaw, yaw_dot, yaw_ddot [15:18]
    data[15] = yaw
    data[16] = 0.0
    data[17] = 0.0

    return UAVFlatState(data)


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_test')

        params = SDFAdapter(_DEFAULT_SDF).extract()
        gains = MellingerGains()  # default gains from controller_node
        self._ctrl = MellingerController(params, gains)

        self._state: UAVState | None = None
        self._t0: float | None = None  # wall-clock time at first odom message

        self._sub = self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 10)
        self._pub = self.create_publisher(ThrustAndTorque, CONTROL_TOPIC, 10)
        self.create_timer(1.0 / CONTROL_RATE, self._control_cb)

        self.get_logger().info(
            f'Trajectory node started — circle R={RADIUS} m, ω={OMEGA} rad/s, '
            f'h={HEIGHT} m, listening on {ODOM_TOPIC}'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        self._state = UAVState(np.array([
            p.x, p.y, p.z,
            lv.x, lv.y, lv.z,
            q.w, q.x, q.y, q.z,
            av.x, av.y, av.z,
        ]))

        if self._t0 is None:
            self._t0 = self.get_clock().now().nanoseconds * 1e-9

    def _control_cb(self) -> None:
        if self._state is None or self._t0 is None:
            return

        t = self.get_clock().now().nanoseconds * 1e-9 - self._t0
        desired = circle_flat_state(t, RADIUS, OMEGA, HEIGHT, YAW)

        u = self._ctrl.compute(desired, self._state)

        cmd = ThrustAndTorque()
        cmd.collective_thrust = float(u[0])
        cmd.torque = Vector3(x=float(u[1]), y=float(u[2]), z=float(u[3]))
        self._pub.publish(cmd)


def main():
    rclpy.init()
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
