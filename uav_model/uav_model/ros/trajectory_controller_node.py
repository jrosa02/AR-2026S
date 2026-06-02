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
"""ROS 2 node: Mellinger controller with FollowWaypoints action interface."""

import enum
import os
import threading

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cf_control_msgs.action import FollowWaypoints
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


class _Mode(enum.Enum):
    IDLE = 'idle'
    FOLLOWING = 'following'
    COMPLETE = 'complete'


def _waypoint_to_flat_state(wp) -> UAVFlatState:
    data = np.zeros(18)
    data[0:3] = [wp.x, wp.y, wp.z]
    data[15] = wp.yaw
    return UAVFlatState(data)


def _hover_at(position: np.ndarray, yaw: float = 0.0) -> UAVFlatState:
    data = np.zeros(18)
    data[0:3] = position
    data[15] = yaw
    return UAVFlatState(data)


class TrajectoryControllerNode(Node):
    """Mellinger controller with a FollowWaypoints action server."""

    def __init__(self):
        """Declare parameters, load UAVParams, create controller and ROS interfaces."""
        super().__init__('trajectory_controller')

        self.declare_parameter('odom_topic', '/uav_model/odom')
        self.declare_parameter('control_topic', '/cf_control/control_command')
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('kx', [6.0, 6.0, 6.0])
        self.declare_parameter('kv', [4.0, 4.0, 4.0])
        self.declare_parameter('kR', [8.0e-3, 8.0e-3, 2.0e-3])
        self.declare_parameter('kOmega', [1.1e-3, 1.1e-3, 5.0e-4])

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        control_topic = self.get_parameter('control_topic').get_parameter_value().string_value
        control_rate = self.get_parameter('control_rate_hz').get_parameter_value().double_value

        kx = np.array(self.get_parameter('kx').get_parameter_value().double_array_value)
        kv = np.array(self.get_parameter('kv').get_parameter_value().double_array_value)
        kR = np.array(self.get_parameter('kR').get_parameter_value().double_array_value)
        kOmega = np.array(self.get_parameter('kOmega').get_parameter_value().double_array_value)

        params = SDFAdapter(_DEFAULT_SDF).extract()
        gains = MellingerGains(kx=kx, kv=kv, kR=kR, kOmega=kOmega)
        self._controller = MellingerController(params, gains)
        self._control_rate = control_rate

        self._setpoint = _hover_at(np.array([0.0, 0.0, 1.0]))
        self._state: UAVState | None = None
        self._lock = threading.Lock()

        self._mode = _Mode.IDLE
        self._waypoints: list = []
        self._wp_idx: int = 0
        self._active_goal_handle = None
        # Set by _execute_cb; signalled from _advance_waypoints when path completes.
        self._done_event: threading.Event | None = None

        # Timer and odom run in one group; action execute runs in its own thread.
        control_cbg = ReentrantCallbackGroup()
        action_cbg = MutuallyExclusiveCallbackGroup()

        self._sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_cb,
            10,
            callback_group=control_cbg,
        )
        self._pub = self.create_publisher(ThrustAndTorque, control_topic, 10)
        self.create_timer(
            1.0 / control_rate,
            self._control_cb,
            callback_group=control_cbg,
        )

        self._action_server = ActionServer(
            self,
            FollowWaypoints,
            'follow_waypoints',
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            execute_callback=self._execute_cb,
            callback_group=action_cbg,
        )

        self.get_logger().info(f'TrajectoryControllerNode started: rate={control_rate} Hz')

    # ------------------------------------------------------------------
    # Odometry callback
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry) -> None:
        """Cache latest odometry as UAVState (world-frame velocity)."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        q_np = np.array([q.w, q.x, q.y, q.z])
        lv_world = MellingerController._quat_to_rot(q_np) @ np.array([lv.x, lv.y, lv.z])

        with self._lock:
            self._state = UAVState(
                np.array(
                    [
                        p.x,
                        p.y,
                        p.z,
                        lv_world[0],
                        lv_world[1],
                        lv_world[2],
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

    # ------------------------------------------------------------------
    # Control timer
    # ------------------------------------------------------------------

    def _control_cb(self) -> None:
        """Advance waypoint state machine, publish control command and feedback."""
        with self._lock:
            if self._state is None:
                return
            state = self._state
            self._advance_waypoints(state)
            setpoint = self._setpoint
            goal_handle = self._active_goal_handle
            wp_idx = self._wp_idx
            n_wps = len(self._waypoints)

        u = self._controller.compute(setpoint, state)

        cmd = ThrustAndTorque()
        cmd.collective_thrust = float(u[0])
        cmd.torque = Vector3(x=float(u[1]), y=float(u[2]), z=float(u[3]))
        self._pub.publish(cmd)

        if goal_handle is not None and n_wps > 0:
            wp = self._waypoints[min(wp_idx, n_wps - 1)]
            dist = float(np.linalg.norm(state.position - np.array([wp.x, wp.y, wp.z])))
            fb = FollowWaypoints.Feedback()
            fb.current_waypoint_index = wp_idx
            fb.total_waypoints = n_wps
            fb.distance_to_current = dist
            goal_handle.publish_feedback(fb)

    def _advance_waypoints(self, state: UAVState) -> None:
        """Advance to next waypoint when within tolerance. Called under lock."""
        if self._mode != _Mode.FOLLOWING:
            return

        wp = self._waypoints[self._wp_idx]
        dist = float(np.linalg.norm(state.position - np.array([wp.x, wp.y, wp.z])))
        if dist < wp.tolerance:
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                self._mode = _Mode.COMPLETE
                if self._done_event is not None:
                    self._done_event.set()
            else:
                self._setpoint = _waypoint_to_flat_state(self._waypoints[self._wp_idx])

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _goal_cb(self, goal_request) -> GoalResponse:
        """Accept all goals."""
        if not goal_request.waypoints:
            self.get_logger().warning('Rejecting empty waypoint list')
            return GoalResponse.REJECT
        self.get_logger().info(f'Accepting goal with {len(goal_request.waypoints)} waypoints')
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        """Accept all cancel requests."""
        self.get_logger().info('Cancel requested — will hover in place')
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle) -> FollowWaypoints.Result:
        """Set up waypoint list and block until complete or cancelled."""
        waypoints = list(goal_handle.request.waypoints)
        done_event = threading.Event()

        with self._lock:
            if self._active_goal_handle is not None:
                try:
                    self._active_goal_handle.abort()
                except RuntimeError:
                    pass
            self._active_goal_handle = goal_handle
            self._done_event = done_event
            self._waypoints = waypoints
            self._wp_idx = 0
            self._mode = _Mode.FOLLOWING
            self._setpoint = _waypoint_to_flat_state(waypoints[0])

        result = FollowWaypoints.Result()

        # Block here — _control_cb advances the state machine and sets done_event.
        while not done_event.wait(timeout=0.05):
            if goal_handle.is_cancel_requested:
                with self._lock:
                    self._mode = _Mode.IDLE
                    if self._state is not None:
                        self._setpoint = _hover_at(self._state.position.copy())
                    reached = self._wp_idx
                    self._active_goal_handle = None
                    self._done_event = None
                goal_handle.canceled()
                result.waypoints_reached = reached
                result.message = 'Cancelled — hovering in place'
                return result
            if not rclpy.ok():
                break

        with self._lock:
            self._mode = _Mode.IDLE
            self._active_goal_handle = None
            self._done_event = None

        goal_handle.succeed()
        result.waypoints_reached = len(waypoints)
        result.message = 'All waypoints reached'
        self.get_logger().info('Path complete — hovering in place')
        return result


def main(args=None):
    """Entry point for the trajectory controller node."""
    rclpy.init(args=args)
    node = TrajectoryControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
