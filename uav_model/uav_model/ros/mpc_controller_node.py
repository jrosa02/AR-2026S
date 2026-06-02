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
"""ROS 2 node: MPC outer loop + Mellinger inner loop with FollowWaypoints action.

Two-layer architecture:
    MPC timer (20 Hz):  translational QP → UAVFlatState setpoint
    Ctrl timer (100 Hz): Mellinger tracks setpoint → ThrustAndTorque
"""

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
from uav_model.model.mpc_controller import MPCController, MPCParams
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


def _waypoint_to_goal(wp) -> tuple:
    """Extract (position, yaw) from a Waypoint message."""
    return np.array([wp.x, wp.y, wp.z]), float(wp.yaw)


class MPCControllerNode(Node):
    """MPC outer loop + Mellinger inner loop with a FollowWaypoints action server."""

    def __init__(self):
        """Declare parameters, build MPC and Mellinger controllers, start timers."""
        super().__init__('mpc_controller')

        # --- Parameters ---
        self.declare_parameter('odom_topic', '/uav_model/odom')
        self.declare_parameter('control_topic', '/cf_control/control_command')
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('mpc_rate_hz', 20.0)

        self.declare_parameter('mpc_horizon', 10)
        self.declare_parameter('mpc_dt', 0.05)
        self.declare_parameter('mpc_q_pos', [10.0, 10.0, 10.0])
        self.declare_parameter('mpc_q_vel', [1.0, 1.0, 1.0])
        self.declare_parameter('mpc_r_acc', [0.1, 0.1, 0.1])
        self.declare_parameter('mpc_q_terminal', [50.0, 50.0, 50.0])
        self.declare_parameter('mpc_v_max', 3.0)
        self.declare_parameter('mpc_a_max', 5.0)

        self.declare_parameter('hover_x', 0.0)
        self.declare_parameter('hover_y', 0.0)
        self.declare_parameter('hover_z', 1.0)
        self.declare_parameter('hover_yaw', 0.0)

        self.declare_parameter('kx', [6.0, 6.0, 6.0])
        self.declare_parameter('kv', [4.0, 4.0, 4.0])
        self.declare_parameter('kR', [8.0e-3, 8.0e-3, 2.0e-3])
        self.declare_parameter('kOmega', [1.1e-3, 1.1e-3, 5.0e-4])

        def _get_f(name):
            return self.get_parameter(name).get_parameter_value().double_value

        def _get_fa(name):
            return np.array(self.get_parameter(name).get_parameter_value().double_array_value)

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        control_topic = self.get_parameter('control_topic').get_parameter_value().string_value
        control_rate = _get_f('control_rate_hz')
        mpc_rate = _get_f('mpc_rate_hz')

        hover_pos = np.array([_get_f('hover_x'), _get_f('hover_y'), _get_f('hover_z')])
        hover_yaw = _get_f('hover_yaw')

        # --- Controllers ---
        cf_params = SDFAdapter(_DEFAULT_SDF).extract()

        mpc_params = MPCParams(
            horizon=self.get_parameter('mpc_horizon').get_parameter_value().integer_value,
            dt=_get_f('mpc_dt'),
            Q_pos=_get_fa('mpc_q_pos'),
            Q_vel=_get_fa('mpc_q_vel'),
            R_acc=_get_fa('mpc_r_acc'),
            Q_terminal=_get_fa('mpc_q_terminal'),
            v_max=_get_f('mpc_v_max'),
            a_max=_get_f('mpc_a_max'),
        )
        self._mpc = MPCController(mpc_params)

        gains = MellingerGains(
            kx=_get_fa('kx'),
            kv=_get_fa('kv'),
            kR=_get_fa('kR'),
            kOmega=_get_fa('kOmega'),
        )
        self._mellinger = MellingerController(cf_params, gains)

        # --- Shared state (guarded by lock) ---
        self._lock = threading.Lock()
        self._uav_state: UAVState | None = None
        self._goal = hover_pos.copy()
        self._goal_yaw = hover_yaw

        # Flat setpoint initialised to hover position
        init_flat = np.zeros(18)
        init_flat[0:3] = hover_pos
        init_flat[15] = hover_yaw
        self._flat_setpoint = UAVFlatState(init_flat)

        self._mode = _Mode.IDLE
        self._waypoints: list = []
        self._wp_idx: int = 0
        self._active_goal_handle = None
        self._done_event: threading.Event | None = None

        # --- Callback groups ---
        # MPC timer in its own group to prevent concurrent solves.
        mpc_cbg = MutuallyExclusiveCallbackGroup()
        # Control timer, odom callback, and action execute share a reentrant group.
        control_cbg = ReentrantCallbackGroup()
        action_cbg = MutuallyExclusiveCallbackGroup()

        # --- ROS interfaces ---
        self._sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_cb,
            10,
            callback_group=control_cbg,
        )
        self._pub = self.create_publisher(ThrustAndTorque, control_topic, 10)

        self.create_timer(1.0 / mpc_rate, self._mpc_cb, callback_group=mpc_cbg)
        self.create_timer(1.0 / control_rate, self._control_cb, callback_group=control_cbg)

        self._action_server = ActionServer(
            self,
            FollowWaypoints,
            'follow_waypoints',
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            execute_callback=self._execute_cb,
            callback_group=action_cbg,
        )

        self.get_logger().info(
            f'MPCControllerNode started: MPC={mpc_rate:.0f} Hz, '
            f'ctrl={control_rate:.0f} Hz, horizon={mpc_params.horizon}'
        )

    # ------------------------------------------------------------------
    # Odometry callback
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry) -> None:
        """Cache latest odometry as UAVState with world-frame velocity."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular

        q_np = np.array([q.w, q.x, q.y, q.z])
        lv_world = MellingerController._quat_to_rot(q_np) @ np.array([lv.x, lv.y, lv.z])

        with self._lock:
            self._uav_state = UAVState(
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
    # MPC timer (20 Hz)
    # ------------------------------------------------------------------

    def _mpc_cb(self) -> None:
        """Solve MPC QP and update flat setpoint. Lock released during solve."""
        with self._lock:
            if self._uav_state is None:
                return
            state = self._uav_state
            goal = self._goal.copy()
            yaw = self._goal_yaw

        # Solve outside lock — _mpc_cb is MutuallyExclusive so no concurrent call
        flat = self._mpc.compute(state, goal, yaw)

        with self._lock:
            self._flat_setpoint = flat

    # ------------------------------------------------------------------
    # Control timer (100 Hz)
    # ------------------------------------------------------------------

    def _control_cb(self) -> None:
        """Run Mellinger on latest flat setpoint and publish ThrustAndTorque."""
        with self._lock:
            if self._uav_state is None:
                return
            state = self._uav_state
            self._advance_waypoints(state)
            flat = self._flat_setpoint
            goal_handle = self._active_goal_handle
            wp_idx = self._wp_idx
            n_wps = len(self._waypoints)

        u = self._mellinger.compute(flat, state)

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
                nxt = self._waypoints[self._wp_idx]
                self._goal, self._goal_yaw = _waypoint_to_goal(nxt)

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _goal_cb(self, goal_request) -> GoalResponse:
        if not goal_request.waypoints:
            self.get_logger().warning('Rejecting empty waypoint list')
            return GoalResponse.REJECT
        self.get_logger().info(f'Accepting goal with {len(goal_request.waypoints)} waypoints')
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
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
            self._goal, self._goal_yaw = _waypoint_to_goal(waypoints[0])

        result = FollowWaypoints.Result()

        while not done_event.wait(timeout=0.05):
            if goal_handle.is_cancel_requested:
                with self._lock:
                    self._mode = _Mode.IDLE
                    if self._uav_state is not None:
                        self._goal = self._uav_state.position.copy()
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
    """Entry point for the MPC controller node."""
    rclpy.init(args=args)
    node = MPCControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
