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
"""ROS2 node: silent trajectory recorder — saves a PNG plot on shutdown/Ctrl-C.

No GUI window is shown while running.  On exit a 4-panel figure is saved:
  - x(t), y(t), z(t) time series
  - x-y top-down path (real 2-D trajectory)
"""

import datetime
import json
import os
import signal
import threading

import matplotlib
import numpy as np
import rclpy
import yaml
from cf_control_msgs.action import FollowWaypoints
from nav_msgs.msg import Odometry
from rclpy.node import Node

matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

_LOCK_FILE = '/tmp/crazyflie_coordinate_plotter.lock'


def _check_lock():
    """Raise RuntimeError if another instance is already running."""
    if os.path.exists(_LOCK_FILE):
        try:
            pid = int(open(_LOCK_FILE).read().strip())
            os.kill(pid, 0)  # signal 0 = check existence only
            raise RuntimeError(
                f'coordinate_plotter_node is already running (PID {pid}). '
                f'Kill it or remove {_LOCK_FILE} to start a new one.'
            )
        except ProcessLookupError:
            pass  # stale lock from a crashed previous run — overwrite it


def _write_lock():
    with open(_LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))


def _remove_lock():
    try:
        os.remove(_LOCK_FILE)
    except FileNotFoundError:
        pass


class CoordinatePlotterNode(Node):
    """Silent odometry recorder; generates a trajectory plot on shutdown."""

    def __init__(self):
        """Set up subscriber and data buffers."""
        super().__init__('coordinate_plotter')
        self.declare_parameter('odom_topic', '/crazyflie/odom')
        self.declare_parameter('save_dir', '/home/developer/ros2_ws/src/plots')
        self.declare_parameter('path_file', '')

        odom_topic = self.get_parameter('odom_topic').value
        self._save_dir = self.get_parameter('save_dir').value
        path_file = self.get_parameter('path_file').value

        self._waypoints: list[tuple[float, float, float]] = []
        if path_file:
            try:
                with open(path_file) as f:
                    data = yaml.safe_load(f)
                self._waypoints = [
                    (float(wp['x']), float(wp['y']), float(wp['z']))
                    for wp in data.get('waypoints', [])
                ]
                self.get_logger().info(f'Loaded {len(self._waypoints)} waypoints from {path_file}')
            except (OSError, KeyError, ValueError, TypeError) as e:
                self.get_logger().warning(f'Could not load path_file {path_file}: {e}')

        self._lock = threading.Lock()
        self._t: list = []
        self._x: list = []
        self._y: list = []
        self._z: list = []
        self._vx: list = []
        self._vy: list = []
        self._vz: list = []
        self._t0: float | None = None
        self._completed_wp_count: int = 0
        self._t_completion: float | None = None  # odom time when last waypoint was advanced

        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(
            FollowWaypoints.Impl.FeedbackMessage,
            '/follow_waypoints/_action/feedback',
            self._wp_feedback_cb,
            10,
        )
        self.get_logger().info(
            f'Recording {odom_topic} — plot will be saved to {self._save_dir} on exit'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        """Append position and velocity sample."""
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        with self._lock:
            self._t.append(t)
            self._x.append(msg.pose.pose.position.x)
            self._y.append(msg.pose.pose.position.y)
            self._z.append(msg.pose.pose.position.z)
            self._vx.append(msg.twist.twist.linear.x)
            self._vy.append(msg.twist.twist.linear.y)
            self._vz.append(msg.twist.twist.linear.z)

    def _wp_feedback_cb(self, msg) -> None:
        """Track how many waypoints have been completed and when the last advance happened."""
        with self._lock:
            new_count = msg.feedback.current_waypoint_index
            if new_count > self._completed_wp_count:
                self._completed_wp_count = new_count
                if self._t:
                    self._t_completion = self._t[-1]

    def get_data(self):
        """Return (t, x, y, z, speed) numpy arrays of the full recorded trajectory."""
        with self._lock:
            vx = np.array(self._vx)
            vy = np.array(self._vy)
            vz = np.array(self._vz)
            return (
                np.array(self._t),
                np.array(self._x),
                np.array(self._y),
                np.array(self._z),
                np.sqrt(vx ** 2 + vy ** 2 + vz ** 2),
            )

    def save_plot(self) -> str | None:
        """Render and save the trajectory figure; return the path or None if no data."""
        t, x, y, z, speed = self.get_data()
        if len(t) < 2:
            self.get_logger().warning('No data collected — nothing to save.')
            return None

        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('UAV Trajectory', fontsize=13)

        gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)
        ax_x = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
        ax_z = fig.add_subplot(gs[2, 0], sharex=ax_x)
        ax_spd = fig.add_subplot(gs[3, 0], sharex=ax_x)
        ax_xy = fig.add_subplot(gs[:, 1])

        with self._lock:
            done = self._waypoints[:self._completed_wp_count]
            t_completion = self._t_completion
            next_wp = (self._waypoints[self._completed_wp_count]
                       if self._completed_wp_count < len(self._waypoints) else None)
        wp_x = [wp[0] for wp in done]
        wp_y = [wp[1] for wp in done]
        wp_z = [wp[2] for wp in done]
        # Waypoints are mapped only up to completion time so the post-orbit
        # portion of the time-series is visually unobstructed.
        t_wp_end = t_completion if (t_completion is not None and t_completion <= t[-1]) else t[-1]
        t_wp = np.linspace(t[0], t_wp_end, len(done)) if done else np.array([])

        for ax, data, wp_vals, label, color in zip(
            [ax_x, ax_y, ax_z],
            [x, y, z],
            [wp_x, wp_y, wp_z],
            ['x [m]', 'y [m]', 'z [m]'],
            ['tab:blue', 'tab:orange', 'tab:green'],
        ):
            ax.plot(t, data, color=color, linewidth=1.2, label='actual')
            if len(t_wp):
                ax.plot(t_wp, wp_vals, ':', color='gray', linewidth=1.0,
                        alpha=0.7, label='waypoints')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        ax_spd.plot(t, speed, color='tab:red', linewidth=1.2)
        ax_spd.set_ylabel('|v| [m/s]')
        ax_spd.set_xlabel('t [s]')
        ax_spd.grid(True, alpha=0.3)

        if t_completion is not None and t[0] <= t_completion <= t[-1]:
            for ax in (ax_x, ax_y, ax_z, ax_spd):
                ax.axvline(t_completion, color='k', linestyle='--',
                           linewidth=0.9, alpha=0.5)
            ax_spd.text(t_completion, ax_spd.get_ylim()[1],
                        ' path done', fontsize=7, va='top', color='dimgray')

        ax_x.set_title('Position vs Time')
        ax_z.tick_params(labelbottom=False)

        # x-y top-down path
        ax_xy.plot(x, y, color='tab:purple', linewidth=1.2)
        ax_xy.plot(x[0], y[0], 'go', markersize=7, label='start')
        ax_xy.plot(x[-1], y[-1], 'rs', markersize=7, label='end')

        if done:
            ax_xy.plot(wp_x, wp_y, 'D--', color='gray', markersize=5,
                       linewidth=0.8, alpha=0.6, label='waypoints')
            for i, (wx, wy) in enumerate(zip(wp_x, wp_y)):
                ax_xy.annotate(str(i), (wx, wy), textcoords='offset points',
                               xytext=(4, 4), fontsize=6, color='dimgray')
        if next_wp is not None:
            ax_xy.plot(next_wp[0], next_wp[1], 'o', color='tab:orange',
                       markersize=8, zorder=5, label='next wp')

        ax_xy.set_xlabel('x [m]')
        ax_xy.set_ylabel('y [m]')
        ax_xy.set_title('Top-down Path (x-y)')
        ax_xy.set_aspect('equal', adjustable='datalim')
        ax_xy.grid(True, alpha=0.3)
        ax_xy.legend(fontsize=8)

        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self._save_dir, f'uav_trajectory_{ts}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def save_ros_graph(self) -> str | None:
        """Save all ROS topic publishers and subscribers to JSON."""
        topics = self.get_topic_names_and_types()
        graph = {}
        for topic, types in sorted(topics):
            pubs = [i.node_name for i in self.get_publishers_info_by_topic(topic)]
            subs = [i.node_name for i in self.get_subscriptions_info_by_topic(topic)]
            graph[topic] = {'types': types, 'publishers': pubs, 'subscribers': subs}

        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self._save_dir, f'ros_graph_{ts}.json')
        with open(path, 'w') as f:
            json.dump(graph, f, indent=2)
        return path

    def save_csv(self) -> str | None:
        """Write trajectory to CSV; return the path or None if no data."""
        t, x, y, z, speed = self.get_data()
        if len(t) < 2:
            return None
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self._save_dir, f'uav_trajectory_{ts}.csv')
        np.savetxt(
            path,
            np.column_stack([t, x, y, z, speed]),
            delimiter=',',
            header='t_s,x_m,y_m,z_m,speed_m_s',
            comments='',
            fmt='%.6f',
        )
        return path


def main(args=None):
    """Entry point: spin silently and save plot on SIGINT/SIGTERM."""
    _check_lock()
    _write_lock()

    rclpy.init(args=args)
    node = CoordinatePlotterNode()

    stop = threading.Event()

    # Signal handlers run on the main thread — they can't interrupt rclpy.spin()
    # directly (it blocks in C), so we spin in a daemon thread and wait on a
    # threading.Event on the main thread, which is always interruptible.
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    stop.wait()

    graph = node.save_ros_graph()
    if graph:
        node.get_logger().info(f'Graph saved → {graph}')
    plot = node.save_plot()
    if plot:
        node.get_logger().info(f'Plot saved → {plot}')
    csv = node.save_csv()
    if csv:
        node.get_logger().info(f'CSV  saved → {csv}')
    _remove_lock()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
