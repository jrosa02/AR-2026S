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

        odom_topic = self.get_parameter('odom_topic').value
        self._save_dir = self.get_parameter('save_dir').value

        self._lock = threading.Lock()
        self._t: list = []
        self._x: list = []
        self._y: list = []
        self._z: list = []
        self._t0: float | None = None

        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.get_logger().info(
            f'Recording {odom_topic} — plot will be saved to {self._save_dir} on exit'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        """Append position sample."""
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        with self._lock:
            self._t.append(t)
            self._x.append(msg.pose.pose.position.x)
            self._y.append(msg.pose.pose.position.y)
            self._z.append(msg.pose.pose.position.z)

    def get_data(self):
        """Return (t, x, y, z) numpy arrays of the full recorded trajectory."""
        with self._lock:
            return (
                np.array(self._t),
                np.array(self._x),
                np.array(self._y),
                np.array(self._z),
            )

    def save_plot(self) -> str | None:
        """Render and save the trajectory figure; return the path or None if no data."""
        t, x, y, z = self.get_data()
        if len(t) < 2:
            self.get_logger().warning('No data collected — nothing to save.')
            return None

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('UAV Trajectory', fontsize=13)

        gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
        ax_x = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
        ax_z = fig.add_subplot(gs[2, 0], sharex=ax_x)
        ax_xy = fig.add_subplot(gs[:, 1])

        n = np.arange(len(x))
        for ax, data, label, color in zip(
            [ax_x, ax_y, ax_z],
            [x, y, z],
            ['x [m]', 'y [m]', 'z [m]'],
            ['tab:blue', 'tab:orange', 'tab:green'],
        ):
            ax.plot(t, data, color=color, linewidth=1.2)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        ax_x.set_title('Position vs Sample')
        ax_z.set_xlabel('sample')

        # x-y top-down path
        ax_xy.plot(x, y, color='tab:purple', linewidth=1.2)
        ax_xy.plot(x[0], y[0], 'go', markersize=7, label='start')
        ax_xy.plot(x[-1], y[-1], 'rs', markersize=7, label='end')
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
        t, x, y, z = self.get_data()
        if len(t) < 2:
            return None
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self._save_dir, f'uav_trajectory_{ts}.csv')
        np.savetxt(
            path,
            np.column_stack([t, x, y, z]),
            delimiter=',',
            header='t_s,x_m,y_m,z_m',
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
