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
"""Differential flatness conversion: flat output → full UAV state + control."""

from dataclasses import dataclass

import numpy as np

from uav_model.model.uav_model import UAVFlatState, UAVState


@dataclass
class FlatOutputResult:
    """Result of the differential flatness inversion."""

    state: UAVState  # Full 13-element state vector
    thrust: float  # Collective thrust [N]
    torques: np.ndarray  # Body torques [N·m], shape (3,)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses Shepperd's method to select the numerically stable branch.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def flat_to_full_state(
    flat: UAVFlatState,
    mass: float,
    gravity: float,
    J: np.ndarray,
) -> FlatOutputResult:
    """Recover full UAV state and control inputs from flat output derivatives.

    Implements the differential flatness map for a quadrotor (Mellinger & Kumar 2011).
    The flat outputs are position r(t) and yaw ψ(t); their time derivatives up to
    snap (d⁴r/dt⁴) and yaw acceleration (d²ψ/dt²) uniquely determine the full state
    and control inputs.

    Args:
        flat:    18-element UAVFlatState
                 [pos(3), vel(3), acc(3), jerk(3), snap(3), yaw, yaw_vel, yaw_acc].
        mass:    Vehicle mass [kg].
        gravity: Gravitational acceleration magnitude [m/s²].
        J:       3x3 body-frame inertia tensor [kg·m²].

    Returns
    -------
        FlatOutputResult(state, thrust, torques) where state is a UAVState.
    """
    pos = flat.position
    vel = flat.velocity
    acc = flat.acceleration
    jerk = flat.jerk
    snap = flat.snap
    yaw = float(flat.yaw)
    yaw_vel = float(flat.yaw_vel)
    yaw_acc = float(flat.yaw_acc)

    g_vec = np.array([0.0, 0.0, gravity])
    t_vec = mass * (acc + g_vec)
    T = float(np.linalg.norm(t_vec))
    z_b = t_vec / T

    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b_num = np.cross(z_b, x_c)
    alpha = float(np.linalg.norm(y_b_num))
    y_b = y_b_num / alpha
    x_b = np.cross(y_b, z_b)

    R = np.column_stack((x_b, y_b, z_b))
    q = _rot_to_quat(R)

    T_dot = mass * float(np.dot(jerk, z_b))
    h_omega = (mass / T) * (jerk - np.dot(jerk, z_b) * z_b)

    omega_x = -float(np.dot(h_omega, y_b))
    omega_y = float(np.dot(h_omega, x_b))

    dx_c_dt = yaw_vel * np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    dy_b_num_dt = np.cross(h_omega, x_c) + np.cross(z_b, dx_c_dt)
    omega_z = -float(np.dot(x_b, dy_b_num_dt)) / alpha

    omega = np.array([omega_x, omega_y, omega_z])

    d2z_b = (mass / T) * (
        snap - (np.dot(snap, z_b) + np.dot(jerk, h_omega)) * z_b - np.dot(jerk, z_b) * h_omega
    ) - (T_dot / T) * h_omega

    dx_c2_dt2 = yaw_acc * np.array([-np.sin(yaw), np.cos(yaw), 0.0]) - yaw_vel**2 * np.array(
        [np.cos(yaw), np.sin(yaw), 0.0]
    )
    dy_b2_num_dt2 = (
        np.cross(d2z_b, x_c) + 2.0 * np.cross(h_omega, dx_c_dt) + np.cross(z_b, dx_c2_dt2)
    )

    alpha_dot = float(np.dot(y_b_num, dy_b_num_dt)) / alpha
    dy_b_dt = (dy_b_num_dt - y_b * float(np.dot(y_b, dy_b_num_dt))) / alpha
    dx_b_dt = np.cross(dy_b_dt, z_b) + np.cross(y_b, h_omega)

    domega_x_dt = -float(np.dot(d2z_b, y_b)) - float(np.dot(h_omega, dy_b_dt))
    domega_y_dt = float(np.dot(d2z_b, x_b)) + float(np.dot(h_omega, dx_b_dt))
    domega_z_dt = -(
        float(np.dot(x_b, dy_b2_num_dt2)) + float(np.dot(dx_b_dt, dy_b_num_dt))
    ) / alpha + (alpha_dot / alpha**2) * float(np.dot(x_b, dy_b_num_dt))

    domega_dt = np.array([domega_x_dt, domega_y_dt, domega_z_dt])

    Jw = J @ omega
    torques = J @ domega_dt + np.cross(omega, Jw)

    state = UAVState(np.concatenate((pos, vel, q, omega)))
    return FlatOutputResult(state=state, thrust=T, torques=torques)
