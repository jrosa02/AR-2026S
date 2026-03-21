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
"""13-state rigid-body UAV dynamics model (pure Python / NumPy)."""

import numpy as np
import rclpy.logging

from uav_model.config_loader.params import UAVParams

_log = rclpy.logging.get_logger('uav_model')

_STATE_DIM = 13


class UAVModel:
    """13-state rigid-body UAV dynamics model (Newton-Euler, fixed-step RK4)."""

    def __init__(self, params: UAVParams):
        """Initialise model and cache physical parameters."""
        self._mass = float(params.mass)
        self._gravity = float(params.gravity)
        self._J = np.array(params.J, dtype=np.float64)
        self._J_inv = np.array(params.J_inv, dtype=np.float64)
        self._num_rotors = int(params.num_rotors)
        self._pos = np.asarray(
            params.rotor_positions, dtype=np.float64
        ).reshape(self._num_rotors, 3).copy()
        self._cT = float(params.motor_constant)
        self._cD = float(params.drag_coefficient)
        self._dirs = np.asarray(
            params.rotor_directions, dtype=np.float64
        ).copy()

        self._x = np.zeros(_STATE_DIM, dtype=np.float64)
        self._x[6] = 1.0  # identity quaternion (qw = 1)

        _log.info(
            f'UAVModel initialised: mass={self._mass:.4f} kg, '
            f'rotors={self._num_rotors}, '
            f'J_diag=[{self._J[0, 0]:.3e}, '
            f'{self._J[1, 1]:.3e}, {self._J[2, 2]:.3e}]'
        )

    def reset(self, x0=None):
        """Reset state to x0, or to default hover-at-origin if None."""
        if x0 is not None:
            self._x[:] = x0
        else:
            self._x[:] = 0.0
            self._x[6] = 1.0
        _log.debug('UAVModel state reset')

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by unit quaternion q = [w, x, y, z]."""
        qv = q[1:4]
        c = np.cross(qv, v)
        return v + 2.0 * (q[0] * c + np.cross(qv, c))

    def _dynamics(self, x, u):
        """Compute state derivative dx = f(x, u)."""
        v = x[3:6]
        q = x[6:10]
        omega = x[10:13]

        T = u[0]
        tau = u[1:4]

        dx = np.empty(_STATE_DIM)

        # dr/dt = v  (eq. 1.6)
        dx[0:3] = v

        # dv/dt = -g*z + (1/m)*R(q)*[0,0,T]  (eq. 1.7)
        thrust_world = self._quat_rotate(q, np.array([0.0, 0.0, T]))
        dx[3:6] = thrust_world / self._mass
        dx[5] -= self._gravity

        # dq/dt = 0.5 * q ⊗ [0, ω]  (eq. 1.13)
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        dx[6] = 0.5 * (-qx * wx - qy * wy - qz * wz)
        dx[7] = 0.5 * (qw * wx + qy * wz - qz * wy)
        dx[8] = 0.5 * (qw * wy - qx * wz + qz * wx)
        dx[9] = 0.5 * (qw * wz + qx * wy - qy * wx)

        # dω/dt = J⁻¹ * (τ − ω × Jω)  (eq. 1.9)
        Jw = self._J @ omega
        dx[10:13] = self._J_inv @ (tau - np.cross(omega, Jw))

        return dx

    def _rk4_step(self, x, u, dt):
        """Integrate one RK4 step. Does not normalise the quaternion."""
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, u, dt):
        """Advance one RK4 step, normalise quaternion, return state view."""
        x_new = self._rk4_step(self._x, u, dt)
        self.quat_normalize(x_new)
        self._x[:] = x_new
        return self._x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def quat_normalize(x):
        """In-place L2 normalisation of the quaternion x[6:10]."""
        q = x[6:10]
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q /= norm

    def rotor_velocities_to_wrench(self, omega):
        """Convert rotor velocities (rad/s) to [T, tx, ty, tz] wrench."""
        omega = np.asarray(omega, dtype=np.float64)
        w2 = omega ** 2
        fi = self._cT * w2                      # thrust per rotor
        T = np.sum(fi)
        tx = np.dot(self._pos[:, 1], fi)        # sum(y_i * F_i)
        ty = -np.dot(self._pos[:, 0], fi)       # sum(-x_i * F_i)
        tz = np.dot(self._dirs, self._cD * w2)  # sum(dir_i * cD * wi^2)
        return np.array([T, tx, ty, tz])

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def state(self):
        """Return a view of the full 13-element state vector."""
        return self._x

    @property
    def position(self):
        """Return a view of the position slice x[0:3]."""
        return self._x[0:3]

    @property
    def velocity(self):
        """Return a view of the velocity slice x[3:6]."""
        return self._x[3:6]

    @property
    def quaternion(self):
        """Return a view of the quaternion slice x[6:10]."""
        return self._x[6:10]

    @property
    def angular_velocity(self):
        """Return a view of the angular velocity slice x[10:13]."""
        return self._x[10:13]
