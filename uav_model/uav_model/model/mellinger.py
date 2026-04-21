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
"""Mellinger trajectory-tracking controller (Mellinger & Kumar 2011).

Implements the full feedback + feedforward control law:
  - Position/velocity feedback (kx, kv gains)
  - Attitude/angular-velocity feedback (kR, kOmega gains)
  - Feedforward thrust and torques from differential flatness inversion

Output: [T, tau_x, tau_y, tau_z] — compatible with UAVModel.step(u, dt).
"""

from dataclasses import dataclass, field

import numpy as np

from uav_model.config_loader.params import UAVParams
from uav_model.model.flat_output import flat_to_full_state
from uav_model.model.uav_model import UAVFlatState, UAVState


@dataclass
class MellingerGains:
    """PD gains for the Mellinger controller.

    Each gain may be a scalar (applied uniformly) or a 3-element array
    (per-axis weighting).
    """

    kx: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0]))
    kv: np.ndarray = field(default_factory=lambda: np.array([2.8, 2.8, 2.8]))
    kR: np.ndarray = field(default_factory=lambda: np.array([8.0e-3, 8.0e-3, 2.0e-3]))
    kOmega: np.ndarray = field(default_factory=lambda: np.array([1.5e-3, 1.5e-3, 5.0e-4]))

    def __post_init__(self):
        """Broadcast scalar gains to 3-element arrays."""
        for name in ('kx', 'kv', 'kR', 'kOmega'):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if v.ndim == 0:
                v = np.full(3, float(v))
            object.__setattr__(self, name, v)


class MellingerController:
    """Mellinger & Kumar (2011) trajectory-tracking controller.

    Given a desired flat-output trajectory and the current UAV state, computes
    the collective thrust and body torques that drive tracking errors to zero.

    Control law overview (all vectors in world frame unless noted):
      1. Position + velocity feedback → corrected desired acceleration a_des
      2. a_des → desired thrust vector F_des, body z-axis z_b_des, thrust T
      3. Desired yaw + z_b_des → desired rotation matrix R_des
      4. Attitude error e_R  = vee(R_des^T R − R^T R_des) / 2
      5. Angular-velocity error e_ω = ω − R^T R_des ω_des   (body frame)
      6. Torques τ = −kR⊙e_R − kΩ⊙e_ω + ω × Jω
    """

    def __init__(self, params: UAVParams, gains: MellingerGains):
        """Initialise controller with physical parameters and control gains.

        Args:
            params: Frozen UAVParams (mass, gravity, inertia tensor).
            gains:  MellingerGains (kx, kv, kR, kOmega).
        """
        self._mass = float(params.mass)
        self._gravity = float(params.gravity)
        self._J = np.array(params.J, dtype=np.float64)
        self._gains = gains

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, desired: UAVFlatState, state: UAVState) -> np.ndarray:
        """Compute control wrench from desired trajectory and current state.

        Args:
            desired: 18-element UAVFlatState with desired pos, vel, acc,
                     jerk, snap, yaw, yaw_rate, yaw_acc.
            state:   13-element UAVState with current pos, vel, quat, omega.

        Returns:
            u: np.ndarray shape (4,) = [T, tau_x, tau_y, tau_z]
        """
        g = self._gains
        m = self._mass
        grav = self._gravity

        # ---- Step 1: position + velocity feedback ----------------------
        e_x = desired.position - state.position
        e_v = desired.velocity - state.velocity
        a_des = desired.acceleration + g.kx * e_x + g.kv * e_v

        # ---- Step 2: desired thrust vector and scalar ------------------
        F_des = m * (a_des + np.array([0.0, 0.0, grav]))
        z_b_des = F_des / np.linalg.norm(F_des)

        R_actual = self._quat_to_rot(state.quaternion)
        T = float(np.dot(F_des, R_actual[:, 2]))   # project onto current z_b

        # ---- Step 3: desired rotation matrix from yaw + z_b_des --------
        R_des = self._rot_des(z_b_des, float(desired.yaw))

        # ---- Step 4: attitude error (vee map) --------------------------
        eR_mat = R_des.T @ R_actual - R_actual.T @ R_des
        e_R = 0.5 * self._vee(eR_mat)

        # ---- Step 5: angular-velocity error (body frame) ---------------
        omega_des = flat_to_full_state(desired, m, grav, self._J).state.angular_velocity
        e_omega = state.angular_velocity - R_actual.T @ R_des @ omega_des

        # ---- Step 6: torque command ------------------------------------
        omega = state.angular_velocity
        gyro = np.cross(omega, self._J @ omega)
        tau = -g.kR * e_R - g.kOmega * e_omega + gyro

        return np.array([T, tau[0], tau[1], tau[2]])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    @staticmethod
    def _vee(M: np.ndarray) -> np.ndarray:
        """Extract axial vector from skew-symmetric matrix M.

        vee([[0, -z, y], [z, 0, -x], [-y, x, 0]]) → [x, y, z]
        """
        return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)

    @staticmethod
    def _rot_des(z_b_des: np.ndarray, yaw: float) -> np.ndarray:
        """Build desired rotation matrix from desired body z-axis and yaw.

        Args:
            z_b_des: Unit vector — desired body z-axis (thrust direction).
            yaw:     Desired heading angle [rad].

        Returns:
            R_des: 3×3 rotation matrix (column-stacked body axes).
        """
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b_des, x_c)
        norm = np.linalg.norm(y_b)
        if norm < 1e-12:
            # Degenerate: z_b_des is parallel to x_c (e.g. near 90° pitch).
            # Fall back to world y-axis as heading reference.
            x_c = np.array([0.0, 1.0, 0.0])
            y_b = np.cross(z_b_des, x_c)
            norm = np.linalg.norm(y_b)
        y_b /= norm
        x_b = np.cross(y_b, z_b_des)
        return np.column_stack((x_b, y_b, z_b_des))
