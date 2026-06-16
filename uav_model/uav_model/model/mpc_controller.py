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
"""Linear MPC controller for translational trajectory planning.

Plans over a receding horizon using a ZOH double-integrator model per axis,
solved as a constrained QP via scipy SLSQP with analytic gradients and
warm-starting.

The MPC optimises translational accelerations only and feeds the resulting
position/velocity/acceleration setpoint into a Mellinger inner-loop
controller that handles attitude dynamics.

Decision variable layout (axis-major):
    z = [ax_0..ax_{N-1}, ay_0..ay_{N-1}, az_0..az_{N-1}]  shape (3N,)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize

from uav_model.model.uav_model import UAVFlatState, UAVState


@dataclass
class MPCParams:
    """Tuning parameters for MPCController."""

    horizon: int = 10
    dt: float = 0.05
    Q_pos: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0]))
    Q_vel: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    R_acc: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    Q_terminal: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, 50.0]))
    v_max: float = 3.0
    a_max: float = 5.0

    def __post_init__(self):
        for name in ('Q_pos', 'Q_vel', 'R_acc', 'Q_terminal'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))


class MPCController:
    """Receding-horizon MPC for UAV translational trajectory planning.

    Uses a ZOH constant-acceleration double-integrator model:
        v_{k+1} = v_k + a_k * dt
        p_{k+1} = p_k + v_k * dt + 0.5 * a_k * dt^2

    The gravity compensation is handled by the Mellinger inner loop;
    MPC sees a world-frame double integrator where a=[0,0,0] corresponds
    to hover (Mellinger adds m*(a + g*z_hat) internally).
    """

    def __init__(self, params: MPCParams) -> None:
        self._p = params
        N, dt = params.horizon, params.dt
        self._N = N

        # --- Propagation matrices for one axis (shape N × N) ---
        # B_vel[k, j] = dt            for j < k  (strictly lower triangular)
        # B_pos[k, j] = (k-j-0.5)*dt² for j < k  (ZOH constant-acceleration)
        B_vel = np.tril(np.ones((N, N)), k=-1) * dt
        B_pos = np.zeros((N, N))
        for k in range(1, N):
            for j in range(k):
                B_pos[k, j] = (k - j - 0.5) * dt ** 2

        self._B_vel = B_vel
        self._B_pos = B_pos

        # Block-diagonal for velocity linear constraint (all 3 axes together)
        self._B_vel_block = block_diag(B_vel, B_vel, B_vel)  # (3N, 3N)

        # Per-step position weights: Q_pos for interior steps, Q_terminal for last
        self._Qp = np.tile(params.Q_pos, (N, 1)).T.copy()  # (3, N)
        for i in range(3):
            self._Qp[i, -1] = params.Q_terminal[i]

        self._acc_bounds = Bounds(
            lb=np.full(3 * N, -params.a_max),
            ub=np.full(3 * N,  params.a_max),
        )

        self._z_prev = np.zeros(3 * N)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        state: UAVState,
        goal: np.ndarray,
        yaw: float = 0.0,
        goal_vel: np.ndarray | None = None,
    ) -> UAVFlatState:
        """Solve the MPC QP and return a UAVFlatState setpoint for Mellinger.

        Args:
            state: Current UAV state (uses position and velocity only).
            goal:  Target position [x, y, z], shape (3,).
            yaw:   Desired heading [rad], passed through to Mellinger.

        Returns:
            UAVFlatState with predicted position, velocity, and optimal
            acceleration as feedforward for the Mellinger inner loop.
        """
        N = self._N
        dt = self._p.dt
        p0 = np.asarray(state.position, dtype=np.float64)
        v0 = np.asarray(state.velocity, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        gv = np.zeros(3) if goal_vel is None else np.asarray(goal_vel, dtype=np.float64)

        # Velocity linear constraint: bounds shift with current velocity
        v_free = np.concatenate([v0[i] * np.ones(N) for i in range(3)])
        vel_con = LinearConstraint(
            self._B_vel_block,
            lb=-self._p.v_max - v_free,
            ub=self._p.v_max - v_free,
        )

        # Warm-start: shift previous solution by 1 step per axis, zero-pad tail
        z_warm = np.concatenate([
            np.append(self._z_prev[i * N:(i + 1) * N][1:], 0.0)
            for i in range(3)
        ])

        result = minimize(
            self._cost_and_grad,
            z_warm,
            args=(p0, v0, goal, gv),
            method='SLSQP',
            jac=True,
            bounds=self._acc_bounds,
            constraints=[vel_con],
            options={'maxiter': 50, 'ftol': 1e-6},
        )
        self._z_prev = result.x.copy()

        z = result.x
        ax, ay, az = z[:N], z[N:2 * N], z[2 * N:]
        a_axes = (ax, ay, az)

        # First acceleration step per axis (axis-major indexing) — feedforward
        a_opt = np.array([z[0], z[N], z[2 * N]])

        # Reconstruct full optimal trajectory to get the terminal predicted state.
        # Using the terminal (N-step-ahead) position/velocity as the Mellinger
        # setpoint gives a meaningful position error (the drone needs to reach the
        # MPC-planned terminal point) and a planned terminal velocity (near zero when
        # close to the goal). A 1-step-ahead setpoint produces a near-zero position
        # error, leaving Mellinger relying almost entirely on the acceleration
        # feedforward and causing oscillation when the drone overshoots.
        P = np.array([
            p0[i] + dt * np.arange(N) * v0[i] + self._B_pos @ a_axes[i]
            for i in range(3)
        ])
        V = np.array([
            v0[i] + self._B_vel @ a_axes[i]
            for i in range(3)
        ])

        flat = np.zeros(18)
        flat[0:3] = P[:, -1]   # terminal predicted position (≈ goal when nearby)
        flat[3:6] = V[:, -1]   # terminal predicted velocity (≈ 0 when stopping)
        flat[6:9] = a_opt      # first-step acceleration as Mellinger feedforward
        flat[15] = yaw
        return UAVFlatState(flat)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cost_and_grad(
        self,
        z: np.ndarray,
        p0: np.ndarray,
        v0: np.ndarray,
        goal: np.ndarray,
        goal_vel: np.ndarray,
    ):
        """Compute QP cost and analytic gradient for SLSQP."""
        N = self._N
        dt = self._p.dt

        ax, ay, az = z[:N], z[N:2 * N], z[2 * N:]
        a_axes = (ax, ay, az)

        # Predicted trajectories (3, N) — free response + controlled response
        P = np.array([
            p0[i] + dt * np.arange(N) * v0[i] + self._B_pos @ a_axes[i]
            for i in range(3)
        ])
        V = np.array([
            v0[i] + self._B_vel @ a_axes[i]
            for i in range(3)
        ])

        ep = P - goal[:, np.newaxis]  # position error (3, N)
        V_err = V - goal_vel[:, np.newaxis]  # velocity error vs desired orbit speed

        J = (
            0.5 * np.sum(self._Qp * ep ** 2)
            + 0.5 * np.sum(self._p.Q_vel[:, np.newaxis] * V_err ** 2)
            + 0.5 * sum(self._p.R_acc[i] * np.dot(a_axes[i], a_axes[i]) for i in range(3))
        )

        Qp_ep = self._Qp * ep
        Qv_V = self._p.Q_vel[:, np.newaxis] * V_err
        grad = np.concatenate([
            self._B_pos.T @ Qp_ep[i] + self._B_vel.T @ Qv_V[i] + self._p.R_acc[i] * a_axes[i]
            for i in range(3)
        ])
        return J, grad
