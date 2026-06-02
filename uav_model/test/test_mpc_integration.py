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
"""Closed-loop MPC + Mellinger integration test (pure Python, no ROS/Gazebo).

Three-rate simulation loop mirrors the ROS node timing:
    MPC:      20 Hz (every 25 physics steps)
    Mellinger: 100 Hz (every  5 physics steps)
    Physics:  500 Hz

Physical parameters are loaded from the Gazebo SDF via SDFAdapter, the same
source used by the real controller and simulator nodes.
"""

import os

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

from uav_model.config_loader.params import UAVParams
from uav_model.config_loader.sdf_adapter import SDFAdapter
from uav_model.model.mellinger import MellingerController, MellingerGains
from uav_model.model.mpc_controller import MPCController, MPCParams
from uav_model.model.uav_model import UAVFlatState, UAVModel, UAVState

_DEFAULT_SDF = os.path.join(
    get_package_share_directory('ros_gz_crazyflie_bringup'),
    'gazebo_files',
    'gazebo',
    'crazyflie',
    'model.sdf',
)

# Simulation timing
_DT = 0.002          # physics step: 500 Hz
_CTRL_EVERY = 5      # Mellinger rate: 100 Hz
_MPC_EVERY = 25      # MPC rate: 20 Hz

# Convergence tolerances (looser than pure Mellinger due to MPC planning lag)
_POS_ATOL = 0.10     # metres
_VEL_ATOL = 0.10     # m/s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def cf_params() -> UAVParams:
    return SDFAdapter(_DEFAULT_SDF).extract()


@pytest.fixture(scope='module')
def mellinger(cf_params) -> MellingerController:
    gains = MellingerGains(
        kx=np.array([6.0, 6.0, 6.0]),
        kv=np.array([4.0, 4.0, 4.0]),
    )
    return MellingerController(cf_params, gains)


@pytest.fixture(scope='module')
def mpc() -> MPCController:
    return MPCController(MPCParams())


@pytest.fixture
def model(cf_params) -> UAVModel:
    m = UAVModel(cf_params)
    m.reset()
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(pos, vel=None, omega=None) -> UAVState:
    if vel is None:
        vel = np.zeros(3)
    if omega is None:
        omega = np.zeros(3)
    return UAVState(np.concatenate([pos, vel, [1.0, 0.0, 0.0, 0.0], omega]))


def _run_simulation(
    model: UAVModel,
    mpc: MPCController,
    mellinger: MellingerController,
    initial_state: UAVState,
    goal: np.ndarray,
    yaw: float = 0.0,
    sim_time: float = 15.0,
) -> UAVState:
    """Run the closed-loop MPC + Mellinger simulation from the given initial state."""
    model.reset(np.array(initial_state, dtype=np.float64))
    mpc._z_prev = np.zeros(3 * mpc._N)  # reset warm-start cache

    # Initialise flat setpoint to current position (hover in place)
    flat_setpoint = UAVFlatState(np.zeros(18))
    flat_setpoint[0:3] = initial_state[0:3]

    hover_thrust = mellinger._mass * mellinger._gravity
    u = np.array([hover_thrust, 0.0, 0.0, 0.0])
    total_steps = int(sim_time / _DT)

    for i in range(total_steps):
        if i % _MPC_EVERY == 0:
            flat_setpoint = mpc.compute(model.state, goal, yaw)
        if i % _CTRL_EVERY == 0:
            u = mellinger.compute(flat_setpoint, model.state)
        model.step(u, _DT)

    return model.state


def _assert_converged(state: UAVState, goal: np.ndarray) -> None:
    np.testing.assert_allclose(
        state.position, goal, atol=_POS_ATOL,
        err_msg=f'position {state.position} did not converge to {goal}',
    )
    np.testing.assert_allclose(
        state.velocity, np.zeros(3), atol=_VEL_ATOL,
        err_msg=f'velocity {state.velocity} did not settle near zero',
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mpc_hover_stationary(model, mpc, mellinger):
    """MPC keeps drone at current position when goal equals initial position."""
    initial = _make_state(pos=np.array([0.0, 0.0, 1.0]))
    goal = np.array([0.0, 0.0, 1.0])
    state = _run_simulation(model, mpc, mellinger, initial, goal, sim_time=10.0)
    _assert_converged(state, goal)


def test_mpc_goto_3d_waypoint(model, mpc, mellinger):
    """MPC drives drone from ground to a 3D position setpoint."""
    initial = _make_state(pos=np.array([0.0, 0.0, 0.0]))
    goal = np.array([1.0, 1.0, 2.0])
    state = _run_simulation(model, mpc, mellinger, initial, goal, sim_time=15.0)
    _assert_converged(state, goal)


def test_mpc_multi_waypoint_sequence(model, mpc, mellinger):
    """MPC tracks a sequence of 3 waypoints; checks each is reached in order."""
    waypoints = [
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.5]),
        np.array([0.0, 1.0, 2.0]),
    ]
    tolerance = 0.15  # metres — coarser because switching mid-flight

    model.reset(np.array(_make_state(pos=np.zeros(3)), dtype=np.float64))
    mpc._z_prev = np.zeros(3 * mpc._N)

    flat_setpoint = UAVFlatState(np.zeros(18))
    hover_thrust = mellinger._mass * mellinger._gravity
    u = np.array([hover_thrust, 0.0, 0.0, 0.0])

    wp_idx = 0
    goal = waypoints[wp_idx]
    reached = [False] * len(waypoints)

    total_steps = int(25.0 / _DT)
    for i in range(total_steps):
        dist = float(np.linalg.norm(model.state.position - goal))
        if dist < tolerance and not reached[wp_idx]:
            reached[wp_idx] = True
            if wp_idx + 1 < len(waypoints):
                wp_idx += 1
                goal = waypoints[wp_idx]
        if i % _MPC_EVERY == 0:
            flat_setpoint = mpc.compute(model.state, goal)
        if i % _CTRL_EVERY == 0:
            u = mellinger.compute(flat_setpoint, model.state)
        model.step(u, _DT)

    assert all(reached), f'Not all waypoints reached: {reached}'


def test_mpc_converges_within_time_limit(model, mpc, mellinger):
    """Drone must reach a 2 m diagonal waypoint within 12 s from the ground."""
    initial = _make_state(pos=np.zeros(3))
    goal = np.array([2.0, 0.0, 2.0])

    model.reset(np.array(initial, dtype=np.float64))
    mpc._z_prev = np.zeros(3 * mpc._N)

    flat_setpoint = UAVFlatState(np.zeros(18))
    hover_thrust = mellinger._mass * mellinger._gravity
    u = np.array([hover_thrust, 0.0, 0.0, 0.0])
    converged_at = None

    total_steps = int(15.0 / _DT)
    for i in range(total_steps):
        if i % _MPC_EVERY == 0:
            flat_setpoint = mpc.compute(model.state, goal)
        if i % _CTRL_EVERY == 0:
            u = mellinger.compute(flat_setpoint, model.state)
        model.step(u, _DT)

        if converged_at is None:
            pos_err = float(np.linalg.norm(model.state.position - goal))
            vel_mag = float(np.linalg.norm(model.state.velocity))
            if pos_err < _POS_ATOL and vel_mag < _VEL_ATOL:
                converged_at = (i + 1) * _DT

    assert converged_at is not None, 'Drone did not converge to goal within 15 s'
    assert converged_at <= 12.0, (
        f'Converged too slowly: t={converged_at:.1f} s (limit 12 s)'
    )
