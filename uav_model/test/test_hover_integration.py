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
"""Closed-loop hover integration test: Mellinger controller + UAVModel, no ROS/Gazebo.

The loop mirrors the Python sim data flow:
    MellingerController.compute() -> [T, tau_x, tau_y, tau_z] -> UAVModel.step()

Physics runs at 500 Hz; controller runs at 100 Hz (every 5 physics steps),
matching the SimNode / ControllerNode rate split used in the real system.

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
from uav_model.model.uav_model import UAVFlatState, UAVModel, UAVState

_DEFAULT_SDF = os.path.join(
    get_package_share_directory('ros_gz_crazyflie_bringup'),
    'gazebo_files',
    'gazebo',
    'crazyflie',
    'model.sdf',
)

# Hover setpoint: [0, 0, 1] m, zero yaw
_HOVER_POS = np.array([0.0, 0.0, 1.0])

# Simulation timing
_DT = 0.002  # physics step: 500 Hz
_CTRL_EVERY = 5  # controller rate: 100 Hz (every 5 physics steps)
_SIM_TIME = 10.0  # seconds

# Convergence tolerances
_POS_ATOL = 0.05  # metres
_VEL_ATOL = 0.05  # m/s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def cf_params() -> UAVParams:
    return SDFAdapter(_DEFAULT_SDF).extract()


@pytest.fixture(scope='module')
def controller(cf_params) -> MellingerController:
    return MellingerController(cf_params, MellingerGains())


@pytest.fixture
def model(cf_params) -> UAVModel:
    m = UAVModel(cf_params)
    m.reset()
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hover_setpoint() -> UAVFlatState:
    data = np.zeros(18)
    data[0:3] = _HOVER_POS
    return UAVFlatState(data)


def _make_state(pos, vel=None, omega=None) -> UAVState:
    if vel is None:
        vel = np.zeros(3)
    if omega is None:
        omega = np.zeros(3)
    return UAVState(np.concatenate([pos, vel, [1.0, 0.0, 0.0, 0.0], omega]))


def _run_hover(
    model: UAVModel, controller: MellingerController, initial_state: UAVState
) -> UAVState:
    """Run the closed-loop hover simulation from the given initial state."""
    model.reset(np.array(initial_state, dtype=np.float64))
    desired = _hover_setpoint()
    params = controller._mass, controller._gravity

    hover_thrust = params[0] * params[1]
    u = np.array([hover_thrust, 0.0, 0.0, 0.0])
    steps = int(_SIM_TIME / _DT)

    for i in range(steps):
        if i % _CTRL_EVERY == 0:
            u = controller.compute(desired, model.state)
        model.step(u, _DT)

    return model.state


def _assert_hover_converged(state: UAVState) -> None:
    np.testing.assert_allclose(
        state.position,
        _HOVER_POS,
        atol=_POS_ATOL,
        err_msg=f'position {state.position} did not converge to {_HOVER_POS}',
    )
    np.testing.assert_allclose(
        state.velocity,
        np.zeros(3),
        atol=_VEL_ATOL,
        err_msg=f'velocity {state.velocity} did not settle near zero',
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hover_from_origin(model, controller):
    """Drone starts 1 m below the setpoint and must climb to hover."""
    state = _run_hover(model, controller, _make_state(pos=np.array([0.0, 0.0, 0.0])))
    _assert_hover_converged(state)


def test_hover_from_offset(model, controller):
    """Drone starts displaced in all three axes and must converge to hover."""
    state = _run_hover(model, controller, _make_state(pos=np.array([0.3, -0.2, 0.8])))
    _assert_hover_converged(state)


def test_hover_from_velocity(model, controller):
    """Drone starts at the hover position but with non-zero velocity and must re-settle."""
    state = _run_hover(
        model,
        controller,
        _make_state(pos=_HOVER_POS.copy(), vel=np.array([0.5, -0.3, 0.2])),
    )
    _assert_hover_converged(state)
