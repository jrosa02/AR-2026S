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
"""Unit tests for the Mellinger controller node logic (no ROS spin required)."""

import numpy as np

from uav_model.config_loader.params import UAVParams
from uav_model.model.mellinger import MellingerController, MellingerGains
from uav_model.model.uav_model import UAVFlatState, UAVState

_MASS = 0.025
_GRAVITY = 9.81
_J = np.diag([1.657e-5, 1.657e-5, 2.9e-5])
_POS = np.array([
    0.031,  0.031, 0.0,
    0.031, -0.031, 0.0,
    -0.031, -0.031, 0.0,
    -0.031,  0.031, 0.0,
], dtype=np.float64)
_DIRS = np.array([-1, 1, -1, 1], dtype=np.int32)


def _make_controller() -> MellingerController:
    params = UAVParams(
        mass=_MASS,
        J=_J,
        J_inv=np.linalg.inv(_J),
        gravity=_GRAVITY,
        num_rotors=4,
        rotor_positions=_POS.copy(),
        motor_constant=1.28192e-8,
        drag_coefficient=8.06428e-5,
        rotor_directions=_DIRS.copy(),
    )
    return MellingerController(params, MellingerGains())


def _hover_flat(x=0.0, y=0.0, z=1.0, yaw=0.0) -> UAVFlatState:
    data = np.zeros(18)
    data[0:3] = [x, y, z]
    data[15] = yaw
    return UAVFlatState(data)


def _level_state(pos, vel=None, omega=None) -> UAVState:
    """Drone at pos with level attitude (w=1), optional vel/omega."""
    if vel is None:
        vel = np.zeros(3)
    if omega is None:
        omega = np.zeros(3)
    return UAVState(np.concatenate([pos, vel, [1.0, 0.0, 0.0, 0.0], omega]))


def test_hover_equilibrium():
    """At the hover target with zero error the thrust should equal m*g and torques ~0."""
    ctrl = _make_controller()
    desired = _hover_flat(x=0.0, y=0.0, z=1.0)
    state = _level_state(pos=np.array([0.0, 0.0, 1.0]))

    u = ctrl.compute(desired, state)

    assert u.shape == (4,)
    assert abs(u[0] - _MASS * _GRAVITY) < 1e-6, f'thrust={u[0]}, expected {_MASS * _GRAVITY}'
    assert np.linalg.norm(u[1:]) < 1e-8, f'torques={u[1:]} should be ~0 at equilibrium'


def test_below_target_increases_thrust():
    """Drone 0.5 m below hover target should receive thrust > m*g to climb."""
    ctrl = _make_controller()
    desired = _hover_flat(x=0.0, y=0.0, z=1.0)
    state = _level_state(pos=np.array([0.0, 0.0, 0.5]))

    u = ctrl.compute(desired, state)

    assert u[0] > _MASS * _GRAVITY, (
        f'thrust={u[0]} should exceed hover thrust {_MASS * _GRAVITY} when below target'
    )


def test_above_target_decreases_thrust():
    """Drone 0.5 m above hover target should receive thrust < m*g to descend."""
    ctrl = _make_controller()
    desired = _hover_flat(x=0.0, y=0.0, z=1.0)
    state = _level_state(pos=np.array([0.0, 0.0, 1.5]))

    u = ctrl.compute(desired, state)

    assert u[0] < _MASS * _GRAVITY, (
        f'thrust={u[0]} should be below hover thrust {_MASS * _GRAVITY} when above target'
    )
