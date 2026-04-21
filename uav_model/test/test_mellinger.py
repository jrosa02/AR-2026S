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
"""Tests for MellingerController."""

import numpy as np
import pytest

from uav_model.config_loader.params import UAVParams
from uav_model.model.mellinger import MellingerController, MellingerGains
from uav_model.model.uav_model import UAVFlatState, UAVState

# CrazyFlie physical constants
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


def _hover_flat() -> UAVFlatState:
    """Flat state representing a stationary hover at the origin, zero yaw."""
    data = np.zeros(18)
    data[8] = 0.0     # zero acceleration (gravity compensated internally)
    return UAVFlatState(data)


def _hover_uav() -> UAVState:
    """Return a UAVState representing a stationary hover at the origin."""
    data = np.zeros(13)
    data[6] = 1.0     # identity quaternion [w=1, x=0, y=0, z=0]
    return UAVState(data)


@pytest.fixture(scope='module')
def cf_params():
    """CrazyFlie-like UAVParams with inline constants."""
    return UAVParams(
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


@pytest.fixture(scope='module')
def controller(cf_params):
    """Return a MellingerController with default gains."""
    return MellingerController(cf_params, MellingerGains())


class TestHoverEquilibrium:
    """At hover, all errors are zero — output should equal gravity compensation."""

    def test_thrust_equals_weight(self, controller):
        """T ≈ m·g when desired and actual are both at hover."""
        u = controller.compute(_hover_flat(), _hover_uav())
        assert u[0] == pytest.approx(_MASS * _GRAVITY, rel=1e-6)

    def test_torques_are_zero(self, controller):
        """All torques ≈ 0 at hover equilibrium."""
        u = controller.compute(_hover_flat(), _hover_uav())
        np.testing.assert_allclose(u[1:], 0.0, atol=1e-12)

    def test_output_shape(self, controller):
        """Output is a 4-element array [T, tau_x, tau_y, tau_z]."""
        u = controller.compute(_hover_flat(), _hover_uav())
        assert u.shape == (4,)


class TestPositionError:
    """A position offset should produce a corrective tilt (non-zero torques)."""

    def test_x_offset_produces_pitch(self, controller):
        """Positive x error → negative pitch torque (nose down to accelerate)."""
        desired = _hover_flat()
        actual = _hover_uav()
        actual = UAVState(actual.copy())
        actual[0] = -0.1   # actual is 0.1 m behind desired in x

        u = controller.compute(desired, actual)
        # Pitch torque (tau_y) should be non-zero and push nose forward
        assert abs(u[2]) > 0.0

    def test_z_offset_increases_thrust(self, controller):
        """Positive altitude error → thrust above hover weight."""
        desired = _hover_flat()
        actual = UAVState(np.zeros(13))
        actual[6] = 1.0        # identity quaternion
        actual[2] = -0.1       # actual is 0.1 m below desired

        u = controller.compute(desired, actual)
        assert u[0] > _MASS * _GRAVITY

    def test_z_offset_below_increases_thrust(self, controller):
        """Negative altitude error (too high) → thrust below hover weight."""
        desired = _hover_flat()
        actual = UAVState(np.zeros(13))
        actual[6] = 1.0
        actual[2] = 0.1        # actual is 0.1 m above desired

        u = controller.compute(desired, actual)
        assert u[0] < _MASS * _GRAVITY


class TestAttitudeError:
    """Test attitude error response.

    With a pure attitude error and no position/velocity error, thrust should
    stay near m·g and torques should correct the roll/pitch.
    """

    def _tilted_state(self, roll_deg: float) -> UAVState:
        """Return a hover state tilted by roll_deg around the x-axis."""
        angle = np.radians(roll_deg)
        qw = np.cos(angle / 2)
        qx = np.sin(angle / 2)
        state = UAVState(np.zeros(13))
        state[6] = qw
        state[7] = qx
        return state

    def test_roll_error_produces_roll_torque(self, controller):
        """A 10° roll error should produce a roll-correcting torque."""
        desired = _hover_flat()
        actual = self._tilted_state(10.0)

        u = controller.compute(desired, actual)
        # Roll torque (tau_x) must be non-zero
        assert abs(u[1]) > 0.0

    def test_thrust_near_weight_for_small_tilt(self, controller):
        """For a small tilt, projected thrust stays close to m·g."""
        desired = _hover_flat()
        actual = self._tilted_state(5.0)

        u = controller.compute(desired, actual)
        # Allow 5% deviation — small tilt, not a large manoeuvre
        assert u[0] == pytest.approx(_MASS * _GRAVITY, rel=0.05)

    def test_roll_torque_sign(self, controller):
        """Positive roll error (right tilt) should produce negative roll torque."""
        desired = _hover_flat()
        actual = self._tilted_state(10.0)   # tilted right

        u = controller.compute(desired, actual)
        assert u[1] < 0.0


class TestGainScaling:
    """Doubling kR should double the attitude torque at constant error."""

    def test_double_kR_doubles_torque(self, cf_params):
        gains_1x = MellingerGains(kR=np.array([4.0e-3, 4.0e-3, 1.0e-3]))
        gains_2x = MellingerGains(kR=np.array([8.0e-3, 8.0e-3, 2.0e-3]))

        ctrl_1x = MellingerController(cf_params, gains_1x)
        ctrl_2x = MellingerController(cf_params, gains_2x)

        desired = _hover_flat()
        actual = UAVState(np.zeros(13))
        actual[6] = np.cos(np.radians(10) / 2)
        actual[7] = np.sin(np.radians(10) / 2)

        u_1x = ctrl_1x.compute(desired, actual)
        u_2x = ctrl_2x.compute(desired, actual)

        np.testing.assert_allclose(u_2x[1:], 2.0 * u_1x[1:], rtol=1e-6)
