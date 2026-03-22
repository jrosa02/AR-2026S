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
"""Tests for UAVModel dynamics (Phase 2: utilities, Phase 3: trajectories)."""

import numpy as np
import pytest

from uav_model.config_loader.params import UAVParams
from uav_model.model.uav_model import UAVModel

# CrazyFlie physical constants (inline, no SDF dependency)
_MASS = 0.025
_GRAVITY = 9.81
_CT = 1.28192e-08
_CD = 8.06428e-05
_J = np.diag([1.657e-5, 1.657e-5, 2.9e-5])

# X-config motor positions [FR, RR, RL, FL] at ±0.031, ±0.031
_POS = np.array([
    0.031,  0.031, 0.0,
    0.031, -0.031, 0.0,
    -0.031, -0.031, 0.0,
    -0.031,  0.031, 0.0,
], dtype=np.float64)

# Alternating CCW/CW: -1, +1, -1, +1
_DIRS = np.array([-1, 1, -1, 1], dtype=np.int32)


@pytest.fixture(scope='module')
def cf_params():
    """Return CrazyFlie-like UAVParams with inline constants."""
    return UAVParams(
        mass=_MASS,
        J=_J,
        J_inv=np.linalg.inv(_J),
        gravity=_GRAVITY,
        num_rotors=4,
        rotor_positions=_POS.copy(),
        motor_constant=_CT,
        drag_coefficient=_CD,
        rotor_directions=_DIRS.copy(),
    )


@pytest.fixture
def model(cf_params):
    """Return a freshly reset UAVModel per test."""
    m = UAVModel(cf_params)
    m.reset()
    return m


# ---------------------------------------------------------------------------
# Phase 2 — UAVModel utilities
# ---------------------------------------------------------------------------

def test_quat_normalize_unit():
    """quat_normalize leaves an already-unit quaternion at norm 1."""
    x = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                 dtype=np.float64)
    UAVModel.quat_normalize(x)
    assert abs(np.linalg.norm(x[6:10]) - 1.0) < 1e-15


def test_quat_normalize_unnormalized():
    """quat_normalize corrects a 2x-scaled quaternion to unit norm."""
    x = np.array([0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
                 dtype=np.float64)
    UAVModel.quat_normalize(x)
    assert abs(np.linalg.norm(x[6:10]) - 1.0) < 1e-15


def test_wrench_symmetric_hover(model):
    """Equal motor speeds on symmetric X-layout produce zero net torque."""
    omega0 = 1000.0
    omega = np.full(4, omega0, dtype=np.float64)
    u = model.rotor_velocities_to_wrench(omega)
    expected_T = 4.0 * _CT * omega0 ** 2
    assert abs(u[0] - expected_T) < 1e-10, f'Thrust mismatch: {u[0]} vs {expected_T}'
    assert abs(u[1]) < 1e-12, f'tau_x non-zero: {u[1]}'
    assert abs(u[2]) < 1e-12, f'tau_y non-zero: {u[2]}'
    assert abs(u[3]) < 1e-12, f'tau_z non-zero: {u[3]}'


def test_wrench_yaw_torque_sign(model):
    """Single CCW rotor (dir=-1) produces negative yaw torque."""
    omega = np.array([1000.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u = model.rotor_velocities_to_wrench(omega)
    assert u[3] < 0.0, f'Expected tau_z < 0 for CCW rotor, got {u[3]}'


def test_hover_equilibrium(model):
    """At exact hover thrust, position and velocity remain zero."""
    u = np.array([_MASS * _GRAVITY, 0.0, 0.0, 0.0], dtype=np.float64)
    pos_before = model.position.copy()
    vel_before = model.velocity.copy()
    for _ in range(100):
        model.step(u, dt=0.001)
    assert np.allclose(model.position, pos_before, atol=1e-12)
    assert np.allclose(model.velocity, vel_before, atol=1e-12)


# ---------------------------------------------------------------------------
# Phase 3 — UAVModel: trajectory tests
# ---------------------------------------------------------------------------

def test_free_fall_z(model):
    """Free-fall from rest matches z(t) = -0.5 * g * t^2."""
    u = np.zeros(4, dtype=np.float64)
    dt = 0.001
    n_steps = 200
    for _ in range(n_steps):
        model.step(u, dt)
    t = n_steps * dt
    z_expected = -0.5 * _GRAVITY * t ** 2
    z_actual = model.position[2]
    rel_err = abs(z_actual - z_expected) / abs(z_expected)
    assert rel_err < 1e-5, f'z={z_actual:.9f}, expected={z_expected:.9f}, rel_err={rel_err:.2e}'


def test_free_fall_xy_unchanged(model):
    """Free-fall from rest produces no horizontal displacement."""
    u = np.zeros(4, dtype=np.float64)
    for _ in range(200):
        model.step(u, 0.001)
    assert abs(model.position[0]) < 1e-15, f'x drifted: {model.position[0]}'
    assert abs(model.position[1]) < 1e-15, f'y drifted: {model.position[1]}'


def test_quat_norm_preserved(model):
    """Quaternion unit norm is preserved over 5000 steps with angular velocity."""
    model.state[10:13] = [0.5, 1.0, 0.3]
    u = np.zeros(4, dtype=np.float64)
    for _ in range(5000):
        model.step(u, 0.001)
    norm = np.linalg.norm(model.quaternion)
    assert abs(norm - 1.0) < 1e-12, f'Quaternion norm drifted to {norm}'
