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
"""Data-driven tests for differential flatness conversion (flat output → full state)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from uav_model.model.flat_output import flat_to_full_state
from uav_model.model.uav_model import UAVFlatState, UAVState

_CSV_PATH = Path(__file__).parent / 'trajectory_from_flat_output_test_data.csv'

# Absolute tolerance for all assertions (pure algebra — no integration error)
_ATOL = 1e-6


# ---------------------------------------------------------------------------
# CSV fixture
# ---------------------------------------------------------------------------

def _load_csv_cases() -> list[dict]:
    """Parse CSV into a list of dicts with float values (test_name stays str)."""
    cases = []
    with open(_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                {k: (v if k == 'test_name' else float(v)) for k, v in row.items()}
            )
    return cases


@pytest.fixture(params=_load_csv_cases(), ids=lambda c: c['test_name'])
def flat_case(request) -> dict:
    """One parametrized test case per CSV row.

    Yields a dict with keys:
        flat       – UAVFlatState (18 elements)
        mass       – float [kg]
        gravity    – float [m/s²]
        J          – ndarray (3×3) inertia tensor
        exp_state  – UAVState (13 elements, ground truth)
        exp_thrust – float [N]
        exp_torques – ndarray (3,) [N·m]
    """
    row = request.param
    J = np.diag([row['in_I_xx'], row['in_I_yy'], row['in_I_zz']])

    flat = UAVFlatState(np.array([
        row['in_pos_x'],   row['in_pos_y'],   row['in_pos_z'],
        row['in_vel_x'],   row['in_vel_y'],   row['in_vel_z'],
        row['in_acc_x'],   row['in_acc_y'],   row['in_acc_z'],
        row['in_jerk_x'],  row['in_jerk_y'],  row['in_jerk_z'],
        row['in_snap_x'],  row['in_snap_y'],  row['in_snap_z'],
        row['in_yaw'],     row['in_yaw_rate'], row['in_yaw_acceleration'],
    ], dtype=np.float64))

    exp_state = UAVState(np.array([
        row['out_pos_x'],   row['out_pos_y'],   row['out_pos_z'],
        row['out_vel_x'],   row['out_vel_y'],   row['out_vel_z'],
        row['out_quat_w'],  row['out_quat_x'],  row['out_quat_y'],  row['out_quat_z'],
        row['out_omega_x'], row['out_omega_y'], row['out_omega_z'],
    ], dtype=np.float64))

    exp_torques = np.array(
        [row['out_torque_x'], row['out_torque_y'], row['out_torque_z']],
        dtype=np.float64,
    )

    return {
        'flat':        flat,
        'mass':        row['in_mass'],
        'gravity':     row['in_gravity'],
        'J':           J,
        'exp_state':   exp_state,
        'exp_thrust':  row['out_thrust'],
        'exp_torques': exp_torques,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(flat_case: dict):
    """Call flat_to_full_state with the fixture's inputs."""
    return flat_to_full_state(
        flat_case['flat'],
        flat_case['mass'],
        flat_case['gravity'],
        flat_case['J'],
    )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_position_and_velocity_passthrough(flat_case):
    """Position and velocity are passed through unchanged from flat output."""
    result = _run(flat_case)
    assert np.allclose(result.state.position, flat_case['exp_state'].position, atol=_ATOL), (
        f'position mismatch: got {result.state.position}, '
        f'expected {flat_case["exp_state"].position}'
    )
    assert np.allclose(result.state.velocity, flat_case['exp_state'].velocity, atol=_ATOL), (
        f'velocity mismatch: got {result.state.velocity}, '
        f'expected {flat_case["exp_state"].velocity}'
    )


def test_quaternion(flat_case):
    """Orientation quaternion matches expected value (sign-flip invariant) and is unit-norm."""
    result = _run(flat_case)
    q = result.state.quaternion
    q_exp = flat_case['exp_state'].quaternion

    # Both q and -q represent the same rotation
    match = np.allclose(q, q_exp, atol=_ATOL) or np.allclose(q, -q_exp, atol=_ATOL)
    assert match, (
        f'quaternion mismatch: got {q}, expected {q_exp} (or its negative)'
    )
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12, (
        f'quaternion not unit-norm: |q| = {np.linalg.norm(q)}'
    )


def test_angular_velocity(flat_case):
    """Body-frame angular velocity matches expected value."""
    result = _run(flat_case)
    assert np.allclose(
        result.state.angular_velocity,
        flat_case['exp_state'].angular_velocity,
        atol=_ATOL,
    ), (
        f'angular_velocity mismatch: got {result.state.angular_velocity}, '
        f'expected {flat_case["exp_state"].angular_velocity}'
    )


def test_thrust(flat_case):
    """Collective thrust matches expected value."""
    result = _run(flat_case)
    assert abs(result.thrust - flat_case['exp_thrust']) < _ATOL, (
        f'thrust mismatch: got {result.thrust:.6f}, expected {flat_case["exp_thrust"]:.6f}'
    )


def test_torques(flat_case):
    """Body torques match expected values."""
    result = _run(flat_case)
    assert np.allclose(result.torques, flat_case['exp_torques'], atol=_ATOL), (
        f'torques mismatch: got {result.torques}, expected {flat_case["exp_torques"]}'
    )
