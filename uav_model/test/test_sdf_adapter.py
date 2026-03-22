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
"""Tests for SDFAdapter: verify UAVParams extracted from model.sdf."""

import math
import os

import numpy as np
import pytest

from ament_index_python.packages import get_package_share_directory

from uav_model.config_loader.sdf_adapter import SDFAdapter


@pytest.fixture(scope='module')
def params():
    """Load UAVParams from the CrazyFlie model.sdf."""
    sdf_dir = get_package_share_directory('ros_gz_crazyflie_bringup')
    sdf_path = os.path.join(
        sdf_dir, 'gazebo_files', 'gazebo', 'crazyflie', 'model.sdf'
    )
    return SDFAdapter(sdf_path).extract()


def test_inertia_positive_definite(params):
    """Inertia tensor is positive-definite."""
    assert np.all(np.linalg.eigvalsh(params.J) > 0)


def test_J_inv_roundtrip(params):
    """J @ J_inv equals identity within numerical tolerance."""
    assert np.allclose(params.J @ params.J_inv, np.eye(3), atol=1e-10)


def test_num_rotors(params):
    """Four rotors are extracted from the SDF."""
    assert params.num_rotors == 4


def test_rotor_positions_arm_length(params):
    """Each rotor arm length is approx 0.0438 m (sqrt(0.031^2 + 0.031^2))."""
    for i in range(params.num_rotors):
        x = params.rotor_positions[3 * i]
        y = params.rotor_positions[3 * i + 1]
        arm = math.sqrt(x ** 2 + y ** 2)
        assert abs(arm - 0.0438) < 5e-4, \
            f'Rotor {i} arm length {arm:.6f} m outside expected range'


def test_rotor_directions_sum(params):
    """Net yaw direction is zero: two CW and two CCW rotors."""
    assert sum(params.rotor_directions) == 0
