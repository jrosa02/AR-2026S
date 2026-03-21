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
"""SDF adapter: extracts UAVParams from a Gazebo model.sdf file."""

import math
import xml.etree.ElementTree as ET

import numpy as np

from uav_model.config_loader.params import UAVParams

GRAVITY = 9.81
MOTOR_PLUGIN_FILENAME = 'gz-sim-multicopter-motor-model-system'


class SDFAdapter:
    """Parse a Gazebo SDF model file and produce a UAVParams dataclass."""

    def __init__(self, sdf_path: str):
        """Load and parse the SDF file at the given path."""
        self._tree = ET.parse(sdf_path)
        self._root = self._tree.getroot()
        self._model = self._root.find('model')
        if self._model is None:
            raise ValueError(f'No <model> element found in {sdf_path}')

    def extract(self) -> UAVParams:
        """Extract all UAV physical parameters from the SDF."""
        mass, J = self._extract_body_inertial()
        J_inv = np.linalg.inv(J)
        motors = self._extract_motors()

        num_rotors = len(motors)
        positions = np.zeros(num_rotors * 3, dtype=np.float64)
        directions = np.zeros(num_rotors, dtype=np.int32)
        motor_constant = motors[0]['motor_constant']
        drag_coefficient = motors[0]['drag_coefficient']

        for i, motor in enumerate(motors):
            positions[3 * i:3 * i + 3] = motor['position']
            directions[i] = motor['direction']

        return UAVParams(
            mass=mass,
            J=J,
            J_inv=J_inv,
            gravity=GRAVITY,
            num_rotors=num_rotors,
            rotor_positions=positions,
            motor_constant=motor_constant,
            drag_coefficient=drag_coefficient,
            rotor_directions=directions,
        )

    def _extract_body_inertial(self):
        """Find the body link (heaviest) and extract mass + inertia tensor."""
        best_link = None
        best_mass = 0.0

        for link in self._model.findall('link'):
            inertial = link.find('inertial')
            if inertial is None:
                continue
            mass_el = inertial.find('mass')
            if mass_el is None:
                continue
            m = float(mass_el.text)
            if m > best_mass:
                best_mass = m
                best_link = link

        if best_link is None:
            raise ValueError('No link with <inertial>/<mass> found in SDF')

        inertia_el = best_link.find('inertial/inertia')
        ixx = float(inertia_el.findtext('ixx', '0'))
        iyy = float(inertia_el.findtext('iyy', '0'))
        izz = float(inertia_el.findtext('izz', '0'))
        ixy = float(inertia_el.findtext('ixy', '0'))
        ixz = float(inertia_el.findtext('ixz', '0'))
        iyz = float(inertia_el.findtext('iyz', '0'))

        J = np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ], dtype=np.float64)

        return best_mass, J

    def _extract_motors(self):
        """Extract motor parameters and positions from MulticopterMotorModel plugins."""
        link_poses = {}
        for link in self._model.findall('link'):
            name = link.get('name', '')
            pose_el = link.find('pose')
            if pose_el is not None:
                coords = [float(v) for v in pose_el.text.strip().split()]
                link_poses[name] = np.array(coords[:3], dtype=np.float64)

        motors = []
        for plugin in self._model.findall('plugin'):
            if plugin.get('filename') != MOTOR_PLUGIN_FILENAME:
                continue

            link_name = plugin.findtext('linkName', '')
            actuator_num = int(plugin.findtext('actuator_number', '0'))
            motor_constant = float(plugin.findtext('motorConstant', '0'))
            drag_coeff = float(
                plugin.findtext('rotorDragCoefficient', '0')
            )
            turning_dir = plugin.findtext('turningDirection', 'ccw')
            direction = -1 if turning_dir == 'ccw' else 1

            if link_name not in link_poses:
                raise ValueError(
                    f'Motor plugin references link "{link_name}" '
                    f'but no <pose> found for it'
                )

            position = link_poses[link_name]
            arm_length = math.sqrt(
                position[0] ** 2 + position[1] ** 2
            )
            if arm_length < 1e-6:
                raise ValueError(
                    f'Motor link "{link_name}" has near-zero arm length'
                )

            motors.append({
                'actuator_number': actuator_num,
                'link_name': link_name,
                'position': position,
                'motor_constant': motor_constant,
                'drag_coefficient': drag_coeff,
                'direction': direction,
            })

        if not motors:
            raise ValueError(
                'No MulticopterMotorModel plugins found in SDF'
            )

        motors.sort(key=lambda m: m['actuator_number'])
        return motors
