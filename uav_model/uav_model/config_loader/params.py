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
"""UAV physical parameters dataclass."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UAVParams:
    """Immutable container for UAV physical parameters extracted from SDF."""

    mass: float
    J: np.ndarray
    J_inv: np.ndarray
    gravity: float
    num_rotors: int
    rotor_positions: np.ndarray
    motor_constant: float
    drag_coefficient: float
    rotor_directions: np.ndarray

    def __post_init__(self):
        """Validate physical parameter constraints."""
        eigvals = np.linalg.eigvalsh(self.J)
        if not np.all(eigvals > 0):
            raise ValueError(
                f'Inertia tensor J must be positive-definite, '
                f'got eigenvalues: {eigvals}'
            )
        if self.mass <= 0:
            raise ValueError(f'Mass must be positive, got {self.mass}')
