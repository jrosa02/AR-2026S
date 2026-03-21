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
"""Pytest session configuration for uav_model tests."""

import os
import sys


def pytest_configure(config):
    """Move PYTHONPATH site-packages dirs to front of sys.path."""
    del config  # unused; pytest hook signature requires the parameter
    for path in os.environ.get('PYTHONPATH', '').split(':'):
        if not path or not os.path.isdir(path):
            continue
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
