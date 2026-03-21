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
"""Launch file for the UAV model simulation node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the UAV model sim node."""
    bringup_share = get_package_share_directory(
        'ros_gz_crazyflie_bringup'
    )
    default_sdf = os.path.join(
        bringup_share,
        'gazebo_files', 'gazebo', 'crazyflie', 'model.sdf',
    )

    uav_model_share = get_package_share_directory('uav_model')
    default_params = os.path.join(
        uav_model_share, 'config', 'sim_params.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'sdf_path', default_value=default_sdf,
            description='Path to the Gazebo SDF model file',
        ),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Path to sim_params.yaml',
        ),
        Node(
            package='uav_model',
            executable='sim_node',
            name='uav_model_sim',
            parameters=[
                LaunchConfiguration('params_file'),
                {'sdf_path': LaunchConfiguration('sdf_path')},
            ],
            output='screen',
        ),
    ])
