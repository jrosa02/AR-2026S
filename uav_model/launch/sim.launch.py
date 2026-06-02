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
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
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

    use_traj = LaunchConfiguration('use_trajectory_controller')
    use_mpc = LaunchConfiguration('use_mpc_controller')
    path_file = LaunchConfiguration('path_file')

    default_mpc_params = os.path.join(
        uav_model_share, 'config', 'mpc_params.yaml',
    )

    # A controller is "external" when the MPC node is used — sim_node only.
    no_builtin_ctrl = PythonExpression(
        ["'", use_traj, "' == 'true' or '", use_mpc, "' == 'true'"]
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
        DeclareLaunchArgument(
            'use_trajectory_controller', default_value='false',
            description='Use TrajectoryControllerNode instead of fixed-hover ControllerNode',
        ),
        DeclareLaunchArgument(
            'use_mpc_controller', default_value='false',
            description='Use MPCControllerNode (MPC outer loop + Mellinger inner loop)',
        ),
        DeclareLaunchArgument(
            'mpc_params_file', default_value=default_mpc_params,
            description='Path to mpc_params.yaml; only used when use_mpc_controller:=true',
        ),
        DeclareLaunchArgument(
            'path_file', default_value='',
            description='YAML waypoint file; used with trajectory or MPC controller',
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
        # Fixed-hover controller (default — disabled when trajectory or MPC is active)
        Node(
            package='uav_model',
            executable='controller_node',
            name='mellinger_controller',
            output='screen',
            condition=UnlessCondition(no_builtin_ctrl),
        ),
        # Trajectory controller (opt-in via use_trajectory_controller:=true)
        Node(
            package='uav_model',
            executable='trajectory_controller_node',
            name='trajectory_controller',
            output='screen',
            condition=IfCondition(use_traj),
        ),
        # MPC controller (opt-in via use_mpc_controller:=true)
        Node(
            package='uav_model',
            executable='mpc_controller_node',
            name='mpc_controller',
            parameters=[LaunchConfiguration('mpc_params_file')],
            output='screen',
            condition=IfCondition(use_mpc),
        ),
        # Path publisher (when path_file non-empty and trajectory or MPC controller active)
        Node(
            package='uav_model',
            executable='path_publisher_node',
            name='path_publisher',
            parameters=[{'path_file': path_file}],
            output='screen',
            condition=IfCondition(
                PythonExpression([
                    "('", use_traj, "' == 'true' or '", use_mpc, "' == 'true')"
                    " and '", path_file, "' != ''"
                ])
            ),
        ),
    ])
