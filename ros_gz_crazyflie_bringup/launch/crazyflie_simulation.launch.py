"""Launch Gazebo simulation with Crazyflie model and ROS 2 bridge."""
# Copyright 2022 Open Source Robotics Foundation, Inc.
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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Configure ROS nodes for launch."""
    gz_ln_arg = DeclareLaunchArgument('gazebo_launch', default_value='True')
    rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Launch RViz2 with path visualization',
    )
    traj_arg = DeclareLaunchArgument(
        'use_trajectory_controller', default_value='false',
        description='Use TrajectoryControllerNode instead of fixed-hover ControllerNode',
    )
    mpc_arg = DeclareLaunchArgument(
        'use_mpc_controller', default_value='false',
        description='Use MPCControllerNode (MPC outer loop + Mellinger inner loop)',
    )
    path_file_arg = DeclareLaunchArgument(
        'path_file', default_value='',
        description='YAML waypoint file; used with trajectory or MPC controller',
    )
    # Setup project paths
    pkg_project_bringup = get_package_share_directory('ros_gz_crazyflie_bringup')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    uav_model_share = get_package_share_directory('uav_model')
    default_mpc_params = os.path.join(uav_model_share, 'config', 'mpc_params.yaml')
    mpc_params_file_arg = DeclareLaunchArgument(
        'mpc_params_file', default_value=default_mpc_params,
        description='Path to mpc_params.yaml; only used when use_mpc_controller:=true',
    )

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        condition=IfCondition(LaunchConfiguration('gazebo_launch')),
        launch_arguments={
            'gz_args': PathJoinSubstitution(
                [pkg_project_bringup, 'gazebo_files', 'gazebo', 'worlds', 'crazyflie_world.sdf -r']
            )
        }.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[
            {
                'config_file': os.path.join(
                    pkg_project_bringup, 'config', 'ros_gz_crazyflie_bridge.yaml'
                ),
            }
        ],
        output='screen',
    )

    # control = Node(
    #     package='ros_gz_crazyflie_control',
    #     executable='control_services',
    #     output='screen',
    #     parameters=[
    #         {'hover_height': 0.5},
    #         {'robot_prefix': '/crazyflie'},
    #         {'incoming_twist_topic': '/cmd_vel'},
    #         {'max_ang_z_rate': 0.4},
    #     ]
    # )

    mixer = Node(
        package='cf_control',
        executable='mixer',
        name='mixer',
        output='screen',
    )

    use_traj = LaunchConfiguration('use_trajectory_controller')
    use_mpc = LaunchConfiguration('use_mpc_controller')
    use_rviz = LaunchConfiguration('use_rviz')
    path_file = LaunchConfiguration('path_file')
    mpc_params_file = LaunchConfiguration('mpc_params_file')

    no_builtin_ctrl = PythonExpression(
        ["'", use_traj, "' == 'true' or '", use_mpc, "' == 'true'"]
    )

    controller = Node(
        package='uav_model',
        executable='controller_node',
        name='mellinger_controller',
        output='screen',
        parameters=[{'odom_topic': '/crazyflie/odom'}],
        condition=UnlessCondition(no_builtin_ctrl),
    )

    trajectory_controller = Node(
        package='uav_model',
        executable='trajectory_controller_node',
        name='trajectory_controller',
        output='screen',
        parameters=[{'odom_topic': '/crazyflie/odom'}],
        condition=IfCondition(use_traj),
    )

    mpc_controller = Node(
        package='uav_model',
        executable='mpc_controller_node',
        name='mpc_controller',
        output='screen',
        parameters=[{'odom_topic': '/crazyflie/odom'}, mpc_params_file],
        condition=IfCondition(use_mpc),
    )

    path_publisher = Node(
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
    )

    path_visualizer = Node(
        package='uav_model',
        executable='path_visualizer_node',
        name='path_visualizer',
        parameters=[{'odom_topic': '/crazyflie/odom', 'path_file': path_file}],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_project_bringup, 'config', 'crazyflie_rviz.rviz')],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            gz_ln_arg,
            rviz_arg,
            traj_arg,
            mpc_arg,
            path_file_arg,
            mpc_params_file_arg,
            gz_sim,
            bridge,
            mixer,
            controller,
            trajectory_controller,
            mpc_controller,
            path_publisher,
            path_visualizer,
            rviz2,
        ]
    )
