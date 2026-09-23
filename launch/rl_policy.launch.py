"""The motion controller node with the rl_policy plugin."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the launch description."""
    as2_config = os.path.join(
        get_package_share_directory('as2_motion_controller'),
        'config', 'motion_controller_default.yaml')
    package_folder = get_package_share_directory('as2_rl_policy')
    plugin_config = os.path.join(package_folder, 'config', 'rl_policy_default.yaml')
    available_modes = os.path.join(package_folder, 'config', 'available_modes.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace', description='Drone namespace',
            default_value=EnvironmentVariable('AEROSTACK2_SIMULATION_DRONE_ID',
                                              default_value='drone0')),
        DeclareLaunchArgument(
            'use_sim_time', description='Use the simulation clock', default_value='true'),
        DeclareLaunchArgument(
            'config_file',
            description='Parameters of this flight (policies, course, mission), applied last'),
        DeclareLaunchArgument('log_level', description='Logging level', default_value='info'),
        Node(
            package='as2_motion_controller',
            executable='as2_motion_controller_node',
            name='controller_manager',
            namespace=LaunchConfiguration('namespace'),
            output='screen',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            parameters=[
                as2_config,
                plugin_config,
                LaunchConfiguration('config_file'),
                # Last, so no file can load another plugin or turn the bypass on.
                {
                    'plugin_name': 'rl_policy',
                    'use_bypass': False,
                    'plugin_available_modes_config_file': available_modes,
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                },
            ],
        ),
    ])
