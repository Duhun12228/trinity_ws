import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('mission'), 'config', 'mission.yaml')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML parameter file'),
        Node(
            package='mission',
            executable='lane_follow_node',
            name='lane_follow_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
