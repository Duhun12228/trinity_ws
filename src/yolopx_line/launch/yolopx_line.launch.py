import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('yolopx_line'), 'config', 'yolopx.yaml')

    params_file = LaunchConfiguration('params_file')
    input_topic = LaunchConfiguration('input_topic')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML parameter file'),
        DeclareLaunchArgument('input_topic', default_value='/camera/color/image_raw/compressed',
                              description='CompressedImage input topic'),
        Node(
            package='yolopx_line',
            executable='yolopx_node',
            name='yolopx_node',
            output='screen',
            parameters=[params_file, {'input_topic': input_topic}],
        ),
    ])
