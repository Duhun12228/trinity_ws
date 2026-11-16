"""
yolopx_node + lane_follow_node + mission nodes (slow, crosswalk, cone) + master_node.

  ros2 launch master master.launch.py              # master dashboard only on screen
  ros2 launch master master.launch.py quiet:=false # every node's log on screen

quiet:=true sends yolopx / lane_follow output to ~/.ros/log/latest/<node>-*.log; a crashed
process is still reported by launch ("process has died").
The camera is started separately.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    quiet = LaunchConfiguration('quiet').perform(context).lower() in ('true', '1', 'yes')
    other_output = 'own_log' if quiet else 'screen'  # own_log: stdout+stderr to a file only
    return [
        Node(
            package='yolopx_line',
            executable='yolopx_node',
            name='yolopx_node',
            output=other_output,
            parameters=[LaunchConfiguration('yolopx_params_file'),
                        {'input_topic': LaunchConfiguration('input_topic')}],
        ),
        Node(
            package='mission',
            executable='lane_follow_node',
            name='lane_follow_node',
            output=other_output,
            parameters=[LaunchConfiguration('lane_params_file')],
        ),
        # mission nodes (trinity_missions); register new ones in master.yaml `missions` too
        Node(
            package='trinity_missions',
            executable='slow_node',
            name='slow_node',
            output=other_output,
            parameters=[LaunchConfiguration('slow_params_file')],
        ),
        Node(
            package='trinity_missions',
            executable='crosswalk_node',
            name='crosswalk_node',
            output=other_output,
            parameters=[LaunchConfiguration('crosswalk_params_file')],
        ),
        Node(
            package='trinity_missions',
            executable='cone_node',
            name='cone_node',
            output=other_output,
            parameters=[LaunchConfiguration('cone_params_file')],
        ),
        Node(
            package='trinity_missions',
            executable='three_road_node',
            name='three_road_node',
            output=other_output,
            parameters=[LaunchConfiguration('three_road_params_file')],
        ),
        Node(
            package='trinity_missions',
            executable='parking_node',
            name='parking_node',
            output=other_output,
            parameters=[LaunchConfiguration('parking_params_file')],
        ),
        Node(
            package='master',
            executable='master_node',
            name='master_node',
            output='screen',
            emulate_tty=True,          # keep colours / line buffering
            output_format='{line}',    # dashboard lines without the [master_node-N] prefix
            parameters=[LaunchConfiguration('params_file'), {'quiet': quiet}],
        ),
    ]


def generate_launch_description():
    master_share = get_package_share_directory('master')
    mission_share = get_package_share_directory('mission')
    yolopx_share = get_package_share_directory('yolopx_line')
    missions_share = get_package_share_directory('trinity_missions')

    return LaunchDescription([
        DeclareLaunchArgument('quiet', default_value='true',
                              description='true: only the master dashboard on screen'),
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(master_share, 'config', 'master.yaml'),
            description='master_node YAML parameter file'),
        DeclareLaunchArgument(
            'lane_params_file',
            default_value=os.path.join(mission_share, 'config', 'mission.yaml'),
            description='lane_follow_node YAML parameter file'),
        DeclareLaunchArgument(
            'yolopx_params_file',
            default_value=os.path.join(yolopx_share, 'config', 'yolopx.yaml'),
            description='yolopx_node YAML parameter file'),
        DeclareLaunchArgument(
            'slow_params_file',
            default_value=os.path.join(missions_share, 'config', 'slow.yaml'),
            description='slow_node YAML parameter file'),
        DeclareLaunchArgument(
            'crosswalk_params_file',
            default_value=os.path.join(missions_share, 'config', 'crosswalk.yaml'),
            description='crosswalk_node YAML parameter file'),
        DeclareLaunchArgument(
            'cone_params_file',
            default_value=os.path.join(missions_share, 'config', 'cone.yaml'),
            description='cone_node YAML parameter file'),
        DeclareLaunchArgument(
            'three_road_params_file',
            default_value=os.path.join(missions_share, 'config', 'three_road.yaml'),
            description='three_road_node YAML parameter file'),
        DeclareLaunchArgument(
            'parking_params_file',
            default_value=os.path.join(missions_share, 'config', 'parking.yaml'),
            description='parking_node YAML parameter file'),
        DeclareLaunchArgument('input_topic', default_value='/camera/color/image_raw/compressed',
                              description='yolopx CompressedImage input topic'),
        OpaqueFunction(function=_nodes),
    ])
