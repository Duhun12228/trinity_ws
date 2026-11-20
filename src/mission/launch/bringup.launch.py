"""
Robot bringup for Trinity: wego teleop_launch.py without what we never use.

  limo_base   drive, /odom, /imu
  URDF        robot / joint state publishers: TF base_link -> imu_link etc. The EKF needs it
              (/imu is in imu_link; without the TF it waits on every IMU message and
              /odometry/filtered drops to ~1 Hz, 2026-11-18)
  ydlidar     /scan
  EKF         /odometry/filtered
  Orbbec      colour only (/camera/color/image_raw(/compressed), camera_info);
              depth, IR and both point clouds off (they cost most of the camera CPU)

Arguments
  tf:=true      also start the camera tilt TF (base_link -> camera_link, for rviz only)
  degree:=0.0   camera tilt for that TF
  depth:=false  turn the depth stream (and its point cloud) back on

  ros2 launch mission bringup.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(pkg, name, args=None, **kw):
    return IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare(pkg), 'launch', name]),
        launch_arguments=(args or {}).items(), **kw)


def generate_launch_description():
    tf = LaunchConfiguration('tf')
    depth = LaunchConfiguration('depth')
    return LaunchDescription([
        DeclareLaunchArgument('tf', default_value='false'),
        DeclareLaunchArgument('degree', default_value='0.0'),
        DeclareLaunchArgument('depth', default_value='false'),

        include('limo_base', 'limo_base.launch.py'),
        include('limo_description', 'load_urdf.launch.py'),
        include('ydlidar_ros2_driver', 'ydlidar.launch.py'),
        include('robot_localization', 'limo_ekf_launch.py'),
        include('orbbec_camera', 'astra_stereo_u3.launch.py', {
            'enable_color': 'true',
            'enable_depth': depth,
            'enable_point_cloud': depth,
            'enable_colored_point_cloud': 'false',
            'enable_ir': 'false',
        }),

        GroupAction([
            include('wego', 'camera_tilt_launch.py',
                    {'degree': LaunchConfiguration('degree')}),
        ], condition=IfCondition(tf)),
    ])
