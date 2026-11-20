"""Real rover: navigator, CAN bridge and the IMU, with the towed train and a site layered over survey.yaml.

The IMU is the imu_processing package (merged from the Controls repository): imu_raw_live reads the ICM-20948,
imu_cal_live applies the saved gyro bias and magnetometer calibration, heading_live fuses them into the Kalman
heading. The navigator takes its yaw rate from /imu/gyr_cal and its heading correction from /imu/heading_kalman.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CONFIG = os.path.join(get_package_share_directory('scout_navigation'), 'config')
SURVEY = os.path.join(CONFIG, 'survey.yaml')
PAYLOAD = os.path.join(CONFIG, 'payload.yaml')
ROVER = os.path.join(CONFIG, 'rover.yaml')


def generate_launch_description():
    """Navigator, CAN bridge and, unless imu:=false, the IMU nodes."""
    imu = LaunchConfiguration('imu')
    controls_imu = IfCondition(PythonExpression(
        ["'", imu, "' == 'true' and '", LaunchConfiguration('imu_source'), "' == 'controls'"]))
    return LaunchDescription([
        DeclareLaunchArgument('site', default_value=SURVEY,
                              description='Site parameters layered over survey.yaml, e.g. south_lawn.yaml'),
        DeclareLaunchArgument('payload', default_value=PAYLOAD,
                              description='Towed-train parameters; pass survey.yaml when driving without the carts'),
        DeclareLaunchArgument('imu', default_value='true',
                              description='Start the IMU; false runs on the chassis yaw rate'),
        DeclareLaunchArgument('imu_source', default_value='controls',
                              description='controls: imu_processing with Kalman heading; driver: gyro-only imu_driver'),
        DeclareLaunchArgument('imu_plot', default_value='false',
                              description='Open the imu_processing heading plot window (needs a display)'),
        # The rover's ICM-20948 answers on bus 1 at 0x68 (scan of 2026-11-20); imu_raw_live's own default is bus 2
        DeclareLaunchArgument('i2c_bus', default_value='1', description='I2C bus of the ICM-20948'),
        DeclareLaunchArgument('i2c_addr', default_value='104', description='I2C address of the ICM-20948 (0x68)'),
        DeclareLaunchArgument('calibration_dir', default_value=os.path.expanduser('~/imu_calibration'),
                              description='Folder with bias_file_gyro.csv and mag_calibration.csv'),

        Node(package='scout_navigation', executable='survey_navigator', name='survey_navigator',
             output='screen', emulate_tty=True,
             parameters=[SURVEY, ROVER, LaunchConfiguration('payload'), LaunchConfiguration('site')],
             # Calibrated gyro and accelerometer from imu_cal_live; imu_driver publishes imu/data itself
             remappings=[('imu/data', PythonExpression(
                 ["'/imu/gyr_cal' if '", LaunchConfiguration('imu_source'), "' == 'controls' else 'imu/data'"]))]),

        Node(package='scout_navigation', executable='scout_can_bridge', name='scout_can_bridge',
             output='screen', parameters=[SURVEY]),

        # Controls IMU chain: 100 Hz, units as in imu_processing (m/s^2, rad/s, T)
        Node(package='imu_processing', executable='imu_raw_live', name='imu_raw_live',
             parameters=[{'i2c_bus': ParameterValue(LaunchConfiguration('i2c_bus'), value_type=int),
                          'i2c_addr': ParameterValue(LaunchConfiguration('i2c_addr'), value_type=int)}],
             # It logs every reading twice a second; keep that in the log file, not the field console
             output='log', condition=controls_imu),
        Node(package='imu_processing', executable='imu_cal_live', name='imu_cal_live', output='screen',
             parameters=[{'calibration_dir': LaunchConfiguration('calibration_dir')}], condition=controls_imu),
        # Weighted towards the calibrated gyro: the rover's magnetometer scatters about 4 uT per reading against
        # a 12 uT horizontal field (replayed on the 20 Nov tests: 2-4x closer to the gyro than the defaults)
        Node(package='imu_processing', executable='heading_live', name='heading_live', output='screen',
             parameters=[{'angle_noise': 1e-4, 'bias_noise': 1e-6, 'mag_noise': 1.0,
                          'initial_bias_variance': 1e-6, 'init_samples': 30}],
             condition=controls_imu),
        Node(package='imu_processing', executable='print_headings', name='print_headings', output='screen',
             condition=IfCondition(PythonExpression(
                 ["'", LaunchConfiguration('imu_plot'), "' == 'true' and '", imu, "' == 'true' and '",
                  LaunchConfiguration('imu_source'), "' == 'controls'"]))),

        Node(package='scout_navigation', executable='imu_driver', name='imu_driver', output='screen',
             condition=IfCondition(PythonExpression(
                 ["'", imu, "' == 'true' and '", LaunchConfiguration('imu_source'), "' == 'driver'"])),
             parameters=[SURVEY]),
    ])
