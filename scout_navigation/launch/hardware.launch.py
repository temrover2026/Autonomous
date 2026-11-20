"""Real rover: navigator, CAN bridge and the IMU, with the towed train and a site layered over survey.yaml.

The IMU is the imu_processing package (merged from the Controls repository): imu_raw_live reads the ICM-20948,
imu_cal_live applies the saved gyro bias and magnetometer calibration, heading_live fuses them into the Kalman
heading. The navigator takes its yaw rate from /imu/gyr_cal and its heading correction from /imu/heading_kalman.
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

CONFIG = os.path.join(get_package_share_directory('scout_navigation'), 'config')
SURVEY = os.path.join(CONFIG, 'survey.yaml')
PAYLOAD = os.path.join(CONFIG, 'payload.yaml')
ROVER = os.path.join(CONFIG, 'rover.yaml')


def find_imu_bus(address=0x68):
    """Number of the LattePanda I2C bus where the ICM-20948 answers (WHO_AM_I 0xEA), or None.

    The LattePanda has two identical DesignWare adapters and Linux numbers them in either order at boot,
    so the IMU has been seen on bus 1 and on bus 2. Only those adapters are probed.
    """
    try:
        from smbus2 import SMBus
    except ImportError:
        return None
    for path in sorted(glob.glob('/sys/bus/i2c/devices/i2c-*/name')):
        with open(path) as file:
            if 'DesignWare' not in file.read():
                continue
        bus = int(path.split('/')[-2].split('-')[1])
        try:
            with SMBus(bus) as smbus:
                if smbus.read_byte_data(address, 0x00) == 0xEA:
                    return bus
        except OSError:
            continue
    return None


def imu_reader(context):
    """imu_raw_live on the bus given, or on the bus found by find_imu_bus with i2c_bus:=auto."""
    bus = LaunchConfiguration('i2c_bus').perform(context)
    actions = []
    if bus == 'auto':
        found = find_imu_bus()
        bus = found if found is not None else 1
        actions.append(LogInfo(msg=f'ICM-20948 found on I2C bus {bus}' if found is not None
                               else 'ICM-20948 not found on any DesignWare I2C bus; trying bus 1'))
    actions.append(Node(package='imu_processing', executable='imu_raw_live', name='imu_raw_live',
                        parameters=[{'i2c_bus': int(bus),
                                     'i2c_addr': int(LaunchConfiguration('i2c_addr').perform(context))}],
                        # It logs every reading twice a second; keep that in the log file, not the field console
                        output='log'))
    return actions


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
        # auto: probe for the ICM-20948 (it moves between bus 1 and 2 across reboots); imu_raw_live's own default is 2
        DeclareLaunchArgument('i2c_bus', default_value='auto', description='I2C bus of the ICM-20948, or auto'),
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
        OpaqueFunction(function=imu_reader, condition=controls_imu),
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
