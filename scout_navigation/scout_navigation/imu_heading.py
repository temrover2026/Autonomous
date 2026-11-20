"""Decides when the IMU Kalman heading (imu_processing heading_live) may correct the navigator's EKF.

The IMU heading is magnetic, in the IMU's own frame: map heading = IMU heading + offset, where the offset holds
the magnetic declination, the IMU's mounting yaw and the grid frame. It goes through three stages:

1. learning: the offset is averaged from GNSS course while the rover drives straight (or set with mode fixed);
2. checking: on the next turn the IMU heading must turn the same way and by the same amount as the EKF, which
   catches a magnetometer whose axes or sign do not match the gyro before it can pull the EKF off;
3. active: the EKF uses it. Persistent rejections send a learned offset back to stage 1.
"""
import numpy as np

from .geodesy import wrap_to_pi


class ImuHeadingAligner:
    """Offset to the map and the learning, checking and active stages of the IMU heading."""

    def __init__(self, mode, offset, samples, learning_yaw_rate, turn_rate, minimum_turn, turn_tolerance,
                 rejection_limit):
        self.mode = mode
        self.samples_needed = samples
        self.learning_yaw_rate = learning_yaw_rate
        self.turn_rate = turn_rate
        self.minimum_turn = minimum_turn
        self.turn_tolerance = turn_tolerance
        self.rejection_limit = rejection_limit
        self.offset = offset if mode == 'fixed' else None
        self.confirmed = False
        self.samples = []
        # During a turn: [IMU heading turned, EKF heading turned, last IMU heading, last EKF heading]
        self.turn = None
        self.rejections = 0

    @property
    def stage(self):
        """disabled, learning, checking or active."""
        if self.mode == 'disabled':
            return 'disabled'
        if self.offset is None:
            return 'learning'
        return 'active' if self.confirmed else 'checking'

    @property
    def progress(self):
        """Fraction of the course samples collected while learning."""
        return len(self.samples) / self.samples_needed

    def to_map(self, imu_heading):
        """IMU heading in the map frame (rad), or None while it or the offset is unknown."""
        if imu_heading is None or self.offset is None:
            return None
        return wrap_to_pi(imu_heading + self.offset)

    def add_course(self, course, imu_heading, yaw_rate):
        """One GNSS course (rad) with the IMU heading at the same moment. Returns a log line or None."""
        if self.stage != 'learning' or abs(yaw_rate) > self.learning_yaw_rate:
            return None
        self.samples.append(wrap_to_pi(course - imu_heading))
        if len(self.samples) < self.samples_needed:
            return None
        mean = np.mean(np.exp(1j * np.array(self.samples)))
        self.samples = []
        # A resultant length of 0.95 is a spread of about 18 deg: the magnetometer was disturbed, try again
        if abs(mean) < 0.95:
            return f'IMU heading offset samples too scattered ({abs(mean):.2f}); relearning'
        self.offset = float(np.angle(mean))
        return (f'IMU heading offset learned: {np.degrees(self.offset):+.1f} deg; '
                'checking it on the next turn before the EKF uses it')

    def track_turn(self, imu_heading, ekf_heading, yaw_rate):
        """Called every control step while checking. Returns (passed, log line) when a turn ends, else None."""
        if self.stage != 'checking':
            self.turn = None
            return None
        if abs(yaw_rate) > self.turn_rate:
            if self.turn is None:
                self.turn = [0.0, 0.0, imu_heading, ekf_heading]
            # Summed step by step so turns past 180 deg are measured whole
            self.turn[0] += wrap_to_pi(imu_heading - self.turn[2])
            self.turn[1] += wrap_to_pi(ekf_heading - self.turn[3])
            self.turn[2:] = [imu_heading, ekf_heading]
            return None
        if self.turn is None:
            return None
        imu_turned, ekf_turned = self.turn[:2]
        self.turn = None
        if abs(ekf_turned) < self.minimum_turn:
            return None
        report = (f'IMU heading turned {np.degrees(imu_turned):+.0f} deg '
                  f'while the EKF turned {np.degrees(ekf_turned):+.0f}')
        if abs(imu_turned - ekf_turned) < self.turn_tolerance:
            self.confirmed = True
            return True, report + ': the IMU heading now corrects the EKF'
        return False, (report + ': not used. Check the magnetometer axes and sign against the gyro, '
                       'and the magnetometer calibration')

    def record(self, applied):
        """Counts consecutive rejections by the EKF gate. Returns a log line when a learned offset is dropped."""
        self.rejections = 0 if applied else self.rejections + 1
        if self.rejections < self.rejection_limit or self.mode != 'learn':
            return None
        self.offset, self.confirmed, self.rejections = None, False, 0
        return f'IMU heading rejected {self.rejection_limit} times in a row; relearning its offset'
