"""Pose estimator: an extended Kalman filter for the rover's position and heading.

Predicts with wheel speed and yaw rate; corrects with GNSS position, GNSS direction of travel and, once its
offset to the map is known, the IMU's Kalman heading (imu_processing heading_live).
"""
import numpy as np

from .geodesy import wrap_to_pi


class PoseEstimator:
    """State is [east, north, heading] of the rover centre (base_link), in metres and radians."""

    def __init__(self, position_process_noise, yaw_process_noise, gnss_noise, course_noise,
                 minimum_course_speed, antenna_offset, initial_heading_noise):
        self.process_noise = np.diag([position_process_noise, position_process_noise,
                                      yaw_process_noise])
        self.gnss_noise = gnss_noise ** 2 * np.eye(2)
        self.course_noise = course_noise ** 2
        self.minimum_course_speed = minimum_course_speed
        self.antenna_offset = antenna_offset

        self.state = np.zeros(3)
        self.covariance = np.diag([gnss_noise ** 2, gnss_noise ** 2, initial_heading_noise ** 2])
        # Set by the navigator each cycle: a standing rover keeps its heading
        self.standing = False

    @property
    def position(self):
        """East and north of the rover centre (m)."""
        return self.state[:2]

    @property
    def heading(self):
        """Heading in radians anticlockwise from east."""
        return self.state[2]

    def lever_arm(self):
        """World-frame vector from base_link to the GNSS antenna."""
        heading = self.state[2]
        return self.antenna_offset * np.array([np.cos(heading), np.sin(heading)])

    def antenna_to_base(self, antenna_position):
        """Rover centre position for a given antenna position."""
        return antenna_position - self.lever_arm()

    def predict(self, speed, yaw_rate, time_step):
        """Moves the state forward by one time step using speed and yaw rate."""
        if self.standing:
            # Gyro bias and process noise would otherwise turn a rover that is not moving
            return
        heading = self.state[2]
        self.state += time_step * np.array([speed * np.cos(heading), speed * np.sin(heading), yaw_rate])
        self.state[2] = wrap_to_pi(self.state[2])

        jacobian = np.eye(3)
        jacobian[0, 2] = -speed * np.sin(heading) * time_step
        jacobian[1, 2] = speed * np.cos(heading) * time_step
        self.covariance = jacobian @ self.covariance @ jacobian.T + self.process_noise * time_step

    def _correct(self, innovation, observation_matrix, noise, hold_heading=False):
        """Kalman correction of state and covariance for one measurement; hold_heading leaves the heading as it is."""
        innovation_covariance = observation_matrix @ self.covariance @ observation_matrix.T + noise
        gain = self.covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        if hold_heading:
            gain[2] = 0.0
        self.state = self.state + gain @ innovation
        self.state[2] = wrap_to_pi(self.state[2])
        # Joseph form: also correct for the gain with the heading row removed
        residual = np.eye(3) - gain @ observation_matrix
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T

    def update_position(self, antenna_position):
        """The antenna is what GNSS measures, so its lever arm enters the observation model.

        While the rover stands, GNSS noise on the lever arm would read as a turn, so the heading is held.
        """
        arm_x, arm_y = self.lever_arm()
        observation_matrix = np.array([[1.0, 0.0, -arm_y], [0.0, 1.0, arm_x]])
        self._correct(antenna_position - self.state[:2] - self.lever_arm(), observation_matrix,
                      self.gnss_noise, hold_heading=self.standing)

    @property
    def heading_variance(self):
        """Variance of the heading estimate (rad^2)."""
        return self.covariance[2, 2]

    def update_course(self, antenna_velocity, yaw_rate):
        """GNSS course bounds yaw drift once moving; the antenna's swing about base_link is removed first.

        Returns the measured course of base_link (rad), or None when the rover is too slow for one or standing:
        at launch the GNSS bridge replays buffered velocities of up to 0.5 m/s with the rover still (23 Nov).
        """
        if self.standing:
            return None
        arm_x, arm_y = self.lever_arm()
        velocity = antenna_velocity - yaw_rate * np.array([-arm_y, arm_x])
        speed = np.linalg.norm(velocity)
        if speed < self.minimum_course_speed:
            return None
        course = np.arctan2(velocity[1], velocity[0])
        innovation = np.array([wrap_to_pi(course - self.state[2])])
        self._correct(innovation, np.array([[0.0, 0.0, 1.0]]),
                      np.array([[self.course_noise * (self.minimum_course_speed / speed) ** 2]]))
        return course

    def update_heading(self, heading, noise, gate):
        """Direct heading measurement (rad, map frame) with standard deviation noise.

        A measurement further than gate standard deviations of the innovation from the estimate is
        rejected, so a disturbed magnetometer cannot drag the heading away. Returns the innovation (rad)
        and whether it was applied.
        """
        innovation = wrap_to_pi(heading - self.state[2])
        if innovation ** 2 > gate ** 2 * (self.covariance[2, 2] + noise ** 2):
            return innovation, False
        self._correct(np.array([innovation]), np.array([[0.0, 0.0, 1.0]]), np.array([[noise ** 2]]))
        return innovation, True
