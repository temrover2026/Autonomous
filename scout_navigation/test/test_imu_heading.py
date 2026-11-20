"""IMU heading stages (learning, checking, active) and the EKF heading update."""
import numpy as np

from scout_navigation.imu_heading import ImuHeadingAligner
from scout_navigation.pose_estimator import PoseEstimator


def aligner(mode='learn'):
    """Five course samples to learn; a 3-rejection limit keeps the tests short."""
    return ImuHeadingAligner(mode, 0.5, 5, 0.1, 0.2, 0.5, 0.26, 3)


def turn(aligner_under_test, imu_direction):
    """Turns the EKF 170 deg left across +-180; the IMU heading turns imu_direction times as far."""
    for step in range(60):
        aligner_under_test.track_turn(wrap(3.0 + imu_direction * step * 0.05), wrap(2.0 + step * 0.05), 0.5)
    return aligner_under_test.track_turn(0.0, 0.0, 0.0)


def wrap(angle):
    return float(np.angle(np.exp(1j * angle)))


def test_offset_learned_from_straight_driving_only():
    learning = aligner()
    learning.add_course(1.0, -0.2, 0.5)
    assert learning.progress == 0
    for _ in range(5):
        learning.add_course(1.0, -0.2, 0.0)
    assert np.isclose(learning.offset, 1.2) and learning.stage == 'checking'


def test_scattered_samples_relearn():
    learning = aligner()
    for course in (0.0, 1.0, 2.0, 3.0, -2.0):
        learning.add_course(course, 0.0, 0.0)
    assert learning.stage == 'learning'


def test_turn_check_passes_and_fails():
    good, flipped = aligner(), aligner()
    good.offset = flipped.offset = 1.0
    assert turn(good, 1.0)[0] and good.stage == 'active'
    assert not turn(flipped, -1.0)[0] and flipped.stage == 'checking'


def test_rejections_relearn_only_a_learned_offset():
    learned, fixed = aligner(), aligner('fixed')
    learned.offset, learned.confirmed, fixed.confirmed = 1.0, True, True
    for _ in range(3):
        learned.record(False)
        fixed.record(False)
    assert learned.stage == 'learning' and fixed.stage == 'active'


def test_disabled_never_learns():
    disabled = aligner('disabled')
    assert disabled.add_course(1.0, 0.0, 0.0) is None and disabled.stage == 'disabled'


def test_ekf_heading_update_and_gate():
    estimator = PoseEstimator(0.01, 0.01, 0.02, 0.2, 0.3, -0.3, 0.35)
    _, applied = estimator.update_heading(0.2, 0.1, 3.0)
    assert applied and 0.0 < estimator.heading < 0.2
    _, applied = estimator.update_heading(np.pi, 0.1, 3.0)
    assert not applied
    assert estimator.update_course(np.array([0.01, 0.0]), 0.0) is None
