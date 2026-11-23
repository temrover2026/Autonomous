"""A standing rover keeps its heading; a moving one is still corrected."""
import numpy as np

from scout_navigation.pose_estimator import PoseEstimator


def estimator():
    """Antenna 0.3 m behind the rover centre, 2 cm GNSS noise, as on the rover."""
    return PoseEstimator(0.01, 0.01, 0.02, 0.2, 0.3, -0.3, 0.35)


def noisy_fixes(count):
    """Antenna positions of a rover standing at the origin facing east."""
    return np.array([-0.3, 0.0]) + np.random.default_rng(0).normal(0.0, 0.02, (count, 2))


def test_standing_rover_keeps_its_heading():
    standing = estimator()
    standing.standing = True
    for fix in noisy_fixes(1800):
        for _ in range(2):
            standing.predict(0.0, 0.004, 0.05)
        standing.update_position(fix)
    assert standing.heading == 0.0
    assert np.isclose(standing.heading_variance, 0.35 ** 2)
    assert np.linalg.norm(standing.position) < 0.01


def test_unheld_heading_drifts_on_the_same_fixes():
    free = estimator()
    for fix in noisy_fixes(1800):
        for _ in range(2):
            free.predict(0.0, 0.004, 0.05)
        free.update_position(fix)
    assert abs(free.heading) > 0.1


def test_moving_rover_is_still_corrected():
    moving = estimator()
    moving.state[2] = 0.2
    for step in range(200):
        moving.predict(1.0, 0.0, 0.05)
        moving.update_course(np.array([1.0, 0.0]), 0.0)
    assert abs(moving.heading) < 0.02


def test_standing_rover_ignores_gnss_course():
    """Buffered velocities replayed at launch must not turn a standing rover."""
    standing = estimator()
    standing.standing = True
    assert standing.update_course(np.array([-0.355, -0.367]), 0.0) is None
    assert standing.heading == 0.0
