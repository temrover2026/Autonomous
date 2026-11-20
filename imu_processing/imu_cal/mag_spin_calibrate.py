#!/usr/bin/env python3
"""Flat (2D) magnetometer calibration with the IMU mounted on the rover.

A ground rover can only spin flat, so the 3D ellipsoid fit of calibrate_mag cannot be used once the IMU
is mounted. Spin the rover slowly in place 2-3 times while this records /imu/mag_raw; the horizontal
readings trace an ellipse whose centre is the rover's hard iron and whose shape is its soft iron.
The fit maps it back to a circle centred on zero, which is all atan2(mag_y, mag_x) needs.

Writes mag_calibration.csv in the format imu_cal_live reads (the old file is kept as a backup),
and stores the recording, the fit and the plots in ~/imu_data/<date_time>_mag_spin/.

    ros2 run imu_processing mag_spin_calibrate --ros-args -p duration:=90.0

The plot window shows the readings live; without a display the plot is only saved.
"""

import csv
import os
import shutil
import threading
import time

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import rclpy                        # noqa: E402
from rclpy.executors import SingleThreadedExecutor     # noqa: E402
from rclpy.node import Node         # noqa: E402
from rclpy.signals import SignalHandlerOptions         # noqa: E402
from sensor_msgs.msg import MagneticField              # noqa: E402

from imu_cal.calibrate_mag import write_calibration    # noqa: E402


def data_folder(test_name):
    """New ~/imu_data/<date_time>_<test_name>/ folder for one test's files."""
    folder = os.path.expanduser(
        os.path.join("~/imu_data", time.strftime("%Y%m%d_%H%M%S_") + test_name))
    os.makedirs(folder, exist_ok=True)
    return folder


def fit_ellipse_2d(x, y):
    """Fit the horizontal readings; returns (centre, 2x2 matrix, radius, report).

    The matrix maps the centred ellipse onto a circle of the ellipse's mean radius, so the
    calibrated field keeps its units (tesla).
    """
    data = np.column_stack((x, y))
    scale = float(np.median(np.linalg.norm(data - data.mean(axis=0), axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The magnetometer readings did not change: did the rover turn?")
    # Normalise around the mean so the least-squares fit is well conditioned
    shift = data.mean(axis=0)
    u, v = ((data - shift) / scale).T
    design = np.column_stack((u * u, u * v, v * v, u, v))
    (a, b, c, d, e), *_ = np.linalg.lstsq(design, np.ones(len(u)), rcond=None)
    quadratic = np.array([[a, b / 2.0], [b / 2.0, c]])
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("The readings do not form an ellipse: spin the rover through full turns.")
    centre = -0.5 * np.linalg.solve(quadratic, np.array([d, e]))
    k = 1.0 + float(centre @ quadratic @ centre)
    shape = quadratic / k
    values, vectors = np.linalg.eigh(shape)
    to_circle = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    # Geometric mean of the semi-axes, back in tesla
    radius = scale / float(np.sqrt(np.sqrt(np.prod(values))))
    matrix = radius * to_circle / scale
    centre_t = centre * scale + shift

    corrected = (matrix @ (data - centre_t).T).T
    lengths = np.linalg.norm(corrected, axis=1)
    spread = float(np.std(lengths) / np.mean(lengths))
    angles = np.degrees(np.arctan2(corrected[:, 1], corrected[:, 0]))
    covered = int(np.count_nonzero(np.histogram(angles, bins=36, range=(-180, 180))[0]))
    axes = 1.0 / np.sqrt(values) * scale
    report = {
        "samples": len(x),
        "hard_iron_x_uT": centre_t[0] * 1e6, "hard_iron_y_uT": centre_t[1] * 1e6,
        "radius_uT": radius * 1e6,
        "axis_ratio": float(axes.max() / axes.min()),
        "radius_spread": spread,
        "heading_coverage_deg": covered * 10,
    }
    if covered < 33:
        raise ValueError(f"Only {covered * 10} deg of heading covered: spin through at least one full turn.")
    if spread > 0.1:
        raise ValueError(f"Poor fit (radius spread {spread:.2f}): spin slower, on level ground, away from steel.")
    return centre_t, matrix, radius, report


class MagSpinRecorder(Node):
    def __init__(self):
        super().__init__("mag_spin_calibrate")
        self.declare_parameter("duration", 90.0)
        self.declare_parameter("calibration_dir", os.path.expanduser("~/imu_calibration"))
        self.declare_parameter("mag_topic", "/imu/mag_raw")
        self.duration = float(self.get_parameter("duration").value)
        self.calibration_dir = self.get_parameter("calibration_dir").value
        self.lock = threading.Lock()
        self.samples = []       # (time, x, y, z) in seconds and tesla
        self.start = time.monotonic()
        self.create_subscription(MagneticField, self.get_parameter("mag_topic").value, self.on_mag, 50)
        self.get_logger().info(f"Recording for {self.duration:.0f} s: spin the rover slowly in place 2-3 times")

    def on_mag(self, msg):
        m = msg.magnetic_field
        with self.lock:
            self.samples.append((time.monotonic() - self.start, m.x, m.y, m.z))

    def snapshot(self):
        with self.lock:
            return np.array(self.samples) if self.samples else np.empty((0, 4))


def draw(figure, axes, data, remaining, fit=None):
    """Raw horizontal readings (left) and their heading over time (right); with the fit once done."""
    for ax in axes:
        ax.clear()
    scatter, timeline = axes
    if len(data):
        scatter.plot(data[:, 1] * 1e6, data[:, 2] * 1e6, ".", ms=2, label="raw")
        timeline.plot(data[:, 0], np.degrees(np.arctan2(data[:, 2], data[:, 1])), ".", ms=2, label="raw heading")
    if fit is not None:
        centre, matrix, radius, _ = fit
        circle = np.linspace(0, 2 * np.pi, 200)
        ellipse = np.linalg.solve(matrix, radius * np.vstack((np.cos(circle), np.sin(circle)))).T + centre
        scatter.plot(ellipse[:, 0] * 1e6, ellipse[:, 1] * 1e6, "-", lw=1.5, label="fitted ellipse")
        corrected = (matrix @ (data[:, 1:3] - centre).T).T
        scatter.plot(corrected[:, 0] * 1e6, corrected[:, 1] * 1e6, ".", ms=2, label="calibrated")
        scatter.plot([centre[0] * 1e6], [centre[1] * 1e6], "x", ms=10, label="hard iron")
        timeline.plot(data[:, 0], np.degrees(np.arctan2(corrected[:, 1], corrected[:, 0])), ".", ms=2,
                      label="calibrated heading")
    scatter.plot([0], [0], "k+", ms=10)
    scatter.set_aspect("equal", adjustable="datalim")
    scatter.set_xlabel("mag x (uT)")
    scatter.set_ylabel("mag y (uT)")
    scatter.set_title("Horizontal field: should be a circle round + after calibration")
    scatter.grid(True)
    scatter.legend(loc="upper right", fontsize=8)
    timeline.set_xlabel("time (s)")
    timeline.set_ylabel("atan2(y, x) (deg)")
    timeline.set_ylim(-180, 180)
    timeline.set_title("Magnetometer heading")
    timeline.grid(True)
    if len(data):
        timeline.legend(loc="upper right", fontsize=8)
    status = f"recording: {remaining:.0f} s left, {len(data)} samples" if remaining > 0 else "recording done"
    figure.suptitle(f"Magnetometer spin calibration ({status})")


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = MagSpinRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    folder = data_folder("mag_spin")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        plt.ion()
        figure.show()
    try:
        while (remaining := node.duration - (time.monotonic() - node.start)) > 0:
            draw(figure, axes, node.snapshot(), remaining)
            if interactive:
                figure.canvas.draw_idle()
                figure.canvas.start_event_loop(0.5)
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped early: fitting what was recorded")
    executor.shutdown(timeout_sec=1.0)

    data = node.snapshot()
    with open(os.path.join(folder, "mag_spin.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "mag_x", "mag_y", "mag_z"])
        writer.writerows(data.tolist())
    try:
        fit = fit_ellipse_2d(data[:, 1], data[:, 2]) if len(data) >= 30 else None
        if fit is None:
            raise ValueError(f"Only {len(data)} samples: is imu_raw_live running?")
    except ValueError as error:
        draw(figure, axes, data, 0)
        figure.savefig(os.path.join(folder, "mag_spin.png"), dpi=110)
        node.get_logger().error(f"Calibration NOT written: {error}  Data: {folder}")
        rclpy.shutdown()
        return

    centre, matrix, radius, report = fit
    draw(figure, axes, data, 0, fit)
    figure.savefig(os.path.join(folder, "mag_spin.png"), dpi=110)

    # z is not calibrated (a flat spin cannot see it) and does not enter the heading: passed through
    offsets = np.array([centre[0], centre[1], 0.0])
    full = np.eye(3)
    full[:2, :2] = matrix
    target = os.path.join(node.calibration_dir, "mag_calibration.csv")
    if os.path.exists(target):
        shutil.copy2(target, os.path.join(folder, "previous_mag_calibration.csv"))
        shutil.copy2(target, target.replace(".csv", time.strftime("_%Y%m%d_%H%M%S.bak.csv")))
    write_calibration(target, offsets, full)
    shutil.copy2(target, os.path.join(folder, "mag_calibration.csv"))
    with open(os.path.join(folder, "fit_report.txt"), "w") as file:
        for key, value in report.items():
            file.write(f"{key}: {value:.3f}\n" if isinstance(value, float) else f"{key}: {value}\n")
    summary = ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in report.items())
    node.get_logger().info(f"Calibration written to {target} ({summary}). Data and plot: {folder}")
    if interactive:
        figure.canvas.start_event_loop(5.0)     # leave the result on screen briefly
    rclpy.shutdown()


if __name__ == "__main__":
    main()
