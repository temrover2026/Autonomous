#!/usr/bin/env python3
"""Flat (2D) magnetometer calibration with the IMU mounted on the rover.

A ground rover can only spin flat, so the 3D ellipsoid fit of calibrate_mag cannot be used once the IMU
is mounted. Spin the rover slowly in place 2-3 times while this records /imu/mag_raw; the horizontal
readings trace an ellipse whose centre is the rover's hard iron and whose shape is its soft iron.
The fit maps it back to a circle centred on zero, which is all atan2(mag_y, mag_x) needs.

Writes mag_calibration.csv in the format imu_cal_live reads (the old file is kept as a backup),
and stores the recording, the fit and the plots in ~/imu_data/<date_time>_mag_spin/.

    ros2 run imu_processing mag_spin_calibrate --ros-args -p duration:=90.0

To fit a recording again without spinning: -p from_csv:=~/imu_data/<date_time>_mag_spin/mag_spin.csv

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


def fit_conic_2d(points):
    """Least-squares ellipse through 2D points; returns (centre, matrix to a circle, radius)."""
    scale = float(np.median(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The magnetometer readings did not change: did the rover turn?")
    # Normalise around the mean so the least-squares fit is well conditioned
    shift = points.mean(axis=0)
    u, v = ((points - shift) / scale).T
    design = np.column_stack((u * u, u * v, v * v, u, v))
    (a, b, c, d, e), *_ = np.linalg.lstsq(design, np.ones(len(u)), rcond=None)
    quadratic = np.array([[a, b / 2.0], [b / 2.0, c]])
    if np.any(np.linalg.eigvalsh(quadratic) <= 0.0):
        raise ValueError("The readings do not form an ellipse: spin the rover through full turns.")
    centre = -0.5 * np.linalg.solve(quadratic, np.array([d, e]))
    shape = quadratic / (1.0 + float(centre @ quadratic @ centre))
    values, vectors = np.linalg.eigh(shape)
    if np.any(values <= 0.0):
        raise ValueError("The readings do not form an ellipse: spin the rover through full turns.")
    # Maps the ellipse onto a circle of its geometric mean radius, so the field keeps its units (tesla)
    radius = scale / float(np.sqrt(np.sqrt(np.prod(values))))
    matrix = radius * (vectors @ np.diag(np.sqrt(values)) @ vectors.T) / scale
    return centre * scale + shift, matrix, radius


def sector_means(points, centre, sectors=36, minimum=3):
    """Mean reading in each direction sector around centre.

    The ICM-20948 magnetometer on the rover scatters by about 5 uT per reading, half the radius of the
    circle; averaging every reading taken while the rover faced the same 10 deg sector removes most of it.
    """
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    index = ((angles + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    return np.array([points[index == k].mean(axis=0) for k in range(sectors) if np.sum(index == k) >= minimum])


def fit_ellipse_2d(x, y):
    """Fit the horizontal readings; returns (centre, 2x2 matrix, radius, report).

    The ellipse is fitted to direction-sector averages (centre refined over three passes, starting from
    the mean, which is the centre when whole turns are recorded); the shape check uses the averages and
    the coverage check uses every reading.
    """
    data = np.column_stack((x, y))
    centre = data.mean(axis=0)
    for _ in range(3):
        means = sector_means(data, centre)
        if len(means) < 24:
            raise ValueError(f"Readings in only {len(means)} of 36 directions: spin through at least one full turn.")
        centre, matrix, radius = fit_conic_2d(means)

    corrected = (matrix @ (data - centre).T).T
    angles = np.degrees(np.arctan2(corrected[:, 1], corrected[:, 0]))
    covered = int(np.count_nonzero(np.histogram(angles, bins=36, range=(-180, 180))[0] >= 3))
    mean_lengths = np.linalg.norm((matrix @ (means - centre).T).T, axis=1)
    spread = float(np.std(mean_lengths) / np.mean(mean_lengths))
    sample_noise = float(np.std(np.linalg.norm(corrected, axis=1) - radius))
    axes = np.linalg.svd(np.linalg.inv(matrix), compute_uv=False)
    report = {
        "samples": len(x), "direction_sectors": len(means),
        "hard_iron_x_uT": centre[0] * 1e6, "hard_iron_y_uT": centre[1] * 1e6,
        "radius_uT": radius * 1e6,
        "axis_ratio": float(axes.max() / axes.min()),
        "radius_spread": spread,
        "reading_noise_uT": sample_noise * 1e6,
        "heading_coverage_deg": covered * 10,
    }
    if covered < 33:
        raise ValueError(f"Only {covered * 10} deg of heading covered: spin through at least one full turn.")
    if spread > 0.1:
        raise ValueError(f"Poor fit (radius spread {spread:.2f}): spin slower, on level ground, away from steel.")
    return centre, matrix, radius, report


class MagSpinRecorder(Node):
    def __init__(self):
        super().__init__("mag_spin_calibrate")
        self.declare_parameter("duration", 90.0)
        # Seconds of "get ready" before recording starts
        self.declare_parameter("countdown", 5.0)
        self.declare_parameter("calibration_dir", os.path.expanduser("~/imu_calibration"))
        self.declare_parameter("mag_topic", "/imu/mag_raw")
        # A saved mag_spin.csv to fit instead of recording
        self.declare_parameter("from_csv", "")
        self.from_csv = os.path.expanduser(self.get_parameter("from_csv").value)
        self.duration = float(self.get_parameter("duration").value)
        self.calibration_dir = self.get_parameter("calibration_dir").value
        self.lock = threading.Lock()
        self.samples = []       # (time, x, y, z) in seconds and tesla
        self.start = time.monotonic() + float(self.get_parameter("countdown").value)
        self.create_subscription(MagneticField, self.get_parameter("mag_topic").value, self.on_mag, 50)
        self.get_logger().info(f"Recording for {self.duration:.0f} s after the countdown: "
                               "spin the rover slowly in place, at least one full turn")

    def on_mag(self, msg):
        m = msg.magnetic_field
        t = time.monotonic() - self.start
        if t < 0.0:
            return      # still counting down
        with self.lock:
            self.samples.append((t, m.x, m.y, m.z))

    def snapshot(self):
        with self.lock:
            return np.array(self.samples) if self.samples else np.empty((0, 4))


def draw(figure, axes, data, remaining, fit=None, stale_since=None, duration=None):
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
    if len(data):
        scatter.legend(loc="upper right", fontsize=8)
    timeline.set_xlabel("time (s)")
    timeline.set_ylabel("atan2(y, x) (deg)")
    timeline.set_ylim(-180, 180)
    timeline.set_title("Magnetometer heading")
    timeline.grid(True)
    if len(data):
        timeline.legend(loc="upper right", fontsize=8)
    status = f"recording: {remaining:.0f} s left, {len(data)} samples" if remaining > 0 else "recording done"
    elapsed = (time.monotonic() - stale_since) if stale_since is not None else 0.0
    if remaining > 0 and elapsed > 2.0:
        status = f"NO DATA for {elapsed:.0f} s: is imu_raw_live running? ({remaining:.0f} s left)"
    figure.suptitle(f"Magnetometer spin calibration ({status})")
    # Large countdown in the middle of the window
    for text in list(figure.texts):
        if text.get_gid() == "countdown":
            text.remove()
    if duration is not None and remaining > duration:
        banner, colour = f"GET READY\nstart spinning in {remaining - duration:.0f}", "tab:red"
    elif remaining > 0:
        banner, colour = f"SPIN SLOWLY\n{remaining:.0f} s", "tab:green"
    else:
        banner = None
    if banner:
        figure.text(0.5, 0.5, banner, ha="center", va="center", fontsize=40, color=colour, alpha=0.35,
                    fontweight="bold", gid="countdown")


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = MagSpinRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    folder = data_folder("mag_spin")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    interactive = matplotlib.get_backend().lower() != "agg" and not node.from_csv
    if interactive:
        plt.ion()
        figure.show()
    if node.from_csv:
        node.samples = np.loadtxt(node.from_csv, delimiter=",", skiprows=1).tolist()
        node.duration, node.start = 0.0, time.monotonic()
        node.get_logger().info(f"Fitting {len(node.samples)} saved readings from {node.from_csv}")
    last_count, stale_since = 0, node.start
    announced = None
    try:
        while (remaining := node.duration - (time.monotonic() - node.start)) > 0:
            data = node.snapshot()
            if len(data) != last_count:
                last_count, stale_since = len(data), time.monotonic()
            draw(figure, axes, data, remaining, stale_since=stale_since if remaining <= node.duration else None,
                 duration=node.duration)
            # The same countdown in the terminal, once per second
            second = int(remaining - node.duration) + 1 if remaining > node.duration else int(remaining)
            if second != announced:
                announced = second
                if remaining > node.duration:
                    node.get_logger().info(f"GET READY: start spinning in {second} s")
                elif second % 5 == 0 or second <= 3:
                    node.get_logger().info(f"Recording: {second} s left, {len(data)} samples")
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
