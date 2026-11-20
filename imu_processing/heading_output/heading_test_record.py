#!/usr/bin/env python3
"""Record and plot a heading test: headings, gyro rates and magnetometer field while the rover is turned.

Plots live (on the screen the node can reach) and stores heading_test.csv and heading_test.png in
~/imu_data/<date_time>_heading_test/. Needs imu_raw_live, imu_cal_live and heading_live running.

    ros2 run imu_processing heading_test_record --ros-args -p duration:=30.0 -p countdown:=5.0

Panels: the gyro, magnetometer and Kalman headings (degrees -180 to 180, anticlockwise positive; the
gyro is shifted to start on the Kalman heading so the three overlap), the calibrated gyro rates, and
the raw and calibrated magnetometer field. All three headings must move together.
"""

import csv
import math
import os
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
from sensor_msgs.msg import Imu, MagneticField         # noqa: E402
from std_msgs.msg import Float64    # noqa: E402

from imu_cal.mag_spin_calibrate import data_folder     # noqa: E402

HEADINGS = ("gyro", "mag", "kalman")
COLUMNS = ("t", "gyro_heading", "mag_heading", "kalman_heading",
           "gyro_x", "gyro_y", "gyro_z",
           "mag_raw_x", "mag_raw_y", "mag_raw_z",
           "mag_cal_x", "mag_cal_y", "mag_cal_z")


class HeadingTestRecorder(Node):
    def __init__(self):
        super().__init__("heading_test_record")
        self.declare_parameter("duration", 30.0)
        self.declare_parameter("countdown", 5.0)
        self.declare_parameter("rate", 30.0)
        self.duration = float(self.get_parameter("duration").value)
        self.lock = threading.Lock()
        self.latest = {}
        self.rows = []
        self.start = time.monotonic() + float(self.get_parameter("countdown").value)
        topics = {"gyro": "/imu/heading/gyro", "mag": "/imu/heading/mag", "kalman": "/imu/heading_kalman"}
        for name, topic in topics.items():
            self.create_subscription(Float64, topic, lambda m, n=name: self.store(n, (m.data,)), 10)
        self.create_subscription(Imu, "/imu/gyr_cal", lambda m: self.store("rates", self.xyz(m.angular_velocity)), 10)
        self.create_subscription(MagneticField, "/imu/mag_raw",
                                 lambda m: self.store("mag_raw", self.xyz(m.magnetic_field)), 10)
        self.create_subscription(MagneticField, "/imu/mag_cal",
                                 lambda m: self.store("mag_cal", self.xyz(m.magnetic_field)), 10)
        self.create_timer(1.0 / float(self.get_parameter("rate").value), self.sample)
        self.get_logger().info(f"Recording headings, gyro and magnetometer for {self.duration:.0f} s "
                               "after the countdown")

    @staticmethod
    def xyz(vector):
        return (vector.x, vector.y, vector.z)

    def store(self, name, values):
        with self.lock:
            self.latest[name] = values

    def sample(self):
        """One row with the latest of everything, once everything has arrived."""
        t = time.monotonic() - self.start
        with self.lock:
            if t < 0.0 or len(self.latest) < 6:
                return
            self.rows.append((t, *(self.latest[n][0] for n in HEADINGS), *self.latest["rates"],
                              *self.latest["mag_raw"], *self.latest["mag_cal"]))

    def snapshot(self):
        with self.lock:
            return np.array(self.rows) if self.rows else np.empty((0, len(COLUMNS)))


def draw(figure, axes, data, remaining, duration):
    headings, rates, mag_raw, mag_cal = axes
    for ax in axes:
        ax.clear()
    if len(data):
        t = data[:, 0]
        unwrapped = {name: np.degrees(np.unwrap(data[:, i])) for i, name in enumerate(HEADINGS, start=1)}
        # The gyro heading counts from wherever heading_live started; start it on the Kalman heading
        unwrapped["gyro"] += unwrapped["kalman"][0] - unwrapped["gyro"][0]
        for name in HEADINGS:
            wrapped = (unwrapped[name] + 180.0) % 360.0 - 180.0
            # Lift the pen where a heading crosses +-180, so the line does not jump across the plot
            wrapped[1:][np.abs(np.diff(wrapped)) > 180.0] = np.nan
            headings.plot(t, wrapped, label=name, lw=1.2)
        for i, axis in enumerate("xyz"):
            rates.plot(t, np.degrees(data[:, 4 + i]), label=f"gyro {axis}", lw=1)
            mag_raw.plot(t, data[:, 7 + i] * 1e6, label=f"raw {axis}", lw=1)
            mag_cal.plot(t, data[:, 10 + i] * 1e6, label=f"calibrated {axis}", lw=1)
        for ax in axes:
            ax.legend(loc="upper left", fontsize=8, ncol=3)
    headings.set_ylabel("heading (deg)")
    headings.set_ylim(-180, 180)
    headings.set_yticks(range(-180, 181, 45))
    headings.set_title("Headings -180 to 180 deg (anticlockwise +; gyro aligned to Kalman at the start)")
    rates.set_ylabel("rate (deg/s)")
    rates.set_title("Gyro (calibrated)")
    mag_raw.set_ylabel("field (uT)")
    mag_raw.set_title("Magnetometer raw")
    mag_cal.set_ylabel("field (uT)")
    mag_cal.set_title("Magnetometer calibrated (x, y used for heading)")
    mag_cal.set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, max(duration, 1.0))
    for text in list(figure.texts):
        if text.get_gid() == "countdown":
            text.remove()
    if remaining > duration:
        status = f"GET READY: starting in {remaining - duration:.0f} s"
        figure.text(0.5, 0.5, f"GET READY\n{remaining - duration:.0f}", ha="center", va="center", fontsize=40,
                    color="tab:red", alpha=0.35, fontweight="bold", gid="countdown")
    elif remaining > 0:
        status = f"recording: {remaining:.0f} s left"
        figure.text(0.5, 0.5, f"TURN NOW\n{remaining:.0f} s", ha="center", va="center", fontsize=40,
                    color="tab:green", alpha=0.25, fontweight="bold", gid="countdown")
    else:
        status = "recording done"
    figure.suptitle(f"IMU heading test ({status})")


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HeadingTestRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    folder = data_folder("heading_test")
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        plt.ion()
        figure.show()
    announced = None
    try:
        while (remaining := node.duration - (time.monotonic() - node.start)) > 0:
            draw(figure, axes, node.snapshot(), remaining, node.duration)
            second = math.ceil(remaining - node.duration) if remaining > node.duration else int(remaining)
            if second != announced:
                announced = second
                if remaining > node.duration:
                    node.get_logger().info(f"GET READY: starting in {second} s")
                elif second % 5 == 0:
                    node.get_logger().info(f"Recording: {second} s left")
            if interactive:
                figure.canvas.draw_idle()
                figure.canvas.start_event_loop(0.3)
            else:
                time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    executor.shutdown(timeout_sec=1.0)

    data = node.snapshot()
    draw(figure, axes, data, 0, node.duration)
    figure.savefig(os.path.join(folder, "heading_test.png"), dpi=110)
    with open(os.path.join(folder, "heading_test.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(data.tolist())
    if len(data) > 1:
        turned = {name: math.degrees(float(np.unwrap(data[:, i])[-1] - data[0, i]))
                  for i, name in enumerate(HEADINGS, start=1)}
        node.get_logger().info("Turned over the test: " + ", ".join(f"{n} {v:+.0f} deg" for n, v in turned.items()))
    node.get_logger().info(f"{len(data)} samples. Data and plot: {folder}")
    if interactive:
        figure.canvas.start_event_loop(5.0)     # leave the result on screen briefly
    rclpy.shutdown()


if __name__ == "__main__":
    main()
