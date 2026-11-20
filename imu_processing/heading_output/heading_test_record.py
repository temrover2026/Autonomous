#!/usr/bin/env python3
"""Record and plot a heading test: gyro, magnetometer and Kalman headings while the rover is turned.

Plots live (on the rover's screen when there is one) and stores heading_test.csv and heading_test.png
in ~/imu_data/<date_time>_heading_test/. Needs imu_raw_live, imu_cal_live and heading_live running.

    ros2 run imu_processing heading_test_record --ros-args -p duration:=60.0

Headings are in degrees, anticlockwise positive (the topics' convention), unwrapped so a full turn
shows as a continuous line. All three must move together for the magnetometer to be trusted.
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
from sensor_msgs.msg import Imu     # noqa: E402
from std_msgs.msg import Float64    # noqa: E402

from imu_cal.mag_spin_calibrate import data_folder     # noqa: E402

SERIES = ("gyro", "mag", "kalman")


class HeadingTestRecorder(Node):
    def __init__(self):
        super().__init__("heading_test_record")
        self.declare_parameter("duration", 60.0)
        self.duration = float(self.get_parameter("duration").value)
        self.lock = threading.Lock()
        self.latest = {}
        self.rows = []      # time, yaw rate, gyro, mag, kalman (rad)
        self.start = time.monotonic()
        topics = {"gyro": "/imu/heading/gyro", "mag": "/imu/heading/mag", "kalman": "/imu/heading_kalman"}
        for name, topic in topics.items():
            self.create_subscription(Float64, topic, lambda m, n=name: self.on_heading(n, m.data), 10)
        self.create_subscription(Imu, "/imu/gyr_cal", self.on_gyro, 10)
        self.create_timer(0.05, self.sample)
        self.get_logger().info(f"Recording headings for {self.duration:.0f} s")

    def on_heading(self, name, value):
        with self.lock:
            self.latest[name] = value

    def on_gyro(self, msg):
        with self.lock:
            self.latest["rate"] = msg.angular_velocity.z

    def sample(self):
        """20 Hz: one row with the latest of everything."""
        with self.lock:
            if len(self.latest) == 4:
                self.rows.append((time.monotonic() - self.start, self.latest["rate"],
                                  *(self.latest[name] for name in SERIES)))

    def snapshot(self):
        with self.lock:
            return np.array(self.rows) if self.rows else np.empty((0, 5))


def draw(figure, axes, data, remaining):
    headings, rate = axes
    headings.clear()
    rate.clear()
    if len(data):
        for column, name in enumerate(SERIES, start=2):
            headings.plot(data[:, 0], np.degrees(np.unwrap(data[:, column])), label=name)
        rate.plot(data[:, 0], np.degrees(data[:, 1]), color="tab:gray")
    headings.set_ylabel("heading (deg, anticlockwise +, unwrapped)")
    headings.set_title("Gyro, magnetometer and Kalman heading: should move together")
    headings.grid(True)
    if len(data):
        headings.legend(loc="upper left")
    rate.set_xlabel("time (s)")
    rate.set_ylabel("yaw rate (deg/s)")
    rate.grid(True)
    status = f"recording: {remaining:.0f} s left" if remaining > 0 else "recording done"
    figure.suptitle(f"Heading test ({status})")


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HeadingTestRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    folder = data_folder("heading_test")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
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
        pass
    executor.shutdown(timeout_sec=1.0)

    data = node.snapshot()
    draw(figure, axes, data, 0)
    figure.savefig(os.path.join(folder, "heading_test.png"), dpi=110)
    with open(os.path.join(folder, "heading_test.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "yaw_rate", "gyro_heading", "mag_heading", "kalman_heading"])
        writer.writerows(data.tolist())
    if len(data) > 1:
        turned = {name: math.degrees(float(np.unwrap(data[:, column])[-1] - data[0, column]))
                  for column, name in enumerate(SERIES, start=2)}
        node.get_logger().info("Turned over the test: " + ", ".join(f"{n} {v:+.0f} deg" for n, v in turned.items()))
    node.get_logger().info(f"Data and plot: {folder}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
