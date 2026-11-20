#!/usr/bin/env bash
# One command per experiment.
#   ./run.sh field [site:=field_today.yaml] [imu:=false] [imu_plot:=true] [record:=false] [lidar_port:=/dev/ttyUSB1] ...
#   ./run.sh sim   [site:=south_lawn.yaml] [gz_args:='-r -s -v 2'] ...
#   ./run.sh stop  stops a running field or sim launch and every node it started
# A bare site or payload file name is looked up in the package's config folder.
set -e
source /opt/ros/humble/setup.bash
source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"
CONFIG="$(ros2 pkg prefix scout_navigation)/share/scout_navigation/config"

mode="$1"
shift || true

# What the launches start, matched on the installed programs so an editor or shell that merely
# mentions a node name is left alone; a node that outlives its launch keeps the IMU, LiDAR or GNSS port busy
PROGRAMS='/lib/[^ ]+/(survey_navigator|scout_can_bridge|imu_raw_live|imu_cal_live|heading_live|print_headings'
PROGRAMS+='|imu_driver|sllidar_node|nmea_tcpclient_driver|rosbridge_websocket|web_video_server|usb_cam_node_exe)( |$)'
PROGRAMS+='|http\.server 8000 --directory [^ ]*/scout_navigation/web|ros2 bag record -o |rtk_gps_bridge\.py'
PROGRAMS+='|scout_navigation\.ntrip_proxy|/ros2 launch scout_navigation '
running() {
    # Never this script or the shell that started it
    pgrep -f -- "$PROGRAMS" | grep -vxE "$$|$PPID" || true
}
if [[ $mode == stop ]]; then
    launches=$(pgrep -f -- '/ros2 launch scout_navigation ' | grep -vxE "$$|$PPID" || true)
    [[ -n $launches ]] && kill -INT $launches
    # On SIGINT the launch closes the bag and stops its nodes; give it up to 20 s
    for _ in $(seq 20); do
        [[ -z $(running) ]] && break
        sleep 1
    done
    left=$(running)
    if [[ -n $left ]]; then
        echo "Stopping $(echo $left | wc -w) processes that did not exit"
        kill $left 2> /dev/null || true
        sleep 2
        left=$(running)
        [[ -n $left ]] && kill -KILL $left 2> /dev/null || true
    fi
    echo "Stopped"
    exit 0
fi
arguments=()
for argument in "$@"; do
    # site:=name.yaml and payload:=name.yaml are looked up in the config folder
    if [[ $argument =~ ^(site|payload):=[^/]+$ ]]; then
        argument="${argument%%:=*}:=$CONFIG/${argument#*:=}"
    fi
    arguments+=("$argument")
done

case "$mode" in
    field)
        if ! ip link show can0 | grep -q "state UP"; then
            sudo ip link set can0 up type can bitrate 500000
        fi
        # imu_processing stops at start-up without its calibration files; say so before anything moves
        if [[ ! " $* " =~ " imu:=false " && ! " $* " =~ " imu_source:=driver " ]]; then
            calibration="$HOME/imu_calibration"
            for argument in "$@"; do
                [[ $argument == calibration_dir:=* ]] && calibration="${argument#*:=}"
            done
            for file in bias_file_gyro.csv mag_calibration.csv; do
                if [[ ! -f "$calibration/$file" ]]; then
                    echo "Missing $calibration/$file: calibrate the IMU first (imu_processing/README.md), or pass imu:=false"
                    exit 1
                fi
            done
        fi
        echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000  (over Tailscale: http://$(hostname):8000)"
        exec ros2 launch scout_navigation field.launch.py "${arguments[@]}"
        ;;
    sim)
        echo "Dashboard: http://localhost:8000"
        exec ros2 launch scout_navigation sim.launch.py "${arguments[@]}"
        ;;
    *)
        sed -n '2,6p' "$0"
        exit 1
        ;;
esac
