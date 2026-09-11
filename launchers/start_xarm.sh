#!/bin/bash

export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash
source "$HOME/Workspace/spinerobot_ros2/install/setup.bash"

echo "======================================"
echo " BART LAB - Starting xArm 6"
echo " Robot IP: 192.168.1.231"
echo "======================================"
echo

ros2 launch xarm_api xarm6_driver.launch.py \
    robot_ip:=192.168.1.231

echo
echo "xArm driver stopped."
read -p "Press Enter to close..."
