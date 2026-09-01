# SpineRobot ROS 2

ROS 2 Jazzy backend for the SpineRobot image-guided spine surgical robotics research platform.

## Owns

- Procedure HFSM / workflow supervision
- ROS 2 interfaces
- Registration and calibration services
- Trajectory management
- Navigation error computation
- Transform authority
- Closed-chain 4-DOF platform mathematical model
- Motion allocation between xArm6 and the 4-DOF fine stage
- Tracking bridge / validation
- Safety supervision and motion command gating
- ros2_control integration for the 4-DOF platform
- Simulation / hardware-in-the-loop support
- Bringup and configuration

## Does not own

- DICOM/MPR rendering and surgeon-facing GUI (`spinerobot_ui`)
- Arduino low-level firmware (`spinerobot_firmware`)
- Vendor source trees such as `xarm_ros2`, MoveIt 2, Plus Toolkit, or SlicerROS2

## Target environment

- Ubuntu 24.04 LTS
- ROS 2 Jazzy
- C++17+
- xArm ROS 2 stack
- ros2_control
- MoveIt 2
- PlusServer / OpenIGTLink

## First integration milestone

1. Real xArm6 moves.
2. Real Polaris measures the tracked body.
3. ROS receives validated tracking.
4. A simulated closed-chain 4-DOF stage is attached to the measured coarse-stage pose.
5. The platform model computes a virtual guide pose.
6. Navigation computes guide-to-trajectory error.
7. RViz/Slicer visualizes the result.
