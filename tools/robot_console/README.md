# BART Robot Console

Small preflight/service UI for the BART Spine Robot system.

Features:

- Start/stop the xArm ROS 2 driver.
- Live xArm connection, state, mode, motion-enabled status, error/warning code, and TCP XYZ.
- xArm state controls: motion enable/disable, clear error/warning, READY (state 0), STOPPED (state 4).
- Start/stop the NDI Polaris ROS node.
- NDI heartbeat from `/tracking/status`, with a one-second timeout independent of marker visibility.
- Live tracked/not-tracked state, quality, XYZ, and last-seen age for each configured marker.
- Combined Start All / Stop All controls.
- Ubuntu application/desktop shortcut.

The console intentionally does not provide Cartesian jogging or velocity commands.
`Set STOPPED` is a software xArm state command and is not a substitute for the physical emergency stop.

## Install

```bash
chmod +x tools/robot_console/install.sh tools/robot_console/run.sh
./tools/robot_console/install.sh
```

The installer creates `~/venvs/spinerobot_console` with `--system-site-packages`,
installs the Python dependencies, and creates an Ubuntu launcher called
`BART Robot Console`.

## Run

```bash
./tools/robot_console/run.sh
```

## Configuration

Edit `tools/robot_console/config.json`.

Defaults match the current lab setup:

- xArm IP: `192.168.1.231`
- namespace: `/xarm`
- Polaris serial port: `/dev/ttyUSB0`
- `8700338.rom` -> Tool marker
- `8700339.rom` -> Robot marker
- `8700340.rom` -> Patient marker

Each marker has a ROS `frame_id` (`tool_marker`, `robot_marker`, or `patient_marker`).
If the physical ROM-to-role mapping differs, change the labels/ROM paths in the JSON.
`heartbeat_timeout_s` controls status expiry. The console refuses to start a
duplicate node when status is already arriving and only stops its own process.

Build `spinerobot_interfaces` and `spinerobot_tracking` before running the console;
see [tracking setup and the patient-marker occlusion check](../../src/spinerobot_tracking/README.md).

If Polaris reports permission denied:

```bash
sudo usermod -aG dialout "$USER"
```

Then log out and back in.

## Verification

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
QT_QPA_PLATFORM=offscreen ~/venvs/spinerobot_console/bin/python tools/robot_console/tests/test_tracking.py
```

### Reopening the console

The console recovers control of local NDI and xArm drivers from this workspace
on ROS domain 42, even if a previous console exited. Start detects existing
local processes before launching another driver; Stop sends SIGINT to matching
local drivers, including orphaned processes. xArm displays LOCAL for a
recoverable driver and REMOTE when status has no matching local process.
Recovery checks the OS user, exact executable path, ROS domain, and process
start identity before signalling. Processes in another workspace or domain
are not stopped. Closing a reopened console does not stop recovered drivers;
use Stop explicitly. Drivers launched by the current window are stopped when
that window closes.
