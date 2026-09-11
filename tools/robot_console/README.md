# BART Robot Console

Small preflight/service UI for the BART Spine Robot system.

Features:

- Start/stop the xArm ROS 2 driver.
- Live xArm connection, state, mode, motion-enabled status, error/warning code, and TCP XYZ.
- xArm state controls: motion enable/disable, clear error/warning, READY (state 0), STOPPED (state 4).
- Start/stop NDI Polaris tracking.
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

If the physical ROM-to-role mapping differs, change the labels/ROM paths in the JSON.

If Polaris reports permission denied:

```bash
sudo usermod -aG dialout "$USER"
```

Then log out and back in.
