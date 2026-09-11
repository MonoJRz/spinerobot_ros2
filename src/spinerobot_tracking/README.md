# spinerobot_tracking

The `ndi_tracker` ROS 2 node owns the Polaris serial connection and publishes one
`spinerobot_interfaces/msg/TrackingStatus` per configured marker per device read
on `/tracking/status` (20 Hz by default).

Build from the workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select spinerobot_interfaces spinerobot_tracking --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
bash launchers/start_ndi.sh
```

The launcher runs the installed ROS node using the console Python environment,
which provides `scikit-surgerynditracker` and ROS dependencies. Override that
interpreter with `NDI_PYTHON` if needed. The console also uses this launcher.
Only one process should own the NDI serial device.

Parameters: `serial_port`, `rom_files` (ordered array), `frame_ids` (matching
ordered array), and `poll_period_s`. The launcher supplies the repository ROMs;
additional `-p name:=value` arguments override defaults.

## Heartbeat and occlusion check

In a second sourced terminal:

```bash
ros2 topic echo /tracking/status
```

For a visible patient marker, expect `frame_id: patient_marker`, `visible: true`,
`valid: true`, `age_sec: 0.0`, a device-reported quality, `reason: ok`, and the ROS
acquisition timestamp. Quality is passed through from NDI; it is not a
calibrated confidence probability. Valid means a finite 4×4 transform and finite,
nonnegative quality, with no clinical acceptance threshold implied.

Cover the patient marker. Its messages continue with `visible: false`,
`valid: false`, `reason: not_visible`, and increasing `age_sec`. Quality is NaN
while invisible; age is infinity until the marker is first seen. Uncover it to
restore visible/valid status and zero age.

Status arrival is the NDI heartbeat, independent of marker visibility. A failed
read exits the node; a blocked read produces no heartbeat. Consumers must time
out missing messages. The console uses a one-second timeout and expires each
marker's state independently.

Visible marker positions are also published as `geometry_msgs/PointStamped` on
`/tracking/<frame_id>/position`, in metres in the `ndi_tracker` coordinate frame.
The console converts these to millimetres for display.

Ctrl+C stops tracking and closes the device.

The tracker, xArm launcher, robot console, and native Spine UI use ROS domain
42. The UI launcher is `../spinerobot_ui/run-ui.sh`; it sources this workspace
and displays NDI heartbeat and marker visibility in Setup.
