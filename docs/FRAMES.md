# Coordinate Frames

Initial frame names:

- `ct_ras`
- `patient_ref`
- `ndi_tracker`
- `spine_marker`
- `robot_marker`
- `robot_base`
- `xarm_flange`
- `platform_base`
- `moving_platform`
- `guide`
- `tool_marker`
- `tool_tip`

## Convention boundary

Medical side:
- RAS
- millimetres where required by medical/OpenIGTLink interfaces

ROS side:
- REP-103 right-handed convention
- metres
- radians

Keep conversions in one adapter and unit-test them.

## Transform ownership

Exactly one source owns each parent-child relation. tf2 is the runtime graph, not the calibration database.
