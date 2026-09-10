#!/usr/bin/env python3
"""
xArm + NDI Polaris Vicra RAW vs KALMAN calibration / validation
===============================================================

Goal
----
Compare the SAME NDI measurements processed in two ways:

    A) RAW: robust outlier rejection + stationary mean
    B) KALMAN: robust outlier rejection + 3-D constant-velocity Kalman filter

The Kalman tuning is selected using CALIBRATION data only.  The selected
filter is then frozen and evaluated on HELD-OUT VALIDATION moves.

This is deliberately an A/B experiment.  It tells you whether Kalman
filtering improves the external relative-motion result.  It does NOT prove
that any remaining disagreement is "NDI error", because xArm physical
positioning and the extrinsic calibration are also in the measurement chain.

Important
---------
This is still a fixed-orientation RELATIVE calibration.  It does not replace
full 6-DOF robot-world/hand-eye + marker-to-TCP calibration.

Safety
------
Bench/test use only.  No patient.  Verify every listed target is reachable
and collision-free before typing CALIBRATE.
"""

import sys
import os
import json
import time
from datetime import datetime

import numpy as np

from xarm.wrapper import XArmAPI
from sksurgerynditracker.nditracker import NDITracker


# =============================================================================
# CONFIGURATION
# =============================================================================

SERIAL_PORT = "/dev/ttyUSB0"

ROM_FILE = (
    "/home/bartlab/Workspace/spinerobot_ros2/"
    "config/ndi_tools/8700340.rom"
)

RESULT_DIR = (
    "/home/bartlab/Workspace/spinerobot_ros2/results/"
    "ndi_xarm_kalman_ab"
)

# Polaris Vicra nominal frame period is about 0.05 s (20 Hz).
KALMAN_DT_SEC = 0.05

# 30 accepted frames ~= 1.5 s at 20 Hz.
NDI_SAMPLES = 30
NDI_CAPTURE_TIMEOUT_SEC = 8.0

# Average the final filtered estimates once the filter has converged.
KALMAN_TAIL_SAMPLES = 10

# Conservative minimum assumed measurement standard deviations.
# These prevent the filter becoming unrealistically over-confident when a
# short static sample happens to have extremely small measured variance.
KALMAN_R_STD_FLOOR_MM = np.array([0.005, 0.005, 0.015], dtype=float)

# Tune process-noise strength on CALIBRATION data only.
# Unit: equivalent acceleration standard deviation [mm/s^2].
KALMAN_ACCEL_STD_CANDIDATES = [
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

ROBOT_SPEED_MM_S = 10.0
ROBOT_ACC_MM_S2 = 50.0
SETTLE_TIME_SEC = 1.5

RETURN_TO_START_AT_END = True

# -----------------------------------------------------------------------------
# CALIBRATION SET
#
# Keep this spatially distributed, but do NOT include the exact held-out
# validation moves below.
#
# The calibration set uses ~25-30 mm motions.
# -----------------------------------------------------------------------------

CALIBRATION_OFFSETS_MM = [
    [ 30.0,   0.0,   0.0],
    [-30.0,   0.0,   0.0],
    [  0.0,  30.0,   0.0],
    [  0.0, -30.0,   0.0],
    [  0.0,   0.0,  30.0],
    [  0.0,   0.0, -30.0],

    [ 25.0,  25.0,   0.0],
    [-25.0,  25.0,   0.0],
    [ 25.0, -25.0,   0.0],
    [-25.0, -25.0,   0.0],

    [ 25.0,   0.0,  25.0],
    [-25.0,   0.0,  25.0],
    [  0.0,  25.0, -25.0],
    [  0.0, -25.0, -25.0],
]

# -----------------------------------------------------------------------------
# HELD-OUT VALIDATION
#
# Includes 20 mm and 40 mm axis sweeps so you can test whether error grows
# with displacement magnitude.  Your previous run already demonstrated
# ±40 mm single-axis moves around this start pose, but STILL check the
# workspace before running.
# -----------------------------------------------------------------------------

VALIDATION_OFFSETS_MM = [
    # 20 mm axis sweep
    [ 20.0,   0.0,   0.0],
    [-20.0,   0.0,   0.0],
    [  0.0,  20.0,   0.0],
    [  0.0, -20.0,   0.0],
    [  0.0,   0.0,  20.0],
    [  0.0,   0.0, -20.0],

    # 40 mm axis sweep: tests displacement-magnitude dependence
    [ 40.0,   0.0,   0.0],
    [-40.0,   0.0,   0.0],
    [  0.0,  40.0,   0.0],
    [  0.0, -40.0,   0.0],
    [  0.0,   0.0,  40.0],
    [  0.0,   0.0, -40.0],

    # Held-out diagonals
    [ 20.0,  20.0,  20.0],
    [ 35.0, -20.0,  15.0],
    [-35.0,  20.0, -15.0],
    [ 20.0,  15.0, -35.0],
]


# =============================================================================
# MATH / ROTATION
# =============================================================================

def nearest_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    r = u @ vt

    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    return r


def average_rotation(rotations):
    m = np.zeros((3, 3), dtype=float)

    for r in rotations:
        m += r

    return nearest_rotation(m)


def matrix_to_rpy_deg(r):
    value = np.clip(-r[2, 0], -1.0, 1.0)
    pitch = np.arcsin(value)

    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])

    return np.rad2deg([roll, pitch, yaw])


def rotation_angle_deg(r):
    c = np.clip(
        (np.trace(r) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.rad2deg(np.arccos(c))
    )


def fit_rotation_kabsch(delta_tracker, delta_base):
    """
    Find R_base_from_tracker minimizing

        d_base ~= R_base_from_tracker @ d_tracker
    """
    x = np.asarray(delta_tracker, dtype=float)
    y = np.asarray(delta_base, dtype=float)

    h = x.T @ y

    u, _, vt = np.linalg.svd(h)

    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    return r


def estimate_uniform_scale(
        delta_tracker,
        delta_base,
        r_base_from_tracker):

    numerator = 0.0
    denominator = 0.0

    for dt, db in zip(
            delta_tracker,
            delta_base):

        rotated = (
            r_base_from_tracker @ dt
        )

        numerator += float(
            np.dot(db, rotated)
        )

        denominator += float(
            np.dot(rotated, rotated)
        )

    if denominator < 1e-12:
        return 1.0

    return numerator / denominator


# =============================================================================
# ROBUST NDI SAMPLE FILTERING
# =============================================================================

def robust_upper_mask(values, sigma=4.0):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 5:
        return np.ones(
            len(values),
            dtype=bool,
        )

    med = np.median(values)

    mad = np.median(
        np.abs(values - med)
    )

    if mad < 1e-12:
        return np.ones(
            len(values),
            dtype=bool,
        )

    robust_sigma = (
        1.4826 * mad
    )

    upper = (
        med
        +
        sigma * robust_sigma
    )

    return values <= upper


def robust_position_mask(
        positions,
        sigma=4.0):

    p = np.asarray(
        positions,
        dtype=float,
    )

    if len(p) < 5:
        return np.ones(
            len(p),
            dtype=bool,
        )

    center = np.median(
        p,
        axis=0,
    )

    residual = np.linalg.norm(
        p - center,
        axis=1,
    )

    med = np.median(residual)

    mad = np.median(
        np.abs(residual - med)
    )

    if mad < 1e-12:
        return np.ones(
            len(p),
            dtype=bool,
        )

    robust_sigma = (
        1.4826 * mad
    )

    upper = (
        med
        +
        sigma * robust_sigma
    )

    return residual <= upper


# =============================================================================
# 3-D CONSTANT-VELOCITY KALMAN FILTER
# =============================================================================

def kalman_filter_position(
        positions,
        accel_std_mm_s2,
        dt=KALMAN_DT_SEC):

    """
    State:
        [x, y, z, vx, vy, vz]

    Measurement:
        [x, y, z]

    Constant-velocity model with white acceleration process noise.

    Returns
    -------
    endpoint_position : (3,)
        Mean of the last KALMAN_TAIL_SAMPLES filtered positions.

    filtered_positions : Nx3

    measurement_std : (3,)
        Static standard deviation used to construct R.
    """

    z_all = np.asarray(
        positions,
        dtype=float,
    )

    if len(z_all) < 3:
        raise RuntimeError(
            "Not enough samples for Kalman filter"
        )

    # Estimate measurement noise from the accepted static samples.
    measured_std = np.std(
        z_all,
        axis=0,
        ddof=1,
    )

    measurement_std = np.maximum(
        measured_std,
        KALMAN_R_STD_FLOOR_MM,
    )

    r_meas = np.diag(
        measurement_std ** 2
    )

    # State transition
    f = np.eye(6)

    f[0, 3] = dt
    f[1, 4] = dt
    f[2, 5] = dt

    # Position measurement matrix
    h = np.zeros((3, 6))

    h[0, 0] = 1.0
    h[1, 1] = 1.0
    h[2, 2] = 1.0

    # Constant-velocity process covariance with white acceleration.
    q = np.zeros((6, 6))

    accel_var = (
        accel_std_mm_s2 ** 2
    )

    q_pos = (
        (dt ** 4) / 4.0
        *
        accel_var
    )

    q_cross = (
        (dt ** 3) / 2.0
        *
        accel_var
    )

    q_vel = (
        (dt ** 2)
        *
        accel_var
    )

    for pos_idx, vel_idx in [
        (0, 3),
        (1, 4),
        (2, 5),
    ]:
        q[pos_idx, pos_idx] = q_pos
        q[pos_idx, vel_idx] = q_cross
        q[vel_idx, pos_idx] = q_cross
        q[vel_idx, vel_idx] = q_vel

    # Initialize from first accepted measurement.
    x = np.zeros(6)

    x[:3] = z_all[0]
    x[3:] = 0.0

    # Position moderately certain; velocity initially uncertain.
    p = np.diag([
        measurement_std[0] ** 2,
        measurement_std[1] ** 2,
        measurement_std[2] ** 2,
        25.0,
        25.0,
        25.0,
    ])

    identity = np.eye(6)

    filtered = []

    for z in z_all:

        # Predict
        x = f @ x
        p = (
            f @ p @ f.T
            +
            q
        )

        # Innovation
        innovation = (
            z
            -
            h @ x
        )

        s = (
            h @ p @ h.T
            +
            r_meas
        )

        k = (
            p
            @ h.T
            @ np.linalg.inv(s)
        )

        # Update
        x = (
            x
            +
            k @ innovation
        )

        # Joseph form is more numerically stable.
        kh = k @ h

        p = (
            (identity - kh)
            @ p
            @ (identity - kh).T
            +
            k @ r_meas @ k.T
        )

        filtered.append(
            x[:3].copy()
        )

    filtered = np.asarray(
        filtered,
        dtype=float,
    )

    tail = min(
        KALMAN_TAIL_SAMPLES,
        len(filtered),
    )

    endpoint = np.mean(
        filtered[-tail:],
        axis=0,
    )

    return (
        endpoint,
        filtered,
        measurement_std,
    )


# =============================================================================
# NDI CAPTURE
# =============================================================================

def capture_ndi_samples(
        tracker,
        samples=NDI_SAMPLES):

    transforms = []
    qualities = []

    last_frame = None
    start_time = time.time()

    target_raw = max(
        samples + 12,
        int(samples * 1.35),
    )

    while len(transforms) < target_raw:

        if (
            time.time()
            -
            start_time
            >
            NDI_CAPTURE_TIMEOUT_SEC
        ):
            break

        (
            port_handles,
            timestamps,
            frame_numbers,
            tracking,
            quality,
        ) = tracker.get_frame()

        if len(tracking) < 1:
            time.sleep(0.003)
            continue

        matrix = np.asarray(
            tracking[0],
            dtype=float,
        )

        if matrix.shape != (4, 4):
            time.sleep(0.003)
            continue

        if not np.all(
            np.isfinite(matrix)
        ):
            time.sleep(0.003)
            continue

        if len(frame_numbers) > 0:

            frame_id = str(
                frame_numbers[0]
            )

            if frame_id == last_frame:
                time.sleep(0.003)
                continue

            last_frame = frame_id

        transforms.append(
            matrix.copy()
        )

        if len(quality) > 0:
            q = float(
                quality[0]
            )
        else:
            q = np.nan

        qualities.append(q)

        time.sleep(0.003)

    if len(transforms) < samples:

        raise RuntimeError(
            f"Only {len(transforms)} valid NDI frames collected; "
            f"need at least {samples}."
        )

    positions = np.asarray(
        [
            t[:3, 3]
            for t in transforms
        ],
        dtype=float,
    )

    rotations = [
        t[:3, :3]
        for t in transforms
    ]

    qualities_np = np.asarray(
        qualities,
        dtype=float,
    )

    mask = robust_position_mask(
        positions
    )

    finite_quality = np.isfinite(
        qualities_np
    )

    if np.count_nonzero(
        finite_quality
    ) >= 5:

        qmask_local = robust_upper_mask(
            qualities_np[finite_quality]
        )

        qmask = np.ones(
            len(qualities_np),
            dtype=bool,
        )

        qmask[
            np.where(finite_quality)[0]
        ] = qmask_local

        mask &= qmask

    accepted_idx = np.where(
        mask
    )[0]

    if len(accepted_idx) < samples:

        center = np.median(
            positions,
            axis=0,
        )

        distance = np.linalg.norm(
            positions - center,
            axis=1,
        )

        accepted_idx = np.argsort(
            distance
        )[:samples]

    else:
        accepted_idx = (
            accepted_idx[:samples]
        )

    accepted_positions = (
        positions[accepted_idx]
    )

    accepted_rotations = [
        rotations[i]
        for i in accepted_idx
    ]

    accepted_quality = (
        qualities_np[accepted_idx]
    )

    accepted_quality = (
        accepted_quality[
            np.isfinite(
                accepted_quality
            )
        ]
    )

    raw_position = np.mean(
        accepted_positions,
        axis=0,
    )

    raw_rotation = average_rotation(
        accepted_rotations
    )

    static_std = np.std(
        accepted_positions,
        axis=0,
    )

    quality_mean = (
        float(
            np.mean(
                accepted_quality
            )
        )
        if len(
            accepted_quality
        ) > 0
        else float("nan")
    )

    return {
        "positions": accepted_positions,
        "rotation": raw_rotation,
        "raw_position": raw_position,
        "static_std_mm": static_std,
        "quality_mean": quality_mean,
        "accepted_count": int(
            len(accepted_positions)
        ),
        "raw_count": int(
            len(transforms)
        ),
    }


# =============================================================================
# XARM
# =============================================================================

def get_arm_pose(arm):

    code, pose = arm.get_position(
        is_radian=False
    )

    if code != 0:
        raise RuntimeError(
            f"xArm get_position failed, code={code}"
        )

    return np.asarray(
        pose,
        dtype=float,
    )


def move_arm_absolute(
        arm,
        pose,
        name):

    print(
        f"\nMoving: {name}"
    )

    print(
        "  TCP target = "
        f"[{pose[0]:.3f}, "
        f"{pose[1]:.3f}, "
        f"{pose[2]:.3f}] mm | "
        f"RPY = "
        f"[{pose[3]:.3f}, "
        f"{pose[4]:.3f}, "
        f"{pose[5]:.3f}] deg"
    )

    code = arm.set_position(
        x=float(pose[0]),
        y=float(pose[1]),
        z=float(pose[2]),
        roll=float(pose[3]),
        pitch=float(pose[4]),
        yaw=float(pose[5]),
        speed=ROBOT_SPEED_MM_S,
        mvacc=ROBOT_ACC_MM_S2,
        is_radian=False,
        wait=True,
        timeout=30,
    )

    if code != 0:

        raise RuntimeError(
            f"xArm movement failed ({name}), code={code}"
        )

    time.sleep(
        SETTLE_TIME_SEC
    )


# =============================================================================
# RELATIVE MOVE ACQUISITION
# =============================================================================

def measure_relative_move(
        arm,
        tracker,
        start_pose,
        offset_mm,
        label):

    offset = np.asarray(
        offset_mm,
        dtype=float,
    )

    move_arm_absolute(
        arm,
        start_pose,
        f"{label}: baseline",
    )

    start_capture = capture_ndi_samples(
        tracker
    )

    target = start_pose.copy()

    target[:3] += offset

    move_arm_absolute(
        arm,
        target,
        (
            f"{label}: offset "
            f"[{offset[0]:+.1f}, "
            f"{offset[1]:+.1f}, "
            f"{offset[2]:+.1f}] mm"
        ),
    )

    end_capture = capture_ndi_samples(
        tracker
    )

    arm_end = get_arm_pose(
        arm
    )

    arm_internal_delta = (
        arm_end[:3]
        -
        start_pose[:3]
    )

    r_delta_tracker = (
        end_capture["rotation"]
        @
        start_capture["rotation"].T
    )

    move_arm_absolute(
        arm,
        start_pose,
        f"{label}: return",
    )

    return {
        "label": label,
        "commanded_base_mm": offset,
        "arm_internal_delta_mm": arm_internal_delta,
        "start": start_capture,
        "end": end_capture,
        "r_delta_tracker": r_delta_tracker,
    }


# =============================================================================
# RAW / KALMAN EXTRACTION
# =============================================================================

def delta_tracker_raw(measurement):

    return (
        measurement["end"]["raw_position"]
        -
        measurement["start"]["raw_position"]
    )


def delta_tracker_kalman(
        measurement,
        accel_std):

    start_endpoint, _, _ = (
        kalman_filter_position(
            measurement[
                "start"
            ]["positions"],
            accel_std,
        )
    )

    end_endpoint, _, _ = (
        kalman_filter_position(
            measurement[
                "end"
            ]["positions"],
            accel_std,
        )
    )

    return (
        end_endpoint
        -
        start_endpoint
    )


def build_delta_array(
        measurements,
        method,
        accel_std=None):

    deltas = []

    for m in measurements:

        if method == "raw":

            d = delta_tracker_raw(m)

        elif method == "kalman":

            d = delta_tracker_kalman(
                m,
                accel_std,
            )

        else:
            raise ValueError(method)

        deltas.append(d)

    return np.asarray(
        deltas,
        dtype=float,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def summarize_errors(errors):

    e = np.asarray(
        errors,
        dtype=float,
    )

    norms = np.linalg.norm(
        e,
        axis=1,
    )

    return {
        "n": int(len(e)),
        "mean_xyz_mm": np.mean(
            e,
            axis=0,
        ).tolist(),
        "std_xyz_mm": (
            np.std(
                e,
                axis=0,
                ddof=1,
            ).tolist()
            if len(e) > 1
            else [
                0.0,
                0.0,
                0.0,
            ]
        ),
        "rmse_xyz_mm": np.sqrt(
            np.mean(
                e ** 2,
                axis=0,
            )
        ).tolist(),
        "mean_3d_mm": float(
            np.mean(norms)
        ),
        "rmse_3d_mm": float(
            np.sqrt(
                np.mean(
                    norms ** 2
                )
            )
        ),
        "max_3d_mm": float(
            np.max(norms)
        ),
        "p95_3d_mm": float(
            np.percentile(
                norms,
                95,
            )
        ),
    }


def evaluate_measurements(
        measurements,
        r_base_from_tracker,
        method,
        accel_std=None):

    rows = []
    errors = []

    r_tracker_from_base = (
        r_base_from_tracker.T
    )

    for m in measurements:

        if method == "raw":

            delta_tracker = (
                delta_tracker_raw(m)
            )

        else:

            delta_tracker = (
                delta_tracker_kalman(
                    m,
                    accel_std,
                )
            )

        measured_base = (
            r_base_from_tracker
            @
            delta_tracker
        )

        commanded = (
            m["commanded_base_mm"]
        )

        error = (
            measured_base
            -
            commanded
        )

        error_3d = float(
            np.linalg.norm(error)
        )

        command_length = float(
            np.linalg.norm(commanded)
        )

        relative_error_pct = (
            100.0
            *
            error_3d
            /
            command_length
            if command_length > 1e-12
            else float("nan")
        )

        r_delta_base = (
            r_base_from_tracker
            @
            m["r_delta_tracker"]
            @
            r_tracker_from_base
        )

        rpy_drift = (
            matrix_to_rpy_deg(
                r_delta_base
            )
        )

        total_rotation_drift = (
            rotation_angle_deg(
                r_delta_base
            )
        )

        errors.append(error)

        rows.append({
            "label": m["label"],
            "commanded_base_mm":
                commanded.tolist(),
            "command_length_mm":
                command_length,
            "measured_base_mm":
                measured_base.tolist(),
            "error_xyz_mm":
                error.tolist(),
            "error_3d_mm":
                error_3d,
            "relative_error_pct":
                relative_error_pct,
            "rotation_drift_rpy_deg":
                rpy_drift.tolist(),
            "rotation_drift_total_deg":
                total_rotation_drift,
            "quality_before":
                m["start"][
                    "quality_mean"
                ],
            "quality_after":
                m["end"][
                    "quality_mean"
                ],
            "static_sd_before_mm":
                m["start"][
                    "static_std_mm"
                ].tolist(),
            "static_sd_after_mm":
                m["end"][
                    "static_std_mm"
                ].tolist(),
        })

    return (
        rows,
        summarize_errors(errors),
    )


def print_evaluation(
        title,
        rows,
        summary):

    print(
        "\n"
        +
        "=" * 112
    )

    print(title)

    print(
        "=" * 112
    )

    print(
        f"{'Test':<8} "
        f"{'|Cmd|':>7} | "
        f"{'Err X':>8} "
        f"{'Err Y':>8} "
        f"{'Err Z':>8} | "
        f"{'3D err':>8} "
        f"{'Rel %':>7} | "
        f"{'Rot err':>8}"
    )

    print(
        "-" * 112
    )

    for row in rows:

        e = row[
            "error_xyz_mm"
        ]

        print(
            f"{row['label']:<8} "
            f"{row['command_length_mm']:>7.2f} | "
            f"{e[0]:>+8.3f} "
            f"{e[1]:>+8.3f} "
            f"{e[2]:>+8.3f} | "
            f"{row['error_3d_mm']:>8.3f} "
            f"{row['relative_error_pct']:>7.3f} | "
            f"{row['rotation_drift_total_deg']:>7.4f}°"
        )

    print("\nSummary:")

    print(
        "  Mean XYZ error [mm] : "
        f"{np.asarray(summary['mean_xyz_mm'])}"
    )

    print(
        "  RMSE XYZ [mm]       : "
        f"{np.asarray(summary['rmse_xyz_mm'])}"
    )

    print(
        f"  RMSE 3D [mm]        : "
        f"{summary['rmse_3d_mm']:.4f}"
    )

    print(
        f"  Mean 3D [mm]        : "
        f"{summary['mean_3d_mm']:.4f}"
    )

    print(
        f"  P95 3D [mm]         : "
        f"{summary['p95_3d_mm']:.4f}"
    )

    print(
        f"  Maximum 3D [mm]     : "
        f"{summary['max_3d_mm']:.4f}"
    )


def summarize_by_command_length(rows):
    """
    Group results by rounded command magnitude so 20 mm and 40 mm
    axis sweeps can be compared directly.
    """

    groups = {}

    for row in rows:

        key = round(
            row[
                "command_length_mm"
            ],
            3,
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row[
                "error_3d_mm"
            ]
        )

    result = {}

    for length, values in sorted(
        groups.items()
    ):

        values = np.asarray(
            values,
            dtype=float,
        )

        result[
            str(length)
        ] = {
            "n": int(
                len(values)
            ),
            "mean_3d_error_mm":
                float(
                    np.mean(values)
                ),
            "rmse_3d_error_mm":
                float(
                    np.sqrt(
                        np.mean(
                            values ** 2
                        )
                    )
                ),
            "max_3d_error_mm":
                float(
                    np.max(values)
                ),
        }

    return result


def print_distance_summary(
        title,
        distance_summary):

    print(
        "\n"
        +
        title
    )

    print(
        "-" * len(title)
    )

    print(
        f"{'Command length':>15} "
        f"{'N':>4} "
        f"{'Mean 3D':>10} "
        f"{'RMSE 3D':>10} "
        f"{'Max 3D':>10}"
    )

    for length_str, data in (
        distance_summary.items()
    ):

        print(
            f"{float(length_str):>12.2f} mm "
            f"{data['n']:>4d} "
            f"{data['mean_3d_error_mm']:>9.3f} "
            f"{data['rmse_3d_error_mm']:>9.3f} "
            f"{data['max_3d_error_mm']:>9.3f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "  python ndi_xarm_kalman_ab.py <XARM_IP>\n\n"
            "Example:\n"
            "  python ndi_xarm_kalman_ab.py 192.168.1.231"
        )

        sys.exit(1)

    xarm_ip = sys.argv[1]

    if not os.path.exists(
        ROM_FILE
    ):

        raise FileNotFoundError(
            ROM_FILE
        )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    ndi_settings = {
        "tracker type":
            "polaris",
        "serial port":
            SERIAL_PORT,
        "romfiles":
            [ROM_FILE],
    }

    print(
        "Connecting to Polaris Vicra..."
    )

    tracker = NDITracker(
        ndi_settings
    )

    tracker.start_tracking()

    print(
        f"Connecting to xArm at "
        f"{xarm_ip}..."
    )

    arm = XArmAPI(
        xarm_ip,
        is_radian=False,
    )

    start_pose = None

    try:

        if not arm.connected:

            raise RuntimeError(
                "Could not connect to xArm"
            )

        code, err_warn = (
            arm.get_err_warn_code()
        )

        if code != 0:

            raise RuntimeError(
                f"get_err_warn_code failed, code={code}"
            )

        print(
            f"xArm error/warning state: "
            f"{err_warn}"
        )

        if (
            len(err_warn) >= 1
            and
            err_warn[0] != 0
        ):

            raise RuntimeError(
                f"xArm has error "
                f"{err_warn[0]}. "
                f"Resolve it before moving."
            )

        for name, result in [
            (
                "motion_enable",
                arm.motion_enable(
                    enable=True
                ),
            ),
            (
                "set_mode",
                arm.set_mode(0),
            ),
            (
                "set_state",
                arm.set_state(
                    state=0
                ),
            ),
        ]:

            if result != 0:

                raise RuntimeError(
                    f"{name} failed, "
                    f"code={result}"
                )

        time.sleep(1.0)

        start_pose = get_arm_pose(
            arm
        )

        print(
            "\n"
            +
            "=" * 80
        )

        print(
            "RAW vs KALMAN NDI CALIBRATION / VALIDATION"
        )

        print(
            "=" * 80
        )

        print(
            "Start XYZ [mm] = "
            f"{start_pose[:3]}"
        )

        print(
            "Start RPY [deg] = "
            f"{start_pose[3:6]}"
        )

        print(
            "\nCALIBRATION OFFSETS:"
        )

        for offset in (
            CALIBRATION_OFFSETS_MM
        ):
            print(
                f"  {offset}"
            )

        print(
            "\nHELD-OUT VALIDATION OFFSETS:"
        )

        for offset in (
            VALIDATION_OFFSETS_MM
        ):
            print(
                f"  {offset}"
            )

        print(
            "\nThis run will compare RAW and KALMAN processing "
            "using the exact same captured NDI frames."
        )

        print(
            "\nThe 20 mm vs 40 mm held-out axis moves are specifically "
            "included to test whether error grows with displacement magnitude."
        )

        print(
            "\nIMPORTANT: verify every listed pose is safe and reachable."
        )

        confirmation = input(
            "\nType CALIBRATE to execute the robot motions: "
        ).strip()

        if confirmation != "CALIBRATE":

            print(
                "Cancelled."
            )

            return

        # =====================================================================
        # ACQUIRE CALIBRATION DATA ONCE
        # =====================================================================

        calibration_measurements = []

        for i, offset in enumerate(
                CALIBRATION_OFFSETS_MM,
                start=1):

            measurement = (
                measure_relative_move(
                    arm,
                    tracker,
                    start_pose,
                    offset,
                    f"CAL{i:02d}",
                )
            )

            calibration_measurements.append(
                measurement
            )

        commanded_cal = np.asarray(
            [
                m[
                    "commanded_base_mm"
                ]
                for m
                in calibration_measurements
            ],
            dtype=float,
        )

        # =====================================================================
        # RAW CALIBRATION
        # =====================================================================

        raw_tracker_cal = (
            build_delta_array(
                calibration_measurements,
                method="raw",
            )
        )

        raw_r_base_from_tracker = (
            fit_rotation_kabsch(
                raw_tracker_cal,
                commanded_cal,
            )
        )

        raw_scale = (
            estimate_uniform_scale(
                raw_tracker_cal,
                commanded_cal,
                raw_r_base_from_tracker,
            )
        )

        (
            raw_cal_rows,
            raw_cal_summary,
        ) = evaluate_measurements(
            calibration_measurements,
            raw_r_base_from_tracker,
            method="raw",
        )

        print_evaluation(
            "RAW CALIBRATION-SET RESIDUALS",
            raw_cal_rows,
            raw_cal_summary,
        )

        print(
            f"\nRAW diagnostic uniform scale = "
            f"{raw_scale:.8f}"
        )

        # =====================================================================
        # KALMAN TUNING ON CALIBRATION SET ONLY
        # =====================================================================

        tuning_results = []

        print(
            "\n"
            +
            "=" * 80
        )

        print(
            "KALMAN PROCESS-NOISE TUNING "
            "(CALIBRATION DATA ONLY)"
        )

        print(
            "=" * 80
        )

        for accel_std in (
            KALMAN_ACCEL_STD_CANDIDATES
        ):

            kal_tracker_cal = (
                build_delta_array(
                    calibration_measurements,
                    method="kalman",
                    accel_std=accel_std,
                )
            )

            kal_r = (
                fit_rotation_kabsch(
                    kal_tracker_cal,
                    commanded_cal,
                )
            )

            (
                _,
                kal_summary,
            ) = evaluate_measurements(
                calibration_measurements,
                kal_r,
                method="kalman",
                accel_std=accel_std,
            )

            kal_scale = (
                estimate_uniform_scale(
                    kal_tracker_cal,
                    commanded_cal,
                    kal_r,
                )
            )

            tuning_results.append({
                "accel_std_mm_s2":
                    float(accel_std),
                "calibration_rmse_3d_mm":
                    kal_summary[
                        "rmse_3d_mm"
                    ],
                "calibration_p95_3d_mm":
                    kal_summary[
                        "p95_3d_mm"
                    ],
                "diagnostic_scale":
                    float(kal_scale),
                "R_base_from_tracker":
                    kal_r,
            })

            print(
                f"accel_std={accel_std:>6.2f} mm/s² | "
                f"RMSE3D={kal_summary['rmse_3d_mm']:.4f} mm | "
                f"P95={kal_summary['p95_3d_mm']:.4f} mm | "
                f"scale={kal_scale:.8f}"
            )

        selected = min(
            tuning_results,
            key=lambda x:
                x[
                    "calibration_rmse_3d_mm"
                ],
        )

        best_accel_std = (
            selected[
                "accel_std_mm_s2"
            ]
        )

        kal_r_base_from_tracker = (
            selected[
                "R_base_from_tracker"
            ]
        )

        kal_scale = (
            selected[
                "diagnostic_scale"
            ]
        )

        print(
            "\nSelected Kalman accel_std = "
            f"{best_accel_std:.3f} mm/s²"
        )

        (
            kal_cal_rows,
            kal_cal_summary,
        ) = evaluate_measurements(
            calibration_measurements,
            kal_r_base_from_tracker,
            method="kalman",
            accel_std=best_accel_std,
        )

        print_evaluation(
            "KALMAN CALIBRATION-SET RESIDUALS",
            kal_cal_rows,
            kal_cal_summary,
        )

        print(
            f"\nKALMAN diagnostic uniform scale = "
            f"{kal_scale:.8f}"
        )

        # =====================================================================
        # ACQUIRE HELD-OUT VALIDATION DATA ONCE
        # =====================================================================

        validation_measurements = []

        for i, offset in enumerate(
                VALIDATION_OFFSETS_MM,
                start=1):

            measurement = (
                measure_relative_move(
                    arm,
                    tracker,
                    start_pose,
                    offset,
                    f"VAL{i:02d}",
                )
            )

            validation_measurements.append(
                measurement
            )

        # =====================================================================
        # RAW HELD-OUT VALIDATION
        # =====================================================================

        (
            raw_val_rows,
            raw_val_summary,
        ) = evaluate_measurements(
            validation_measurements,
            raw_r_base_from_tracker,
            method="raw",
        )

        print_evaluation(
            "RAW HELD-OUT VALIDATION",
            raw_val_rows,
            raw_val_summary,
        )

        # =====================================================================
        # KALMAN HELD-OUT VALIDATION
        #
        # IMPORTANT: best_accel_std and Kalman frame rotation were selected
        # using calibration data only.  They are frozen here.
        # =====================================================================

        (
            kal_val_rows,
            kal_val_summary,
        ) = evaluate_measurements(
            validation_measurements,
            kal_r_base_from_tracker,
            method="kalman",
            accel_std=best_accel_std,
        )

        print_evaluation(
            "KALMAN HELD-OUT VALIDATION",
            kal_val_rows,
            kal_val_summary,
        )

        # =====================================================================
        # DISTANCE DEPENDENCE
        # =====================================================================

        raw_distance = (
            summarize_by_command_length(
                raw_val_rows
            )
        )

        kal_distance = (
            summarize_by_command_length(
                kal_val_rows
            )
        )

        print_distance_summary(
            "RAW ERROR vs COMMANDED DISPLACEMENT",
            raw_distance,
        )

        print_distance_summary(
            "KALMAN ERROR vs COMMANDED DISPLACEMENT",
            kal_distance,
        )

        # =====================================================================
        # FINAL A/B INTERPRETATION
        # =====================================================================

        raw_rmse = (
            raw_val_summary[
                "rmse_3d_mm"
            ]
        )

        kal_rmse = (
            kal_val_summary[
                "rmse_3d_mm"
            ]
        )

        improvement_mm = (
            raw_rmse
            -
            kal_rmse
        )

        improvement_pct = (
            100.0
            *
            improvement_mm
            /
            raw_rmse
            if raw_rmse > 1e-12
            else 0.0
        )

        print(
            "\n"
            +
            "=" * 80
        )

        print(
            "RAW vs KALMAN HELD-OUT COMPARISON"
        )

        print(
            "=" * 80
        )

        print(
            f"RAW validation RMSE 3D    : "
            f"{raw_rmse:.4f} mm"
        )

        print(
            f"KALMAN validation RMSE 3D : "
            f"{kal_rmse:.4f} mm"
        )

        print(
            f"Difference                 : "
            f"{improvement_mm:+.4f} mm"
        )

        print(
            f"Relative improvement       : "
            f"{improvement_pct:+.2f}%"
        )

        if improvement_pct >= 10.0:

            interpretation = (
                "Kalman filtering produced a meaningful held-out "
                "improvement. Keep investigating its latency and dynamic "
                "tracking behavior before using it for closed-loop control."
            )

        elif improvement_pct > 0.0:

            interpretation = (
                "Kalman filtering improved the result only slightly. "
                "The dominant residual is probably not random NDI jitter."
            )

        else:

            interpretation = (
                "Kalman filtering did not improve held-out accuracy. "
                "Do not use it as an accuracy-correction algorithm; focus "
                "on extrinsic/hand-eye calibration and systematic error."
            )

        print(
            "\n"
            +
            interpretation
        )

        print(
            "\nIf 40 mm errors are consistently about twice the 20 mm "
            "errors, that supports a scale/systematic effect. Kalman "
            "filtering cannot remove a true scale or spatial-calibration error."
        )

        # =====================================================================
        # SAVE
        # =====================================================================

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        serializable_tuning = []

        for item in tuning_results:

            serializable_tuning.append({
                "accel_std_mm_s2":
                    item[
                        "accel_std_mm_s2"
                    ],
                "calibration_rmse_3d_mm":
                    item[
                        "calibration_rmse_3d_mm"
                    ],
                "calibration_p95_3d_mm":
                    item[
                        "calibration_p95_3d_mm"
                    ],
                "diagnostic_scale":
                    item[
                        "diagnostic_scale"
                    ],
            })

        output = {
            "timestamp":
                timestamp,
            "xarm_ip":
                xarm_ip,
            "ndi_rom":
                ROM_FILE,
            "ndi_samples_per_pose":
                NDI_SAMPLES,
            "kalman_dt_sec":
                KALMAN_DT_SEC,
            "kalman_tail_samples":
                KALMAN_TAIL_SAMPLES,
            "kalman_r_std_floor_mm":
                KALMAN_R_STD_FLOOR_MM.tolist(),
            "kalman_candidates":
                serializable_tuning,
            "selected_kalman_accel_std_mm_s2":
                best_accel_std,
            "start_pose_xarm":
                start_pose.tolist(),
            "calibration_offsets_mm":
                CALIBRATION_OFFSETS_MM,
            "validation_offsets_mm":
                VALIDATION_OFFSETS_MM,

            "raw_R_base_from_tracker":
                raw_r_base_from_tracker.tolist(),
            "raw_diagnostic_scale":
                float(raw_scale),
            "raw_calibration_summary":
                raw_cal_summary,
            "raw_validation_summary":
                raw_val_summary,
            "raw_distance_summary":
                raw_distance,

            "kalman_R_base_from_tracker":
                kal_r_base_from_tracker.tolist(),
            "kalman_diagnostic_scale":
                float(kal_scale),
            "kalman_calibration_summary":
                kal_cal_summary,
            "kalman_validation_summary":
                kal_val_summary,
            "kalman_distance_summary":
                kal_distance,

            "held_out_rmse_improvement_mm":
                float(improvement_mm),
            "held_out_rmse_improvement_pct":
                float(improvement_pct),

            "raw_validation_rows":
                raw_val_rows,
            "kalman_validation_rows":
                kal_val_rows,

            "notes": (
                "Kalman filter is applied to translation only. "
                "Rotation is still robustly averaged from the same NDI "
                "frames. This remains a relative fixed-orientation "
                "experiment, not full 6-DOF hand-eye calibration."
            ),
        }

        result_file = os.path.join(
            RESULT_DIR,
            f"ndi_xarm_kalman_ab_{timestamp}.json",
        )

        with open(
            result_file,
            "w",
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
            )

        print(
            "\nSaved result:"
        )

        print(
            f"  {result_file}"
        )

    finally:

        if (
            RETURN_TO_START_AT_END
            and
            start_pose is not None
        ):

            try:

                move_arm_absolute(
                    arm,
                    start_pose,
                    (
                        "Final return to "
                        "original start pose"
                    ),
                )

            except Exception as exc:

                print(
                    f"WARNING: final return failed: "
                    f"{exc}"
                )

        print(
            "\nStopping NDI tracker..."
        )

        try:
            tracker.stop_tracking()
        except Exception:
            pass

        try:
            tracker.close()
        except Exception:
            pass

        print(
            "Disconnecting xArm..."
        )

        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
