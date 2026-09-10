#!/usr/bin/env python3
"""
HandEyeCalibration_v3.py
========================

Modern xArm6 + NDI Polaris Vicra robot-world / hand-eye calibration.

This version combines the calibration work developed in the recent tests:

* direct Polaris Vicra acquisition using scikit-surgerynditracker
* robust unique-frame capture and MAD outlier rejection
* 30-frame stationary averaging
* proper xArm fixed-axis XYZ RPY conversion
* automatic, repeatable calibration/validation pose sets
* OpenCV robot-world/hand-eye calibration (SHAH and LI)
* solver selection using calibration-set cross-validation only
* robust refit after rejecting bad calibration poses
* completely held-out validation poses
* residual-vs-workspace-distance analysis
* 20 mm versus 40 mm validation to detect growing error
* optional LOW-DIMENSIONAL similarity/scale diagnostic fitted on calibration
  data and evaluated only on held-out validation
* JSON + NPZ output for later ROS2 / Slicer use

Frames
------
B = xArm robot base
G = xArm TCP/gripper frame
T = NDI tracker frame
M = rigid NDI marker frame

Measured at pose i
------------------
A_i = ^B T_Gi   from xArm
C_i = ^T T_Mi   from Polaris

Unknown constants
-----------------
X = ^G T_M      marker -> TCP/gripper
Y = ^B T_T      tracker -> robot base

Rigid-body equation
-------------------
        ^B T_Gi  ^G T_M  =  ^B T_T  ^T T_Mi

        A_i X = Y C_i

IMPORTANT
---------
The similarity scale model in this script is DIAGNOSTIC ONLY.  It is not
automatically applied to the saved rigid hand-eye transforms.  If a scale
model improves held-out validation repeatedly across independent sessions,
then it can later be considered as a SYSTEM compensation layer.  It should
not be described as "NDI intrinsic calibration" unless independent metrology
proves that the error belongs to the NDI tracker.

Safety
------
Bench / phantom testing only. Verify all listed poses are collision-free and
reachable before typing CALIBRATE.
"""

import json
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from xarm.wrapper import XArmAPI
from sksurgerynditracker.nditracker import NDITracker

REQUIRED_OPENCV_APIS = [
    "calibrateRobotWorldHandEye",
    "CALIB_ROBOT_WORLD_HAND_EYE_SHAH",
    "CALIB_ROBOT_WORLD_HAND_EYE_LI",
]

missing = [
    name
    for name in REQUIRED_OPENCV_APIS
    if not hasattr(cv2, name)
]

if missing:
    raise RuntimeError(
        "\nOpenCV installation is incompatible with this "
        "hand-eye calibration.\n"
        f"OpenCV version: {cv2.__version__}\n"
        f"cv2 location: {cv2.__file__}\n"
        f"Missing APIs: {missing}\n\n"
        "Recommended fix inside the ndi venv:\n"
        "  python -m pip uninstall -y "
        "opencv-python opencv-contrib-python "
        "opencv-python-headless "
        "opencv-contrib-python-headless\n"
        "  python -m pip install "
        "'opencv-python==4.14.0.94'\n"
    )

# =============================================================================
# USER CONFIGURATION
# =============================================================================

XARM_IP = "192.168.1.231"

SERIAL_PORT = "/dev/ttyUSB0"

ROM_FILE = (
    "/home/bartlab/Workspace/spinerobot_ros2/"
    "config/ndi_tools/8700340.rom"
)

RESULT_DIR = (
    "/home/bartlab/Workspace/spinerobot_ros2/"
    "results/handeye_v3"
)

# Conservative motion while calibrating.
ROBOT_SPEED_MM_S = 10.0
ROBOT_ACC_MM_S2 = 50.0
SETTLE_TIME_SEC = 1.5

# Polaris capture.
NDI_SAMPLES = 30
NDI_CAPTURE_TIMEOUT_SEC = 8.0

# Minimum accepted frames after outlier rejection.
MIN_ACCEPTED_NDI_FRAMES = 20

# Robust calibration-pose rejection.
POSE_OUTLIER_SIGMA = 3.5

# Cross-validation folds used to choose SHAH vs LI.
CV_FOLDS = 4

# The system-scale diagnostic is fitted ONLY on calibration poses.
# It is NOT automatically applied to the rigid transform.
ENABLE_SIMILARITY_DIAGNOSTIC = True

RETURN_TO_START_AT_END = True


# =============================================================================
# AUTOMATIC POSE DESIGN
#
# Each row:
# [dx_mm, dy_mm, dz_mm, droll_deg, dpitch_deg, dyaw_deg]
#
# The calibration poses span translations and rotations.
# The validation set deliberately contains paired 20 mm and 40 mm motions
# so that growth with workspace displacement can be tested independently.
#
# Reduce these values if your mechanical setup does not have safe clearance.
# =============================================================================

CALIBRATION_POSE_OFFSETS = [
    [   0,   0,   0,    0,   0,   0],

    [  30,   0,   0,   10,   0,   0],
    [ -30,   0,   0,  -10,   0,   0],
    [   0,  30,   0,    0,  10,   0],
    [   0, -30,   0,    0, -10,   0],
    [   0,   0,  30,    0,   0,  10],
    [   0,   0, -30,    0,   0, -10],

    [  25,  25,   0,   10,   8,   0],
    [ -25,  25,   0,  -10,   8,   0],
    [  25, -25,   0,   10,  -8,   0],
    [ -25, -25,   0,  -10,  -8,   0],

    [  25,   0,  25,    8,   0,  10],
    [ -25,   0,  25,   -8,   0,  10],
    [   0,  25, -25,    0,   8, -10],
    [   0, -25, -25,    0,  -8, -10],

    [  20,  20,  20,   12,  -8,  10],
    [ -20,  20, -20,  -12,   8, -10],
    [  20, -20, -20,    8,  12, -10],
]

# Held-out poses. None of these are used to estimate the hand-eye transform.
VALIDATION_POSE_OFFSETS = [
    # Inner workspace shell: 20 mm.
    [  20,   0,   0,    5,  -5,   7],
    [ -20,   0,   0,   -5,   5,  -7],
    [   0,  20,   0,    7,   5,  -5],
    [   0, -20,   0,   -7,  -5,   5],
    [   0,   0,  20,    5,  -7,   5],
    [   0,   0, -20,   -5,   7,  -5],

    # Outer workspace shell: 40 mm.
    [  40,   0,   0,    5, -10,  12],
    [ -40,   0,   0,   -5,  10, -12],
    [   0,  40,   0,   12,   5,  -8],
    [   0, -40,   0,  -12,  -5,   8],
    [   0,   0,  40,    8, -12,   5],
    [   0,   0, -40,   -8,  12,  -5],
]


# =============================================================================
# TRANSFORM UTILITIES
# =============================================================================

def make_transform(rotation, translation):
    """Create 4x4 homogeneous transform."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert_transform(T):
    """Fast inverse for a rigid 4x4 transform."""
    T = np.asarray(T, dtype=float)
    Rm = T[:3, :3]
    t = T[:3, 3]

    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t

    return Ti


def nearest_rotation(matrix):
    """Project a 3x3 matrix to the nearest proper rotation."""
    U, _, Vt = np.linalg.svd(matrix)
    Rm = U @ Vt

    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt

    return Rm


def average_rotation(rotations):
    """Average rotations without averaging Euler angles."""
    total = np.zeros((3, 3), dtype=float)

    for Rm in rotations:
        total += np.asarray(Rm, dtype=float)

    return nearest_rotation(total)


def rx(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ])


def ry(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array([
        [  c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,   c],
    ])


def rz(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def xarm_rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    xArm uses XYZ FIXED angles:

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Writing the formula explicitly avoids intrinsic/extrinsic Euler ambiguity.
    """
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    return rz(yaw) @ ry(pitch) @ rx(roll)


def rotation_angle_deg(Rm):
    """Geodesic magnitude of a rotation."""
    value = np.clip(
        (np.trace(Rm) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(value)
        )
    )


def transform_rotation_error_deg(T_error):
    return rotation_angle_deg(
        T_error[:3, :3]
    )


# =============================================================================
# ROBUST STATISTICS
# =============================================================================

def mad_upper_mask(values, sigma=3.5):
    """
    Keep values below median + sigma * robust_sigma.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 5:
        return np.ones(len(values), dtype=bool)

    median = np.median(values)
    mad = np.median(
        np.abs(
            values - median
        )
    )

    if mad < 1e-12:
        return np.ones(len(values), dtype=bool)

    robust_sigma = 1.4826 * mad
    limit = median + sigma * robust_sigma

    return values <= limit


def robust_position_mask(positions, sigma=4.0):
    positions = np.asarray(
        positions,
        dtype=float,
    )

    center = np.median(
        positions,
        axis=0,
    )

    radial_error = np.linalg.norm(
        positions - center,
        axis=1,
    )

    return mad_upper_mask(
        radial_error,
        sigma=sigma,
    )


def summarize_scalar_errors(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "rmse": float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        ),
        "median": float(np.median(values)),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "max": float(np.max(values)),
    }


# =============================================================================
# POLARIS VICRA
# =============================================================================

class PolarisReader:
    def __init__(self):
        if not os.path.exists(ROM_FILE):
            raise FileNotFoundError(
                ROM_FILE
            )

        settings = {
            "tracker type": "polaris",
            "serial port": SERIAL_PORT,
            "romfiles": [ROM_FILE],
        }

        print(
            "Connecting directly to Polaris Vicra..."
        )

        self.tracker = NDITracker(
            settings
        )

        self.tracker.start_tracking()

    def capture_pose(self, samples=NDI_SAMPLES):
        """
        Collect unique valid NDI frames and return a robust averaged ^T T_M.

        Returns
        -------
        dict:
          transform          4x4 ^T T_M
          position_std_mm    XYZ static SD
          quality_mean       NDI quality/error field
          accepted_frames
          raw_valid_frames
        """

        transforms = []
        qualities = []

        last_frame_id = None
        start_time = time.time()

        # Gather extra frames so outlier removal still leaves >= samples.
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
            ) = self.tracker.get_frame()

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

                if frame_id == last_frame_id:
                    time.sleep(0.003)
                    continue

                last_frame_id = frame_id

            transforms.append(
                matrix.copy()
            )

            if len(quality) > 0:
                q = float(
                    quality[0]
                )
            else:
                q = float("nan")

            qualities.append(q)

            time.sleep(0.003)

        if len(transforms) < samples:
            raise RuntimeError(
                f"Only {len(transforms)} valid NDI frames captured; "
                f"need at least {samples}."
            )

        positions = np.asarray(
            [
                T[:3, 3]
                for T in transforms
            ],
            dtype=float,
        )

        rotations = [
            T[:3, :3]
            for T in transforms
        ]

        qualities = np.asarray(
            qualities,
            dtype=float,
        )

        position_mask = robust_position_mask(
            positions,
            sigma=4.0,
        )

        # Quality-based outlier rejection if finite values are available.
        finite_q = np.isfinite(
            qualities
        )

        quality_mask = np.ones(
            len(transforms),
            dtype=bool,
        )

        if np.count_nonzero(
            finite_q
        ) >= 5:
            local_mask = mad_upper_mask(
                qualities[finite_q],
                sigma=4.0,
            )

            quality_mask[
                np.where(finite_q)[0]
            ] = local_mask

        mask = (
            position_mask
            &
            quality_mask
        )

        accepted_indices = np.where(
            mask
        )[0]

        # If robust rejection was too aggressive, keep the frames nearest
        # to the static median.
        if len(accepted_indices) < MIN_ACCEPTED_NDI_FRAMES:
            center = np.median(
                positions,
                axis=0,
            )

            distance = np.linalg.norm(
                positions - center,
                axis=1,
            )

            accepted_indices = np.argsort(
                distance
            )[:samples]

        # Limit to requested stationary sample count.
        if len(accepted_indices) > samples:
            accepted_indices = accepted_indices[:samples]

        accepted_positions = positions[
            accepted_indices
        ]

        accepted_rotations = [
            rotations[i]
            for i
            in accepted_indices
        ]

        accepted_quality = qualities[
            accepted_indices
        ]

        accepted_quality = accepted_quality[
            np.isfinite(
                accepted_quality
            )
        ]

        T_tm = np.eye(4, dtype=float)

        T_tm[:3, 3] = np.mean(
            accepted_positions,
            axis=0,
        )

        T_tm[:3, :3] = average_rotation(
            accepted_rotations
        )

        std_xyz = np.std(
            accepted_positions,
            axis=0,
        )

        q_mean = (
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
            "transform": T_tm,
            "position_std_mm": std_xyz,
            "quality_mean": q_mean,
            "accepted_frames": int(
                len(
                    accepted_indices
                )
            ),
            "raw_valid_frames": int(
                len(
                    transforms
                )
            ),
        }

    def close(self):
        try:
            self.tracker.stop_tracking()
        except Exception:
            pass

        try:
            self.tracker.close()
        except Exception:
            pass


# =============================================================================
# XARM
# =============================================================================

class XArmReader:
    def __init__(self, ip):
        print(
            f"Connecting to xArm at {ip}..."
        )

        self.arm = XArmAPI(
            ip,
            is_radian=False,
        )

        if not self.arm.connected:
            raise RuntimeError(
                f"Could not connect to xArm at {ip}"
            )

        code, err_warn = (
            self.arm.get_err_warn_code()
        )

        if code != 0:
            raise RuntimeError(
                f"get_err_warn_code failed, code={code}"
            )

        print(
            f"xArm error/warning state: {err_warn}"
        )

        if (
            len(err_warn) >= 1
            and
            err_warn[0] != 0
        ):
            raise RuntimeError(
                f"xArm error code {err_warn[0]}; "
                "resolve before calibration."
            )

        for operation, code in [
            (
                "motion_enable",
                self.arm.motion_enable(
                    enable=True
                ),
            ),
            (
                "set_mode",
                self.arm.set_mode(0),
            ),
            (
                "set_state",
                self.arm.set_state(
                    state=0
                ),
            ),
        ]:
            if code != 0:
                raise RuntimeError(
                    f"{operation} failed, code={code}"
                )

        time.sleep(1.0)

    def get_pose_vector(self):
        code, pose = self.arm.get_position(
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

    def get_transform(self):
        """
        Returns ^B T_G.
        """
        pose = self.get_pose_vector()

        R_bg = xarm_rpy_deg_to_matrix(
            pose[3],
            pose[4],
            pose[5],
        )

        return make_transform(
            R_bg,
            pose[:3],
        )

    def move_absolute(self, pose, name):
        pose = np.asarray(
            pose,
            dtype=float,
        )

        print(
            f"\nMoving: {name}"
        )

        print(
            "  XYZ [mm] = "
            f"[{pose[0]:.3f}, "
            f"{pose[1]:.3f}, "
            f"{pose[2]:.3f}]"
        )

        print(
            "  RPY [deg] = "
            f"[{pose[3]:.3f}, "
            f"{pose[4]:.3f}, "
            f"{pose[5]:.3f}]"
        )

        code = self.arm.set_position(
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

    def disconnect(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


# =============================================================================
# DATA ACQUISITION
# =============================================================================

def pose_from_offset(start_pose, offset):
    """
    Construct an xArm command by adding translation and fixed-RPY increments
    to the measured starting pose.

    The offsets are chosen only to create pose diversity; they do not need
    to represent pure local-axis rotations.
    """
    target = np.asarray(
        start_pose,
        dtype=float,
    ).copy()

    offset = np.asarray(
        offset,
        dtype=float,
    )

    target[:3] += offset[:3]
    target[3:6] += offset[3:6]

    return target


def capture_pose_pair(
        arm,
        ndi,
        target_pose,
        label):

    arm.move_absolute(
        target_pose,
        label,
    )

    # Robot is stationary at this point.  Capture robot transform and then
    # average NDI frames. Static acquisition makes synchronization error small.
    T_bg = arm.get_transform()

    ndi_capture = ndi.capture_pose(
        NDI_SAMPLES
    )

    T_tm = ndi_capture[
        "transform"
    ]

    print(
        "  NDI SD [mm] = "
        f"{ndi_capture['position_std_mm']}"
    )

    print(
        "  NDI quality/error = "
        f"{ndi_capture['quality_mean']:.4f}"
    )

    return {
        "label": label,
        "target_pose": np.asarray(
            target_pose,
            dtype=float,
        ),
        "T_base_gripper": T_bg,
        "T_tracker_marker": T_tm,
        "ndi_position_std_mm":
            ndi_capture[
                "position_std_mm"
            ],
        "ndi_quality_mean":
            ndi_capture[
                "quality_mean"
            ],
        "ndi_accepted_frames":
            ndi_capture[
                "accepted_frames"
            ],
    }


# =============================================================================
# ROBOT-WORLD / HAND-EYE SOLVER
# =============================================================================

def solve_robot_world_handeye(
        samples,
        method):

    """
    Solve:

        ^B T_Gi  ^G T_M
        =
        ^B T_T   ^T T_Mi

    OpenCV expects:
        ^camera T_world
        ^gripper T_base

    Relabel:
        world   = Tracker T
        camera  = Marker M
        base    = Robot Base B
        gripper = Robot TCP G

    Inputs:
        ^M T_T = inv(^T T_M)
        ^G T_B = inv(^B T_G)

    OpenCV outputs:
        ^T T_B
        ^M T_G

    Desired:
        ^B T_T = inverse(^T T_B)
        ^G T_M = inverse(^M T_G)
    """

    if len(samples) < 4:
        raise ValueError(
            "Need at least four diverse poses."
        )

    R_world2cam = []
    t_world2cam = []

    R_base2gripper = []
    t_base2gripper = []

    for sample in samples:
        T_bg = sample[
            "T_base_gripper"
        ]

        T_tm = sample[
            "T_tracker_marker"
        ]

        T_mt = invert_transform(
            T_tm
        )

        T_gb = invert_transform(
            T_bg
        )

        R_world2cam.append(
            T_mt[:3, :3]
        )

        t_world2cam.append(
            T_mt[:3, 3].reshape(3, 1)
        )

        R_base2gripper.append(
            T_gb[:3, :3]
        )

        t_base2gripper.append(
            T_gb[:3, 3].reshape(3, 1)
        )

    (
        R_base2world,
        t_base2world,
        R_gripper2cam,
        t_gripper2cam,
    ) = cv2.calibrateRobotWorldHandEye(
        R_world2cam,
        t_world2cam,
        R_base2gripper,
        t_base2gripper,
        method=method,
    )

    # OpenCV outputs ^T T_B.
    T_tb = make_transform(
        R_base2world,
        np.asarray(
            t_base2world
        ).reshape(3),
    )

    # OpenCV outputs ^M T_G.
    T_mg = make_transform(
        R_gripper2cam,
        np.asarray(
            t_gripper2cam
        ).reshape(3),
    )

    # Convert to desired directions.
    T_bt = invert_transform(
        T_tb
    )

    T_gm = invert_transform(
        T_mg
    )

    return T_bt, T_gm


# =============================================================================
# RESIDUALS
# =============================================================================

def closure_for_sample(
        sample,
        T_bt,
        T_gm):

    """
    Compare the same marker pose in robot-base coordinates:

      robot prediction:
          ^B T_M(robot) = ^B T_G ^G T_M

      tracker measurement:
          ^B T_M(ndi)   = ^B T_T ^T T_M

    Perfect calibration gives equal transforms.
    """

    T_bg = sample[
        "T_base_gripper"
    ]

    T_tm = sample[
        "T_tracker_marker"
    ]

    T_bm_robot = (
        T_bg
        @
        T_gm
    )

    T_bm_ndi = (
        T_bt
        @
        T_tm
    )

    T_error = (
        invert_transform(
            T_bm_robot
        )
        @
        T_bm_ndi
    )

    translation_vector_base = (
        T_bm_ndi[:3, 3]
        -
        T_bm_robot[:3, 3]
    )

    return {
        "T_bm_robot":
            T_bm_robot,
        "T_bm_ndi":
            T_bm_ndi,
        "translation_error_vector_mm":
            translation_vector_base,
        "translation_error_mm":
            float(
                np.linalg.norm(
                    translation_vector_base
                )
            ),
        "rotation_error_deg":
            transform_rotation_error_deg(
                T_error
            ),
    }


def evaluate_samples(
        samples,
        T_bt,
        T_gm):

    rows = []

    for sample in samples:
        closure = closure_for_sample(
            sample,
            T_bt,
            T_gm,
        )

        row = {
            "label":
                sample[
                    "label"
                ],
            "translation_error_vector_mm":
                closure[
                    "translation_error_vector_mm"
                ],
            "translation_error_mm":
                closure[
                    "translation_error_mm"
                ],
            "rotation_error_deg":
                closure[
                    "rotation_error_deg"
                ],
            "robot_marker_position_base_mm":
                closure[
                    "T_bm_robot"
                ][:3, 3],
            "ndi_marker_position_base_mm":
                closure[
                    "T_bm_ndi"
                ][:3, 3],
        }

        rows.append(row)

    translation_errors = np.asarray(
        [
            row[
                "translation_error_mm"
            ]
            for row in rows
        ],
        dtype=float,
    )

    rotation_errors = np.asarray(
        [
            row[
                "rotation_error_deg"
            ]
            for row in rows
        ],
        dtype=float,
    )

    return {
        "rows": rows,
        "translation_summary":
            summarize_scalar_errors(
                translation_errors
            ),
        "rotation_summary":
            summarize_scalar_errors(
                rotation_errors
            ),
    }


def print_evaluation(title, evaluation):
    print(
        "\n"
        +
        "=" * 88
    )

    print(title)

    print(
        "=" * 88
    )

    print(
        f"{'Pose':<10} "
        f"{'Ex mm':>9} "
        f"{'Ey mm':>9} "
        f"{'Ez mm':>9} | "
        f"{'3D mm':>9} "
        f"{'Rot deg':>9}"
    )

    print(
        "-" * 88
    )

    for row in evaluation[
        "rows"
    ]:
        e = row[
            "translation_error_vector_mm"
        ]

        print(
            f"{row['label']:<10} "
            f"{e[0]:>+9.3f} "
            f"{e[1]:>+9.3f} "
            f"{e[2]:>+9.3f} | "
            f"{row['translation_error_mm']:>9.3f} "
            f"{row['rotation_error_deg']:>9.4f}"
        )

    t = evaluation[
        "translation_summary"
    ]

    r = evaluation[
        "rotation_summary"
    ]

    print(
        "\nTranslation:"
    )

    print(
        f"  Mean = {t['mean']:.4f} mm"
    )

    print(
        f"  RMSE = {t['rmse']:.4f} mm"
    )

    print(
        f"  P95  = {t['p95']:.4f} mm"
    )

    print(
        f"  Max  = {t['max']:.4f} mm"
    )

    print(
        "Rotation:"
    )

    print(
        f"  Mean = {r['mean']:.4f} deg"
    )

    print(
        f"  RMSE = {r['rmse']:.4f} deg"
    )

    print(
        f"  P95  = {r['p95']:.4f} deg"
    )

    print(
        f"  Max  = {r['max']:.4f} deg"
    )


# =============================================================================
# SOLVER CROSS-VALIDATION
# =============================================================================

def deterministic_folds(
        n,
        num_folds):

    num_folds = max(
        2,
        min(
            int(num_folds),
            n,
        ),
    )

    indices = np.arange(n)

    return [
        indices[
            fold_index::num_folds
        ]
        for fold_index
        in range(
            num_folds
        )
    ]


def cross_validate_method(
        samples,
        method,
        num_folds=CV_FOLDS):

    folds = deterministic_folds(
        len(samples),
        num_folds,
    )

    all_translation_errors = []
    all_rotation_errors = []

    all_indices = np.arange(
        len(samples)
    )

    for validation_indices in folds:
        validation_indices = set(
            validation_indices.tolist()
        )

        train_samples = [
            sample
            for index, sample
            in enumerate(samples)
            if index not in validation_indices
        ]

        fold_samples = [
            sample
            for index, sample
            in enumerate(samples)
            if index in validation_indices
        ]

        if len(train_samples) < 4:
            continue

        try:
            T_bt, T_gm = (
                solve_robot_world_handeye(
                    train_samples,
                    method,
                )
            )

            evaluation = evaluate_samples(
                fold_samples,
                T_bt,
                T_gm,
            )

            all_translation_errors.extend(
                [
                    row[
                        "translation_error_mm"
                    ]
                    for row
                    in evaluation[
                        "rows"
                    ]
                ]
            )

            all_rotation_errors.extend(
                [
                    row[
                        "rotation_error_deg"
                    ]
                    for row
                    in evaluation[
                        "rows"
                    ]
                ]
            )

        except Exception:
            # A failed fold makes the method less attractive.
            return {
                "translation_rmse_mm":
                    float("inf"),
                "rotation_rmse_deg":
                    float("inf"),
            }

    if not all_translation_errors:
        return {
            "translation_rmse_mm":
                float("inf"),
            "rotation_rmse_deg":
                float("inf"),
        }

    all_translation_errors = np.asarray(
        all_translation_errors,
        dtype=float,
    )

    all_rotation_errors = np.asarray(
        all_rotation_errors,
        dtype=float,
    )

    return {
        "translation_rmse_mm":
            float(
                np.sqrt(
                    np.mean(
                        all_translation_errors ** 2
                    )
                )
            ),
        "rotation_rmse_deg":
            float(
                np.sqrt(
                    np.mean(
                        all_rotation_errors ** 2
                    )
                )
            ),
    }


# =============================================================================
# ROBUST REFIT
# =============================================================================

def robust_refit(
        samples,
        method):

    T_bt, T_gm = (
        solve_robot_world_handeye(
            samples,
            method,
        )
    )

    initial = evaluate_samples(
        samples,
        T_bt,
        T_gm,
    )

    translation_errors = np.asarray(
        [
            row[
                "translation_error_mm"
            ]
            for row
            in initial[
                "rows"
            ]
        ],
        dtype=float,
    )

    rotation_errors = np.asarray(
        [
            row[
                "rotation_error_deg"
            ]
            for row
            in initial[
                "rows"
            ]
        ],
        dtype=float,
    )

    translation_mask = mad_upper_mask(
        translation_errors,
        sigma=POSE_OUTLIER_SIGMA,
    )

    rotation_mask = mad_upper_mask(
        rotation_errors,
        sigma=POSE_OUTLIER_SIGMA,
    )

    keep_mask = (
        translation_mask
        &
        rotation_mask
    )

    inlier_samples = [
        sample
        for sample, keep
        in zip(
            samples,
            keep_mask,
        )
        if keep
    ]

    if (
        len(inlier_samples) >= 8
        and
        len(inlier_samples)
        < len(samples)
    ):
        rejected = [
            sample[
                "label"
            ]
            for sample, keep
            in zip(
                samples,
                keep_mask,
            )
            if not keep
        ]

        print(
            "\nRobust refit rejected calibration poses:"
        )

        print(
            "  "
            +
            ", ".join(
                rejected
            )
        )

        T_bt, T_gm = (
            solve_robot_world_handeye(
                inlier_samples,
                method,
            )
        )

    else:
        inlier_samples = list(
            samples
        )

        rejected = []

    return (
        T_bt,
        T_gm,
        inlier_samples,
        rejected,
    )


# =============================================================================
# ERROR-GROWTH ANALYSIS
# =============================================================================

def workspace_growth_analysis(
        evaluation,
        calibration_centroid):

    rows = evaluation[
        "rows"
    ]

    distances = []
    errors = []

    for row in rows:
        p = np.asarray(
            row[
                "robot_marker_position_base_mm"
            ],
            dtype=float,
        )

        distance = float(
            np.linalg.norm(
                p - calibration_centroid
            )
        )

        distances.append(
            distance
        )

        errors.append(
            row[
                "translation_error_mm"
            ]
        )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    if (
        len(distances) < 3
        or
        np.std(distances) < 1e-9
    ):
        slope = float("nan")
        intercept = float("nan")
        correlation = float("nan")

    else:
        slope, intercept = np.polyfit(
            distances,
            errors,
            1,
        )

        correlation = float(
            np.corrcoef(
                distances,
                errors,
            )[0, 1]
        )

    return {
        "distance_mm":
            distances,
        "error_mm":
            errors,
        "slope_mm_error_per_mm":
            float(slope),
        "slope_mm_error_per_100mm":
            float(slope * 100.0),
        "intercept_mm":
            float(intercept),
        "correlation":
            correlation,
    }


def shell_summary(
        validation_samples,
        validation_evaluation):

    groups = {
        "20_mm_shell": [],
        "40_mm_shell": [],
    }

    for sample, row in zip(
        validation_samples,
        validation_evaluation[
            "rows"
        ],
    ):
        start_xyz = np.asarray(
            sample[
                "target_pose"
            ][:3],
            dtype=float,
        )

        # label already contains VAL; determine shell from stored original offset.
        offset = np.asarray(
            sample[
                "pose_offset"
            ][:3],
            dtype=float,
        )

        distance = float(
            np.linalg.norm(
                offset
            )
        )

        if abs(
            distance - 20.0
        ) < 1e-6:
            groups[
                "20_mm_shell"
            ].append(
                row[
                    "translation_error_mm"
                ]
            )

        elif abs(
            distance - 40.0
        ) < 1e-6:
            groups[
                "40_mm_shell"
            ].append(
                row[
                    "translation_error_mm"
                ]
            )

    result = {}

    for name, values in groups.items():
        if values:
            result[name] = (
                summarize_scalar_errors(
                    np.asarray(
                        values,
                        dtype=float,
                    )
                )
            )

    return result


# =============================================================================
# LOW-DIMENSIONAL SIMILARITY / SCALE DIAGNOSTIC
# =============================================================================

def fit_isotropic_similarity_on_positions(
        evaluation):

    """
    Fit a low-dimensional relation:

        p_robot ~= c_robot + s * R_delta * (p_ndi - c_ndi)

    using only CALIBRATION pose positions.

    This is a DIAGNOSTIC to test scale-like growing error.
    It is not merged into the rigid hand-eye transform.
    """

    p_robot = np.asarray(
        [
            row[
                "robot_marker_position_base_mm"
            ]
            for row
            in evaluation[
                "rows"
            ]
        ],
        dtype=float,
    )

    p_ndi = np.asarray(
        [
            row[
                "ndi_marker_position_base_mm"
            ]
            for row
            in evaluation[
                "rows"
            ]
        ],
        dtype=float,
    )

    c_robot = np.mean(
        p_robot,
        axis=0,
    )

    c_ndi = np.mean(
        p_ndi,
        axis=0,
    )

    X = p_ndi - c_ndi
    Y = p_robot - c_robot

    # Kabsch rotation between the already-nearly-aligned point clouds.
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)

    R_delta = Vt.T @ U.T

    if np.linalg.det(
        R_delta
    ) < 0:
        Vt[-1, :] *= -1
        R_delta = Vt.T @ U.T

    X_rot = (
        R_delta
        @
        X.T
    ).T

    numerator = float(
        np.sum(
            Y * X_rot
        )
    )

    denominator = float(
        np.sum(
            X_rot * X_rot
        )
    )

    scale = (
        numerator
        /
        denominator
        if denominator > 1e-12
        else 1.0
    )

    return {
        "scale":
            float(scale),
        "R_delta":
            R_delta,
        "center_robot":
            c_robot,
        "center_ndi":
            c_ndi,
    }


def evaluate_similarity(
        evaluation,
        model):

    errors = []
    rows = []

    scale = float(
        model[
            "scale"
        ]
    )

    R_delta = np.asarray(
        model[
            "R_delta"
        ],
        dtype=float,
    )

    c_robot = np.asarray(
        model[
            "center_robot"
        ],
        dtype=float,
    )

    c_ndi = np.asarray(
        model[
            "center_ndi"
        ],
        dtype=float,
    )

    for row in evaluation[
        "rows"
    ]:
        p_robot = np.asarray(
            row[
                "robot_marker_position_base_mm"
            ],
            dtype=float,
        )

        p_ndi = np.asarray(
            row[
                "ndi_marker_position_base_mm"
            ],
            dtype=float,
        )

        p_corrected = (
            c_robot
            +
            scale
            *
            R_delta
            @
            (
                p_ndi
                -
                c_ndi
            )
        )

        error_vector = (
            p_corrected
            -
            p_robot
        )

        error = float(
            np.linalg.norm(
                error_vector
            )
        )

        errors.append(
            error
        )

        rows.append({
            "label":
                row[
                    "label"
                ],
            "error_vector_mm":
                error_vector,
            "error_mm":
                error,
        })

    return {
        "rows": rows,
        "summary":
            summarize_scalar_errors(
                np.asarray(
                    errors,
                    dtype=float,
                )
            ),
    }


# =============================================================================
# JSON CONVERSION
# =============================================================================

def to_jsonable(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            key:
                to_jsonable(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            to_jsonable(
                item
            )
            for item in value
        ]

    return value


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    arm = None
    ndi = None
    start_pose = None

    try:
        arm = XArmReader(
            XARM_IP
        )

        ndi = PolarisReader()

        start_pose = arm.get_pose_vector()

        print(
            "\n"
            +
            "=" * 82
        )

        print(
            "HAND-EYE CALIBRATION V3"
        )

        print(
            "=" * 82
        )

        print(
            "Starting xArm pose:"
        )

        print(
            f"  XYZ [mm] = "
            f"{start_pose[:3]}"
        )

        print(
            f"  RPY [deg] = "
            f"{start_pose[3:6]}"
        )

        print(
            f"\nCalibration poses: "
            f"{len(CALIBRATION_POSE_OFFSETS)}"
        )

        print(
            f"Held-out validation poses: "
            f"{len(VALIDATION_POSE_OFFSETS)}"
        )

        print(
            "\nThe calibration set contains both translation and rotation."
        )

        print(
            "The validation set is NEVER used to estimate the rigid transform."
        )

        print(
            "\nVALIDATION WORKSPACE SHELLS:"
        )

        print(
            "  ±20 mm axis motions"
        )

        print(
            "  ±40 mm axis motions"
        )

        print(
            "\nCheck that ALL target poses are collision-free and reachable."
        )

        print(
            "Marker 8700340 must remain rigidly attached throughout the run."
        )

        confirmation = input(
            "\nType CALIBRATE to execute all motions: "
        ).strip()

        if confirmation != "CALIBRATE":
            print(
                "Cancelled."
            )
            return

        # ---------------------------------------------------------------------
        # Acquire calibration poses.
        # ---------------------------------------------------------------------

        calibration_samples = []

        for index, offset in enumerate(
            CALIBRATION_POSE_OFFSETS,
            start=1,
        ):
            target = pose_from_offset(
                start_pose,
                offset,
            )

            sample = capture_pose_pair(
                arm,
                ndi,
                target,
                f"CAL{index:02d}",
            )

            sample[
                "pose_offset"
            ] = np.asarray(
                offset,
                dtype=float,
            )

            calibration_samples.append(
                sample
            )

        # ---------------------------------------------------------------------
        # Choose solver on calibration data only.
        # ---------------------------------------------------------------------

        methods = {
            "SHAH":
                cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
            "LI":
                cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
        }

        cv_results = {}

        print(
            "\n"
            +
            "=" * 82
        )

        print(
            "SOLVER CROSS-VALIDATION "
            "(CALIBRATION POSES ONLY)"
        )

        print(
            "=" * 82
        )

        for name, method in methods.items():
            cv_result = cross_validate_method(
                calibration_samples,
                method,
                CV_FOLDS,
            )

            cv_results[
                name
            ] = cv_result

            print(
                f"{name:<8} | "
                f"translation CV RMSE = "
                f"{cv_result['translation_rmse_mm']:.4f} mm | "
                f"rotation CV RMSE = "
                f"{cv_result['rotation_rmse_deg']:.4f} deg"
            )

        selected_name = min(
            cv_results,
            key=lambda name:
                cv_results[name][
                    "translation_rmse_mm"
                ],
        )

        selected_method = methods[
            selected_name
        ]

        print(
            f"\nSelected solver: "
            f"{selected_name}"
        )

        # ---------------------------------------------------------------------
        # Robust final fit using calibration poses only.
        # ---------------------------------------------------------------------

        (
            T_bt,
            T_gm,
            calibration_inliers,
            rejected_calibration_labels,
        ) = robust_refit(
            calibration_samples,
            selected_method,
        )

        calibration_evaluation = (
            evaluate_samples(
                calibration_samples,
                T_bt,
                T_gm,
            )
        )

        print_evaluation(
            "FINAL CALIBRATION-SET CLOSURE",
            calibration_evaluation,
        )

        print(
            "\n^B T_T  Tracker -> Robot Base:"
        )

        print(
            T_bt
        )

        print(
            "\n^G T_M  Marker -> xArm TCP:"
        )

        print(
            T_gm
        )

        # ---------------------------------------------------------------------
        # Acquire held-out validation poses.
        # ---------------------------------------------------------------------

        validation_samples = []

        for index, offset in enumerate(
            VALIDATION_POSE_OFFSETS,
            start=1,
        ):
            target = pose_from_offset(
                start_pose,
                offset,
            )

            sample = capture_pose_pair(
                arm,
                ndi,
                target,
                f"VAL{index:02d}",
            )

            sample[
                "pose_offset"
            ] = np.asarray(
                offset,
                dtype=float,
            )

            validation_samples.append(
                sample
            )

        validation_evaluation = (
            evaluate_samples(
                validation_samples,
                T_bt,
                T_gm,
            )
        )

        print_evaluation(
            "HELD-OUT HAND-EYE VALIDATION",
            validation_evaluation,
        )

        # ---------------------------------------------------------------------
        # 20 mm vs 40 mm shell analysis.
        # ---------------------------------------------------------------------

        shell_results = shell_summary(
            validation_samples,
            validation_evaluation,
        )

        print(
            "\n"
            +
            "=" * 82
        )

        print(
            "20 mm vs 40 mm GROWING-ERROR TEST"
        )

        print(
            "=" * 82
        )

        for shell_name, summary in (
            shell_results.items()
        ):
            print(
                f"{shell_name:<15} | "
                f"N={summary['n']:>2d} | "
                f"RMSE={summary['rmse']:.4f} mm | "
                f"Mean={summary['mean']:.4f} mm | "
                f"P95={summary['p95']:.4f} mm | "
                f"Max={summary['max']:.4f} mm"
            )

        # ---------------------------------------------------------------------
        # Error-vs-workspace-distance slope.
        # ---------------------------------------------------------------------

        calibration_marker_positions = np.asarray(
            [
                row[
                    "robot_marker_position_base_mm"
                ]
                for row
                in calibration_evaluation[
                    "rows"
                ]
            ],
            dtype=float,
        )

        calibration_centroid = np.mean(
            calibration_marker_positions,
            axis=0,
        )

        growth = workspace_growth_analysis(
            validation_evaluation,
            calibration_centroid,
        )

        print(
            "\n"
            +
            "=" * 82
        )

        print(
            "WORKSPACE ERROR-GROWTH DIAGNOSTIC"
        )

        print(
            "=" * 82
        )

        print(
            f"Error slope = "
            f"{growth['slope_mm_error_per_mm']:.6f} "
            f"mm error/mm workspace distance"
        )

        print(
            f"            = "
            f"{growth['slope_mm_error_per_100mm']:.4f} "
            f"mm error per 100 mm"
        )

        print(
            f"Correlation(error,distance) = "
            f"{growth['correlation']:.4f}"
        )

        # ---------------------------------------------------------------------
        # Optional similarity/scale A/B diagnostic.
        # ---------------------------------------------------------------------

        similarity_model = None
        similarity_calibration = None
        similarity_validation = None

        if ENABLE_SIMILARITY_DIAGNOSTIC:
            similarity_model = (
                fit_isotropic_similarity_on_positions(
                    calibration_evaluation
                )
            )

            similarity_calibration = (
                evaluate_similarity(
                    calibration_evaluation,
                    similarity_model,
                )
            )

            similarity_validation = (
                evaluate_similarity(
                    validation_evaluation,
                    similarity_model,
                )
            )

            rigid_rmse = (
                validation_evaluation[
                    "translation_summary"
                ][
                    "rmse"
                ]
            )

            compensated_rmse = (
                similarity_validation[
                    "summary"
                ][
                    "rmse"
                ]
            )

            improvement_pct = (
                100.0
                *
                (
                    rigid_rmse
                    -
                    compensated_rmse
                )
                /
                rigid_rmse
                if rigid_rmse > 1e-12
                else 0.0
            )

            print(
                "\n"
                +
                "=" * 82
            )

            print(
                "OPTIONAL SYSTEM-SCALE DIAGNOSTIC"
            )

            print(
                "=" * 82
            )

            print(
                "This model is fitted on CALIBRATION positions only."
            )

            print(
                f"Fitted isotropic scale = "
                f"{similarity_model['scale']:.8f}"
            )

            print(
                f"Rigid held-out RMSE = "
                f"{rigid_rmse:.4f} mm"
            )

            print(
                f"Similarity held-out RMSE = "
                f"{compensated_rmse:.4f} mm"
            )

            print(
                f"Held-out improvement = "
                f"{improvement_pct:+.2f}%"
            )

            if improvement_pct >= 15.0:
                print(
                    "\nMeaningful held-out improvement detected. "
                    "Repeat this experiment across several independent "
                    "sessions before considering a system compensation layer."
                )

            elif improvement_pct > 0.0:
                print(
                    "\nOnly a small held-out improvement. "
                    "Do not add scale compensation yet."
                )

            else:
                print(
                    "\nThe similarity model did not improve held-out accuracy. "
                    "Do not use scale compensation."
                )

        # ---------------------------------------------------------------------
        # Save everything.
        # ---------------------------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        json_path = os.path.join(
            RESULT_DIR,
            f"handeye_v3_{timestamp}.json",
        )

        npz_path = os.path.join(
            RESULT_DIR,
            f"handeye_v3_{timestamp}.npz",
        )

        result = {
            "timestamp":
                timestamp,
            "xarm_ip":
                XARM_IP,
            "ndi_rom":
                ROM_FILE,
            "ndi_samples_per_pose":
                NDI_SAMPLES,
            "selected_solver":
                selected_name,
            "solver_cross_validation":
                cv_results,
            "rejected_calibration_poses":
                rejected_calibration_labels,

            "T_base_from_tracker":
                T_bt,
            "T_gripper_from_marker":
                T_gm,

            "calibration_translation_summary":
                calibration_evaluation[
                    "translation_summary"
                ],
            "calibration_rotation_summary":
                calibration_evaluation[
                    "rotation_summary"
                ],

            "validation_translation_summary":
                validation_evaluation[
                    "translation_summary"
                ],
            "validation_rotation_summary":
                validation_evaluation[
                    "rotation_summary"
                ],

            "shell_results":
                shell_results,
            "workspace_growth":
                growth,

            "similarity_model":
                similarity_model,
            "similarity_calibration":
                similarity_calibration,
            "similarity_validation":
                similarity_validation,

            "calibration_rows":
                calibration_evaluation[
                    "rows"
                ],
            "validation_rows":
                validation_evaluation[
                    "rows"
                ],

            "notes": (
                "Rigid robot-world/hand-eye AX=ZB calibration. "
                "Direct Polaris capture. xArm fixed XYZ RPY. "
                "Similarity compensation is diagnostic only and is "
                "not merged into the saved rigid transforms."
            ),
        }

        with open(
            json_path,
            "w",
        ) as file:
            json.dump(
                to_jsonable(
                    result
                ),
                file,
                indent=2,
            )

        np.savez(
            npz_path,
            T_base_from_tracker=T_bt,
            T_gripper_from_marker=T_gm,
            calibration_centroid_base_mm=
                calibration_centroid,
        )

        latest_npz_path = os.path.join(
            RESULT_DIR,
            "latest_handeye.npz",
        )

        np.savez(
            latest_npz_path,
            T_base_from_tracker=T_bt,
            T_gripper_from_marker=T_gm,
            calibration_centroid_base_mm=
                calibration_centroid,
        )

        print(
            "\nSaved:"
        )

        print(
            f"  {json_path}"
        )

        print(
            f"  {npz_path}"
        )

        print(
            f"  {latest_npz_path}"
        )

    finally:
        if (
            arm is not None
            and
            start_pose is not None
            and
            RETURN_TO_START_AT_END
        ):
            try:
                arm.move_absolute(
                    start_pose,
                    "Return to original start pose",
                )
            except Exception as exc:
                print(
                    f"WARNING: return-to-start failed: {exc}"
                )

        if ndi is not None:
            ndi.close()

        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()
