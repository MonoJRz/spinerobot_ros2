from __future__ import annotations

import json
import math
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QTimer, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import rclpy
    from rclpy.node import Node
    from spinerobot_interfaces.msg import TrackingStatus
    from geometry_msgs.msg import PointStamped
    from rclpy.qos import qos_profile_sensor_data
    from xarm_msgs.msg import RobotMsg
    from xarm_msgs.srv import Call, SetInt16, SetInt16ById
    ROS_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    rclpy = None
    Node = object
    RobotMsg = None
    Call = None
    SetInt16 = None
    SetInt16ById = None
    ROS_IMPORT_ERROR = exc


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from local_processes import discover, stop as stop_local_process

CONFIG_PATH = Path(
    os.environ.get("BART_ROBOT_CONSOLE_CONFIG", str(HERE / "config.json"))
).expanduser()

APP_STYLE = """
QWidget { background: #081311; color: #dcebea; font-family: "DejaVu Sans"; font-size: 13px; }
QMainWindow { background: #06100f; }
QGroupBox {
    border: 1px solid #294b4d; border-radius: 8px; margin-top: 14px;
    padding: 12px; font-weight: 700; color: #d9eeee;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QLabel#Title { font-size: 23px; font-weight: 700; color: #eef8f7; }
QLabel#Subtitle, QLabel#Key { color: #7faaa9; }
QLabel#Value { color: #e4f1f0; font-weight: 600; }
QPushButton {
    min-height: 34px; border: 1px solid #315d60; border-radius: 5px;
    padding: 4px 12px; background: #102927; color: #e6f3f2; font-weight: 600;
}
QPushButton:hover { background: #17413e; }
QPushButton:pressed { background: #0c201f; }
QPushButton#Primary { background: #166d77; border-color: #238f9b; }
QPushButton#Danger { background: #542522; border-color: #8d413a; }
QTextEdit {
    border: 1px solid #294b4d; border-radius: 6px; background: #050d0c;
    color: #a8c9c7; font-family: monospace; font-size: 11px;
}
"""


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_repo_path(value: str) -> Path:
    candidate = Path(os.path.expandvars(value)).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


class StatusBadge(QLabel):
    COLORS = {
        "ready": ("#83e3a1", "#123324", "#2b6b49"),
        "warning": ("#ffd978", "#352c12", "#776029"),
        "error": ("#ff9389", "#3b1d1a", "#814039"),
        "waiting": ("#9bb8b6", "#172322", "#38504e"),
        "info": ("#80d8e5", "#113139", "#28646e"),
    }

    def __init__(self, text: str = "Waiting", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self.set_status(text, "waiting")

    def set_status(self, text: str, kind: str = "waiting") -> None:
        fg, bg, border = self.COLORS.get(kind, self.COLORS["waiting"])
        self.setText(text)
        self.setStyleSheet(
            f"QLabel {{ color: {fg}; background: {bg}; border: 1px solid {border}; "
            "border-radius: 10px; padding: 4px 9px; font-weight: 700; }"
        )


class MarkerRow(QFrame):
    def __init__(self, label: str, rom_name: str, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setHorizontalSpacing(12)

        name = QLabel(label)
        name.setObjectName("Value")
        rom = QLabel(rom_name)
        rom.setObjectName("Key")

        self.badge = StatusBadge("Waiting")
        self.position = QLabel("XYZ  —")
        self.position.setObjectName("Value")
        self.quality = QLabel("Quality  —")
        self.quality.setObjectName("Key")
        self.age = QLabel("Last seen  —")
        self.age.setObjectName("Key")

        layout.addWidget(name, 0, 0)
        layout.addWidget(rom, 1, 0)
        layout.addWidget(self.badge, 0, 1, 2, 1)
        layout.addWidget(self.position, 0, 2)
        layout.addWidget(self.quality, 1, 2)
        layout.addWidget(self.age, 0, 3, 2, 1)
        layout.setColumnStretch(2, 1)

    def reset(self) -> None:
        self.badge.set_status("NO DATA", "waiting")
        self.position.setText("XYZ  —")
        self.quality.setText("Quality  —")
        self.age.setText("Last seen  —")

    def set_marker(self, visible, quality, position, age_s) -> None:
        if visible:
            self.badge.set_status("TRACKED", "ready")
            if position is not None:
                x, y, z = position
                self.position.setText(f"XYZ  {x:7.1f}  {y:7.1f}  {z:7.1f} mm")
            else:
                self.position.setText("XYZ  —")
            self.quality.setText(
                f"Quality  {quality:.3f}"
                if quality is not None and math.isfinite(quality)
                else "Quality  —"
            )
        else:
            self.badge.set_status("NOT TRACKED", "warning")
            self.position.setText("XYZ  —")
            self.quality.setText("Quality  —")

        if age_s is None or not math.isfinite(age_s):
            self.age.setText("Last seen  —")
        elif age_s < 1.0:
            self.age.setText(f"Last seen  {age_s * 1000.0:.0f} ms")
        else:
            self.age.setText(f"Last seen  {age_s:.1f} s")


if ROS_IMPORT_ERROR is None:
    class XArmNode(Node):
        def __init__(self, bridge: "RosBridge", namespace: str):
            super().__init__("bart_robot_console")
            namespace = "/" + namespace.strip("/")

            self.create_subscription(
                RobotMsg, f"{namespace}/robot_states", bridge._on_robot_state, 10
            )
            self.create_subscription(
                TrackingStatus, "/tracking/status", bridge._on_tracking_status,
                qos_profile_sensor_data,
            )
            for frame in bridge._marker_frames:
                self.create_subscription(
                    PointStamped, f"/tracking/{frame}/position",
                    lambda msg, frame=frame: bridge._on_tracking_position(frame, msg),
                    qos_profile_sensor_data,
                )
            self.motion_enable = self.create_client(SetInt16ById, f"{namespace}/motion_enable")
            self.clean_error = self.create_client(Call, f"{namespace}/clean_error")
            self.clean_warn = self.create_client(Call, f"{namespace}/clean_warn")
            self.set_state = self.create_client(SetInt16, f"{namespace}/set_state")


class RosBridge(QObject):
    ndi_connection_changed = Signal(bool)
    marker_update = Signal(int, bool, object, object, object)
    robot_connection_changed = Signal(bool)
    robot_state_changed = Signal(object)
    service_result = Signal(str, bool, str)
    log = Signal(str)

    def __init__(self, xarm_config: dict[str, Any], ndi_config: dict[str, Any], parent=None):
        super().__init__(parent)
        self.node = None
        self._owns_rclpy = False
        self._last_state_time = 0.0
        self._connected = False
        self._marker_frames = [item["frame_id"] for item in ndi_config["markers"]]
        self._marker_times = {}
        self._positions = {}
        self._last_tracking_time = None
        self.ndi_connected = False
        self._heartbeat_timeout = float(ndi_config.get("heartbeat_timeout_s", 1.0))
        self._dof = int(xarm_config.get("dof", 6))

        if ROS_IMPORT_ERROR is not None:
            QTimer.singleShot(
                0,
                lambda: self.log.emit(
                    f"ROS/xarm_msgs import failed: {ROS_IMPORT_ERROR}. Start through run.sh."
                ),
            )
        else:
            try:
                if not rclpy.ok():
                    rclpy.init(args=None, domain_id=42)
                    self._owns_rclpy = True
                self.node = XArmNode(self, xarm_config.get("namespace", "/xarm"))
            except Exception as exc:
                QTimer.singleShot(
                    0, lambda message=str(exc): self.log.emit(f"Failed to create ROS node: {message}")
                )

        self.spin_timer = QTimer(self)
        self.spin_timer.timeout.connect(self._spin_once)
        self.spin_timer.start(10)

        self.watchdog = QTimer(self)
        self.watchdog.timeout.connect(self._check_connection)
        self.watchdog.start(250)

    def _spin_once(self) -> None:
        if self.node is None or rclpy is None or not rclpy.ok():
            return
        try:
            # Drain enough callbacks for status plus positions at the tracker rate.
            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            self.log.emit(f"ROS spin error: {exc}")

    def _on_robot_state(self, msg) -> None:
        self._last_state_time = time.monotonic()
        if not self._connected:
            self._connected = True
            self.robot_connection_changed.emit(True)

        mask = (1 << self._dof) - 1
        self.robot_state_changed.emit(
            {
                "state": int(msg.state),
                "mode": int(msg.mode),
                "motion_enabled": (int(msg.mt_able) & mask) == mask,
                "err": int(msg.err),
                "warn": int(msg.warn),
                "pose": list(msg.pose),
            }
        )

    def _check_connection(self) -> None:
        now = time.monotonic()
        if self.ndi_connected and now - self._last_tracking_time > self._heartbeat_timeout:
            self.ndi_connected = False
            self.ndi_connection_changed.emit(False)
        for index, seen in list(self._marker_times.items()):
            if now - seen > self._heartbeat_timeout:
                self.marker_update.emit(index, False, None, None, None)
                del self._marker_times[index]
        if self._connected and time.monotonic() - self._last_state_time > 1.0:
            self._connected = False
            self.robot_connection_changed.emit(False)

    def _on_tracking_position(self, frame, msg) -> None:
        self._positions[frame] = (time.monotonic(), (
            msg.point.x * 1000, msg.point.y * 1000, msg.point.z * 1000,
        ))

    def _on_tracking_status(self, msg) -> None:
        now = time.monotonic()
        self._last_tracking_time = now
        if not self.ndi_connected:
            self.ndi_connected = True
            self.ndi_connection_changed.emit(True)
        if msg.frame_id not in self._marker_frames:
            return
        index = self._marker_frames.index(msg.frame_id)
        self._marker_times[index] = now
        position_time, position = self._positions.get(msg.frame_id, (0, None))
        if now - position_time > self._heartbeat_timeout:
            position = None
        self.marker_update.emit(index, bool(msg.visible and msg.valid),
                                msg.quality, position, msg.age_sec)

    def _call(self, client, request, label: str) -> None:
        if self.node is None:
            self.service_result.emit(label, False, "ROS node unavailable")
            return
        if not client.service_is_ready():
            self.service_result.emit(label, False, "Service unavailable")
            return

        future = client.call_async(request)
        future.add_done_callback(
            lambda f, action=label: self._service_done(action, f)
        )

    def _service_done(self, label: str, future) -> None:
        try:
            response = future.result()
            ret = int(getattr(response, "ret", -1))
            message = str(getattr(response, "message", "")).strip()
            detail = f"ret={ret}" + (f" — {message}" if message else "")
            self.service_result.emit(label, ret == 0, detail)
        except Exception as exc:
            self.service_result.emit(label, False, str(exc))

    def motion_enable(self, enabled: bool) -> None:
        if self.node is None or SetInt16ById is None:
            self.service_result.emit("Motion enable", False, "xarm_msgs unavailable")
            return
        request = SetInt16ById.Request()
        request.id = 8
        request.data = 1 if enabled else 0
        self._call(self.node.motion_enable, request, "Enable motion" if enabled else "Disable motion")

    def clear_error(self) -> None:
        if self.node is None or Call is None:
            self.service_result.emit("Clear error", False, "xarm_msgs unavailable")
            return
        self._call(self.node.clean_error, Call.Request(), "Clear error")

    def clear_warning(self) -> None:
        if self.node is None or Call is None:
            self.service_result.emit("Clear warning", False, "xarm_msgs unavailable")
            return
        self._call(self.node.clean_warn, Call.Request(), "Clear warning")

    def set_robot_state(self, state: int) -> None:
        if self.node is None or SetInt16 is None:
            self.service_result.emit("Set state", False, "xarm_msgs unavailable")
            return
        request = SetInt16.Request()
        request.data = int(state)
        self._call(self.node.set_state, request, f"Set state {state}")

    def shutdown(self) -> None:
        self.spin_timer.stop()
        self.watchdog.stop()
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self._owns_rclpy and rclpy is not None and rclpy.ok():
            rclpy.shutdown()


class RobotConsole(QMainWindow):
    STATE_NAMES = {0: "READY", 1: "RUNNING", 2: "SLEEPING", 3: "PAUSED", 4: "STOPPED", 5: "CONFIG CHANGED"}
    MODE_NAMES = {0: "POSITION", 1: "SERVOJ", 2: "TEACHING JOINT"}

    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.xarm_cfg = self.config["xarm"]
        self.ndi_cfg = self.config["ndi"]

        self.setWindowTitle("BART Robot Console")
        self.resize(1180, 760)
        self.setMinimumSize(980, 680)
        self._build_ui()

        self.xarm_process = QProcess(self)
        self.xarm_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.xarm_process.readyReadStandardOutput.connect(self._read_xarm_output)
        self.xarm_process.started.connect(self._xarm_process_started)
        self.xarm_process.finished.connect(self._xarm_process_finished)

        self.ndi_process = QProcess(self)
        self.ndi_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.ndi_process.readyReadStandardOutput.connect(self._read_ndi_output)
        self.ndi_process.finished.connect(self._ndi_finished)
        self.ndi_process.errorOccurred.connect(
            lambda _error: self.log(f"NDI process error: {self.ndi_process.errorString()}")
        )
        self.ros = RosBridge(self.xarm_cfg, self.ndi_cfg, parent=self)
        self.ros.ndi_connection_changed.connect(self._on_ndi_connection)
        self.ros.marker_update.connect(self._on_marker_update)
        self.ros.robot_connection_changed.connect(self._on_robot_connection)
        self.ros.robot_state_changed.connect(self._on_robot_state)
        self.ros.service_result.connect(self._on_service_result)
        self.ros.log.connect(self.log)

        self.marker_rows: list[MarkerRow] = []
        self._populate_markers()

        if ROS_IMPORT_ERROR is not None:
            self.xarm_robot_badge.set_status("ROS ERROR", "error")
    
    def _reset_ndi_markers(self) -> None:
        for row in self.marker_rows:
            row.reset()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("BART ROBOT CONSOLE")
        title.setObjectName("Title")
        subtitle = QLabel("xArm 6 + NDI Polaris Vicra   |   preflight & service control")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.ros_badge = StatusBadge("ROS STARTING")
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(self.ros_badge)
        outer.addLayout(header)

        top_actions = QHBoxLayout()
        start_all = QPushButton("START ALL")
        start_all.setObjectName("Primary")
        stop_all = QPushButton("STOP ALL")
        stop_all.setObjectName("Danger")
        start_all.clicked.connect(self.start_all)
        stop_all.clicked.connect(self.stop_all)
        top_actions.addWidget(start_all)
        top_actions.addWidget(stop_all)
        top_actions.addStretch(1)
        outer.addLayout(top_actions)

        content = QHBoxLayout()
        content.setSpacing(12)

        xarm_group = QGroupBox("xARM 6")
        xarm_layout = QVBoxLayout(xarm_group)
        grid = QGridLayout()

        self.xarm_process_badge = StatusBadge("STOPPED")
        self.xarm_robot_badge = StatusBadge("OFFLINE")
        self.xarm_motion_badge = StatusBadge("UNKNOWN")
        self.xarm_state_value = QLabel("—")
        self.xarm_mode_value = QLabel("—")
        self.xarm_error_value = QLabel("—")
        self.xarm_warn_value = QLabel("—")
        self.xarm_tcp_value = QLabel("—")
        for widget in [
            self.xarm_state_value, self.xarm_mode_value, self.xarm_error_value,
            self.xarm_warn_value, self.xarm_tcp_value
        ]:
            widget.setObjectName("Value")

        rows = [
            ("Driver process", self.xarm_process_badge),
            ("Robot link", self.xarm_robot_badge),
            ("Motion", self.xarm_motion_badge),
            ("State", self.xarm_state_value),
            ("Mode", self.xarm_mode_value),
            ("Error code", self.xarm_error_value),
            ("Warning code", self.xarm_warn_value),
            ("TCP XYZ", self.xarm_tcp_value),
        ]
        for i, (text, widget) in enumerate(rows):
            key = QLabel(text)
            key.setObjectName("Key")
            grid.addWidget(key, i, 0)
            grid.addWidget(widget, i, 1)
        xarm_layout.addLayout(grid)

        driver_buttons = QHBoxLayout()
        start_driver = QPushButton("Start driver")
        stop_driver = QPushButton("Stop driver")
        stop_driver.setObjectName("Danger")
        start_driver.clicked.connect(self.start_xarm)
        stop_driver.clicked.connect(self.stop_xarm)
        driver_buttons.addWidget(start_driver)
        driver_buttons.addWidget(stop_driver)
        xarm_layout.addLayout(driver_buttons)

        controls = QGridLayout()
        buttons = [
            ("Enable motion", lambda: self.ros.motion_enable(True), 0, 0, ""),
            ("Disable motion", lambda: self.ros.motion_enable(False), 0, 1, ""),
            ("Clear error", self.ros_clear_error, 1, 0, ""),
            ("Clear warning", self.ros_clear_warning, 1, 1, ""),
            ("Set READY", lambda: self.ros.set_robot_state(0), 2, 0, ""),
            ("Set STOPPED", lambda: self.ros.set_robot_state(4), 2, 1, "Danger"),
        ]
        for text, slot, row, col, obj in buttons:
            button = QPushButton(text)
            if obj:
                button.setObjectName(obj)
            button.clicked.connect(slot)
            controls.addWidget(button, row, col)
        xarm_layout.addLayout(controls)
        xarm_layout.addStretch(1)

        ndi_group = QGroupBox("NDI POLARIS VICRA")
        ndi_layout = QVBoxLayout(ndi_group)
        ndi_top = QHBoxLayout()
        key = QLabel("Tracking system")
        key.setObjectName("Key")
        self.ndi_badge = StatusBadge("STOPPED")
        ndi_top.addWidget(key)
        ndi_top.addStretch(1)
        ndi_top.addWidget(self.ndi_badge)
        ndi_layout.addLayout(ndi_top)

        self.marker_container = QVBoxLayout()
        ndi_layout.addLayout(self.marker_container)

        ndi_buttons = QHBoxLayout()
        start_tracking = QPushButton("Start tracking")
        stop_tracking = QPushButton("Stop tracking")
        stop_tracking.setObjectName("Danger")
        start_tracking.clicked.connect(self.start_ndi)
        stop_tracking.clicked.connect(self.stop_ndi)
        ndi_buttons.addWidget(start_tracking)
        ndi_buttons.addWidget(stop_tracking)
        ndi_layout.addLayout(ndi_buttons)
        ndi_layout.addStretch(1)

        content.addWidget(xarm_group, 1)
        content.addWidget(ndi_group, 1)
        outer.addLayout(content, 1)

        log_group = QGroupBox("SYSTEM LOG")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(185)
        log_layout.addWidget(self.log_view)
        outer.addWidget(log_group)

    def _populate_markers(self) -> None:
        for item in self.ndi_cfg.get("markers", []):
            rom_name = Path(item["rom"]).stem
            row = MarkerRow(item.get("label", rom_name), rom_name)
            self.marker_rows.append(row)
            self.marker_container.addWidget(row)

    def log(self, text: str) -> None:
        self.log_view.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def start_xarm(self) -> None:
        if self.xarm_process.state() != QProcess.ProcessState.NotRunning:
            self.log("xArm driver is already running from this console")
            return
        if discover("xarm", REPO_ROOT):
            self.xarm_process_badge.set_status("LOCAL", "info")
            self.log("xArm driver already running locally; Stop can control it")
            return
        if self.xarm_robot_badge.text() == "ONLINE":
            self.log("xArm robot_states is online; no matching local driver found")
            return

        ip = str(self.xarm_cfg.get("robot_ip", "192.168.1.231"))
        install_setup = REPO_ROOT / "install" / "setup.bash"
        pieces = ["export ROS_DOMAIN_ID=42", "source /opt/ros/jazzy/setup.bash"]
        if install_setup.is_file():
            pieces.append(f"source {shlex.quote(str(install_setup))}")
        pieces.append(
            "exec ros2 launch xarm_api xarm6_driver.launch.py "
            f"robot_ip:={shlex.quote(ip)}"
        )
        self.log(f"Starting xArm driver for {ip}")
        self.xarm_process.setProgram("/bin/bash")
        self.xarm_process.setArguments(["-lc", " && ".join(pieces)])
        self.xarm_process.start()

    def _stop_driver(self, kind: str, process: QProcess) -> None:
        local = discover(kind, REPO_ROOT)
        if local:
            for driver in local:
                try:
                    if stop_local_process(driver, kind, REPO_ROOT):
                        self.log(f"Stopping local {kind} driver (PID {driver.pid})")
                except OSError as exc:
                    self.log(f"Cannot stop {kind} PID {driver.pid}: {exc}")
        if process.state() != QProcess.ProcessState.NotRunning:
            # SIGINT lets ros2 launch shut down its children rather than orphaning them.
            try:
                os.kill(process.processId(), signal.SIGINT)
            except ProcessLookupError:
                pass
            if not process.waitForFinished(5000):
                self.log(f"{kind} is still shutting down; waiting for graceful exit")
        elif not local:
            self.log(f"No local {kind} driver on domain 42 found")

    def stop_xarm(self) -> None:
        self._stop_driver("xarm", self.xarm_process)

    def _xarm_process_started(self) -> None:
        self.xarm_process_badge.set_status("RUNNING", "info")
        self.log("xArm ROS 2 launch process started")

    def _xarm_process_finished(self, exit_code: int, _status) -> None:
        if self.xarm_robot_badge.text() == "ONLINE":
            self.xarm_process_badge.set_status(
                "LOCAL" if discover("xarm", REPO_ROOT) else "REMOTE",
                "info",
            )
        else:
            self.xarm_process_badge.set_status(
                "STOPPED",
                "waiting",
            )

        self.log(
            f"xArm launch process exited with code {exit_code}"
        )

    def _read_xarm_output(self) -> None:
        data = bytes(self.xarm_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.log(f"xArm | {line.strip()}")

    def _on_robot_connection(self, connected: bool) -> None:
        if connected:
            self.xarm_robot_badge.set_status("ONLINE", "ready")
            self.ros_badge.set_status("ROS ONLINE", "ready")

            if (
                self.xarm_process.state()
                == QProcess.ProcessState.NotRunning
            ):
                self.xarm_process_badge.set_status(
                    "LOCAL" if discover("xarm", REPO_ROOT) else "REMOTE",
                    "info",
                )

            self.log("xArm robot_states received")

        else:
            self.xarm_robot_badge.set_status(
                "OFFLINE",
                "warning",
            )

            self._clear_robot_state_display()

            self.log("xArm robot_states timed out")

    def _on_robot_state(self, state: dict[str, Any]) -> None:
        state_code, mode_code = state["state"], state["mode"]
        self.xarm_state_value.setText(f"{self.STATE_NAMES.get(state_code, 'UNKNOWN')} ({state_code})")
        self.xarm_mode_value.setText(f"{self.MODE_NAMES.get(mode_code, 'UNKNOWN')} ({mode_code})")
        self.xarm_motion_badge.set_status(
            "ENABLED" if state["motion_enabled"] else "DISABLED",
            "ready" if state["motion_enabled"] else "warning",
        )
        self.xarm_error_value.setText(str(state["err"]))
        self.xarm_warn_value.setText(str(state["warn"]))
        pose = state.get("pose", [])
        self.xarm_tcp_value.setText(
            f"{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f} mm"
            if len(pose) >= 3 else "—"
        )

    def _clear_robot_state_display(self) -> None:
        self.xarm_motion_badge.set_status("UNKNOWN", "waiting")
        self.xarm_state_value.setText("—")
        self.xarm_mode_value.setText("—")
        self.xarm_error_value.setText("—")
        self.xarm_warn_value.setText("—")
        self.xarm_tcp_value.setText("—")

    def _on_service_result(self, label: str, success: bool, detail: str) -> None:
        self.log(f"{label}: {'OK' if success else 'FAILED'} — {detail}")

    def ros_clear_error(self) -> None:
        self.ros.clear_error()

    def ros_clear_warning(self) -> None:
        self.ros.clear_warning()

    def start_ndi(self) -> None:
        if self.ndi_process.state() != QProcess.ProcessState.NotRunning:
            self.log("NDI ROS node is already running from this console")
            return
        if discover("ndi", REPO_ROOT):
            self.log("NDI already running locally; Stop can control it")
            return
        if self.ros.ndi_connected:
            self.log("NDI is online; no matching local node found")
            return
        if self.ros.node is None:
            self.log("Cannot start NDI: ROS node unavailable. Build the interfaces and use run.sh.")
            return
        markers = self.ndi_cfg["markers"]
        args = [str(REPO_ROOT / "launchers/start_ndi.sh")]
        for key, value in {
            "serial_port": self.ndi_cfg.get("serial_port", "/dev/ttyUSB0"),
            "poll_period_s": self.ndi_cfg.get("poll_period_s", 0.05),
            "rom_files": [str(resolve_repo_path(item["rom"])) for item in markers],
            "frame_ids": [item["frame_id"] for item in markers],
        }.items():
            args.extend(["-p", f"{key}:={json.dumps(value)}"])
        self.ndi_process.setProgram("/bin/bash")
        self.ndi_process.setArguments(args)
        self.ndi_badge.set_status("STARTING", "info")
        self.log("Starting NDI ROS node; waiting for /tracking/status")
        self.ndi_process.start()

    def stop_ndi(self) -> None:
        self._stop_driver("ndi", self.ndi_process)

    def _on_ndi_connection(self, connected: bool) -> None:
        self.ndi_badge.set_status("ONLINE" if connected else "NO HEARTBEAT",
                                  "ready" if connected else "warning")
        if not connected:
            self._reset_ndi_markers()
        self.log("NDI /tracking/status " + ("received" if connected else "timed out"))

    def _on_marker_update(self, index, visible, quality, position, age_s) -> None:
        if 0 <= index < len(self.marker_rows):
            self.marker_rows[index].set_marker(visible, quality, position, age_s)

    def _read_ndi_output(self) -> None:
        data = bytes(self.ndi_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.log(f"NDI | {line.strip()}")

    def _ndi_finished(self, exit_code, _status) -> None:
        self.log(f"NDI ROS process exited with code {exit_code}")
        if not self.ros.ndi_connected:
            self._reset_ndi_markers()
            self.ndi_badge.set_status("STOPPED" if exit_code == 0 else "ERROR",
                                      "waiting" if exit_code == 0 else "error")

    def start_all(self) -> None:
        self.start_xarm()
        self.start_ndi()

    def stop_all(self) -> None:
        self.stop_ndi()
        self.stop_xarm()

    def closeEvent(self, event) -> None:
        if self.ndi_process.state() != QProcess.ProcessState.NotRunning:
            self.stop_ndi()
        if self.xarm_process.state() != QProcess.ProcessState.NotRunning:
            self.stop_xarm()
        self.ros.shutdown()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("BART Robot Console")
    app.setOrganizationName("BART LAB")
    app.setStyleSheet(APP_STYLE)
    try:
        window = RobotConsole()
    except Exception as exc:
        QMessageBox.critical(None, "BART Robot Console", f"Failed to start:\\n\\n{exc}")
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
