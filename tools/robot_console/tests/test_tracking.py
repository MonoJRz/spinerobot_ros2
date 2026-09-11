"""Run in the sourced ROS workspace with the console Python environment."""
import importlib.machinery
import importlib.util
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QApplication
from builtin_interfaces.msg import Time
from spinerobot_interfaces.msg import TrackingStatus

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


tracker_module = load('ndi_tracker', ROOT / 'src/spinerobot_tracking/scripts/ndi_tracker')
console = load('robot_console', ROOT / 'tools/robot_console/main.py')


class TrackingTests(unittest.TestCase):
    def test_occlusion_keeps_heartbeat_and_increases_age(self):
        node = Mock()
        node.frames = ['patient_marker']
        node.last_seen = [None]
        node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
        node.position_pubs = {'patient_marker': Mock()}
        visible = np.eye(4)
        visible[0, 3] = 100
        node.tracker.get_frame.side_effect = [
            (None, None, None, [visible], [0.92]),
            (None, None, None, [np.full((4, 4), np.nan)], [np.nan]),
            (None, None, None, [visible], [0.92]),
        ]
        with patch.object(tracker_module.time, 'monotonic', side_effect=[10., 12., 13.]):
            for _ in range(3):
                tracker_module.TrackingNode.poll(node)
        messages = [call.args[0] for call in node.status_pub.publish.call_args_list]
        self.assertEqual([(m.visible, m.valid, m.age_sec, m.reason) for m in messages],
                         [(True, True, 0., 'ok'), (False, False, 2., 'not_visible'),
                          (True, True, 0., 'ok')])
        self.assertAlmostEqual(node.position_pubs['patient_marker'].publish.call_args.args[0].point.x, .1)

    def test_failed_read_does_not_publish_healthy_heartbeat(self):
        node = Mock()
        node.tracker.get_frame.side_effect = OSError('device disconnected')
        with self.assertRaises(OSError):
            tracker_module.TrackingNode.poll(node)
        node.status_pub.publish.assert_not_called()

    def test_console_heartbeat_independent_of_visibility_and_expires(self):
        app = QApplication.instance() or QApplication([])
        cfg = console.load_config()
        bridge = console.RosBridge(cfg['xarm'], cfg['ndi'])
        self.assertIsNotNone(bridge.node, str(console.ROS_IMPORT_ERROR))
        connections, markers = [], []
        bridge.ndi_connection_changed.connect(connections.append)
        bridge.marker_update.connect(lambda *args: markers.append(args))
        try:
            msg = TrackingStatus(frame_id='patient_marker', visible=True, valid=True,
                                 quality=.92, age_sec=0., reason='ok')
            bridge._on_tracking_status(msg)
            msg.visible = False
            msg.valid = False
            msg.age_sec = 2.
            bridge._on_tracking_status(msg)
            self.assertEqual(connections, [True])
            self.assertFalse(markers[-1][1])
            with patch.object(console.time, 'monotonic', return_value=time.monotonic() + 2.):
                bridge._check_connection()
            self.assertEqual(connections, [True, False])
            self.assertFalse(bridge.ndi_connected)
        finally:
            bridge.shutdown()


if __name__ == '__main__':
    unittest.main()
