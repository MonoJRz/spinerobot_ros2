import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_processes import LocalProcess, discover, stop


class LocalProcessTests(unittest.TestCase):
    def test_reopened_controller_recovers_and_stops_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / 'install/spinerobot_tracking/lib/spinerobot_tracking/ndi_tracker'
            script.parent.mkdir(parents=True)
            script.write_text('import time\ntime.sleep(60)\n')
            driver = subprocess.Popen([sys.executable, str(script)],
                                      env={**os.environ, 'ROS_DOMAIN_ID': '42'},
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                first = discover('ndi', repo)
                self.assertEqual([p.pid for p in first], [driver.pid])
                # A new console has no Popen/QProcess from the first console.
                reopened = discover('ndi', repo)
                self.assertEqual(reopened, first)
                self.assertTrue(stop(reopened[0], 'ndi', repo))
                driver.wait(timeout=5)
                self.assertEqual(discover('ndi', repo), [])
            finally:
                if driver.poll() is None:
                    driver.kill()
                driver.wait()

    def test_other_domain_and_workspace_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / 'install/spinerobot_tracking/lib/spinerobot_tracking/ndi_tracker'
            script.parent.mkdir(parents=True)
            script.write_text('import time\ntime.sleep(60)\n')
            driver = subprocess.Popen([sys.executable, str(script)],
                                      env={**os.environ, 'ROS_DOMAIN_ID': '0'})
            try:
                self.assertEqual(discover('ndi', repo), [])
                self.assertEqual(discover('ndi', repo / 'other'), [])
            finally:
                driver.kill()
                driver.wait()

    def test_reused_pid_identity_cannot_be_stopped(self):
        process = LocalProcess(os.getpid(), 'not-the-start-time')
        self.assertFalse(stop(process, 'ndi', Path('/nonexistent-workspace')))


if __name__ == '__main__':
    unittest.main()
