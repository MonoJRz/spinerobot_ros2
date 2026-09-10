import time
import numpy as np
from sksurgerynditracker.nditracker import NDITracker

ROM_DIR = "/home/bartlab/Workspace/spinerobot_ros2/config/ndi_tools"

ROM_NAMES = [
    "8700338",
    "8700339",
    "8700340",
]

SETTINGS = {
    "tracker type": "polaris",
    "serial port": "/dev/ttyUSB0",
    "romfiles": [
        f"{ROM_DIR}/8700338.rom",
        f"{ROM_DIR}/8700339.rom",
        f"{ROM_DIR}/8700340.rom",
    ],
}

tracker = NDITracker(SETTINGS)
tracker.start_tracking()

print("Polaris tracking started")
print("Ctrl+C to stop\n")

try:

    while True:

        (
            port_handles,
            timestamps,
            frame_numbers,
            tracking,
            quality,
        ) = tracker.get_frame()

        for i, matrix in enumerate(tracking):

            matrix = np.asarray(matrix)

            # Ignore invisible tools
            if np.isnan(matrix).any():
                continue

            x = matrix[0, 3]
            y = matrix[1, 3]
            z = matrix[2, 3]

            print(
                f"{ROM_NAMES[i]} | "
                f"Handle={port_handles[i]} | "
                f"X={x:8.2f} mm "
                f"Y={y:8.2f} mm "
                f"Z={z:8.2f} mm | "
                f"Quality={quality[i]:.3f}"
            )

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping tracker")

finally:

    tracker.stop_tracking()
    tracker.close()