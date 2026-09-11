#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${NDI_PYTHON:-${HOME}/venvs/spinerobot_console/bin/python}"
export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash
source "${REPO_ROOT}/install/setup.bash"
# exec keeps the ROS node as the managed process and forwards stop signals.
TRACKING_PREFIX="$(ros2 pkg prefix spinerobot_tracking)"
exec "$PYTHON" "${TRACKING_PREFIX}/lib/spinerobot_tracking/ndi_tracker" --ros-args \
    -p "rom_files:=['${REPO_ROOT}/config/ndi_tools/8700338.rom', '${REPO_ROOT}/config/ndi_tools/8700339.rom', '${REPO_ROOT}/config/ndi_tools/8700340.rom']" \
    "$@"
