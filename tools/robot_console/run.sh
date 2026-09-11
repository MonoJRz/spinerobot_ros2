#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${HOME}/venvs/spinerobot_console"

export ROS_DOMAIN_ID=42
source /opt/ros/jazzy/setup.bash

if [[ -f "${REPO_ROOT}/install/setup.bash" ]]; then
    source "${REPO_ROOT}/install/setup.bash"
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "BART Robot Console is not installed yet."
    echo "Run: ${SCRIPT_DIR}/install.sh"
    command -v notify-send >/dev/null 2>&1 && notify-send "BART Robot Console" "Run tools/robot_console/install.sh first." || true
    exit 1
fi

exec "${VENV}/bin/python" "${SCRIPT_DIR}/main.py"
