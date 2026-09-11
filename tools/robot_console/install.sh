#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${HOME}/venvs/spinerobot_console"
APP_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${APP_DIR}/bart-robot-console.desktop"

echo "======================================"
echo " BART LAB - Robot Console Installer"
echo "======================================"
echo

source /opt/ros/jazzy/setup.bash

if [[ -f "${REPO_ROOT}/install/setup.bash" ]]; then
    source "${REPO_ROOT}/install/setup.bash"
else
    echo "WARNING: ${REPO_ROOT}/install/setup.bash was not found."
    echo "Build the ROS workspace before using xArm controls."
    echo
fi

mkdir -p "${HOME}/venvs"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[1/3] Creating ${VENV}"
    python3 -m venv --system-site-packages "${VENV}"
else
    echo "[1/3] Reusing ${VENV}"
fi

echo "[2/3] Installing Python dependencies"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "[3/3] Installing Ubuntu application shortcut"
mkdir -p "${APP_DIR}"

cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=BART Robot Console
Comment=xArm 6 and NDI Polaris preflight console
Exec=${SCRIPT_DIR}/run.sh
Icon=${REPO_ROOT}/assets/BART_console_logo.png
Terminal=false
Categories=Development;Science;
StartupNotify=true
EOF

chmod +x "${DESKTOP_FILE}"
chmod +x "${SCRIPT_DIR}/run.sh"

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "${APP_DIR}" >/dev/null 2>&1 || true

if [[ -d "${HOME}/Desktop" ]]; then
    cp "${DESKTOP_FILE}" "${HOME}/Desktop/BART Robot Console.desktop"
    chmod +x "${HOME}/Desktop/BART Robot Console.desktop"
    command -v gio >/dev/null 2>&1 && gio set "${HOME}/Desktop/BART Robot Console.desktop" metadata::trusted true >/dev/null 2>&1 || true
fi

echo
echo "Installation complete."
echo "Run: ${SCRIPT_DIR}/run.sh"
echo "Or search Ubuntu applications for: BART Robot Console"
