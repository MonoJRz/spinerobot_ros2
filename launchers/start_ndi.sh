#!/bin/bash

PYTHON="$HOME/venvs/spinerobot_backend/bin/python"
SCRIPT="$HOME/Workspace/spinerobot_ros2/test_vicra.py"

echo "======================================"
echo " BART LAB - Starting NDI Polaris Vicra"
echo "======================================"
echo
echo "Python: $PYTHON"
echo "Script: $SCRIPT"
echo

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Backend Python environment not found:"
    echo "$PYTHON"
    read -p "Press Enter to close..."
    exit 1
fi

"$PYTHON" -c \
"from sksurgerynditracker.nditracker import NDITracker; print('NDITracker module OK')"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: sksurgerynditracker is not installed in backend environment."
    read -p "Press Enter to close..."
    exit 1
fi

echo
echo "Starting Polaris..."
echo

"$PYTHON" "$SCRIPT"

echo
echo "NDI process stopped."
read -p "Press Enter to close..."
