#!/bin/bash
# Double-click in Finder to start Local FaceSwap macOS in Terminal.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "No .venv found. Install first (see README.md), then double-click again."
    read -r -p "Press Enter to close."
    exit 1
fi
exec .venv/bin/python app.py
