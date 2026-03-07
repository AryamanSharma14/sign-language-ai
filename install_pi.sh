#!/bin/bash
set -e

echo "[INSTALL] Updating apt..."
sudo apt update
sudo apt install -y python3-pip libatlas-base-dev libjpeg-dev libopenblas-dev

echo "[INSTALL] Installing Python packages..."
pip3 install --break-system-packages \
    opencv-python-headless \
    mediapipe \
    numpy \
    scikit-learn \
    joblib \
    pyserial

echo "[INSTALL] Checking RPi.GPIO..."
python3 -c "import RPi.GPIO" 2>/dev/null && echo "[GPIO] RPi.GPIO OK" \
    || echo "[GPIO] RPi.GPIO not found — install manually if on Pi hardware"

echo ""
echo "[DONE] Run with:"
echo "  python3 edge_inference.py --headless"
echo "  python3 edge_inference.py           (with display)"
