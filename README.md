# Real-Time Sign Language Gesture Recognition

A Python MVP that detects hand gestures via webcam using MediaPipe hand landmarks,
classifies them with a scikit-learn SVC, and displays predictions in real time.

---

## Supported Gestures

| Label | Description |
|-------|-------------|
| A | Closed fist / ASL A |
| B | Flat hand, fingers together / ASL B |
| C | Curved hand / ASL C |
| PEACE | Two-finger V shape |
| OK | Thumb and index finger circle |

---

## Setup

```bash
cd sign-language-ai
pip install -r requirements.txt
```

> **Python 3.10+** required (uses `X | Y` union type hints).

---

## Usage (3 Steps)

### Step 1 — Collect Data

```bash
python collect_data.py
```

- Enter a gesture label (e.g. `A`) when prompted
- Hold your hand in front of the webcam
- Press **`s`** to save a sample, **`q`** to quit
- Collect ~100 samples per gesture (5 gestures = ~500 total rows)
- Repeat for each gesture label

Samples are appended to `dataset/gestures.csv`.

### Step 2 — Train Model

```bash
python train_model.py
```

- Loads `dataset/gestures.csv`
- Trains KNN (k=5) and SVC (rbf, C=10) classifiers
- Prints accuracy, classification report, and confusion matrix for both
- Saves the SVC model to `model.pkl`

Expected results with 100 samples/gesture: KNN >90%, SVC >95%.

### Step 3 — Run Recognition

```bash
python run_recognition.py
```

- Loads `model.pkl`
- Opens webcam with live hand tracking
- Displays predicted gesture and confidence score
- Green label = high confidence (≥ 0.60), Yellow = lower confidence
- Press **`q`** to quit, **`r`** to reset the prediction buffer

---

## Tips for Better Accuracy

- **Lighting**: Use consistent, good lighting — avoid backlight
- **Background**: Plain backgrounds help hand detection
- **Distance**: Keep hand 30–60 cm from camera
- **Variety**: Collect samples at different angles and hand positions
- **Balance**: Collect the same number of samples per gesture
- **PEACE vs B confusion**: These are visually similar — collect more samples
  with slight angle variations to improve separation

---

## Windows-Specific Notes

- The scripts use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` for faster webcam startup on Windows.
  If your webcam doesn't open, try changing the index to `1` or `2`.
- If you have multiple cameras, try indices 0, 1, 2 to find your webcam.
- MediaPipe may show warnings on first run — these are harmless.

---

## Project Structure

```
sign-language-ai/
├── dataset/
│   └── gestures.csv      # label + 63 landmark features per row
├── utils.py              # shared: landmark extraction + normalization
├── collect_data.py       # Step 1: webcam → CSV
├── train_model.py        # Step 2: CSV → model.pkl
├── run_recognition.py    # Step 3: webcam → live prediction
├── platform_io.py        # Hardware abstraction layer (GPIO / Serial)
├── model.pkl             # saved SVC model (created by train_model.py)
├── requirements.txt      # PC dependencies
└── requirements_pi.txt   # Raspberry Pi / embedded dependencies
```

---

## Embedded Systems Demo

This project includes a hardware abstraction layer (`platform_io.py`) that makes the
recognizer behave like an embedded system — printing boot messages, routing gesture
events to GPIO pins or a serial port, and capping framerate for resource-constrained hardware.

### Platform Modes

| Command | Platform | Window | Resolution | FPS cap |
|---------|----------|--------|------------|---------|
| `python run_recognition.py` | PC | yes | 640×480 | none |
| `python run_recognition.py --headless` | PC | no | 640×480 | none |
| `python run_recognition.py --platform rpi` | RPi | no | 320×240 | 10 |
| `python run_recognition.py --platform arduino` | Arduino | no | 320×240 | 10 |

Auto-detection reads `/proc/device-tree/model`; falls back to `pc`.

### GPIO Pin Map

Each gesture drives a distinct BCM pin HIGH when detected:

| Gesture | BCM Pin |
|---------|---------|
| A | 17 |
| B | 27 |
| C | 22 |
| PEACE | 23 |
| OK | 24 |

On PC/simulation the pin state is printed to the terminal:
```
[09:14:32] [GPIO] Pin 17 HIGH  |  Gesture: A  |  Confidence: 0.94
```

### Arduino Serial Protocol

When `--platform arduino` is used, each gesture detection writes a line to `/dev/ttyUSB0`:
```
GESTURE:A:94\n   ← label : confidence_as_integer_percent
```
If the serial port is unavailable it falls back to console simulation automatically.

### Arduino Wiring Diagram

```
Raspberry Pi 4          Arduino / LED array
─────────────           ────────────────────
BCM 17 (Pin 11) ──────► LED_A  (anode → 220Ω → GND)
BCM 27 (Pin 13) ──────► LED_B
BCM 22 (Pin 15) ──────► LED_C
BCM 23 (Pin 16) ──────► LED_PEACE
BCM 24 (Pin 18) ──────► LED_OK
GND    (Pin 6)  ──────► GND (common)
```

Or use TX (Pin 8) → Arduino RX with `--platform arduino` for serial control.

### Raspberry Pi Setup

1. Flash **Raspberry Pi OS Lite** (64-bit) — no desktop required
2. Enable camera: `sudo raspi-config` → Interface Options → Camera
3. Install dependencies:
   ```bash
   pip install -r requirements_pi.txt
   ```
4. Copy project files to the Pi (scp or git clone)
5. Run headless:
   ```bash
   python run_recognition.py --platform rpi
   ```
   Press **Ctrl+C** to quit.

> `RPi.GPIO` installs automatically only on aarch64 (Pi hardware).
> On PC it is skipped and GPIO output is simulated in the terminal.

---

## Verification

```bash
# Check imports
python -c "import cv2, mediapipe, sklearn, numpy, joblib; print('OK')"

# Check dataset shape (should be (N, 64) where N = total samples)
python -c "import numpy as np; d=np.genfromtxt('dataset/gestures.csv',delimiter=',',dtype=str,skip_header=1); print(d.shape)"
```

---

## Edge AI Inference (Embedded Linux)

`edge_inference.py` is a purpose-built inference script for **Raspberry Pi / embedded Linux**.
It replaces `run_recognition.py` on the Pi with features designed for constrained hardware:
real performance metrics, frame skipping to reduce CPU load, a gesture-to-command translation
layer, and direct GPIO control via a standalone `GpioController`.

### What's different from `run_recognition.py`

| Feature | `run_recognition.py` | `edge_inference.py` |
|---------|----------------------|---------------------|
| Target | PC / demo | Embedded Linux / Pi |
| Frame skip | No | Yes (default: every 2nd frame) |
| FPS cap | Embedded platforms only | Always (default: 15 FPS) |
| Metrics (FPS / inference ms) | No | Yes |
| Gesture→Command mapping | No | Yes |
| GPIO module | `platform_io.py` | `gpio_controller.py` (standalone) |
| Camera open | `CAP_DSHOW` (Windows) | No flag (Linux `/dev/video0`) |

### CLI Flags

```bash
python edge_inference.py                     # windowed + GPIO simulation
python edge_inference.py --headless          # terminal only, no window
python edge_inference.py --skip-frames 2    # process every 2nd frame (default)
python edge_inference.py --skip-frames 1    # process every frame (higher CPU)
python edge_inference.py --fps-cap 15       # target FPS (default: 15)
python edge_inference.py --fps-cap 5        # visibly slow — ~5 FPS
python edge_inference.py --no-gpio          # disable GPIO entirely
```

### Performance Metrics

Every frame tracks and displays:

| Metric | Description |
|--------|-------------|
| `FPS` | Rolling 30-frame average frames per second |
| `Inference ms` | Time spent in MediaPipe + SVC predict |
| `Frame ms` | Total loop iteration time |

### Gesture-to-Command Map

| Gesture | Command |
|---------|---------|
| A | YES |
| B | NO |
| C | CONFIRM |
| PEACE | HELLO |
| OK | OK |

### Terminal Output (headless)

```
[EDGE] Gesture: A  →  Command: YES
[EDGE] Confidence: 0.91 | FPS: 13 | Inference: 28 ms | Frame: 35 ms
```

### Display Overlay (windowed)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [camera feed + hand landmarks]                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Gesture: A  →  YES              Confidence: 0.91            │
│ FPS: 13  Inference: 28ms  Frame: 35ms  [q=quit r=reset]     │
└─────────────────────────────────────────────────────────────┘
```

### Raspberry Pi Quick-Start

```bash
# 1. Copy project to Pi (or git clone)
# 2. Run the installer
bash install_pi.sh

# 3. Run headless inference
python3 edge_inference.py --headless

# 4. Run with display (if monitor attached)
python3 edge_inference.py
```

`install_pi.sh` installs `opencv-python-headless`, `mediapipe`, `numpy`,
`scikit-learn`, `joblib`, and `pyserial` via pip, and checks for `RPi.GPIO`.

### GPIO Wiring (same as Embedded Systems Demo above)

| Gesture | BCM Pin |
|---------|---------|
| A | 17 |
| B | 27 |
| C | 22 |
| PEACE | 23 |
| OK | 24 |

On PC, GPIO is automatically simulated — no hardware required:
```
[GPIO] Pins [17, 27, 22, 23, 24] configured as OUTPUT (simulated)
[GPIO] Pin 17 HIGH → A (sim)
```

On real Pi hardware, `RPi.GPIO` is imported automatically and actual pins switch state.
