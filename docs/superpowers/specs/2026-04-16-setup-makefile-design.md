# Setup & One-Command Workflow — Design Spec
**Date:** 2026-04-16
**Project:** sign-language-ai (Embedded OS Course Project)

---

## Goal

Make the project runnable on both PC and Raspberry Pi VM after a single command post-clone. Targeted at a course demo showing the system works on embedded Linux (Raspberry Pi OS x86 VM).

---

## Scope

1. `Makefile` as the single entry point for all project workflows
2. Auto-download of `hand_landmarker.task` (MediaPipe model, ~8MB)
3. Python venv creation and dependency installation
4. Pi VM setup documentation added to README
5. `TODO.md` tracking project completion for the course

Out of scope: MicroPython, cloud deployment, CI/CD.

---

## Architecture

### Makefile Targets

| Target | Command | Description |
|---|---|---|
| `setup` | `make setup` | Creates `.venv`, installs `requirements.txt`, downloads `hand_landmarker.task` |
| `setup-pi` | `make setup-pi` | Same but uses `requirements_pi.txt` (run inside Pi VM) |
| `run` | `make run` | Runs `inference/run_recognition.py` (PC windowed) |
| `run-headless` | `make run-headless` | Runs headless mode |
| `web` | `make web` | Starts Flask web dashboard at localhost:5000 |
| `train` | `make train` | Runs full training pipeline (collect → train) |
| `test` | `make test` | Runs `pytest tests/ -v` |
| `clean` | `make clean` | Removes `.venv/`, `models/`, `dataset/`, downloaded model |

### Venv Strategy

- Venv created at `.venv/` in project root (already git-ignored pattern)
- All `make` targets activate the venv before running Python
- Windows: `.venv\Scripts\python` / Linux+Pi: `.venv/bin/python`
- Makefile detects OS via `$(OS)` variable (`Windows_NT` vs other)

### MediaPipe Model Auto-Download

- `hand_landmarker.task` downloaded via Python `urllib.request` inline in the Makefile setup target
- Source: official MediaPipe GCS bucket (from README)
- Skips download if file already exists (idempotent)
- Size: ~8MB, one-time download

### Pi VM Workflow

1. Install VirtualBox (free)
2. Download Raspberry Pi Desktop x86 ISO from raspberrypi.com
3. Create VM (1–2GB RAM, 20GB disk)
4. Boot VM, open terminal
5. Clone repo: `git clone <repo>`
6. Run: `make setup-pi`
7. Demo: `make run-headless` or `make web`

---

## Files Changed

| File | Action |
|---|---|
| `Makefile` | Create — all workflow targets |
| `README.md` | Add Pi VM Setup section + update Quick Start |
| `TODO.md` | Create — course project completion checklist |
| `.gitignore` | Add `.venv/` if not already present |

---

## Project Completion TODO

Tasks needed to fully demo for the course:

- [x] Core ML pipeline (gestures, predictor, utils)
- [x] Training pipeline (collect_data, train_model)
- [x] Inference scripts (run_recognition, edge_inference)
- [x] Web dashboard (Flask + WebSocket + Chart.js)
- [x] Hardware abstraction (GPIO, platform_io)
- [x] Tests (pytest suite)
- [ ] Setup tooling (Makefile, auto-download) ← this spec
- [ ] Collect training data (~100 samples × 5 gestures)
- [ ] Train and verify model (>95% accuracy target)
- [ ] Verify inference runs on PC
- [ ] Set up Pi Desktop VM
- [ ] Verify inference runs on Pi VM
- [ ] Record or live-demo both platforms for course submission

---

## Constraints

- No global Python package installs (everything in `.venv`)
- `make` is the only global tool required on PC (install via `choco install make`)
- VirtualBox is the only additional software for Pi demo
- Script must be idempotent (safe to re-run)
