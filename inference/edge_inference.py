# inference/edge_inference.py
"""
Edge inference script for Raspberry Pi / embedded Linux.

Usage:
    python -m inference.edge_inference
    python -m inference.edge_inference --headless
    python -m inference.edge_inference --skip-frames 2 --fps-cap 15 --no-gpio
"""

import argparse
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections
from collections import deque

import config
from core.utils import landmarks_to_feature
from core.predictor import EWMAPredictor
from core.gestures import COMMAND_MAP
from hardware.gpio_controller import GpioController

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 128), -1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=config.EDGE_SKIP_FRAMES)
    parser.add_argument("--fps-cap", type=float, default=config.EDGE_FPS_CAP)
    parser.add_argument("--no-gpio", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    headless = args.headless
    skip_frames = args.skip_frames
    fps_cap = args.fps_cap

    gpio = GpioController(simulate=args.no_gpio)
    gpio.setup()

    print(f"Loading model from '{config.MODEL_PATH}'...")
    payload = joblib.load(config.MODEL_PATH)
    predictor = EWMAPredictor(
        payload["model"], payload["label_map"],
        threshold=config.CONFIDENCE_THRESHOLD
    )

    base_options = mp_python.BaseOptions(model_asset_path=config.MP_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(config.CAM_INDEX_PC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RES_EMBEDDED[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RES_EMBEDDED[1])

    fps_history: deque = deque(maxlen=30)
    start_time = time.time()
    frame_count = 0
    label, confidence = "", 0.0
    inference_ms = 0.0

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if frame_count % skip_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.time() - start_time) * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                t0 = time.perf_counter()
                result = detector.detect_for_video(mp_image, timestamp_ms)
                inference_ms = (time.perf_counter() - t0) * 1000

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    if not headless:
                        draw_landmarks(frame, landmarks, w, h)
                    feature = landmarks_to_feature(landmarks)
                    if feature is not None:
                        label, confidence = predictor.predict(feature)
                    else:
                        predictor.reset()
                        label, confidence = "", 0.0
                else:
                    predictor.reset()
                    label, confidence = "", 0.0

                if label:
                    command = COMMAND_MAP.get(label, label)
                    gpio.set_gesture(label)
                    if headless:
                        print(f"[EDGE] Gesture: {label}  \u2192  Command: {command}")
                        print(f"[EDGE] Confidence: {confidence:.2f} | FPS: {int(sum(fps_history)/max(len(fps_history),1))} | Inference: {inference_ms:.0f} ms")

            frame_count += 1
            frame_elapsed = time.time() - frame_start
            if frame_elapsed > 0:
                fps_history.append(1.0 / frame_elapsed)

            if not headless:
                fps_now = int(sum(fps_history) / max(len(fps_history), 1))
                command = COMMAND_MAP.get(label, "")
                cv2.putText(frame, f"Gesture: {label}  ->  {command}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}  FPS: {fps_now}  Inf: {inference_ms:.0f}ms",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Edge Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            target_frame_time = 1.0 / fps_cap if fps_cap else 0
            sleep_time = target_frame_time - (time.time() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[EDGE] Interrupted.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        detector.close()
        gpio.cleanup()


if __name__ == "__main__":
    main()
