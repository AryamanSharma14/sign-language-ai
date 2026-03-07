# training/collect_data.py
"""
Collect gesture data from webcam and save to dataset/gestures.csv.

Usage:
    python -m training.collect_data A
    python -m training.collect_data THANK_YOU

Controls:
    s - Save current frame's landmarks as a sample
    q - Quit
"""

import os
import sys
import csv
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

import config
from core.gestures import GESTURE_LABELS
from core.utils import landmarks_to_feature

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
FLASH_DURATION = 15


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 128), -1)


def get_label_from_user() -> str:
    if len(sys.argv) >= 2:
        label = sys.argv[1].strip().upper()
        if label in GESTURE_LABELS:
            return label
        print(f"Invalid label '{label}'. Choose from: {GESTURE_LABELS}")
        sys.exit(1)
    print(f"Available gestures: {GESTURE_LABELS}")
    while True:
        label = input("Enter gesture label to collect: ").strip().upper()
        if label in GESTURE_LABELS:
            return label
        print(f"  Invalid label. Choose from: {GESTURE_LABELS}")


def ensure_csv_exists():
    os.makedirs(os.path.dirname(config.DATASET_PATH), exist_ok=True)
    if not os.path.exists(config.DATASET_PATH):
        with open(config.DATASET_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(config.FEATURE_DIM)]
            writer.writerow(header)


def main():
    label = get_label_from_user()
    ensure_csv_exists()

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

    sample_count = 0
    flash_counter = 0
    start_time = time.time()

    cap = cv2.VideoCapture(config.CAM_INDEX_PC, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAM_INDEX_PC)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            feature = None
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                draw_landmarks(frame, lms, w, h)
                feature = landmarks_to_feature(lms)

            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Gesture: {label}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            status_color = (0, 255, 0) if feature is not None else (0, 100, 255)
            status_text = "Hand detected" if feature is not None else "No hand"
            cv2.putText(frame, status_text, (w - 200, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if flash_counter > 0:
                cv2.putText(frame, "SAVED!", (w // 2 - 60, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 128), 3)
                flash_counter -= 1

            cv2.putText(frame, "s=save  q=quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if feature is not None:
                    with open(config.DATASET_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + feature.tolist())
                    sample_count += 1
                    flash_counter = FLASH_DURATION
                else:
                    print("No hand detected — nothing saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"\nDone. Collected {sample_count} samples for '{label}'.")
        print(f"Data saved to: {config.DATASET_PATH}")


if __name__ == "__main__":
    main()
