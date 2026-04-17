"""
Asset download and setup helper for sign-language-ai.
Ensures hand_landmarker.task and model.pkl are available before inference.
"""

import os
import sys
from pathlib import Path
import urllib.request
import shutil


HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
CHUNK_SIZE = 8192


def download_hand_landmarker(output_path: str) -> bool:
    """
    Download MediaPipe hand landmarker model if it doesn't exist.

    Args:
        output_path: Where to save the model file

    Returns:
        True if download succeeded or file already exists
    """
    output_path = Path(output_path)

    if output_path.exists():
        return True

    print(f"[Setup] Downloading hand_landmarker.task (~8 MB)...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                sys.stdout.write(f"\r[Setup] {percent:.1f}% downloaded")
                sys.stdout.flush()

        urllib.request.urlretrieve(HAND_LANDMARKER_URL, output_path, show_progress)
        print("\n[Setup] hand_landmarker.task downloaded successfully")
        return True

    except Exception as e:
        print(f"\n[Error] Failed to download hand_landmarker.task: {e}")
        return False


def ensure_assets(model_path: str = None, model_pkl_path: str = None) -> bool:
    """
    Ensure all required assets exist. Downloads missing files.

    Args:
        model_path: Path to hand_landmarker.task (default: hand_landmarker.task in cwd)
        model_pkl_path: Path to model.pkl (default: models/model.pkl in cwd)

    Returns:
        True if all assets are available or successfully downloaded
    """
    from config import MP_MODEL_PATH, MODEL_PATH

    model_path = model_path or MP_MODEL_PATH
    model_pkl_path = model_pkl_path or MODEL_PATH

    model_path_obj = Path(model_path)
    model_pkl_obj = Path(model_pkl_path)

    # Download hand landmarker if missing
    if not model_path_obj.exists():
        if not download_hand_landmarker(model_path):
            print("[Warning] hand_landmarker.task not available")
            return False

    # Check model.pkl
    if not model_pkl_obj.exists():
        print(f"[Error] {model_pkl_path} not found")
        print("[Info] Please train the model first:")
        print("  python -m training.collect_data")
        print("  python -m training.train_model")
        return False

    return True


def setup_environment(venv_path: str = None) -> bool:
    """
    Set up the Python environment (placeholder for future expansion).

    Args:
        venv_path: Path to virtual environment

    Returns:
        True if setup successful
    """
    return True


if __name__ == "__main__":
    print("Sign Language AI - Setup Helper")
    print("=" * 50)

    if ensure_assets():
        print("[Success] All assets ready. You can now run:")
        print("  python -m inference.run_recognition")
        print("  python -m web.app")
    else:
        print("[Setup] Some assets missing. See instructions above.")
        sys.exit(1)
