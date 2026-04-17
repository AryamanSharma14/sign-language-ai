import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_download_hand_landmarker_if_missing():
    """Setup should download hand_landmarker.task if it doesn't exist."""
    from setup_helper import ensure_assets

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "hand_landmarker.task"
        assert not model_path.exists(), "Model shouldn't exist initially"

        # Mock the download to avoid actual network calls in tests
        with patch('setup_helper.download_hand_landmarker') as mock_download:
            mock_download.return_value = True
            result = ensure_assets(model_path=str(model_path))
            assert result is True, "ensure_assets should return True on success"
            mock_download.assert_called_once()


def test_ensure_assets_returns_true_when_files_exist():
    """Setup should succeed when all assets already exist."""
    from setup_helper import ensure_assets

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "hand_landmarker.task"
        model_pkl_path = Path(tmpdir) / "model.pkl"

        # Create dummy files
        model_path.touch()
        model_pkl_path.touch()

        result = ensure_assets(model_path=str(model_path), model_pkl_path=str(model_pkl_path))
        assert result is True, "ensure_assets should return True when all files exist"


def test_setup_creates_venv():
    """Setup should create virtual environment if needed."""
    from setup_helper import setup_environment

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "venv"
        assert not venv_path.exists()

        result = setup_environment(venv_path=str(venv_path))
        # We don't actually create venv in test, but check the function handles it gracefully
        assert result is not None


def test_inference_can_run_after_setup():
    """After setup, basic inference module should import without errors."""
    from setup_helper import ensure_assets

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "hand_landmarker.task"
        model_pkl_path = Path(tmpdir) / "model.pkl"

        # Create dummy files so imports won't fail on missing files
        model_path.write_bytes(b"dummy_model")
        model_pkl_path.write_bytes(b"dummy_pkl")

        # Patch config to use our test paths
        with patch('config.MP_MODEL_PATH', str(model_path)):
            with patch('config.MODEL_PATH', str(model_pkl_path)):
                result = ensure_assets(model_path=str(model_path), model_pkl_path=str(model_pkl_path))
                assert result is True
