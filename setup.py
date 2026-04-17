#!/usr/bin/env python3
"""
Setup script for Sign Language AI.
Run this after cloning to download required assets.

Usage:
    python setup.py
"""

import sys
from pathlib import Path
from setup_helper import ensure_assets

if __name__ == "__main__":
    print("Sign Language AI - Setup")
    print("=" * 60)

    if ensure_assets():
        print("\n[OK] Setup complete! All assets are ready.")
        print("\nNext steps:")
        print("  1. Install dependencies:")
        print("     pip install -r requirements.txt")
        print("\n  2. Run inference:")
        print("     python -m inference.run_recognition")
        print("\n  3. Or start the web dashboard:")
        print("     python -m web.app")
        sys.exit(0)
    else:
        print("\n[FAIL] Setup incomplete. See errors above.")
        sys.exit(1)
