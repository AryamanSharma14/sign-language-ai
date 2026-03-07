# core/utils.py
"""Landmark extraction, normalization, and palm-orientation feature computation."""

import numpy as np
from config import FEATURE_DIM


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract raw x/y/z coords from 21 MediaPipe landmarks -> shape (63,)."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def _palm_normal(points: np.ndarray) -> np.ndarray:
    """
    Compute palm-normal vector from wrist-relative landmark positions.
    Uses cross product of (wrist->index_mcp) and (wrist->pinky_mcp).
    Landmarks 5 = index MCP, 17 = pinky MCP (0-indexed after wrist subtraction).
    Returns unit vector of shape (3,). Returns zeros if degenerate.
    """
    v1 = points[5]   # index MCP (wrist-relative)
    v2 = points[17]  # pinky MCP (wrist-relative)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (normal / norm).astype(np.float32)


def normalize_landmarks(raw: np.ndarray) -> np.ndarray | None:
    """
    Normalize landmarks to be translation- and scale-invariant, then append
    the palm-normal vector for rotation awareness.

    Steps:
      1. Subtract wrist (landmark 0) -> translation invariant
      2. Divide by max Euclidean distance from wrist -> scale invariant
      3. Append 3-element palm-normal vector (cross product of two palm edges)

    Returns array of shape (66,), or None if hand is degenerate.
    """
    points = raw.reshape(21, 3)
    wrist = points[0].copy()
    points = points - wrist

    dists = np.linalg.norm(points, axis=1)
    max_dist = dists.max()
    if max_dist < 1e-6:
        return None

    points = points / max_dist
    normal = _palm_normal(points)

    return np.concatenate([points.flatten(), normal]).astype(np.float32)


def landmarks_to_feature(hand_landmarks) -> np.ndarray | None:
    """Extract landmarks then normalize. Returns array of shape (FEATURE_DIM,) or None."""
    raw = extract_landmarks(hand_landmarks)
    return normalize_landmarks(raw)
