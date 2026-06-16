# storage/unknown_handler.py
"""
Unknown Face Handler — saves unrecognised face crops for later enrollment.
Time-based deduplication to avoid flooding disk with similar faces.
"""

import cv2
import time
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UNKNOWN_FACES_DIR


class UnknownFaceHandler:
    """
    Saves unknown face crops for later enrollment.
    Deduplication: skip save if a very similar face was saved < MIN_INTERVAL_SEC ago.
    (Simple time-based; can upgrade to embedding-based clustering.)
    """

    MIN_INTERVAL_SEC = 10.0

    def __init__(self, save_dir=UNKNOWN_FACES_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved: dict[int, float] = {}   # track_id → timestamp
        self._counter = 0

    def save(self, face_crop, track_id: int = -1) -> str | None:
        """
        Save a face crop if enough time has passed since last save.
        Returns the file path if saved, None otherwise.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        now = time.time()
        if now - self._last_saved.get(track_id, 0.0) < self.MIN_INTERVAL_SEC:
            return None

        self._counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{ts}_t{track_id}_{self._counter:04d}.jpg"
        save_path = self.save_dir / filename

        try:
            cv2.imwrite(str(save_path), face_crop)
            self._last_saved[track_id] = now
            print(f"[Unknown] Saved unknown face: {filename}")
            return str(save_path)
        except Exception as e:
            print(f"[Unknown] Failed to save face crop: {e}")
            return None
