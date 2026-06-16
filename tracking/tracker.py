# tracking/tracker.py
"""
PersonTracker — YOLOv8n + ByteTrack wrapper.
Maintains a track cache: {track_id → identity_info}
Recognition is triggered only on new tracks or re-check intervals.
"""

import time
import os

os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

import torch
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRACK_TIMEOUT_SEC, RECHECK_INTERVAL_SEC, YOLO_CONF_THRESHOLD)


class PersonTracker:
    """
    Wraps YOLOv8n + ByteTrack.
    Maintains a track cache: {track_id → identity_info}
    Recognition is triggered only on:
      - New tracks (never seen before)
      - Tracks past their re-check interval (identity drift prevention)
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        self.track_cache: dict[int, dict] = {}
        # {track_id: {'identity': str, 'score': float, 'last_seen': float, 'last_recognised': float}}
        self.frame_count = 0

    def update(self, frame) -> tuple[list, list]:
        """
        Run YOLO + ByteTrack on frame.
        Returns:
          tracks_to_recognise: list of {track_id, bbox, crop}
          all_tracks:          list of {track_id, bbox, identity_info}
        """
        self.frame_count += 1
        now = time.time()

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],                          # Person class only
            conf=YOLO_CONF_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            return [], []

        boxes = results[0].boxes
        track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        bboxes = boxes.xyxy.cpu().numpy().astype(int).tolist()

        tracks_to_recognise = []
        all_tracks = []

        for track_id, bbox in zip(track_ids, bboxes):
            x1, y1, x2, y2 = bbox

            # Clamp to frame boundaries
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]

            cache_entry = self.track_cache.get(track_id)
            needs_recognition = (
                cache_entry is None or
                (now - cache_entry['last_recognised']) > RECHECK_INTERVAL_SEC
            )

            if cache_entry:
                cache_entry['last_seen'] = now
            else:
                self.track_cache[track_id] = {
                    'identity': None, 'score': 0.0,
                    'last_seen': now, 'last_recognised': 0.0
                }

            if needs_recognition and crop.size > 0:
                tracks_to_recognise.append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'crop': crop
                })

            all_tracks.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'identity': self.track_cache[track_id].get('identity', 'Pending')
            })

        # Clean stale tracks
        for tid in list(self.track_cache.keys()):
            if now - self.track_cache[tid]['last_seen'] > TRACK_TIMEOUT_SEC:
                del self.track_cache[tid]

        return tracks_to_recognise, all_tracks

    def update_track_identity(self, track_id: int, identity: str, score: float):
        """Update the identity for a specific track after recognition."""
        if track_id in self.track_cache:
            self.track_cache[track_id]['identity'] = identity
            self.track_cache[track_id]['score'] = score
            self.track_cache[track_id]['last_recognised'] = time.time()
