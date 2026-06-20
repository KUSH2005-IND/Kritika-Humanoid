# recognition/face_detector.py
"""
SCRFD Face Detector — lightweight face detection optimised for CPU.
Returns face bounding boxes + 5-point landmarks for alignment.

Key tuning notes:
  - input_size=(480,480): better for large/close faces (phone photos, webcam at 1-2m)
  - input_size=(640,640): better for small/distant faces (group photos)
  - Two-pass fallback: tries 480 first, retries at 640 if nothing found
"""

from insightface.model_zoo import get_model
from pathlib import Path
import numpy as np
import cv2

from config import SCRFD_INPUT_SIZE, SCRFD_CONF_THRESHOLD


class SCRFDDetector:
    """
    Lightweight face detector optimised for CPU.
    Returns face bounding boxes + 5-point landmarks for alignment.
    """

    def __init__(self, model_path: str, input_size=None, conf_thresh=None, nms_thresh=0.4):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"[SCRFDDetector] Model not found: {model_path}\n"
                f"Run 'python setup_models.py' to download models."
            )
        self.input_size  = input_size  or SCRFD_INPUT_SIZE
        self.conf_thresh = conf_thresh or SCRFD_CONF_THRESHOLD
        self.nms_thresh  = nms_thresh
        self.model_path  = model_path

        self.model = get_model(model_path)
        self.model.prepare(
            ctx_id=-1,
            input_size=self.input_size,
            det_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh
        )

    def _prepare(self, input_size):
        """Re-prepare model with a different input size."""
        self.model.prepare(
            ctx_id=-1,
            input_size=input_size,
            det_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh
        )

    def _run_detect(self, img: np.ndarray) -> list[dict]:
        """Run detection on img with current model input_size. Returns list of dicts."""
        bboxes, kpss = self.model.detect(img)
        results = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            det = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'score': float(score)
            }
            if kpss is not None and i < len(kpss):
                det['landmarks'] = kpss[i].astype(int).tolist()
            results.append(det)
        return results

    def detect(self, img: np.ndarray) -> list[dict]:
        """
        Detect faces in img with two-pass fallback:
          Pass 1: input_size=(480,480) — best for large/close faces
          Pass 2: input_size=(640,640) — fallback for small/distant faces

        Returns list of:
          {'bbox': [x1,y1,x2,y2], 'landmarks': [[x,y]×5], 'score': float}
        """
        if img is None or img.size == 0:
            return []

        # Pass 1 — primary size (480x480, tuned for phone photos + webcam)
        if self.input_size != (480, 480):
            self._prepare((480, 480))
        results = self._run_detect(img)

        # Pass 2 — fallback to 640x640 for small/distant faces
        if not results:
            self._prepare((640, 640))
            results = self._run_detect(img)
            # Restore primary size for next call
            self._prepare((480, 480))

        # Ensure model is always left at primary size
        if self.model.input_size != self.input_size:
            self._prepare(self.input_size)

        return results

    @staticmethod
    def align_face(img: np.ndarray, landmarks: list, size=112) -> np.ndarray:
        """5-point landmark alignment for ArcFace input."""
        dst = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014],
            [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        src = np.array(landmarks, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return cv2.resize(img, (size, size))
        return cv2.warpAffine(img, M, (size, size))