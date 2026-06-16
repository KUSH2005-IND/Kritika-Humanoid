# recognition/face_detector.py
"""
SCRFD Face Detector — lightweight face detection optimised for CPU.
Returns face bounding boxes + 5-point landmarks for alignment.
"""

from insightface.model_zoo import get_model
from pathlib import Path
import numpy as np
import cv2

class SCRFDDetector:
    """
    Lightweight face detector optimised for CPU.
    Returns face bounding boxes + 5-point landmarks for alignment.
    """

    def __init__(self, model_path: str, input_size=(640, 640), conf_thresh=0.5, nms_thresh=0.4):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"[SCRFDDetector] Model not found: {model_path}\n"
                f"Run 'python setup_models.py' to download models."
            )
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=-1, input_size=input_size, det_thresh=conf_thresh, nms_thresh=nms_thresh)
        self.conf_thresh = conf_thresh

    def detect(self, img: np.ndarray) -> list[dict]:
        """
        Returns list of:
          {'bbox': [x1,y1,x2,y2], 'landmarks': [[x,y]×5], 'score': float}
        """
        if img is None or img.size == 0:
            return []
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
