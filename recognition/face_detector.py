# recognition/face_detector.py
"""
SCRFD Face Detector — lightweight face detection optimised for CPU.
Returns face bounding boxes + 5-point landmarks for alignment.
"""

import cv2
import numpy as np
import onnxruntime as ort


class SCRFDDetector:
    """
    Lightweight face detector optimised for CPU.
    Returns face bounding boxes + 5-point landmarks for alignment.
    """

    def __init__(self, model_path: str, input_size=(640, 640), conf_thresh=0.5, nms_thresh=0.4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # Tune to CPU core count
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_name = self.session.get_inputs()[0].name

        # Determine output structure from model
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._num_outputs = len(self._output_names)

    def preprocess(self, img: np.ndarray) -> tuple:
        """Resize + normalise. Returns blob and scale factors."""
        h, w = img.shape[:2]
        ih, iw = self.input_size
        scale_x, scale_y = w / iw, h / ih
        resized = cv2.resize(img, (iw, ih))
        blob = (resized.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]   # NCHW
        return blob, scale_x, scale_y

    def detect(self, img: np.ndarray) -> list[dict]:
        """
        Returns list of:
          {'bbox': [x1,y1,x2,y2], 'landmarks': [[x,y]×5], 'score': float}
        """
        if img is None or img.size == 0:
            return []

        blob, sx, sy = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})

        # SCRFD models from InsightFace have varying output layouts.
        # Common layout for SCRFD with keypoints: 9 outputs
        #   3 stride groups × (score_map, bbox_map, kps_map)
        # Simpler layout: scores, bboxes, landmarks
        detections = self._parse_outputs(outputs, sx, sy)
        return self._nms(detections)

    def _parse_outputs(self, outputs, sx, sy) -> list[dict]:
        """Parse SCRFD outputs based on model structure."""
        detections = []

        if self._num_outputs >= 9:
            # Multi-stride SCRFD output: 3 strides × (scores, bboxes, kps)
            fmc = 3
            for i in range(fmc):
                score_blob = outputs[i]
                bbox_blob = outputs[i + fmc]
                kps_blob = outputs[i + fmc * 2] if self._num_outputs > fmc * 2 else None

                scores = score_blob.flatten()
                for j, score in enumerate(scores):
                    if score < self.conf_thresh:
                        continue

                    # Decode bounding box
                    bbox_data = bbox_blob.reshape(-1, 4)
                    if j >= len(bbox_data):
                        continue
                    x1, y1, x2, y2 = bbox_data[j] * [sx, sy, sx, sy]

                    det = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'score': float(score)
                    }

                    # Decode landmarks
                    if kps_blob is not None:
                        kps_data = kps_blob.reshape(-1, 10)
                        if j < len(kps_data):
                            kps = kps_data[j].reshape(5, 2) * [[sx, sy]]
                            det['landmarks'] = kps.astype(int).tolist()

                    detections.append(det)
        else:
            # Simple output layout: scores, bboxes, [landmarks]
            scores = outputs[0].squeeze()
            bboxes = outputs[1].squeeze()
            landmarks = outputs[2].squeeze() if len(outputs) > 2 else None

            if scores.ndim == 0:
                scores = scores.reshape(1)
                bboxes = bboxes.reshape(1, 4)
                if landmarks is not None:
                    landmarks = landmarks.reshape(1, 10)

            for i, score in enumerate(scores):
                if score < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = bboxes[i] * [sx, sy, sx, sy]
                det = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'score': float(score)
                }
                if landmarks is not None:
                    kps = landmarks[i].reshape(5, 2) * [[sx, sy]]
                    det['landmarks'] = kps.astype(int).tolist()
                detections.append(det)

        return detections

    def _nms(self, detections: list) -> list:
        """Non-maximum suppression."""
        if not detections:
            return []
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1 + 1) * np.maximum(0, iy2 - iy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= self.nms_thresh)[0] + 1]
        return [detections[i] for i in keep]

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
