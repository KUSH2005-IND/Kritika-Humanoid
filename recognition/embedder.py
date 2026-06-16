# recognition/embedder.py
"""
ArcFace R50 — 512-D L2-normalised face embeddings via ONNX Runtime.
Cosine similarity reduces to dot product after normalisation.
"""

import numpy as np
import onnxruntime as ort
import cv2


class ArcFaceEmbedder:
    """
    ArcFace R50 — 512-D L2-normalised embeddings.
    Cosine similarity reduces to dot product after normalisation.
    """

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Get expected input shape from model
        input_shape = self.session.get_inputs()[0].shape
        # Typically [batch, channels, height, width] = [1, 3, 112, 112]
        self.input_h = input_shape[2] if isinstance(input_shape[2], int) else 112
        self.input_w = input_shape[3] if isinstance(input_shape[3], int) else 112

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """ArcFace expects 112×112 RGB, normalised to [-1, 1]."""
        if face_img.shape[:2] != (self.input_h, self.input_w):
            face_img = cv2.resize(face_img, (self.input_w, self.input_h))
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
        return blob.transpose(2, 0, 1)[np.newaxis]          # NCHW

    def embed(self, face_img: np.ndarray) -> np.ndarray:
        """Returns 512-D L2-normalised embedding vector."""
        blob = self.preprocess(face_img)
        embedding = self.session.run(None, {self.input_name: blob})[0][0]
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return embedding.astype(np.float32)
        return (embedding / norm).astype(np.float32)

    def embed_batch(self, face_imgs: list[np.ndarray]) -> np.ndarray:
        """Batch embedding for enrollment — more efficient than sequential."""
        if not face_imgs:
            return np.array([], dtype=np.float32)
        blobs = np.vstack([self.preprocess(f) for f in face_imgs])
        embeddings = self.session.run(None, {self.input_name: blobs})[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Prevent division by zero
        return (embeddings / norms).astype(np.float32)
