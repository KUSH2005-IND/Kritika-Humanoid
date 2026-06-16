# config.py
# ── Modular Person Recognition System — Configuration ────────────────

import os

# ── Base Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model Paths ──────────────────────────────────────────────────────
YOLO_MODEL_PATH    = "yolov8n.pt"
SCRFD_MODEL_PATH   = os.path.join(BASE_DIR, "models", "det_10g.onnx")
ARCFACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "arcface_r50.onnx")

# ── Database Paths ───────────────────────────────────────────────────
KNOWN_DB_ROOT      = os.path.join(BASE_DIR, "database", "known")
FAISS_INDEX_PATH   = os.path.join(BASE_DIR, "database", "embeddings", "faiss_index.bin")
FAISS_META_PATH    = os.path.join(BASE_DIR, "database", "embeddings", "faiss_meta.pkl")
UNKNOWN_FACES_DIR  = os.path.join(BASE_DIR, "unknown_faces")
DB_PATH            = os.path.join(BASE_DIR, "database", "presence.db")

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_INDEX       = 1
CAMERA_BACKEND     = "DSHOW"   # Windows: DSHOW avoids MSMF USB issues
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480

# ── Recognition ──────────────────────────────────────────────────────
RECOGNITION_THRESHOLD   = 0.42     # Cosine similarity (ArcFace R50 calibrated)
UNKNOWN_SAVE_THRESHOLD  = 0.35     # Below this → save as unknown
RECHECK_INTERVAL_SEC    = 30.0     # Re-run recognition every N seconds per track

# ── Detection ────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.45

# ── Tracking ─────────────────────────────────────────────────────────
TRACK_TIMEOUT_SEC   = 5.0          # Remove track if unseen for N seconds

# ── Presence ─────────────────────────────────────────────────────────
PRESENCE_TIMEOUT_SEC = 8.0         # Remove from present list if unseen for N seconds

# ── Performance ──────────────────────────────────────────────────────
ORT_NUM_THREADS = 4                # Match to available physical cores
