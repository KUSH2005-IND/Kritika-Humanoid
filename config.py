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
RECHECK_INTERVAL_SEC    = 15.0     # Re-run recognition every N seconds per track
                                   # (was 30.0; halved to speed up "Catch" purge)
 
# ── Detection ────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD  = 0.45
SCRFD_INPUT_SIZE     = (480, 480)  # 480 catches large faces (phone/webcam); 640 used as fallback
SCRFD_CONF_THRESHOLD = 0.35
 
# ── Tracking ─────────────────────────────────────────────────────────
TRACK_TIMEOUT_SEC    = 5.0         # Remove track if unseen for N seconds
 
# ── Presence ─────────────────────────────────────────────────────────
PRESENCE_TIMEOUT_SEC = 8.0         # Remove from present list if unseen for N seconds
 
# ── Performance ──────────────────────────────────────────────────────
ORT_NUM_THREADS = 4                # Match to available physical cores
 
# ── Unknown Person Management ─────────────────────────────────────────
# Instances (quality face crops) needed before a cluster is offered for naming
UNKNOWN_ACCUMULATE_TARGET  = 20
 
# Minimum gap between successive crop saves per cluster.
# With RECHECK_INTERVAL_SEC = 15s, the effective floor is already 15s;
# this guard protects against future recheck reductions.
UNKNOWN_MIN_INTERVAL_SEC   = 2.5
 
# Quality gate: face bounding-box side must be at least this many pixels
UNKNOWN_MIN_FACE_SIZE      = 60
 
# Quality gate: SCRFD detection confidence must meet this floor
UNKNOWN_SCRFD_CONF_GATE    = 0.50
 
# Cosine similarity threshold for re-matching a new track to a detached cluster
# (same unknown person re-entered frame)
UNKNOWN_MATCH_THRESHOLD    = 0.38
 
# Cosine similarity threshold for purging a detached cluster when a known
# recognition fires (slightly below RECOGNITION_THRESHOLD to account for
# averaged / suboptimal embeddings in the cluster mean)
UNKNOWN_PURGE_THRESHOLD    = 0.37
 
# Minimum crop count before a cluster is written to disk (restart persistence)
UNKNOWN_MIN_PERSIST_COUNT  = 5




# # config.py
# # ── Modular Person Recognition System — Configuration ────────────────

# import os

# # ── Base Paths ───────────────────────────────────────────────────────
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # ── Model Paths ──────────────────────────────────────────────────────
# YOLO_MODEL_PATH    = "yolov8n.pt"
# SCRFD_MODEL_PATH   = os.path.join(BASE_DIR, "models", "det_10g.onnx")
# ARCFACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "arcface_r50.onnx")

# # ── Database Paths ───────────────────────────────────────────────────
# KNOWN_DB_ROOT      = os.path.join(BASE_DIR, "database", "known")
# FAISS_INDEX_PATH   = os.path.join(BASE_DIR, "database", "embeddings", "faiss_index.bin")
# FAISS_META_PATH    = os.path.join(BASE_DIR, "database", "embeddings", "faiss_meta.pkl")
# UNKNOWN_FACES_DIR  = os.path.join(BASE_DIR, "unknown_faces")
# DB_PATH            = os.path.join(BASE_DIR, "database", "presence.db")

# # ── Camera ───────────────────────────────────────────────────────────
# CAMERA_INDEX       = 1
# CAMERA_BACKEND     = "DSHOW"   # Windows: DSHOW avoids MSMF USB issues
# FRAME_WIDTH        = 640
# FRAME_HEIGHT       = 480

# # ── Recognition ──────────────────────────────────────────────────────
# RECOGNITION_THRESHOLD   = 0.42     # Cosine similarity (ArcFace R50 calibrated)
# UNKNOWN_SAVE_THRESHOLD  = 0.35     # Below this → save as unknown
# RECHECK_INTERVAL_SEC    = 30.0     # Re-run recognition every N seconds per track

# # ── Detection ────────────────────────────────────────────────────────
# YOLO_CONF_THRESHOLD = 0.45
# SCRFD_INPUT_SIZE = (480, 480)  # 480 catches large faces (phone/webcam), # 640 used as fallback for small/distant faces
# SCRFD_CONF_THRESHOLD = 0.35

# # ── Tracking ─────────────────────────────────────────────────────────
# TRACK_TIMEOUT_SEC   = 5.0          # Remove track if unseen for N seconds

# # ── Presence ─────────────────────────────────────────────────────────
# PRESENCE_TIMEOUT_SEC = 8.0         # Remove from present list if unseen for N seconds

# # ── Performance ──────────────────────────────────────────────────────
# ORT_NUM_THREADS = 4                # Match to available physical cores
