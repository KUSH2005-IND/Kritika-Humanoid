# FIX.md — Kritika-Humanoid: Complete Fix Guide

> All fixes required for smooth, production-adjacent execution on Windows + USB webcam + CPU-only target (Intel i7 8th Gen).  
> Fixes are ordered by severity: 🔴 Blocker → 🟠 High → 🟡 Medium → 🟢 Low.

---

## FIX-01 🔴 Camera fails to open on Windows (MSMF error)

**Symptom**
```
[ WARN:0] videoio(MSMF): OnReadSample() is called with error status: -1072873851
[App] Camera read failed. Exiting.
```

**Root cause**  
OpenCV on Windows defaults to the Media Source Foundation (MSMF) backend. USB webcams frequently fail with MSMF due to driver conflicts or Windows locking the device handle before OpenCV acquires it. DirectShow (DSHOW) bypasses this layer and talks to the USB camera directly.

**Fix — `app.py` line 107**

```python
# BEFORE
cap = cv2.VideoCapture(CAMERA_INDEX)

# AFTER
import platform
if platform.system() == "Windows":
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX)
```

**Fix — `config.py` (add after CAMERA_INDEX)**

```python
# ── Camera ───────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
CAMERA_BACKEND  = "DSHOW"   # Windows: DSHOW avoids MSMF USB issues
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
```

**If DSHOW also fails — camera index scan**  
USB webcams are sometimes index `1` when a built-in IR/Hello camera occupies index `0`. Run this once to find the correct index:

```python
# camera_scan.py — run once, then set CAMERA_INDEX in config.py
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

**If still failing** — kill any process holding the camera (browser tabs, Teams, Zoom, Discord, OBS) and retry. Only one process can own a USB camera handle at a time on Windows.

---

## FIX-02 🔴 `setup_models.py` copies wrong SCRFD model

**Root cause**  
`buffalo_l` ships `det_10g.onnx` (~17 MB, 10GFlop variant). The setup script globs `det_*.onnx` and copies it to `scrfd_500m_bnkps.onnx`. The filenames are interchangeable to the filesystem but the two models have **different output shapes** — `det_10g` has a different stride/anchor layout than `scrfd_500m`. If the wrong binary ends up at `models/scrfd_500m_bnkps.onnx`, inference silently produces garbage detections.

**Fix — `setup_models.py` `download_insightface_models()`**

```python
# BEFORE — name mismatch, wrong file may be copied
scrfd_candidates = list(insightface_dir.glob("det_*.onnx"))
if scrfd_candidates:
    src = scrfd_candidates[0]
    dst = models_dir / "scrfd_500m_bnkps.onnx"

# AFTER — copy with original name, update config to match
scrfd_candidates = list(insightface_dir.glob("det_*.onnx"))
if scrfd_candidates:
    src = scrfd_candidates[0]
    dst = models_dir / src.name          # keep original filename
    config_name = src.name               # e.g. "det_10g.onnx"
    print(f"  ✓ Copying SCRFD model as: {config_name}")
    if not dst.exists():
        shutil.copy2(str(src), str(dst))
```

Then update `config.py` to match whatever filename was actually copied:

```python
# config.py — match the actual filename from buffalo_l
SCRFD_MODEL_PATH = os.path.join(BASE_DIR, "models", "det_10g.onnx")
```

**Alternative (cleanest):** Remove the custom ONNX wrappers entirely for detection and use InsightFace's `FaceAnalysis` directly. It handles all model path resolution and output decoding internally (see FIX-03).

---

## FIX-03 🔴 SCRFD multi-stride output decode is incorrect

**Root cause**  
`face_detector.py` `_parse_outputs()` multi-stride branch (triggered when model has ≥9 outputs) iterates `score_blob.flatten()` and uses the flat index `j` against `bbox_blob.reshape(-1, 4)`. This is missing the per-stride anchor grid decode — raw SCRFD bbox outputs are delta offsets relative to anchor grids, not pixel coordinates. Without decoding the anchor grid, coordinates on strides 16 and 32 are wrong.

The simple output branch (`else`) works correctly with `buffalo_l`'s `det_10g.onnx` which is why enrollment ran successfully — it hit the else path.

**Fix option A (recommended) — replace SCRFDDetector with InsightFace native**

Replace `recognition/face_detector.py` with a thin wrapper around InsightFace's own model:

```python
# recognition/face_detector.py — InsightFace-native replacement
from insightface.model_zoo import get_model
import numpy as np
import cv2

class SCRFDDetector:
    def __init__(self, model_path: str, input_size=(640, 640), conf_thresh=0.5, nms_thresh=0.4):
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=-1, input_size=input_size)
        self.conf_thresh = conf_thresh

    def detect(self, img: np.ndarray) -> list[dict]:
        if img is None or img.size == 0:
            return []
        bboxes, kpss = self.model.detect(img, thresh=self.conf_thresh)
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
        dst = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014],
            [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        src = np.array(landmarks, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return cv2.resize(img, (size, size))
        return cv2.warpAffine(img, M, (size, size))
```

**Fix option B** — keep custom decoder but add proper anchor grid decode per InsightFace's reference SCRFD implementation (significantly more complex; only worth it if you need full control over the inference path).

---

## FIX-04 🟠 `torch.load` monkey-patch has global scope

**Root cause**  
`tracking/tracker.py` replaces `torch.load` at module level:

```python
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load
```

This mutates the global `torch` module object. Any library imported after `tracker.py` that calls `torch.load` will get the patched version with `weights_only=False`, bypassing PyTorch's security model for all subsequent loads in the process.

**Fix — use environment variable instead**

```python
# tracking/tracker.py — top of file, before ultralytics import
import os
os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

# Remove the monkey-patch block entirely:
# _original_load = torch.load         ← DELETE
# def safe_load(*args, **kwargs):     ← DELETE
#     kwargs['weights_only'] = False  ← DELETE
#     return _original_load(*args, **kwargs)  ← DELETE
# torch.load = safe_load              ← DELETE

from ultralytics import YOLO
```

Alternatively, upgrade to `ultralytics>=8.3.0` which sets `weights_only` internally and no patch is needed.

---

## FIX-05 🟠 Recognition pipeline blocks the camera loop

**Root cause**  
ONNX inference for SCRFD + ArcFace on CPU takes 30–80 ms per face. With 3 simultaneous unknown tracks all requiring recognition in the same frame, `app.py`'s main loop blocks for 90–240 ms, dropping effective FPS to 4–10. The ByteTrack `RECHECK_INTERVAL_SEC=30` limits how often this happens, but first-frame recognition of multiple people is always serial.

**Fix — async recognition via ThreadPoolExecutor**

```python
# app.py — add at top of file
from concurrent.futures import ThreadPoolExecutor
import threading

# In main(), after initialising modules:
executor = ThreadPoolExecutor(max_workers=2)  # 2 workers for 2 ONNX sessions

def recognise_track(track, face_detector, embedder, db, identity_engine, unknown_handler, tracker):
    crop = track['crop']
    track_id = track['track_id']
    faces = face_detector.detect(crop)
    if not faces:
        tracker.update_track_identity(track_id, 'Unknown', 0.0)
        return None
    best_face = max(faces, key=lambda d: d['score'])
    if 'landmarks' in best_face:
        aligned = SCRFDDetector.align_face(crop, best_face['landmarks'])
    else:
        x1, y1, x2, y2 = best_face['bbox']
        ch, cw = crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(cw, x2), min(ch, y2)
        face_region = crop[y1:y2, x1:x2]
        if face_region.size == 0:
            tracker.update_track_identity(track_id, 'Unknown', 0.0)
            return None
        aligned = cv2.resize(face_region, (112, 112))
    query_emb = embedder.embed(aligned)
    name, score = db.majority_vote_search(query_emb, top_k=7)
    decision = identity_engine.decide(name, score)
    tracker.update_track_identity(track_id, decision['identity'], decision['score'])
    if not decision['is_known']:
        unknown_handler.save(aligned)
        return None
    return decision['identity']

# In the camera loop, replace the sequential recognition block with:
futures = [
    executor.submit(recognise_track, track, face_detector, embedder,
                    db, identity_engine, unknown_handler, tracker)
    for track in tracks_to_recognise
]
current_identities = [f.result() for f in futures if f.result() is not None]
```

> Note: ONNX Runtime sessions are not thread-safe by default. Set `sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL` in both `SCRFDDetector` and `ArcFaceEmbedder`, or instantiate separate session objects per worker.

---

## FIX-06 🟡 `UnknownFaceHandler` deduplication is global, not per-track

**Root cause**  
`_last_saved` is a single timestamp. If 3 unknown persons appear simultaneously, only the first gets saved; the 10-second cooldown blocks the other 2 regardless of whether they are different individuals.

**Fix — per-track cooldown dict**

```python
# storage/unknown_handler.py

class UnknownFaceHandler:
    MIN_INTERVAL_SEC = 10.0

    def __init__(self, save_dir=UNKNOWN_FACES_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved: dict[int, float] = {}   # track_id → timestamp
        self._counter = 0

    def save(self, face_crop, track_id: int = -1) -> str | None:
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
```

Update the call in `app.py`:
```python
# BEFORE
unknown_handler.save(aligned)

# AFTER
unknown_handler.save(aligned, track_id=track_id)
```

---

## FIX-07 🟡 Enrollment uses single-image `embed()` instead of `embed_batch()`

**Root cause**  
`recognition/enrollment.py` collects face crops in a list and calls `embedder.embed(aligned)` per image inside a loop. `ArcFaceEmbedder.embed_batch()` exists and does a single batched ONNX forward pass — 2–4× faster for initial enrollment with many images.

**Fix — `recognition/enrollment.py` `enroll_from_folder()`**

```python
# BEFORE — sequential, one ONNX call per image
for img_path in image_paths:
    ...
    emb = self.embedder.embed(aligned)
    embeddings.append(emb)

# AFTER — collect aligned crops, then single batch call
aligned_faces = []
for img_path in image_paths:
    ...
    aligned_faces.append(aligned)

if not aligned_faces:
    print(f"[Enrollment] No valid faces found for {name}")
    return 0

emb_matrix = self.embedder.embed_batch(aligned_faces)  # single forward pass
self.db.add_person(name, emb_matrix)
print(f"[Enrollment] Enrolled '{name}' with {len(aligned_faces)} embeddings ({skipped} skipped)")
return len(aligned_faces)
```

---

## FIX-08 🟡 `database.py` `get_session_duration()` is semantically wrong

**Root cause**  
The method calculates session duration by running `MIN(ts)` and `MAX(ts)` on `entry` events only, which gives the span from first ever entry to last ever entry — not the duration of a single session. The docstring acknowledges this.

**Fix — pair entry/exit events per session**

```python
def get_session_duration(self, name: str, date: str = None) -> dict:
    """
    Returns entry/exit pairs for the most recent session for a given person.
    If no exit is recorded (person still present), last_seen = last entry.
    """
    date_filter = f"AND date(ts, 'unixepoch') = '{date}'" if date else ""
    query = f"""
    SELECT event, ts FROM presence_log
    WHERE name = ? {date_filter}
    ORDER BY ts ASC
    """
    try:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, (name,)).fetchall()
        if not rows:
            return {}
        # Find last entry/exit pair
        last_entry = None
        last_exit = None
        for event, ts in rows:
            if event == 'entry':
                last_entry = ts
                last_exit = None   # reset — new session started
            elif event == 'exit' and last_entry is not None:
                last_exit = ts
        if last_entry is None:
            return {}
        end_ts = last_exit if last_exit else last_entry
        return {
            'arrived': last_entry,
            'last_seen': end_ts,
            'minutes': (end_ts - last_entry) / 60.0,
            'still_present': last_exit is None
        }
    except sqlite3.Error as e:
        print(f"[Database] Error querying session duration for {name}: {e}")
        return {}
```

---

## FIX-09 🟡 Dead config constant `RECOGNITION_INTERVAL_FRAMES`

**Root cause**  
`config.py` defines `RECOGNITION_INTERVAL_FRAMES = 5` but `tracker.py` never imports or uses it. Only `RECHECK_INTERVAL_SEC` drives recognition gating. Dead constants create confusion about what the system actually does.

**Fix — remove from `config.py`**

```python
# config.py — REMOVE this line entirely:
RECOGNITION_INTERVAL_FRAMES = 5    # ← DELETE, unused
```

And remove from `config.py`'s `tracker.py` import:

```python
# tracking/tracker.py — BEFORE
from config import (RECOGNITION_INTERVAL_FRAMES, TRACK_TIMEOUT_SEC,
                    RECHECK_INTERVAL_SEC, YOLO_CONF_THRESHOLD)

# AFTER
from config import (TRACK_TIMEOUT_SEC, RECHECK_INTERVAL_SEC, YOLO_CONF_THRESHOLD)
```

---

## FIX-10 🟢 No graceful error when model files are missing

**Root cause**  
If `setup_models.py` was not run and `.onnx` files are absent, `SCRFDDetector.__init__` throws a cryptic ONNX Runtime `InvalidGraph` or `FileNotFound` traceback with no pointer to the fix.

**Fix — add existence checks in `face_detector.py` and `embedder.py`**

```python
# recognition/face_detector.py __init__ — add before ort.InferenceSession
from pathlib import Path
if not Path(model_path).exists():
    raise FileNotFoundError(
        f"[SCRFDDetector] Model not found: {model_path}\n"
        f"Run 'python setup_models.py' to download models."
    )

# recognition/embedder.py __init__ — same guard
from pathlib import Path
if not Path(model_path).exists():
    raise FileNotFoundError(
        f"[ArcFaceEmbedder] Model not found: {model_path}\n"
        f"Run 'python setup_models.py' to download models."
    )
```

---

## FIX-11 🟢 `FAISS majority_vote_search` `top_k` not clamped to enrollment size

**Root cause**  
With a small enrollment (e.g. 2 people, 8 total vectors), `top_k=7` returns all 8 vectors. The vote aggregation still works, but cumulative score ordering becomes unreliable when one person has far more enrolled images than another — that person will win by volume, not by similarity.

**Fix — `recognition/faiss_db.py` `majority_vote_search()`**

```python
def majority_vote_search(self, query: np.ndarray, top_k=7) -> tuple[str, float]:
    # Clamp top_k to avoid over-polling small databases
    effective_k = min(top_k, max(1, self.index.ntotal // 2))
    candidates = self.search(query, top_k=effective_k)
    ...
```

---

## Quick-Reference: Apply Order

| # | File | Severity | Description |
|---|------|----------|-------------|
| FIX-01 | `app.py`, `config.py` | 🔴 | DSHOW backend for Windows USB camera |
| FIX-02 | `setup_models.py`, `config.py` | 🔴 | Correct SCRFD model filename on copy |
| FIX-03 | `recognition/face_detector.py` | 🔴 | Replace broken multi-stride SCRFD decoder |
| FIX-04 | `tracking/tracker.py` | 🟠 | Remove global `torch.load` monkey-patch |
| FIX-05 | `app.py` | 🟠 | Async recognition via ThreadPoolExecutor |
| FIX-06 | `storage/unknown_handler.py`, `app.py` | 🟡 | Per-track unknown face deduplication |
| FIX-07 | `recognition/enrollment.py` | 🟡 | Use `embed_batch()` for enrollment speed |
| FIX-08 | `storage/database.py` | 🟡 | Fix `get_session_duration()` entry/exit logic |
| FIX-09 | `config.py`, `tracking/tracker.py` | 🟢 | Remove unused `RECOGNITION_INTERVAL_FRAMES` |
| FIX-10 | `recognition/face_detector.py`, `embedder.py` | 🟢 | Graceful FileNotFoundError for missing models |
| FIX-11 | `recognition/faiss_db.py` | 🟢 | Clamp `top_k` to enrolled vector count |

---

## Minimum viable fix set to get a running demo

If you want the fastest path to a working camera feed with recognition:

1. **FIX-01** — camera opens
2. **FIX-02** — correct model path in `config.py`
3. **FIX-03 option A** — replace SCRFD wrapper with InsightFace native
4. **FIX-04** — remove torch patch (prevents potential silent inference errors)

FIX-05 through FIX-11 are quality/robustness improvements that won't block the initial demo run.
