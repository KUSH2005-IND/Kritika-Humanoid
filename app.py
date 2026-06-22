# app.py
"""
Modular Person Recognition System — Main Pipeline
Real-time person detection → tracking → face recognition → presence management
 
Target: Intel i7 8th Gen CPU, ~22-28 FPS effective
 
Keybindings (OpenCV window must be in focus):
  Q — quit
  E — enroll whoever is currently in frame as a new person (terminal prompt)
  U — name a pending unknown cluster (terminal prompt, amber box = ready)
"""
 
import cv2
import time
import sys
import os
import platform
import threading
import numpy as np
from pathlib import Path
 
# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from config import *
from recognition.face_detector import SCRFDDetector
from recognition.embedder import ArcFaceEmbedder
from recognition.faiss_db import FaceDatabase
from recognition.identity import IdentityDecision
from recognition.enrollment import EnrollmentPipeline
from tracking.tracker import PersonTracker
from presence.manager import PresenceManager
from storage.unknown_handler import UnknownPersonManager
 
 
# ── Live Enrollment State ────────────────────────────────────────────────────
# Shared between main loop and enrollment thread — all protected by _enroll_lock
_enroll_lock        = threading.Lock()
_enroll_pending     = False   # E was pressed, waiting for name input
_enroll_name        = None    # Name typed in terminal (None until submitted)
_enroll_crops       = []      # Accumulated face crops captured while enrolling
_enroll_active      = False   # Actively capturing multi-frame crops
_enroll_frame_count = 0       # How many crops captured so far
_ENROLL_TARGET      = 15      # Number of frames to capture before committing
 
# ── Unknown Naming State ─────────────────────────────────────────────────────
# Shared between main loop and U-key thread — protected by _ukey_lock
_ukey_lock    = threading.Lock()
_ukey_pending = False   # U-key flow is active in a background thread
 
 
def _terminal_name_prompt():
    """Runs in a background thread — reads name from terminal without blocking camera."""
    global _enroll_name, _enroll_active, _enroll_frame_count, _enroll_crops
    print("\n" + "=" * 50)
    print("  LIVE ENROLLMENT")
    print("  Stay in frame. Type the person's name and press Enter.")
    print("  Leave blank and press Enter to cancel.")
    print("=" * 50)
    name = input("  Name: ").strip()
    with _enroll_lock:
        if name:
            _enroll_name        = name
            _enroll_active      = True
            _enroll_frame_count = 0
            _enroll_crops       = []
            print(f"  [Enroll] Capturing {_ENROLL_TARGET} frames for '{name}'...")
        else:
            print("  [Enroll] Cancelled.")
            globals()['_enroll_pending'] = False
 
 
def _terminal_unknown_prompt(unknown_mgr, embedder, db, tracker):
    """
    Runs in a background thread — lists clusters ready for naming and prompts operator.
    Camera loop never pauses; this thread interacts only via terminal I/O and
    brief lock acquisitions on unknown_mgr.
    """
    global _ukey_pending
 
    ready = unknown_mgr.ready_clusters
    if not ready:
        print("\n[Unknown] No clusters ready to name yet.")
        with _ukey_lock:
            _ukey_pending = False
        return
 
    print("\n" + "=" * 58)
    print("  PENDING UNKNOWNS")
    print("  " + "─" * 54)
 
    numbered = []
    for c in ready:
        first_t = time.strftime("%H:%M", time.localtime(c.first_seen))
        idx_str = f"[{len(numbered) + 1}]"
        print(f"  {idx_str}  {c.display_label:<14}  —  {c.count:2d} instances  —  {first_t}")
        numbered.append(c)
 
    # Show accumulating clusters for awareness, but they can't be selected
    accumulating = [c for c in unknown_mgr.all_clusters if c.count < UNKNOWN_ACCUMULATE_TARGET]
    for c in accumulating:
        first_t = time.strftime("%H:%M", time.localtime(c.first_seen))
        print(f"       {c.display_label:<14}  —  {c.count:2d} instances  —  (accumulating)")
 
    print("  " + "─" * 54)
    sel = input("  Enter number to name (blank = cancel): ").strip()
 
    if not sel:
        print("  [Unknown] Cancelled.")
        with _ukey_lock:
            _ukey_pending = False
        return
 
    try:
        idx = int(sel) - 1
        if not (0 <= idx < len(numbered)):
            raise ValueError("out of range")
        chosen = numbered[idx]
    except ValueError:
        print("  [Unknown] Invalid selection — enter a number from the list.")
        with _ukey_lock:
            _ukey_pending = False
        return
 
    name = input(f"  Name for {chosen.display_label}: ").strip()
    if not name:
        print("  [Unknown] Cancelled.")
        with _ukey_lock:
            _ukey_pending = False
        return
 
    # enroll_cluster handles the cancelled-flag race internally
    success = unknown_mgr.enroll_cluster(
        cluster_id    = chosen.cluster_id,
        name          = name,
        embedder      = embedder,
        db            = db,
        known_db_root = KNOWN_DB_ROOT,
    )
 
    if success:
        # Invalidate all cached track identities so re-recognition fires immediately
        for tid in list(tracker.track_cache.keys()):
            tracker.track_cache[tid]['last_recognised'] = 0.0
        print(f"  [Unknown] ✓ '{name}' enrolled.")
        print(f"  [Unknown] ✓ FAISS: {db.total_vectors} vectors for "
              f"{len(db.enrolled_names)} persons\n")
 
    with _ukey_lock:
        _ukey_pending = False
 
 
# ── Overlay ──────────────────────────────────────────────────────────────────
 
def draw_overlay(
    frame,
    all_tracks,
    presence_list,
    enroll_active,
    enroll_name,
    enroll_frame_count,
    unknown_mgr=None,
):
    """
    Draw bounding boxes, labels, presence panel, and bottom HUD.
 
    Bounding box colour states:
      Green          — known, recognised person
      Red            — unknown, cluster accumulating  (< UNKNOWN_ACCUMULATE_TARGET)
      Amber          — unknown, cluster ready to name (≥ UNKNOWN_ACCUMULATE_TARGET)
      Grey           — pending (recognition not yet run this track)
      Orange/Blue    — live E-key enrollment in progress
    """
 
    for track in all_tracks:
        x1, y1, x2, y2 = track['bbox']
        track_id = track.get('track_id')
        identity = track.get('identity', 'Pending')
 
        if enroll_active:
            # Blue-ish during live capture phase
            colour = (255, 140, 0)
            label  = f"Capturing... {enroll_frame_count}/{_ENROLL_TARGET}"
 
        elif identity in (None, 'Pending'):
            # Grey — recognition hasn't run yet for this track
            colour = (140, 140, 140)
            label  = 'Pending'
 
        elif identity.startswith('Unknown'):
            # Unknown cluster — check if it's ready for naming
            count    = 0
            is_ready = False
            if unknown_mgr is not None and track_id is not None:
                _, count = unknown_mgr.get_track_status(track_id)
                is_ready = count >= UNKNOWN_ACCUMULATE_TARGET
 
            if is_ready:
                # Amber/orange — operator should press U
                colour = (0, 165, 255)    # BGR: B=0 G=165 R=255
                label  = f"{identity} ✓"
            else:
                # Red — still accumulating
                colour = (0, 0, 200)
                label  = f"{identity} ({count})" if count > 0 else identity
 
        else:
            # Known, recognised person
            colour = (0, 200, 0)
            label  = identity
 
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
 
    # ── Presence panel (top-right) ───────────────────────────────────────────
    panel_x = frame.shape[1] - 200
    panel_h = 26 + 22 * max(len(presence_list), 1)
    cv2.rectangle(frame, (panel_x, 0), (frame.shape[1], panel_h), (30, 30, 30), -1)
    cv2.putText(frame, "PRESENT", (panel_x + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if presence_list:
        for i, name in enumerate(presence_list):
            cv2.putText(frame, f"  {name}", (panel_x + 5, 42 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    else:
        cv2.putText(frame, "  (none)", (panel_x + 5, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
 
    # ── Bottom HUD ───────────────────────────────────────────────────────────
    h, w = frame.shape[:2]
    if enroll_active and enroll_name:
        bar_text = f"  ENROLLING: {enroll_name}  [{enroll_frame_count}/{_ENROLL_TARGET} frames]"
        cv2.rectangle(frame, (0, h - 32), (w, h), (0, 100, 200), -1)
        cv2.putText(frame, bar_text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        ready_count = unknown_mgr.ready_count() if unknown_mgr else 0
        if ready_count > 0:
            hint = f"E: enroll  |  U: name unknowns ({ready_count} ready)  |  Q: quit"
        else:
            hint = "E: enroll  |  Q: quit"
        cv2.putText(frame, hint, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
 
    return frame
 
 
# ── Enrollment commit ─────────────────────────────────────────────────────────
 
def commit_enrollment(name, crops, embedder, db, tracker):
    """Embed all captured crops and add to FAISS. Invalidate all track caches."""
    if not crops:
        print(f"  [Enroll] No valid crops captured for '{name}'. Aborted.")
        return
 
    print(f"  [Enroll] Embedding {len(crops)} crops for '{name}'...")
    emb_matrix = embedder.embed_batch(crops)
    db.add_person(name, emb_matrix)
 
    # Also save images to database/known/<name>/ for persistence across restarts
    person_dir = Path(KNOWN_DB_ROOT) / name
    person_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, crop in enumerate(crops):
        cv2.imwrite(str(person_dir / f"live_{ts}_{i:03d}.jpg"), crop)
 
    # Force all cached tracks to re-run recognition immediately
    for tid in list(tracker.track_cache.keys()):
        tracker.track_cache[tid]['last_recognised'] = 0.0
 
    print(f"  [Enroll] ✓ '{name}' enrolled with {len(crops)} embeddings.")
    print(f"  [Enroll] ✓ Images saved to database/known/{name}/")
    print(f"  [Enroll] ✓ FAISS now has {db.total_vectors} vectors for "
          f"{len(db.enrolled_names)} persons\n")
 
 
# ── Main loop ─────────────────────────────────────────────────────────────────
 
def main():
    global _enroll_pending, _enroll_name, _enroll_active, _enroll_frame_count, _enroll_crops
 
    import argparse
    parser = argparse.ArgumentParser(description="Modular Person Recognition System")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Camera index to use")
    args = parser.parse_args()
    camera_index = args.camera
 
    print("=" * 60)
    print("  MODULAR PERSON RECOGNITION SYSTEM")
    print("  Target: Intel i7 8th Gen CPU · No GPU")
    print("=" * 60)
    print()
 
    # ── Initialise modules ───────────────────────────────────────────────────
    print("[App] Loading models...")
 
    print("[App]   → SCRFD face detector...")
    face_detector = SCRFDDetector(SCRFD_MODEL_PATH)
 
    print("[App]   → ArcFace embedder...")
    embedder = ArcFaceEmbedder(ARCFACE_MODEL_PATH)
 
    print("[App]   → FAISS database...")
    db = FaceDatabase(index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
 
    print("[App]   → Identity decision engine...")
    identity_engine = IdentityDecision(threshold=RECOGNITION_THRESHOLD)
 
    print("[App]   → YOLOv8n person tracker...")
    tracker = PersonTracker(YOLO_MODEL_PATH)
 
    print("[App]   → Presence manager...")
    presence_mgr = PresenceManager()
 
    print("[App]   → Unknown person manager...")
    unknown_mgr = UnknownPersonManager()
 
    enrollment = EnrollmentPipeline(face_detector, embedder, db)
 
    print("[App] All modules loaded successfully.")
    print()
 
    # ── Optional: first-run enrollment ──────────────────────────────────────
    if db.index.ntotal == 0:
        print("[App] No enrolled persons found. Running enrollment...")
        results = enrollment.enroll_all(KNOWN_DB_ROOT)
        if results:
            print(f"[App] Enrollment complete: {results}")
        else:
            print("[App] No persons enrolled. Press E while someone is in frame to enroll them.")
    else:
        print(f"[App] Database loaded: {db.total_vectors} vectors for "
              f"{len(db.enrolled_names)} persons")
        print(f"[App] Enrolled: {db.enrolled_names}")
 
    # ── Open camera ──────────────────────────────────────────────────────────
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
 
    if not cap.isOpened():
        print(f"[App] ERROR: Cannot open camera (index={camera_index})")
        print("[App] Check camera connection and CAMERA_INDEX in config.py "
              "or pass --camera <index>")
        return
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[App] Camera opened: {actual_w}×{actual_h}")
    print("[App] Controls: E = enroll  |  U = name unknowns  |  Q = quit")
    print()
 
    fps_time    = time.time()
    frame_count = 0
 
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[App] Camera read failed. Exiting.")
                break
 
            frame_count += 1
 
            # ── Track persons ────────────────────────────────────────────────
            tracks_to_recognise, all_tracks = tracker.update(frame)
 
            # ── CHANGE 1: Detach unknown clusters for timed-out tracks ───────
            # tracker.track_cache is the ground truth for live tracks.
            # Any track_id absent from it has been pruned by ByteTrack (> TRACK_TIMEOUT_SEC).
            live_track_ids = set(tracker.track_cache.keys())
            for tid in unknown_mgr.get_active_track_ids():
                if tid not in live_track_ids:
                    unknown_mgr.detach_track(tid)
 
            # ── Live enrollment — crop capture phase ─────────────────────────
            with _enroll_lock:
                currently_active = _enroll_active
                current_name     = _enroll_name
 
            if currently_active and all_tracks:
                # Grab the largest person crop in frame
                best_track = max(all_tracks, key=lambda t: (
                    (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])
                ))
                x1, y1, x2, y2 = best_track['bbox']
                crop = frame[y1:y2, x1:x2]
                faces = face_detector.detect(crop)
 
                if faces:
                    best_face = max(faces, key=lambda d: d['score'])
                    if 'landmarks' in best_face:
                        aligned = SCRFDDetector.align_face(crop, best_face['landmarks'])
                    else:
                        fx1, fy1, fx2, fy2 = best_face['bbox']
                        ch, cw = crop.shape[:2]
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(cw, fx2), min(ch, fy2)
                        face_region = crop[fy1:fy2, fx1:fx2]
                        if face_region.size > 0:
                            aligned = cv2.resize(face_region, (112, 112))
                        else:
                            aligned = None
 
                    if aligned is not None:
                        with _enroll_lock:
                            _enroll_crops.append(aligned)
                            _enroll_frame_count += 1
                            done           = _enroll_frame_count >= _ENROLL_TARGET
                            crops_snapshot = list(_enroll_crops)
                            name_snapshot  = _enroll_name
 
                        if done:
                            commit_enrollment(name_snapshot, crops_snapshot,
                                              embedder, db, tracker)
                            with _enroll_lock:
                                _enroll_active      = False
                                _enroll_pending     = False
                                _enroll_name        = None
                                _enroll_crops       = []
                                _enroll_frame_count = 0
 
            # ── Standard recognition on stale / new tracks ───────────────────
            current_identities = []
 
            for track in tracks_to_recognise:
                crop     = track['crop']
                track_id = track['track_id']
 
                faces = face_detector.detect(crop)
                if not faces:
                    tracker.update_track_identity(track_id, 'Unknown', 0.0)
                    continue
 
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
                        continue
                    aligned = cv2.resize(face_region, (112, 112))
 
                query_emb = embedder.embed(aligned)
                name, score = db.majority_vote_search(query_emb, top_k=7)
                decision = identity_engine.decide(name, score)
 
                if decision['is_known']:
                    # ── CHANGE 2: The Catch — purge clusters on known recognition ──
                    # Case 1: active cluster for this track
                    purged = unknown_mgr.purge_by_track(track_id, decision['identity'])
                    # Case 2: no active cluster — search detached clusters
                    if not purged:
                        unknown_mgr.purge_by_embedding(query_emb, decision['identity'])
 
                    tracker.update_track_identity(track_id, decision['identity'],
                                                  decision['score'])
                    current_identities.append(decision['identity'])
 
                else:
                    # ── CHANGE 3: Route to UnknownPersonManager ───────────────
                    display_label = unknown_mgr.handle_unknown(
                        face_crop   = aligned,
                        embedding   = query_emb,
                        track_id    = track_id,
                        scrfd_score = best_face['score'],
                        face_bbox   = best_face['bbox'],
                    )
                    # Store cluster label in track cache so overlay shows correct label
                    tracker.update_track_identity(track_id, display_label, decision['score'])
 
            # ── Collect cached known identities (non-recheck tracks) ─────────
            for track in all_tracks:
                identity = track.get('identity')
                if identity and identity not in ('Unknown', 'Pending', None) \
                        and not identity.startswith('Unknown #'):
                    if identity not in current_identities:
                        current_identities.append(identity)
 
            # ── Presence ─────────────────────────────────────────────────────
            events = presence_mgr.update(current_identities)
            for name in events['entries']:
                print(f"[Presence] ENTRY → {name}")
            for name in events['exits']:
                print(f"[Presence] EXIT  ← {name}")
 
            # ── CHANGE 4: Draw overlay — pass unknown_mgr for colour states ──
            with _enroll_lock:
                ea  = _enroll_active
                en  = _enroll_name
                efc = _enroll_frame_count
 
            frame = draw_overlay(
                frame, all_tracks, presence_mgr.present,
                ea, en, efc,
                unknown_mgr=unknown_mgr,
            )
 
            # ── FPS counter ───────────────────────────────────────────────────
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_time
                if elapsed > 0:
                    fps      = 30 / elapsed
                    fps_time = time.time()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    print(f"[App] FPS: {fps:.1f} | Present: {presence_mgr.present} "
                          f"| Clusters: {unknown_mgr.ready_count()} ready")
 
            cv2.imshow("Kritika — Person Recognition", frame)
 
            # ── Keypress handling ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord('q'):
                print("\n[App] Quit requested. Shutting down...")
                break
 
            elif key == ord('e'):
                with _enroll_lock:
                    already_pending = _enroll_pending or _enroll_active
                if already_pending:
                    print("[Enroll] Already enrolling — please wait.")
                elif not all_tracks:
                    print("[Enroll] No person detected in frame. Move closer to camera.")
                else:
                    with _enroll_lock:
                        _enroll_pending = True
                    t = threading.Thread(target=_terminal_name_prompt, daemon=True)
                    t.start()
 
            elif key == ord('u'):
                with _ukey_lock:
                    already = _ukey_pending
                if already:
                    print("[Unknown] Already in naming flow — please wait.")
                else:
                    rc = unknown_mgr.ready_count()
                    if rc == 0:
                        print("[Unknown] No clusters ready yet. "
                              "(Amber boxes appear when a cluster reaches "
                              f"{UNKNOWN_ACCUMULATE_TARGET} instances.)")
                    else:
                        with _ukey_lock:
                            _ukey_pending = True
                        t = threading.Thread(
                            target=_terminal_unknown_prompt,
                            args=(unknown_mgr, embedder, db, tracker),
                            daemon=True,
                        )
                        t.start()
 
    except KeyboardInterrupt:
        print("\n[App] Interrupted. Shutting down...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[App] Cleanup complete.")
 
 
if __name__ == "__main__":
    main()










# # app.py
# """
# Modular Person Recognition System — Main Pipeline
# Real-time person detection → tracking → face recognition → presence management
 
# Target: Intel i7 8th Gen CPU, ~22-28 FPS effective
 
# Keybindings (OpenCV window must be in focus):
#   Q — quit
#   E — enroll whoever is currently in frame as a new person (terminal prompt)
# """
 
# import cv2
# import time
# import sys
# import os
# import platform
# import threading
# import numpy as np
# from pathlib import Path
 
# # Ensure project root is on path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
# from config import *
# from recognition.face_detector import SCRFDDetector
# from recognition.embedder import ArcFaceEmbedder
# from recognition.faiss_db import FaceDatabase
# from recognition.identity import IdentityDecision
# from recognition.enrollment import EnrollmentPipeline
# from tracking.tracker import PersonTracker
# from presence.manager import PresenceManager
# from storage.unknown_handler import UnknownFaceHandler
 
 
# # ── Live Enrollment State ────────────────────────────────────────────────────
# # Shared between main loop and enrollment thread — all protected by _enroll_lock
# _enroll_lock        = threading.Lock()
# _enroll_pending     = False   # E was pressed, waiting for name input
# _enroll_name        = None    # Name typed in terminal (None until submitted)
# _enroll_crops       = []      # Accumulated face crops captured while enrolling
# _enroll_active      = False   # Actively capturing multi-frame crops
# _enroll_frame_count = 0       # How many crops captured so far
# _ENROLL_TARGET      = 15      # Number of frames to capture before committing
 
 
# def _terminal_name_prompt():
#     """Runs in a background thread — reads name from terminal without blocking camera."""
#     global _enroll_name, _enroll_active, _enroll_frame_count, _enroll_crops
#     print("\n" + "=" * 50)
#     print("  LIVE ENROLLMENT")
#     print("  Stay in frame. Type the person's name and press Enter.")
#     print("  Leave blank and press Enter to cancel.")
#     print("=" * 50)
#     name = input("  Name: ").strip()
#     with _enroll_lock:
#         if name:
#             _enroll_name        = name
#             _enroll_active      = True
#             _enroll_frame_count = 0
#             _enroll_crops       = []
#             print(f"  [Enroll] Capturing {_ENROLL_TARGET} frames for '{name}'...")
#         else:
#             print("  [Enroll] Cancelled.")
#             globals()['_enroll_pending'] = False
 
 
# def draw_overlay(frame, all_tracks, presence_list, enroll_active, enroll_name, enroll_frame_count):
#     """Draw bounding boxes, labels, presence panel, and enrollment HUD."""
 
#     for track in all_tracks:
#         x1, y1, x2, y2 = track['bbox']
#         identity = track.get('identity', 'Pending')
 
#         if enroll_active:
#             # Blue box during capture phase
#             colour = (255, 140, 0)
#             label  = f"Capturing... {enroll_frame_count}/{_ENROLL_TARGET}"
#         elif identity not in ('Unknown', 'Pending', None):
#             colour = (0, 200, 0)
#             label  = identity
#         else:
#             colour = (0, 0, 200)
#             label  = identity or 'Pending'
 
#         cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
#         cv2.putText(frame, label, (x1, y1 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
 
#     # ── Presence panel (top-right) ───────────────────────────────────────
#     panel_x = frame.shape[1] - 200
#     panel_h = 26 + 22 * max(len(presence_list), 1)
#     cv2.rectangle(frame, (panel_x, 0), (frame.shape[1], panel_h), (30, 30, 30), -1)
#     cv2.putText(frame, "PRESENT", (panel_x + 5, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
#     if presence_list:
#         for i, name in enumerate(presence_list):
#             cv2.putText(frame, f"  {name}", (panel_x + 5, 42 + i * 22),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
#     else:
#         cv2.putText(frame, "  (none)", (panel_x + 5, 42),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
 
#     # ── Enrollment HUD (bottom bar) ──────────────────────────────────────
#     h, w = frame.shape[:2]
#     if enroll_active and enroll_name:
#         bar_text = f"  ENROLLING: {enroll_name}  [{enroll_frame_count}/{_ENROLL_TARGET} frames]"
#         cv2.rectangle(frame, (0, h - 32), (w, h), (0, 100, 200), -1)
#         cv2.putText(frame, bar_text, (8, h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#     else:
#         # Always show the hint
#         cv2.putText(frame, "E: enroll  Q: quit", (8, h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
 
#     return frame
 
 
# def commit_enrollment(name, crops, embedder, db, tracker):
#     """Embed all captured crops and add to FAISS. Invalidate all track caches."""
#     if not crops:
#         print(f"  [Enroll] No valid crops captured for '{name}'. Aborted.")
#         return
 
#     print(f"  [Enroll] Embedding {len(crops)} crops for '{name}'...")
#     emb_matrix = embedder.embed_batch(crops)
#     db.add_person(name, emb_matrix)
 
#     # Also save images to database/known/<name>/ for persistence across restarts
#     person_dir = Path(KNOWN_DB_ROOT) / name
#     person_dir.mkdir(parents=True, exist_ok=True)
#     ts = time.strftime("%Y%m%d_%H%M%S")
#     for i, crop in enumerate(crops):
#         cv2.imwrite(str(person_dir / f"live_{ts}_{i:03d}.jpg"), crop)
 
#     # Force all cached tracks to re-run recognition immediately
#     for tid in list(tracker.track_cache.keys()):
#         tracker.track_cache[tid]['last_recognised'] = 0.0
 
#     print(f"  [Enroll] ✓ '{name}' enrolled with {len(crops)} embeddings.")
#     print(f"  [Enroll] ✓ Images saved to database/known/{name}/")
#     print(f"  [Enroll] ✓ FAISS now has {db.total_vectors} vectors for {len(db.enrolled_names)} persons\n")
 
 
# def main():
#     global _enroll_pending, _enroll_name, _enroll_active, _enroll_frame_count, _enroll_crops
 
#     import argparse
#     parser = argparse.ArgumentParser(description="Modular Person Recognition System")
#     parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index to use")
#     args = parser.parse_args()
#     camera_index = args.camera
 
#     print("=" * 60)
#     print("  MODULAR PERSON RECOGNITION SYSTEM")
#     print("  Target: Intel i7 8th Gen CPU · No GPU")
#     print("=" * 60)
#     print()
 
#     # ── Initialise modules ───────────────────────────────────────────────
#     print("[App] Loading models...")
 
#     print("[App]   → SCRFD face detector...")
#     face_detector = SCRFDDetector(SCRFD_MODEL_PATH)
 
#     print("[App]   → ArcFace embedder...")
#     embedder = ArcFaceEmbedder(ARCFACE_MODEL_PATH)
 
#     print("[App]   → FAISS database...")
#     db = FaceDatabase(index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
 
#     print("[App]   → Identity decision engine...")
#     identity_engine = IdentityDecision(threshold=RECOGNITION_THRESHOLD)
 
#     print("[App]   → YOLOv8n person tracker...")
#     tracker = PersonTracker(YOLO_MODEL_PATH)
 
#     print("[App]   → Presence manager...")
#     presence_mgr = PresenceManager()
 
#     print("[App]   → Unknown face handler...")
#     unknown_handler = UnknownFaceHandler()
 
#     enrollment = EnrollmentPipeline(face_detector, embedder, db)
 
#     print("[App] All modules loaded successfully.")
#     print()
 
#     # ── Optional: first-run enrollment ──────────────────────────────────
#     if db.index.ntotal == 0:
#         print("[App] No enrolled persons found. Running enrollment...")
#         results = enrollment.enroll_all(KNOWN_DB_ROOT)
#         if results:
#             print(f"[App] Enrollment complete: {results}")
#         else:
#             print("[App] No persons enrolled. Press E while someone is in frame to enroll them.")
#     else:
#         print(f"[App] Database loaded: {db.total_vectors} vectors for {len(db.enrolled_names)} persons")
#         print(f"[App] Enrolled: {db.enrolled_names}")
 
#     # ── Open camera ──────────────────────────────────────────────────────
#     if platform.system() == "Windows":
#         cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     else:
#         cap = cv2.VideoCapture(camera_index)
 
#     if not cap.isOpened():
#         print(f"[App] ERROR: Cannot open camera (index={camera_index})")
#         print("[App] Check camera connection and CAMERA_INDEX in config.py or pass --camera <index>")
#         return
 
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 
#     actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"[App] Camera opened: {actual_w}×{actual_h}")
#     print("[App] Controls: E = enroll person in frame | Q = quit")
#     print()
 
#     fps_time    = time.time()
#     frame_count = 0
 
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("[App] Camera read failed. Exiting.")
#                 break
 
#             frame_count += 1
 
#             # ── Track persons ────────────────────────────────────────────
#             tracks_to_recognise, all_tracks = tracker.update(frame)
 
#             # ── Live enrollment — crop capture phase ─────────────────────
#             with _enroll_lock:
#                 currently_active = _enroll_active
#                 current_name     = _enroll_name
 
#             if currently_active and all_tracks:
#                 # Grab the largest person crop in frame (most centred/closest)
#                 best_track = max(all_tracks, key=lambda t: (
#                     (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])
#                 ))
#                 x1, y1, x2, y2 = best_track['bbox']
#                 crop = frame[y1:y2, x1:x2]
#                 faces = face_detector.detect(crop)
 
#                 if faces:
#                     best_face = max(faces, key=lambda d: d['score'])
#                     if 'landmarks' in best_face:
#                         aligned = SCRFDDetector.align_face(crop, best_face['landmarks'])
#                     else:
#                         fx1, fy1, fx2, fy2 = best_face['bbox']
#                         ch, cw = crop.shape[:2]
#                         fx1, fy1 = max(0, fx1), max(0, fy1)
#                         fx2, fy2 = min(cw, fx2), min(ch, fy2)
#                         face_region = crop[fy1:fy2, fx1:fx2]
#                         if face_region.size > 0:
#                             aligned = cv2.resize(face_region, (112, 112))
#                         else:
#                             aligned = None
 
#                     if aligned is not None:
#                         with _enroll_lock:
#                             _enroll_crops.append(aligned)
#                             _enroll_frame_count += 1
#                             done  = _enroll_frame_count >= _ENROLL_TARGET
#                             crops_snapshot = list(_enroll_crops)
#                             name_snapshot  = _enroll_name
 
#                         if done:
#                             # Commit and reset
#                             commit_enrollment(name_snapshot, crops_snapshot,
#                                               embedder, db, tracker)
#                             with _enroll_lock:
#                                 _enroll_active      = False
#                                 _enroll_pending     = False
#                                 _enroll_name        = None
#                                 _enroll_crops       = []
#                                 _enroll_frame_count = 0
 
#             # ── Standard recognition on non-enroll frames ────────────────
#             current_identities = []
#             for track in tracks_to_recognise:
#                 crop     = track['crop']
#                 track_id = track['track_id']
 
#                 faces = face_detector.detect(crop)
#                 if not faces:
#                     tracker.update_track_identity(track_id, 'Unknown', 0.0)
#                     continue
 
#                 best_face = max(faces, key=lambda d: d['score'])
 
#                 if 'landmarks' in best_face:
#                     aligned = SCRFDDetector.align_face(crop, best_face['landmarks'])
#                 else:
#                     x1, y1, x2, y2 = best_face['bbox']
#                     ch, cw = crop.shape[:2]
#                     x1, y1 = max(0, x1), max(0, y1)
#                     x2, y2 = min(cw, x2), min(ch, y2)
#                     face_region = crop[y1:y2, x1:x2]
#                     if face_region.size == 0:
#                         tracker.update_track_identity(track_id, 'Unknown', 0.0)
#                         continue
#                     aligned = cv2.resize(face_region, (112, 112))
 
#                 query_emb = embedder.embed(aligned)
#                 name, score = db.majority_vote_search(query_emb, top_k=7)
#                 decision = identity_engine.decide(name, score)
 
#                 tracker.update_track_identity(track_id, decision['identity'], decision['score'])
 
#                 if not decision['is_known']:
#                     unknown_handler.save(aligned, track_id=track_id)
#                 else:
#                     current_identities.append(decision['identity'])
 
#             # ── Collect cached known identities ──────────────────────────
#             for track in all_tracks:
#                 identity = track.get('identity')
#                 if identity and identity not in ('Unknown', 'Pending', None):
#                     if identity not in current_identities:
#                         current_identities.append(identity)
 
#             # ── Presence ─────────────────────────────────────────────────
#             events = presence_mgr.update(current_identities)
#             for name in events['entries']:
#                 print(f"[Presence] ENTRY → {name}")
#             for name in events['exits']:
#                 print(f"[Presence] EXIT  ← {name}")
 
#             # ── Draw overlay ─────────────────────────────────────────────
#             with _enroll_lock:
#                 ea  = _enroll_active
#                 en  = _enroll_name
#                 efc = _enroll_frame_count
 
#             frame = draw_overlay(frame, all_tracks, presence_mgr.present, ea, en, efc)
 
#             # ── FPS counter ───────────────────────────────────────────────
#             if frame_count % 30 == 0:
#                 elapsed = time.time() - fps_time
#                 if elapsed > 0:
#                     fps      = 30 / elapsed
#                     fps_time = time.time()
#                     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#                     print(f"[App] FPS: {fps:.1f} | Present: {presence_mgr.present}")
 
#             cv2.imshow("Kritika — Person Recognition", frame)
 
#             # ── Keypress handling ─────────────────────────────────────────
#             key = cv2.waitKey(1) & 0xFF
 
#             if key == ord('q'):
#                 print("\n[App] Quit requested. Shutting down...")
#                 break
 
#             elif key == ord('e'):
#                 with _enroll_lock:
#                     already_pending = _enroll_pending or _enroll_active
#                 if already_pending:
#                     print("[Enroll] Already enrolling — please wait.")
#                 elif not all_tracks:
#                     print("[Enroll] No person detected in frame. Move closer to camera.")
#                 else:
#                     with _enroll_lock:
#                         _enroll_pending = True
#                     # Spawn terminal prompt in background so camera loop keeps running
#                     t = threading.Thread(target=_terminal_name_prompt, daemon=True)
#                     t.start()
 
#     except KeyboardInterrupt:
#         print("\n[App] Interrupted. Shutting down...")
#     finally:
#         try:
#             cap.release()
#         except Exception:
#             pass
#         cv2.destroyAllWindows()
#         print("[App] Cleanup complete.")
 
 
# if __name__ == "__main__":
#     main()
