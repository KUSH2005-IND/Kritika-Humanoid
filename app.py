# app.py
"""
Modular Person Recognition System — Main Pipeline
Real-time person detection → tracking → face recognition → presence management

Target: Intel i7 8th Gen CPU, ~22-28 FPS effective
"""

import cv2
import time
import sys
import os

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
from storage.unknown_handler import UnknownFaceHandler


def draw_overlay(frame, all_tracks, presence_list):
    """Draw bounding boxes, labels, and presence panel."""
    for track in all_tracks:
        x1, y1, x2, y2 = track['bbox']
        identity = track.get('identity', 'Pending')
        colour = (0, 200, 0) if identity not in ('Unknown', 'Pending', None) else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{identity}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # Presence panel (top-right)
    panel_x = frame.shape[1] - 200
    panel_h = 26 + 22 * max(len(presence_list), 1)
    cv2.rectangle(frame, (panel_x, 0), (frame.shape[1], panel_h),
                  (30, 30, 30), -1)
    cv2.putText(frame, "PRESENT", (panel_x + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if presence_list:
        for i, name in enumerate(presence_list):
            cv2.putText(frame, f"  {name}", (panel_x + 5, 42 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    else:
        cv2.putText(frame, "  (none)", (panel_x + 5, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    return frame


def main():
    print("=" * 60)
    print("  MODULAR PERSON RECOGNITION SYSTEM")
    print("  Target: Intel i7 8th Gen CPU · No GPU")
    print("=" * 60)
    print()

    # ── Initialise modules ──────────────────────────────────────────
    print("[App] Loading models...")

    print("[App]   → SCRFD face detector...")
    face_detector = SCRFDDetector(SCRFD_MODEL_PATH)

    print("[App]   → ArcFace embedder...")
    embedder = ArcFaceEmbedder(ARCFACE_MODEL_PATH)

    print("[App]   → FAISS database...")
    db = FaceDatabase(
        index_path=FAISS_INDEX_PATH,
        meta_path=FAISS_META_PATH
    )

    print("[App]   → Identity decision engine...")
    identity_engine = IdentityDecision(threshold=RECOGNITION_THRESHOLD)

    print("[App]   → YOLOv8n person tracker...")
    tracker = PersonTracker(YOLO_MODEL_PATH)

    print("[App]   → Presence manager...")
    presence_mgr = PresenceManager()

    print("[App]   → Unknown face handler...")
    unknown_handler = UnknownFaceHandler()

    print("[App] All modules loaded successfully.")
    print()

    # ── Optional: first-run enrollment ──────────────────────────────
    if db.index.ntotal == 0:
        print("[App] No enrolled persons found. Running enrollment...")
        enrollment = EnrollmentPipeline(face_detector, embedder, db)
        results = enrollment.enroll_all(KNOWN_DB_ROOT)
        if results:
            print(f"[App] Enrollment complete: {results}")
        else:
            print("[App] No persons enrolled. Add images to database/known/<PersonName>/ and restart.")
    else:
        print(f"[App] Database loaded: {db.total_vectors} vectors for {len(db.enrolled_names)} persons")
        print(f"[App] Enrolled: {db.enrolled_names}")

    # ── Camera loop ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[App] ERROR: Cannot open camera (index={CAMERA_INDEX})")
        print("[App] Check camera connection and CAMERA_INDEX in config.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[App] Camera opened: {actual_w}×{actual_h}")
    print("[App] Starting recognition pipeline. Press 'q' to quit.")
    print()

    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[App] Camera read failed. Exiting.")
                break

            frame_count += 1

            # ── Track persons ────────────────────────────────────────────
            tracks_to_recognise, all_tracks = tracker.update(frame)

            # ── Recognise flagged tracks ─────────────────────────────────
            current_identities = []
            for track in tracks_to_recognise:
                crop = track['crop']
                track_id = track['track_id']

                # Detect face within person crop
                faces = face_detector.detect(crop)
                if not faces:
                    tracker.update_track_identity(track_id, 'Unknown', 0.0)
                    continue

                best_face = max(faces, key=lambda d: d['score'])

                # Align face for embedding
                if 'landmarks' in best_face:
                    aligned = SCRFDDetector.align_face(crop, best_face['landmarks'])
                else:
                    x1, y1, x2, y2 = best_face['bbox']
                    # Clamp to crop boundaries
                    ch, cw = crop.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(cw, x2), min(ch, y2)
                    face_region = crop[y1:y2, x1:x2]
                    if face_region.size == 0:
                        tracker.update_track_identity(track_id, 'Unknown', 0.0)
                        continue
                    aligned = cv2.resize(face_region, (112, 112))

                # Generate embedding and search
                query_emb = embedder.embed(aligned)
                name, score = db.majority_vote_search(query_emb, top_k=7)
                decision = identity_engine.decide(name, score)

                tracker.update_track_identity(track_id, decision['identity'], decision['score'])

                if not decision['is_known']:
                    unknown_handler.save(aligned)
                else:
                    current_identities.append(decision['identity'])

            # ── Also collect identities from cached tracks ────────────────
            for track in all_tracks:
                identity = track.get('identity')
                if identity and identity not in ('Unknown', 'Pending', None):
                    if identity not in current_identities:
                        current_identities.append(identity)

            # ── Update presence ──────────────────────────────────────────
            events = presence_mgr.update(current_identities)
            for name in events['entries']:
                print(f"[Presence] ENTRY → {name}")
            for name in events['exits']:
                print(f"[Presence] EXIT  ← {name}")

            # ── Draw and display ─────────────────────────────────────────
            frame = draw_overlay(frame, all_tracks, presence_mgr.present)

            # FPS counter
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_time
                if elapsed > 0:
                    fps = 30 / elapsed
                    fps_time = time.time()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    print(f"[App] FPS: {fps:.1f} | Present: {presence_mgr.present}")

            cv2.imshow("Person Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[App] Quit requested. Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[App] Interrupted. Shutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[App] Cleanup complete.")


if __name__ == "__main__":
    main()
