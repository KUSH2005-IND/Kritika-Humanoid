# test_sanity.py
import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from recognition.face_detector import SCRFDDetector
from recognition.embedder import ArcFaceEmbedder
from recognition.faiss_db import FaceDatabase
from recognition.identity import IdentityDecision
from tracking.tracker import PersonTracker
from presence.manager import PresenceManager
from storage.unknown_handler import UnknownFaceHandler

def test_pipeline():
    print("=" * 60)
    print("  RUNNING SANITY CHECK ON PIPELINE MODULES")
    print("=" * 60)
    
    # 1. Test Configurations
    print("\n[Test 1] Checking configuration constants...")
    print(f"  YOLO model path: {YOLO_MODEL_PATH}")
    print(f"  SCRFD model path: {SCRFD_MODEL_PATH}")
    print(f"  ArcFace model path: {ARCFACE_MODEL_PATH}")
    print(f"  FAISS index path: {FAISS_INDEX_PATH}")
    print(f"  FAISS metadata path: {FAISS_META_PATH}")
    print(f"  Presence DB path: {DB_PATH}")
    
    # 2. Test Face Detector
    print("\n[Test 2] Initialising SCRFD Face Detector...")
    try:
        detector = SCRFDDetector(SCRFD_MODEL_PATH)
        print("  ✓ SCRFD Detector initialised successfully.")
    except Exception as e:
        print(f"  ✗ SCRFD Detector initialisation FAILED: {e}")
        return False
        
    # Find a sample image
    sample_img_path = None
    known_dir = os.path.join(BASE_DIR, "database", "known")
    if os.path.exists(known_dir):
        for root, dirs, files in os.walk(known_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_img_path = os.path.join(root, file)
                    break
            if sample_img_path:
                break
                
    if not sample_img_path:
        print("  ✗ No sample image found in database/known to test detection.")
        return False
        
    print(f"  Testing detector on: {os.path.basename(sample_img_path)}")
    img = cv2.imread(sample_img_path)
    if img is None:
        print(f"  ✗ Failed to read image: {sample_img_path}")
        return False
        
    try:
        detections = detector.detect(img)
        print(f"  ✓ SCRFD Detector finished. Found {len(detections)} faces.")
        for i, det in enumerate(detections):
            print(f"    - Face {i+1}: BBox {det['bbox']}, Score {det['score']:.4f}")
    except Exception as e:
        print(f"  ✗ SCRFD Detector execution FAILED: {e}")
        return False
        
    # 3. Test Embedder
    print("\n[Test 3] Initialising ArcFace Embedder...")
    try:
        embedder = ArcFaceEmbedder(ARCFACE_MODEL_PATH)
        print("  ✓ ArcFace Embedder initialised successfully.")
    except Exception as e:
        print(f"  ✗ ArcFace Embedder initialisation FAILED: {e}")
        return False
        
    if detections:
        best_face = max(detections, key=lambda d: d['score'])
        if 'landmarks' in best_face:
            aligned = SCRFDDetector.align_face(img, best_face['landmarks'])
        else:
            x1, y1, x2, y2 = best_face['bbox']
            aligned = cv2.resize(img[max(0, y1):y2, max(0, x1):x2], (112, 112))
            
        try:
            emb = embedder.embed(aligned)
            print(f"  ✓ ArcFace Embedder generated embedding of shape: {emb.shape}")
        except Exception as e:
            print(f"  ✗ ArcFace Embedder execution FAILED: {e}")
            return False
            
    # 4. Test FAISS DB
    print("\n[Test 4] Initialising FAISS database...")
    try:
        db = FaceDatabase(index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
        print(f"  ✓ FAISS database initialised. Total vectors: {db.total_vectors}")
        print(f"  ✓ Enrolled individuals: {db.enrolled_names}")
    except Exception as e:
        print(f"  ✗ FAISS database initialisation FAILED: {e}")
        return False
        
    if detections and db.total_vectors > 0:
        try:
            name, score = db.majority_vote_search(emb, top_k=7)
            print(f"  ✓ Search query returned: '{name}' with vote/mean score {score:.4f}")
        except Exception as e:
            print(f"  ✗ FAISS database search FAILED: {e}")
            return False
            
    # 5. Test Identity Decision
    print("\n[Test 5] Initialising Identity Decision...")
    try:
        engine = IdentityDecision(threshold=RECOGNITION_THRESHOLD)
        print("  ✓ Identity Decision initialised successfully.")
        if detections and db.total_vectors > 0:
            decision = engine.decide(name, score)
            print(f"    - Decision: {decision}")
    except Exception as e:
        print(f"  ✗ Identity Decision initialisation FAILED: {e}")
        return False
        
    # 6. Test Tracker
    print("\n[Test 6] Initialising YOLOv8n Person Tracker...")
    try:
        tracker = PersonTracker(YOLO_MODEL_PATH)
        print("  ✓ Person Tracker initialised successfully.")
    except Exception as e:
        print(f"  ✗ Person Tracker initialisation FAILED: {e}")
        return False
        
    try:
        # Run tracker on a dummy frame (zeros)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks_to_rec, all_tracks = tracker.update(dummy_frame)
        print(f"  ✓ Tracker executed on dummy frame. Tracks to recognize: {len(tracks_to_rec)}, All tracks: {len(all_tracks)}")
    except Exception as e:
        print(f"  ✗ Tracker execution FAILED: {e}")
        return False
        
    # 7. Test Presence Manager
    print("\n[Test 7] Initialising Presence Manager...")
    try:
        presence_mgr = PresenceManager()
        print("  ✓ Presence Manager initialised successfully.")
        # Test presence update
        events = presence_mgr.update(["TestPerson"])
        print(f"    - Event check 1 (TestPerson entry): {events}")
        events = presence_mgr.update([])
        print(f"    - Event check 2 (no one detected): {events}")
    except Exception as e:
        print(f"  ✗ Presence Manager FAILED: {e}")
        return False
        
    # 8. Test Unknown Face Handler
    print("\n[Test 8] Initialising Unknown Face Handler...")
    try:
        unknown_handler = UnknownFaceHandler()
        print("  ✓ Unknown Face Handler initialised successfully.")
        if detections:
            save_path = unknown_handler.save(aligned, track_id=999)
            print(f"    - Saved unknown face to: {save_path}")
            # Try saving again immediately to test cooldown (should return None)
            save_path_cooldown = unknown_handler.save(aligned, track_id=999)
            print(f"    - Immediate retry save (cooldown test): {save_path_cooldown} (Expected: None)")
    except Exception as e:
        print(f"  ✗ Unknown Face Handler FAILED: {e}")
        return False
        
    print("\n" + "=" * 60)
    print("  ✓ ALL PIPELINE MODULE MODULES ARE FUNCTIONAL!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
