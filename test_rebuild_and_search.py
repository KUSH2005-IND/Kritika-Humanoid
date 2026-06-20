# # test_rebuild_and_search.py
# import os
# import sys
# import cv2
# import numpy as np

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from config import *
# from recognition.face_detector import SCRFDDetector
# from recognition.embedder import ArcFaceEmbedder
# from recognition.faiss_db import FaceDatabase
# from recognition.enrollment import EnrollmentPipeline

# def test_rebuild():
#     detector = SCRFDDetector(SCRFD_MODEL_PATH)
#     embedder = ArcFaceEmbedder(ARCFACE_MODEL_PATH)
    
#     # Create an in-memory/temporary database (we won't save it to the default files)
#     print("\n--- Initialising fresh database in-memory ---")
#     db = FaceDatabase(dim=512, index_path=None, meta_path=None)
    
#     pipeline = EnrollmentPipeline(detector, embedder, db)
    
#     print("\n--- Running enrollment pipeline on known folders ---")
#     results = pipeline.enroll_all(KNOWN_DB_ROOT)
#     print(f"Enrollment results: {results}")
#     print(f"Total vectors in fresh index: {db.total_vectors}")
    
#     # Now run search on adesh1.jpg
#     img_path = os.path.join(BASE_DIR, "database", "known", "Kushagra Srivastava", "kush_img20.jpeg")
#     img = cv2.imread(img_path)
    
#     detections = detector.detect(img)
#     if not detections:
#         print("No face detected in test image!")
#         return
        
#     best_face = max(detections, key=lambda d: d['score'])
#     aligned = SCRFDDetector.align_face(img, best_face['landmarks'])
#     emb = embedder.embed(aligned)
    
#     print("\n--- Fresh FAISS search (top 10) ---")
#     candidates = db.search(emb, top_k=10)
#     for i, (name, score) in enumerate(candidates):
#         print(f"Rank {i+1}: {name} -> Score: {score:.6f}")
        
#     print("\n--- Fresh Majority Vote Search (top_k=7) ---")
#     name, score = db.majority_vote_search(emb, top_k=7)
#     print(f"Result: {name} with score {score:.6f}")

# if __name__ == "__main__":
#     test_rebuild()


# debug_scrfd.py
import os, sys, cv2
import numpy as np
from PIL import Image, ExifTags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SCRFD_MODEL_PATH
from recognition.face_detector import SCRFDDetector

def read_exif_aware(path):
    pil_img = Image.open(path)
    exif = pil_img._getexif()
    if exif:
        orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
        if orientation_key and orientation_key in exif:
            orientation = exif[orientation_key]
            rotation_map = {3: 180, 6: 270, 8: 90}
            if orientation in rotation_map:
                pil_img = pil_img.rotate(rotation_map[orientation], expand=True)
    return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

detector = SCRFDDetector(SCRFD_MODEL_PATH)

failing = [
    "kush_img7.jpeg", "kush_img10.jpeg", "kush_img12.jpeg",
    "kush_img15.jpeg", "kush_img16.jpeg", "kush_img18.jpeg",
    "kush_img19.jpeg", "kush_img22.jpeg"
]

base = r"C:\Projects\Kritika-Face recognition\database\known\Kushagra Srivastava"

for fname in failing:
    path = os.path.join(base, fname)
    
    # Method 1: raw cv2
    img_cv = cv2.imread(path)
    det_cv = detector.detect(img_cv) if img_cv is not None else []
    
    # Method 2: EXIF-aware
    img_exif = read_exif_aware(path)
    det_exif = detector.detect(img_exif) if img_exif is not None else []
    
    print(f"{fname}:")
    print(f"  cv2.imread   → shape={img_cv.shape if img_cv is not None else None}, detections={len(det_cv)}, scores={[round(d['score'],3) for d in det_cv]}")
    print(f"  exif_aware   → shape={img_exif.shape if img_exif is not None else None}, detections={len(det_exif)}, scores={[round(d['score'],3) for d in det_exif]}")