# test_search.py
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from recognition.face_detector import SCRFDDetector
from recognition.embedder import ArcFaceEmbedder
from recognition.faiss_db import FaceDatabase

def test_search():
    detector = SCRFDDetector(SCRFD_MODEL_PATH)
    embedder = ArcFaceEmbedder(ARCFACE_MODEL_PATH)
    db = FaceDatabase(index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
    
    img_path = os.path.join(BASE_DIR, "database", "known", "Dr Adesh Kr Pandey", "adesh1.jpg")
    img = cv2.imread(img_path)
    
    detections = detector.detect(img)
    if not detections:
        print("No face detected!")
        return
        
    best_face = max(detections, key=lambda d: d['score'])
    aligned = SCRFDDetector.align_face(img, best_face['landmarks'])
    emb = embedder.embed(aligned)
    
    print("\n--- Raw FAISS search (top 10) ---")
    candidates = db.search(emb, top_k=10)
    for i, (name, score) in enumerate(candidates):
        print(f"Rank {i+1}: {name} -> Score: {score:.6f}")
        
    print("\n--- Majority Vote Search (top_k=7) ---")
    name, score = db.majority_vote_search(emb, top_k=7)
    print(f"Result: {name} with score {score:.6f}")

if __name__ == "__main__":
    test_search()
