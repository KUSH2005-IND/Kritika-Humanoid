# recognition/enrollment.py
"""
Enrollment Pipeline — processes folders of person images to build the face database.
Recommended: 10–20 images per person, varied lighting and angles.
"""

import cv2
from pathlib import Path
import numpy as np

from recognition.face_detector import SCRFDDetector
from recognition.embedder import ArcFaceEmbedder
from recognition.faiss_db import FaceDatabase


class EnrollmentPipeline:
    """
    Enrolls a person from a folder of images.
    Recommended: 10–20 images per person, varied lighting and angles.
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, detector: SCRFDDetector, embedder: ArcFaceEmbedder, db: FaceDatabase):
        self.detector = detector
        self.embedder = embedder
        self.db = db

    def enroll_from_folder(self, name: str, folder: str) -> int:
        """
        Processes all images in folder, extracts aligned faces, generates embeddings.
        Returns number of successfully enrolled images.
        """
        folder = Path(folder)
        image_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_paths.extend(folder.glob(f"*{ext}"))
            image_paths.extend(folder.glob(f"*{ext.upper()}"))

        # Deduplicate (case-insensitive glob may return same files)
        seen = set()
        unique_paths = []
        for p in image_paths:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)
        image_paths = unique_paths

        if not image_paths:
            print(f"[Enrollment] No images found in {folder}")
            return 0

        embeddings = []
        skipped = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[Enrollment] Could not read {img_path.name}, skipping")
                skipped += 1
                continue

            detections = self.detector.detect(img)
            if not detections:
                print(f"[Enrollment] No face detected in {img_path.name}, skipping")
                skipped += 1
                continue

            # Use highest-confidence face if multiple detected
            best = max(detections, key=lambda d: d['score'])

            if 'landmarks' in best:
                aligned = SCRFDDetector.align_face(img, best['landmarks'])
            else:
                x1, y1, x2, y2 = best['bbox']
                # Clamp to image boundaries
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    skipped += 1
                    continue
                aligned = cv2.resize(face_crop, (112, 112))

            emb = self.embedder.embed(aligned)
            embeddings.append(emb)

        if not embeddings:
            print(f"[Enrollment] No valid faces found for {name}")
            return 0

        emb_matrix = np.vstack(embeddings)
        self.db.add_person(name, emb_matrix)
        print(f"[Enrollment] Enrolled '{name}' with {len(embeddings)} embeddings ({skipped} skipped)")
        return len(embeddings)

    def enroll_all(self, database_root: str) -> dict:
        """Batch enroll from database/known/ directory structure."""
        db_root = Path(database_root)
        if not db_root.exists():
            print(f"[Enrollment] Database root not found: {db_root}")
            return {}

        results = {}
        for person_dir in sorted(db_root.iterdir()):
            if person_dir.is_dir():
                count = self.enroll_from_folder(person_dir.name, str(person_dir))
                results[person_dir.name] = count
        return results
