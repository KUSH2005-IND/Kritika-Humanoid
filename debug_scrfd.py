# debug_scrfd.py
import os, sys, cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insightface.model_zoo import get_model
from config import SCRFD_MODEL_PATH

model = get_model(SCRFD_MODEL_PATH)
model.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.3, nms_thresh=0.4)

path = r"C:\Projects\Kritika-Face recognition\database\known\Kushagra Srivastava\kush_img12.jpeg"
img = cv2.imread(path)
h, w = img.shape[:2]
print(f"Original: {w}x{h}")

for max_side in [1600, 960, 640, 480, 320]:
    prescale = min(max_side / max(h, w), 1.0)
    rw, rh = int(w * prescale), int(h * prescale)
    prescaled = cv2.resize(img, (rw, rh))

    box_scale = min(640 / rw, 640 / rh)
    nw, nh = int(rw * box_scale), int(rh * box_scale)
    resized = cv2.resize(prescaled, (nw, nh))

    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized

    bboxes, _ = model.detect(padded)
    scores = [round(float(b[4]), 4) for b in bboxes[:5]] if len(bboxes) else []
    print(f"max_side={max_side:4d} → prescaled={rw}x{rh} → letterboxed={nw}x{nh}, dets={len(bboxes)}, top scores={scores}")