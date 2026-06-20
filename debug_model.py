# debug_ort.py
import cv2
import numpy as np
import onnxruntime as ort
from config import SCRFD_MODEL_PATH

sess = ort.InferenceSession(SCRFD_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
print(f"Input name: {input_name}")

path = r"C:\Projects\Kritika-Face recognition\database\known\Kushagra Srivastava\kush_img12.jpeg"
img = cv2.imread(path)
h, w = img.shape[:2]
print(f"Original: {w}x{h}")

def preprocess(img, size=640):
    # Letterbox
    scale = size / max(img.shape[:2])
    nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    # Normalize
    blob = padded.astype(np.float32) / 255.0
    blob = (blob - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    blob = blob[:, :, ::-1]  # BGR to RGB
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC to NCHW
    return blob.astype(np.float32), scale

for size in [640, 480, 320]:
    blob, scale = preprocess(img, size)
    print(f"\nsize={size}, blob shape={blob.shape}")
    outputs = sess.run(None, {input_name: blob})
    print(f"  num outputs: {len(outputs)}")
    for i, o in enumerate(outputs):
        print(f"  output[{i}] shape={o.shape}, max={o.max():.4f}, min={o.min():.4f}")