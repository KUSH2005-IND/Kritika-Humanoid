# setup_models.py
"""
Model download and setup script.
Downloads SCRFD, ArcFace, and YOLOv8n models for the recognition pipeline.
"""

import os
import sys
import shutil
from pathlib import Path


def setup_directories():
    """Create all required directories."""
    dirs = [
        "models",
        "database/known",
        "database/embeddings",
        "unknown_faces",
    ]
    base = Path(os.path.dirname(os.path.abspath(__file__)))
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {d}/")


def download_insightface_models():
    """Download SCRFD and ArcFace models via InsightFace."""
    print("\n[Setup] Downloading InsightFace models (buffalo_l pack)...")
    print("        This includes SCRFD (face detection) and ArcFace (face embedding).")
    print("        First download may take a few minutes.\n")

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)
        print("  ✓ InsightFace models downloaded successfully.")

        # Find the downloaded model files
        home = Path.home()
        insightface_dir = home / ".insightface" / "models" / "buffalo_l"

        if insightface_dir.exists():
            models_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "models"

            # Copy SCRFD model
            scrfd_candidates = list(insightface_dir.glob("det_*.onnx"))
            if scrfd_candidates:
                src = scrfd_candidates[0]
                dst = models_dir / src.name
                config_name = src.name
                print(f"  ✓ Copying SCRFD model as: {config_name}")
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                else:
                    print(f"  ✓ SCRFD model already exists: {dst.name}")
            else:
                print("  ⚠ SCRFD model not found in InsightFace downloads.")
                print(f"    Check: {insightface_dir}")

            # Copy ArcFace model
            arcface_candidates = list(insightface_dir.glob("w600k_r50.onnx"))
            if not arcface_candidates:
                arcface_candidates = list(insightface_dir.glob("*arcface*.onnx"))
            if not arcface_candidates:
                # buffalo_l uses w600k_r50 as the recognition model
                arcface_candidates = list(insightface_dir.glob("*.onnx"))
                arcface_candidates = [f for f in arcface_candidates if "det" not in f.name
                                      and "gender" not in f.name and "age" not in f.name]

            if arcface_candidates:
                src = arcface_candidates[0]
                dst = models_dir / "arcface_r50.onnx"
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    print(f"  ✓ Copied ArcFace model: {src.name} → {dst.name}")
                else:
                    print(f"  ✓ ArcFace model already exists: {dst.name}")
            else:
                print("  ⚠ ArcFace model not found in InsightFace downloads.")
                print(f"    Check: {insightface_dir}")
                print(f"    Available files: {[f.name for f in insightface_dir.glob('*.onnx')]}")
        else:
            print(f"  ⚠ InsightFace model directory not found: {insightface_dir}")

    except ImportError:
        print("  ✗ InsightFace not installed. Run: pip install insightface")
        return False
    except Exception as e:
        print(f"  ✗ Error downloading InsightFace models: {e}")
        return False

    return True


def download_yolo_model():
    """Download YOLOv8n model via ultralytics."""
    print("\n[Setup] Downloading YOLOv8n model...")
    try:
        import torch
        _original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = safe_load
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("  ✓ YOLOv8n model ready.")
        return True
    except ImportError:
        print("  ✗ Ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"  ✗ Error downloading YOLOv8n: {e}")
        return False


def verify_setup():
    """Verify all models and directories are in place."""
    print("\n[Setup] Verifying setup...")
    base = Path(os.path.dirname(os.path.abspath(__file__)))

    checks = {
        "SCRFD model": base / "models" / "scrfd_500m_bnkps.onnx",
        "ArcFace model": base / "models" / "arcface_r50.onnx",
        "YOLOv8n model": Path("yolov8n.pt"),
        "Known DB directory": base / "database" / "known",
        "Embeddings directory": base / "database" / "embeddings",
        "Unknown faces directory": base / "unknown_faces",
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name}: {path} ({size_mb:.1f} MB)")
            else:
                print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("  PERSON RECOGNITION SYSTEM — MODEL SETUP")
    print("=" * 60)

    print("\n[Step 1] Creating directories...")
    setup_directories()

    print("\n[Step 2] Downloading InsightFace models...")
    insightface_ok = download_insightface_models()

    print("\n[Step 3] Downloading YOLOv8n...")
    yolo_ok = download_yolo_model()

    print("\n[Step 4] Verification...")
    all_ok = verify_setup()

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓ SETUP COMPLETE — All models and directories ready!")
        print()
        print("  Next steps:")
        print("  1. Add person images to database/known/<PersonName>/")
        print("     (10-20 JPG images per person, varied lighting/angles)")
        print("  2. Run: python app.py")
    else:
        print("  ⚠ SETUP INCOMPLETE — Some items are missing.")
        print("  Check the errors above and re-run this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
