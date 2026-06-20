# camera_scan.py
import cv2
import platform

def scan_cameras():
    print("Scanning for available cameras...")
    found_any = False
    for i in range(5):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
            
        if cap.isOpened():
            print(f"  ✓ Camera found at index {i}")
            cap.release()
            found_any = True
        else:
            # Try without DSHOW just in case
            if platform.system() == "Windows":
                cap_no_dshow = cv2.VideoCapture(i)
                if cap_no_dshow.isOpened():
                    print(f"  ✓ Camera found at index {i} (without DirectShow)")
                    cap_no_dshow.release()
                    found_any = True
    if not found_any:
        print("  ✗ No working cameras detected!")

if __name__ == "__main__":
    scan_cameras()
