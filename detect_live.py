"""
Metin2 Live Object Detection — YOLOv8

Captures the game window and runs YOLOv8 inference in real-time,
displaying bounding boxes overlaid on the game feed in a separate window.

Prerequisites:
    pip install ultralytics opencv-python numpy

Usage:
    python detect_live.py
    python detect_live.py --model path/to/best.pt
    python detect_live.py --conf 0.4
    python detect_live.py --no-preview    # run headless (no OpenCV window)

Controls (in the preview window):
    Q / ESC  — quit
    +/-      — raise/lower confidence threshold
"""

import argparse
import ctypes
import ctypes.wintypes as wintypes
import time
import sys
import numpy as np
import cv2
from ultralytics import YOLO

# ========================
#  CONFIG
# ========================

DEFAULT_MODEL = "runs/detect/metin2_detect/boulder_v1/weights/best.pt"
DEFAULT_CONF = 0.35
WINDOW_KEYWORD = "nothyr"  # partial match for the game window title

# Class colors (BGR) — boulder is bright green for visibility
CLASS_COLORS = {
    "Boss":    (0, 0, 255),     # red
    "Buff":    (255, 255, 0),   # cyan
    "Enemy":   (0, 165, 255),   # orange
    "Me":      (255, 0, 255),   # magenta
    "Statue":  (128, 128, 128), # gray
    "boulder": (0, 255, 0),     # GREEN — primary target
}
DEFAULT_COLOR = (255, 255, 255)


# ========================
#  WIN32 WINDOW CAPTURE
# ========================

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Win32 constants
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
PW_RENDERFULLCONTENT = 0x00000002

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
GetWindowTextW = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
IsWindowVisible = user32.IsWindowVisible


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


def find_window(keyword):
    """Find windows whose title contains the keyword."""
    results = []
    def callback(hwnd, _):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buf, length + 1)
                if keyword.lower() in buf.value.lower():
                    results.append((hwnd, buf.value))
        return True
    EnumWindows(EnumWindowsProc(callback), 0)
    return results


def capture_window(hwnd):
    """Capture a window's contents using PrintWindow (works even if occluded)."""
    # Get window dimensions
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top

    if w <= 0 or h <= 0:
        return None

    # Create compatible DC and bitmap
    hwnd_dc = user32.GetDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    gdi32.SelectObject(mem_dc, bitmap)

    # PrintWindow captures even if the window is behind others
    result = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)

    if not result:
        # Fallback to BitBlt
        gdi32.BitBlt(mem_dc, 0, 0, w, h, hwnd_dc, 0, 0, SRCCOPY)

    # Extract pixel data
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h  # top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), DIB_RGB_COLORS)

    # Cleanup
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)

    # Convert to numpy (BGRA -> BGR)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return img[:, :, :3].copy()  # drop alpha


# ========================
#  DETECTION + DRAWING
# ========================

def draw_detections(frame, results, conf_threshold):
    """Draw bounding boxes and labels on the frame."""
    if not results or len(results) == 0:
        return frame, []

    detections = []
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return frame, detections

    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        thickness = 3 if cls_name == "boulder" else 2

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
        })

    return frame, detections


def draw_hud(frame, fps, conf_threshold, det_count):
    """Draw FPS and info overlay."""
    h, w = frame.shape[:2]
    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info = f"FPS: {fps:.1f}  |  Conf: {conf_threshold:.2f} (+/- to adjust)  |  Detections: {det_count}  |  Q to quit"
    cv2.putText(frame, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


# ========================
#  MAIN LOOP
# ========================

def main():
    parser = argparse.ArgumentParser(description="Metin2 Live Object Detection")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLOv8 .pt file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--no-preview", action="store_true", help="Run without OpenCV window")
    parser.add_argument("--window", default=WINDOW_KEYWORD, help="Game window title keyword")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Model loaded. Classes: {model.names}")

    # Find game window
    matches = find_window(args.window)
    if not matches:
        print(f"ERROR: No window found matching '{args.window}'")
        print("Make sure the game is running.")
        sys.exit(1)

    if len(matches) > 1:
        print("Multiple windows found:")
        for i, (_, title) in enumerate(matches):
            print(f"  [{i}] {title}")
        idx = int(input("Pick: ").strip() or "0")
        hwnd = matches[idx][0]
        title = matches[idx][1]
    else:
        hwnd = matches[0][0]
        title = matches[0][1]

    print(f"Capturing: \"{title}\"")
    print(f"Confidence threshold: {args.conf}")
    print(f"Preview: {'OFF' if args.no_preview else 'ON'}")
    print(f"Controls: Q/ESC=quit, +/-=adjust confidence\n")

    conf_threshold = args.conf
    fps = 0
    frame_count = 0
    fps_start = time.time()
    last_good_frame = None

    cv2.namedWindow("Metin2 Detection", cv2.WINDOW_NORMAL)

    while True:
        loop_start = time.time()

        # Capture
        frame = capture_window(hwnd)
        if frame is None or frame.mean() < 5:
            # Dark or failed frame — reuse last good frame
            if last_good_frame is not None:
                frame = last_good_frame
            else:
                time.sleep(0.05)
                continue
        else:
            last_good_frame = frame

        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
            device="cpu",
        )

        # Draw
        frame, detections = draw_detections(frame, results, conf_threshold)

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Print boulder detections to console
        boulders = [d for d in detections if d["class"] == "boulder"]
        if boulders:
            coords = ", ".join(
                f"({d['bbox'][0]},{d['bbox'][1]})-({d['bbox'][2]},{d['bbox'][3]}) {d['confidence']:.0%}"
                for d in boulders
            )
            print(f"[boulder] {len(boulders)} detected: {coords}")

        if not args.no_preview:
            frame = draw_hud(frame, fps, conf_threshold, len(detections))

            # Resize for display if the frame is very large
            h, w = frame.shape[:2]
            max_display_w = 1280
            if w > max_display_w:
                scale = max_display_w / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            cv2.imshow("Metin2 Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == ord("+") or key == ord("="):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"  Confidence threshold: {conf_threshold:.2f}")
            elif key == ord("-") or key == ord("_"):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"  Confidence threshold: {conf_threshold:.2f}")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()