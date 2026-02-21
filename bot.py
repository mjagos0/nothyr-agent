"""
Metin2 Bot — Serialized main loop with YOLOv8 detection

Architecture:
    A single main loop captures one frame every --loop-interval seconds (default: 3)
    and runs checks in priority order, stopping at the first match:
        1. CAPTCHA  — if captcha popup detected, solve it
        2. RESPAWN  — if respawn button detected, click it + Ctrl+H
        3. STUCK    — if N consecutive low-diff frames, run unstuck routine
        3b. BUFFS   — if spell icon inactive, re-apply buffs (Ctrl+H, F2, F3, F4)
        4. METIN    — if boulder detected, Shift+Right-click it
        5. ENEMY    — if enemy detected, left-click it
        6. IDLE     — click random position to roam

Usage:
    python bot.py                                   # nothing enabled
    python bot.py --captcha --respawn --stuck --metin --enemy # all checks active
    python bot.py --metin --enemy --space --key1    # detection + key helpers
    python bot.py --stuck --metin                   # only stuck + metin

Flags:
    --loop-interval N    Seconds between main-loop ticks (default: 3)
    --captcha            Enable captcha detection+solving
    --respawn            Enable respawn detection (click respawn + Ctrl+H)
    --stuck              Enable stuck detection
    --metin              Enable boulder/metin detection (Shift+Right-click)
    --enemy              Enable enemy detection (left-click)
    --save-detections N  Save annotated YOLO frame every N ticks (0=off)
    --show-detections    Show live OpenCV window with YOLO bounding boxes
    --space              Hold spacebar while focused
    --key1 [N]           Enable '1' key loop, mean interval N sec (default: 3)
    --keyz [N]           Enable 'Z' key loop, mean interval N sec (default: 1)
    --save-every N       Screenshot every N seconds (0=off)
    --save-key KEY       Hotkey to manually take screenshot (default: F9)
    --stuck-thresh F     Frame-diff threshold for stuck (default: 3.0)
    --stuck-consec N     Consecutive low-diff frames before unstuck (default: 5)
    --stuck-key KEY      Hotkey to manually run stuck check (default: F11)
    --captcha-key KEY    Hotkey to run captcha solver (default: F10)
    --captcha-data DIR   Captcha template root dir (default: capcha-data)

Screen Capture:
    Uses GDI PrintWindow API which renders the window content directly,
    regardless of whether the window is partially off-screen or occluded.
    Falls back to BitBlt if PrintWindow returns a black frame.

Hotkeys:
    --pause-key   (F6)   Pause / resume bot
    --save-key    (F9)   Manual screenshot
    --captcha-key (F10)  Run captcha solver
    --stuck-key   (F11)  Manual stuck check
"""

import ctypes, ctypes.wintypes as wt
import time, random, threading, sys, argparse, os, glob, re
import numpy as np, cv2
from datetime import datetime
from ultralytics import YOLO

# ── Constants / Win32 ──────────────────────────────────────────────────────

KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP    = 0x0002
INPUT_KEYBOARD     = 1
INPUT_MOUSE        = 0
MOUSEEVENTF_LDOWN  = 0x0002
MOUSEEVENTF_LUP    = 0x0004
MOUSEEVENTF_RDOWN  = 0x0008
MOUSEEVENTF_RUP    = 0x0010
PW_RENDERFULL      = 2
BI_RGB             = 0

SCAN = {"1": 0x02, "z": 0x2C, "space": 0x39, "lshift": 0x2A, "h": 0x23, "lctrl": 0x1D, "esc": 0x01,
        "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f7": 0x41}
VK_FKEYS = {f"F{i}": 0x6F + i for i in range(1, 13)}

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk",ctypes.c_ushort),("wScan",ctypes.c_ushort),
                ("dwFlags",ctypes.c_ulong),("time",ctypes.c_ulong),
                ("dwExtraInfo",ctypes.POINTER(ctypes.c_ulong))]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx",ctypes.c_long),("dy",ctypes.c_long),
                ("mouseData",ctypes.c_ulong),("dwFlags",ctypes.c_ulong),
                ("time",ctypes.c_ulong),("dwExtraInfo",ctypes.POINTER(ctypes.c_ulong))]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki",KEYBDINPUT),("mi",MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type",ctypes.c_ulong),("union",INPUT_UNION)]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize",ctypes.c_uint32),("biWidth",ctypes.c_long),
                ("biHeight",ctypes.c_long),("biPlanes",ctypes.c_uint16),
                ("biBitCount",ctypes.c_uint16),("biCompression",ctypes.c_uint32),
                ("biSizeImage",ctypes.c_uint32),("biXPelsPerMeter",ctypes.c_long),
                ("biYPelsPerMeter",ctypes.c_long),("biClrUsed",ctypes.c_uint32),
                ("biClrImportant",ctypes.c_uint32)]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader",BITMAPINFOHEADER),("bmiColors",ctypes.c_uint32*3)]

u32 = ctypes.windll.user32
gdi = ctypes.windll.gdi32
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

# ── Global state ───────────────────────────────────────────────────────────

target_hwnd = None
running = True
paused = False
space_held = False
lock = threading.Lock()
boulder_count = 0
_latest_vis_frame = None       # shared frame for the detection viewer thread
_vis_frame_lock = threading.Lock()
_shutdown_event = threading.Event()  # used for Ctrl+C responsive main thread

ts = lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3]
focused = lambda: u32.GetForegroundWindow() == target_hwnd

# ── Helpers ────────────────────────────────────────────────────────────────

def find_window(keyword):
    results = []
    def cb(hwnd, _):
        if u32.IsWindowVisible(hwnd):
            n = u32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                u32.GetWindowTextW(hwnd, buf, n + 1)
                if keyword.lower() in buf.value.lower():
                    results.append((hwnd, buf.value))
        return True
    u32.EnumWindows(EnumWindowsProc(cb), 0)
    return results

def _send_key(scan, up=False):
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def send_scan(scan, up=False):
    with lock: _send_key(scan, up)

def press_key(scan):
    with lock:
        _send_key(scan); time.sleep(0.05); _send_key(scan, up=True)

def right_click():
    with lock:
        for flags in (MOUSEEVENTF_RDOWN, MOUSEEVENTF_RUP):
            inp = INPUT(type=INPUT_MOUSE)
            inp.union.mi.dwFlags = flags
            u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
            time.sleep(0.05)

def move_to(hwnd, x, y):
    pt = wt.POINT(x, y)
    u32.ClientToScreen(hwnd, ctypes.byref(pt))
    u32.SetCursorPos(pt.x, pt.y)

def human_move_to(hwnd, x, y, duration=None):
    """Move cursor from current position to (x, y) in client coords with human-like motion."""
    # Get current cursor position
    cur = wt.POINT()
    u32.GetCursorPos(ctypes.byref(cur))
    # Convert target to screen coords
    tgt = wt.POINT(x, y)
    u32.ClientToScreen(hwnd, ctypes.byref(tgt))
    sx, sy = cur.x, cur.y
    ex, ey = tgt.x, tgt.y
    dist = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
    if duration is None:
        duration = random.uniform(0.15, 0.35) + dist / random.uniform(1800, 2800)
    steps = max(int(dist / random.uniform(3, 6)), 12)
    # Generate slightly curved path using bezier with a random control point
    ctrl_x = (sx + ex) / 2 + random.uniform(-40, 40)
    ctrl_y = (sy + ey) / 2 + random.uniform(-40, 40)
    for i in range(steps + 1):
        t = i / steps
        # Ease-in-out (slow start and end)
        t = t * t * (3 - 2 * t)
        # Quadratic bezier
        bx = (1-t)**2 * sx + 2*(1-t)*t * ctrl_x + t**2 * ex
        by = (1-t)**2 * sy + 2*(1-t)*t * ctrl_y + t**2 * ey
        # Add tiny jitter
        jx = int(bx + random.gauss(0, 0.8))
        jy = int(by + random.gauss(0, 0.8))
        u32.SetCursorPos(jx, jy)
        time.sleep(max(0, duration / steps + random.uniform(-0.002, 0.002)))

def left_click():
    with lock:
        for flags in (MOUSEEVENTF_LDOWN, MOUSEEVENTF_LUP):
            inp = INPUT(type=INPUT_MOUSE)
            inp.union.mi.dwFlags = flags
            u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
            time.sleep(0.05)

def capture_window(hwnd):
    """Capture the client area of hwnd using GDI PrintWindow.
    Works regardless of window position (even partially off-screen)."""
    return _capture_window_gdi(hwnd)


def _capture_window_gdi(hwnd):
    """Legacy GDI-based capture (PrintWindow + BitBlt fallback)."""
    wr = wt.RECT(); u32.GetWindowRect(hwnd, ctypes.byref(wr))
    fw, fh = wr.right - wr.left, wr.bottom - wr.top
    if fw <= 0 or fh <= 0: return None

    cr = wt.RECT(); u32.GetClientRect(hwnd, ctypes.byref(cr))
    cw, ch = cr.right - cr.left, cr.bottom - cr.top
    origin = wt.POINT(0, 0); u32.ClientToScreen(hwnd, ctypes.byref(origin))

    def _grab(src_hdc, src_x, src_y, w, h):
        mdc = gdi.CreateCompatibleDC(src_hdc)
        bmp = gdi.CreateCompatibleBitmap(src_hdc, w, h)
        gdi.SelectObject(mdc, bmp)
        return mdc, bmp

    def _read(mdc, bmp, w, h):
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth, bmi.bmiHeader.biHeight = w, -h
        bmi.bmiHeader.biPlanes, bmi.bmiHeader.biBitCount = 1, 32
        bmi.bmiHeader.biCompression = BI_RGB
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        gdi.DeleteObject(bmp); gdi.DeleteDC(mdc)
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

    # Try PrintWindow first (works even if partially occluded)
    hdc = u32.GetDC(hwnd)
    mdc, bmp = _grab(hdc, 0, 0, fw, fh)
    u32.PrintWindow(hwnd, mdc, PW_RENDERFULL)
    img = _read(mdc, bmp, fw, fh)
    u32.ReleaseDC(hwnd, hdc)

    bx, by = origin.x - wr.left, origin.y - wr.top
    client = img[by:by+ch, bx:bx+cw, :3].copy()

    if client.mean() >= 5:
        return client

    # Fallback: BitBlt from screen DC (works with DirectX, needs visible window)
    sdc = u32.GetDC(0)
    mdc, bmp = _grab(sdc, 0, 0, cw, ch)
    gdi.BitBlt(mdc, 0, 0, cw, ch, sdc, origin.x, origin.y, 0x00CC0020)  # SRCCOPY
    img = _read(mdc, bmp, cw, ch)
    u32.ReleaseDC(0, sdc)

    return img[:, :, :3].copy()

def save_screenshot(tag="manual"):
    frame = capture_window(target_hwnd)
    if frame is None:
        print(f"  [{ts()}] [SCREENSHOT] capture failed"); return
    os.makedirs("screenshots", exist_ok=True)
    path = f"screenshots/{tag}_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(path, frame)
    print(f"  [{ts()}] [SCREENSHOT] {path}")

def frame_diff_pct(prev_frame, frame):
    """Return percentage of pixels that changed between two frames."""
    if prev_frame is None or frame is None: return 100.0
    if prev_frame.shape != frame.shape: return 100.0
    diff_per_pixel = np.max(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16)), axis=2)
    changed = np.count_nonzero(diff_per_pixel > 10)
    total = diff_per_pixel.shape[0] * diff_per_pixel.shape[1]
    return changed / total * 100.0


def save_stuck_visualization(prev_frame, frame, diff, thresh):
    """Save a side-by-side visualization when stuck is detected."""
    try:
        os.makedirs("stuck-logs", exist_ok=True)
        h, w = frame.shape[:2]
        diff_raw = np.abs(frame.astype(float) - prev_frame.astype(float))
        diff_gray = np.mean(diff_raw, axis=2).astype(np.uint8)
        heatmap = cv2.applyColorMap(cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        pad = 4
        canvas = np.full((h + 60, w * 3 + pad * 2, 3), 30, dtype=np.uint8)
        canvas[30:30+h, 0:w] = prev_frame
        canvas[30:30+h, w+pad:2*w+pad] = frame
        canvas[30:30+h, 2*w+2*pad:3*w+2*pad] = heatmap
        font, scale, color = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255)
        cv2.putText(canvas, "Previous", (10, 22), font, scale, color, 1)
        cv2.putText(canvas, "Current", (w + pad + 10, 22), font, scale, color, 1)
        cv2.putText(canvas, f"Diff Heatmap ({diff:.2f}% changed)", (2*w + 2*pad + 10, 22), font, scale, (0, 180, 255), 1)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(canvas, f"STUCK DETECTED  |  {stamp}  |  {diff:.2f}% changed  thresh={thresh}%",
                    (10, h + 50), font, 0.5, (0, 0, 255), 1)
        path = f"stuck-logs/stuck_{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(path, canvas)
        print(f"  [{ts()}] [STUCK] visualization saved: {path}")
    except Exception as e:
        print(f"  [{ts()}] [STUCK] failed to save visualization: {e}")

# ── Unstuck routine ────────────────────────────────────────────────────────

def unstuck_routine(hwnd):
    """Click 30 random positions in the center 50% of the client area over ~3 seconds."""
    cr = wt.RECT(); u32.GetClientRect(hwnd, ctypes.byref(cr))
    cw, ch = cr.right - cr.left, cr.bottom - cr.top
    if cw <= 0 or ch <= 0:
        return
    # Center 50% region: from 25% to 75% of width and height
    x_min, x_max = int(cw * 0.25), int(cw * 0.75)
    y_min, y_max = int(ch * 0.25), int(ch * 0.75)

    print(f"  [{ts()}] [UNSTUCK] clicking 30 random positions in center region ({x_min},{y_min})-({x_max},{y_max})")
    for i in range(30):
        if not running:
            break
        rx = random.randint(x_min, x_max)
        ry = random.randint(y_min, y_max)
        move_to(hwnd, rx, ry)
        time.sleep(random.uniform(0.03, 0.06))
        left_click()
        time.sleep(random.uniform(0.05, 0.10))  # ~3s total for 30 clicks

    print(f"  [{ts()}] [UNSTUCK] done clicking")


def run_unstuck(thresh, prev_frame):
    """Pause bot, run unstuck clicks, repeat until no longer stuck (single-frame recheck).
    Returns the latest frame."""
    global paused, space_held

    was_paused = paused
    paused = True
    press_key(SCAN["space"])
    print(f"  [{ts()}] [UNSTUCK] bot paused, starting unstuck routine...")

    frame = prev_frame
    attempt = 0
    while running:
        attempt += 1
        print(f"  [{ts()}] [UNSTUCK] attempt #{attempt}")
        unstuck_routine(target_hwnd)
        time.sleep(0.5)
        new_frame = capture_window(target_hwnd)
        if new_frame is None or new_frame.mean() < 5:
            continue
        pct = frame_diff_pct(frame, new_frame)
        frame = new_frame
        if pct >= thresh:
            print(f"  [{ts()}] [UNSTUCK] no longer stuck after attempt #{attempt} ({pct:.2f}%)")
            break
        print(f"  [{ts()}] [UNSTUCK] still stuck ({pct:.2f}%), retrying...")

    paused = was_paused
    if not paused:
        print(f"  [{ts()}] [UNSTUCK] bot resumed")
    return frame

# ── Captcha solver ─────────────────────────────────────────────────────────

# Captcha popup dimensions as ratios (the popup itself is a fixed game UI element,
# Captcha popup is a fixed-size UI element (does NOT scale with resolution).
# Original measured dimensions on 1920x1009: popup at (815,381)-(1103,627)
# That's 288px wide × 246px tall. The header template sits at the top.
CAPTCHA_POPUP_W = 288   # fixed pixel width of the captcha popup
CAPTCHA_POPUP_H = 246   # fixed pixel height of the captcha popup

def find_captcha_region(frame, header_tmpl, confidence=0.95):
    """Search the full frame for the captcha header template at multiple scales.
    Uses TM_CCOEFF_NORMED which gives low scores for non-matching regions.
    Always logs the best match score. Returns (x1, y1, x2, y2) or None."""
    if frame is None or header_tmpl is None:
        return None
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(header_tmpl, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]
    fh, fw = frame_gray.shape[:2]

    best_val, best_loc, best_scale = -1, None, 1.0

    for scale in np.arange(0.5, 1.55, 0.05):
        nw, nh = int(tw * scale), int(th * scale)
        if nw > fw or nh > fh or nw < 10 or nh < 10:
            continue
        resized = cv2.resize(tmpl_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val, best_loc, best_scale = max_val, max_loc, scale
        except cv2.error:
            continue

    print(f"  [{ts()}] [CAPTCHA] header match: {best_val:.3f} (s={best_scale:.2f} loc={best_loc}) need>{confidence}")

    if best_val < confidence or best_loc is None:
        return None

    x1 = best_loc[0]
    y1 = best_loc[1]
    popup_w = int(CAPTCHA_POPUP_W * best_scale)
    popup_h = int(CAPTCHA_POPUP_H * best_scale)
    x2 = min(x1 + popup_w, fw)
    y2 = min(y1 + popup_h, fh)
    return (x1, y1, x2, y2)

# Store the loaded header template globally so run_captcha can use it
_captcha_header_tmpl = None

def detect_captcha_popup(header_tmpl):
    """Check if captcha popup is visible by searching the full frame for the header."""
    frame = capture_window(target_hwnd)
    return find_captcha_region(frame, header_tmpl) is not None

# ── Respawn detection ──────────────────────────────────────────────────────

# Respawn button measured on 1536x780:  (61,67)-(239,92)  →  178x25 px
# As ratios of screen:  x: 3.97%–15.56%   y: 8.59%–11.79%
# Center: ~9.77% x, ~10.19% y
RESPAWN_BTN_X_RATIO = (61 / 1536, 239 / 1536)   # left, right
RESPAWN_BTN_Y_RATIO = (67 / 780,   92 / 780)     # top, bottom

_respawn_btn_tmpl = None   # loaded from respawn/respawn-button.png
_buff_icon_tmpl = None     # loaded from spells/non-active-spell.png

def find_respawn_button(frame, tmpl, confidence=0.90):
    """Search the full frame for the respawn button template at multiple scales.
    Returns (center_x, center_y) in frame coords or None."""
    if frame is None or tmpl is None:
        return None
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]
    fh, fw = frame_gray.shape[:2]

    best_val, best_loc, best_scale = -1, None, 1.0

    for scale in np.arange(0.5, 1.55, 0.05):
        nw, nh = int(tw * scale), int(th * scale)
        if nw > fw or nh > fh or nw < 10 or nh < 10:
            continue
        resized = cv2.resize(tmpl_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val, best_loc, best_scale = max_val, max_loc, scale
        except cv2.error:
            continue

    print(f"  [{ts()}] [RESPAWN] button match: {best_val:.3f} (s={best_scale:.2f} loc={best_loc}) need>{confidence}")

    if best_val < confidence or best_loc is None:
        return None

    # Return center of the matched region
    nw, nh = int(tw * best_scale), int(th * best_scale)
    cx = best_loc[0] + nw // 2
    cy = best_loc[1] + nh // 2
    return (cx, cy)

def do_respawn(frame):
    """Click the respawn button, then press Ctrl+H to go to town."""
    btn = find_respawn_button(frame, _respawn_btn_tmpl)
    if btn is None:
        return False

    cx, cy = btn
    # Drag the cursor onto the button — the game ignores clicks when
    # the cursor spawns directly on the target without moving across it
    offset_x = cx - random.randint(30, 60)
    offset_y = cy - random.randint(10, 20)
    print(f"  [{ts()}] [RESPAWN] dragging onto button at ({cx},{cy})")
    move_to(target_hwnd, offset_x, offset_y)
    time.sleep(random.uniform(0.1, 0.2))
    human_move_to(target_hwnd, cx, cy)
    time.sleep(random.uniform(0.08, 0.15))
    left_click()
    time.sleep(random.uniform(0.3, 0.5))

    # Ctrl+H to go to town
    print(f"  [{ts()}] [RESPAWN] pressing Ctrl+H")
    send_scan(SCAN["lctrl"])
    time.sleep(0.05)
    press_key(SCAN["h"])
    time.sleep(0.05)
    send_scan(SCAN["lctrl"], up=True)

    print(f"  [{ts()}] [RESPAWN] done")
    return True

# ── Buff detection ────────────────────────────────────────────────────────

# Icon region measured on 1536x780:  (919,745)-(951,766)  →  32x21 px
# Stored as ratios so it scales with any resolution
BUFF_ICON_X_RATIO = (919 / 1536, 951 / 1536)   # left, right
BUFF_ICON_Y_RATIO = (745 / 780,  766 / 780)     # top, bottom

def find_buff_icon(frame, tmpl, confidence=0.80):
    """Check a small region of the frame for the non-active-spell icon.
    Uses multi-scale template matching like respawn/captcha detection.
    Returns True if the icon is found (buffs are NOT active), False otherwise."""
    if frame is None or tmpl is None:
        return False
    fh, fw = frame.shape[:2]

    # Extract the search region (with some padding)
    x1 = max(0, int(BUFF_ICON_X_RATIO[0] * fw) - 15)
    x2 = min(fw, int(BUFF_ICON_X_RATIO[1] * fw) + 15)
    y1 = max(0, int(BUFF_ICON_Y_RATIO[0] * fh) - 15)
    y2 = min(fh, int(BUFF_ICON_Y_RATIO[1] * fh) + 15)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]
    rh, rw = roi_gray.shape[:2]

    best_val = -1

    for scale in np.arange(0.5, 1.55, 0.05):
        nw, nh = int(tw * scale), int(th * scale)
        if nw > rw or nh > rh or nw < 5 or nh < 5:
            continue
        resized = cv2.resize(tmpl_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(roi_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
        except cv2.error:
            continue

    print(f"  [{ts()}] [BUFFS] icon match: {best_val:.3f} need>{confidence}")
    return best_val >= confidence


def do_buffs():
    """Apply buffs: Ctrl+H, then F2/F3/F4 with 2s gaps, then Ctrl+H again."""
    print(f"  [{ts()}] [BUFFS] non-active spell detected — applying buffs")

    # Ctrl+H
    send_scan(SCAN["lctrl"])
    time.sleep(0.05)
    press_key(SCAN["h"])
    time.sleep(0.05)
    send_scan(SCAN["lctrl"], up=True)

    # F2, F3, F4 with 2s between each
    press_key(SCAN["f2"])
    print(f"  [{ts()}] [BUFFS] pressed F2")
    time.sleep(2.0)

    press_key(SCAN["f3"])
    print(f"  [{ts()}] [BUFFS] pressed F3")
    time.sleep(2.0)

    press_key(SCAN["f4"])
    print(f"  [{ts()}] [BUFFS] pressed F4")

    # Ctrl+H again
    time.sleep(0.05)
    send_scan(SCAN["lctrl"])
    time.sleep(0.05)
    press_key(SCAN["h"])
    time.sleep(0.05)
    send_scan(SCAN["lctrl"], up=True)

    print(f"  [{ts()}] [BUFFS] done")

# Captcha categories — these are the exact words that appear in the captcha text
# and also the folder names in capcha-data/
ITEM_KEYWORDS = ["boty", "dyky", "helmu", "nahrdelnik", "naramek", "nausnice", "opasek", "stit", "vejir"]

_ocr_reader = None

def init_ocr():
    """Pre-initialize easyocr reader (downloads models if needed)."""
    global _ocr_reader
    import easyocr
    print(f"  [{ts()}] [OCR] initializing easyocr...")
    _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print(f"  [{ts()}] [OCR] ready")

def ocr_captcha_class(frame_crop):
    """OCR the captcha crop, find item keyword in text."""
    if _ocr_reader is None:
        print(f"  [{ts()}] [CAPTCHA] OCR not initialized"); return None

    results = _ocr_reader.readtext(frame_crop, detail=0)
    text = " ".join(results).lower()
    print(f"  [{ts()}] [CAPTCHA] OCR text: {text!r}")

    for keyword in ITEM_KEYWORDS:
        if keyword in text:
            print(f"  [{ts()}] [CAPTCHA] matched: {keyword!r}")
            return keyword

    print(f"  [{ts()}] [CAPTCHA] no known item in OCR text"); return None

def load_templates(class_dir):
    templates = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG"):
        for path in glob.glob(os.path.join(class_dir, ext)):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                templates.append((os.path.splitext(os.path.basename(path))[0], img))
    return templates

def prepare_template(tmpl):
    if tmpl.shape[2] == 4:
        bgr = tmpl[:, :, :3]
        _, mask = cv2.threshold(tmpl[:, :, 3], 128, 255, cv2.THRESH_BINARY)
        return bgr, mask
    return tmpl, None

def multiscale_match(gray, tmpl_bgr, tmpl_mask, scales, early_exit=0.95):
    tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]
    sh, sw = gray.shape[:2]
    best_score, best_loc, best_scale, best_size = -1, (0,0), 1.0, (tw,th)
    for s in scales:
        nw, nh = int(tw*s), int(th*s)
        if nw < 10 or nh < 10 or nw > sw or nh > sh: continue
        rt = cv2.resize(tmpl_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        rm = cv2.resize(tmpl_mask, (nw, nh), interpolation=cv2.INTER_NEAREST) if tmpl_mask is not None else None
        try:
            res = cv2.matchTemplate(gray, rt, cv2.TM_CCOEFF_NORMED, mask=rm) if rm is not None else cv2.matchTemplate(gray, rt, cv2.TM_CCOEFF_NORMED)
        except cv2.error: continue
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best_score:
            best_score, best_loc, best_scale, best_size = mv, ml, s, (nw, nh)
            if best_score >= early_exit:
                return best_score, best_loc, best_scale, best_size
    return best_score, best_loc, best_scale, best_size

def solve_captcha(crop, captcha_dir, class_name):
    """Run captcha solver on cropped frame. Returns (item_result, target_result).
    Target position is calculated arithmetically (fixed offset inside the fixed-size popup).
    Item matching uses a narrow scale range since the popup doesn't scale."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Popup is fixed-size so items are always ~same size; narrow range is sufficient
    scales = np.arange(0.7, 1.35, 0.05)

    class_dir_path = os.path.join(captcha_dir, class_name)
    if not os.path.isdir(class_dir_path):
        print(f"  [{ts()}] [CAPTCHA] class dir not found: {class_dir_path}"); return None, None
    templates = load_templates(class_dir_path)
    if not templates:
        print(f"  [{ts()}] [CAPTCHA] no templates in {class_dir_path}"); return None, None

    best = None
    for name, tmpl_img in templates:
        bgr, mask = prepare_template(tmpl_img)
        score, loc, scale, (mw, mh) = multiscale_match(gray, bgr, mask, scales)
        if best is None or score > best[0]:
            best = (score, loc, scale, mw, mh, name)
            if score >= 0.95:
                break   # confident match, skip remaining templates

    # Target square position: fixed offset (242, 198) from popup top-left.
    # Since crop IS the popup, the target center is simply (242, 198) in crop coords.
    CAPTCHA_TARGET_OFFSET = (242, 198)
    tgt_cx, tgt_cy = CAPTCHA_TARGET_OFFSET
    # Build a small bounding box for visualization (~20px square)
    tgt = {"center": (tgt_cx, tgt_cy),
           "tl": (tgt_cx - 10, tgt_cy - 10),
           "br": (tgt_cx + 10, tgt_cy + 10),
           "score": 1.0}

    if best is None: return None, None
    score, loc, scale, mw, mh, name = best
    item = {"name": name, "score": score, "center": (loc[0]+mw//2, loc[1]+mh//2),
            "tl": loc, "br": (loc[0]+mw, loc[1]+mh)}
    return item, tgt

def run_captcha(captcha_dir):
    """Capture captcha region, OCR class name, solve, save debug image."""
    frame = capture_window(target_hwnd)
    if frame is None:
        print(f"  [{ts()}] [CAPTCHA] capture failed"); return

    # Find captcha popup by searching for header template
    region = find_captcha_region(frame, _captcha_header_tmpl)
    if region is None:
        print(f"  [{ts()}] [CAPTCHA] popup not found in frame"); return

    x1, y1, x2, y2 = region
    h, w = frame.shape[:2]
    if x2 > w or y2 > h:
        print(f"  [{ts()}] [CAPTCHA] region out of bounds ({w}x{h})"); return
    crop = frame[y1:y2, x1:x2]
    print(f"  [{ts()}] [CAPTCHA] popup found at ({x1},{y1})-({x2},{y2})")

    # OCR to detect class
    class_name = ocr_captcha_class(crop)
    if class_name is None:
        class_name = "nahrdelnik"  # fallback
        print(f"  [{ts()}] [CAPTCHA] OCR failed, falling back to '{class_name}'")

    print(f"  [{ts()}] [CAPTCHA] solving for '{class_name}'...")
    item, tgt = solve_captcha(crop, captcha_dir, class_name)

    # Draw debug visualization on the crop
    vis = crop.copy()
    if item:
        cv2.rectangle(vis, item["tl"], item["br"], (0,255,0), 3)
        cv2.circle(vis, item["center"], 8, (0,0,255), -1)
        cv2.putText(vis, f"{item['name']} ({item['score']:.3f})", (item["tl"][0], item["tl"][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        print(f"  [{ts()}] [CAPTCHA] item={item['name']} score={item['score']:.4f} center={item['center']}")
    else:
        print(f"  [{ts()}] [CAPTCHA] no item found")

    if tgt:
        cv2.rectangle(vis, tgt["tl"], tgt["br"], (255,0,255), 3)
        cv2.circle(vis, tgt["center"], 8, (255,0,0), -1)
        cv2.putText(vis, f"TARGET ({tgt['score']:.3f})", (tgt["tl"][0], tgt["tl"][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        print(f"  [{ts()}] [CAPTCHA] target center={tgt['center']} score={tgt['score']:.4f}")

    os.makedirs("capcha-solution", exist_ok=True)
    path = f"capcha-solution/solve_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(path, vis)
    print(f"  [{ts()}] [CAPTCHA] saved: {path}")

    # Click item, then human-move to target and click again
    if item and tgt:
        ox, oy = x1, y1
        src_x, src_y = item["center"][0] + ox, item["center"][1] + oy
        dst_x, dst_y = tgt["center"][0] + ox, tgt["center"][1] + oy
        print(f"  [{ts()}] [CAPTCHA] clicking ({src_x},{src_y}) → ({dst_x},{dst_y})")

        u32.SetForegroundWindow(target_hwnd)
        time.sleep(random.uniform(0.08, 0.15))

        # Human-move to the found item, then click it
        human_move_to(target_hwnd, src_x, src_y)
        time.sleep(random.uniform(0.05, 0.12))
        left_click()
        time.sleep(random.uniform(0.15, 0.30))

        # Human-move to the target, then click it
        human_move_to(target_hwnd, dst_x, dst_y)
        time.sleep(random.uniform(0.05, 0.12))
        left_click()

        print(f"  [{ts()}] [CAPTCHA] done")

# ── Hotkey listener (generic) ──────────────────────────────────────────────

def hotkey_listener(vk_code, key_name, callback):
    was = False
    while running:
        state = u32.GetAsyncKeyState(vk_code) & 0x8000
        if state and not was:
            callback(key_name)
        was = bool(state)
        time.sleep(0.05)

# ── Threads ────────────────────────────────────────────────────────────────

def pause_callback(key_name):
    global paused, space_held
    paused = not paused
    print(f"  [{ts()}] [{key_name}] {'PAUSED' if paused else 'RESUMED'}")
    if paused and space_held:
        send_scan(SCAN["space"], up=True); space_held = False

def key_loop(name, scan, mean):
    while running:
        d = max(0.5, random.gauss(mean, 0.3))
        time.sleep(d)
        if not running or paused: continue
        if focused():
            press_key(scan)
            print(f"  [{ts()}] [{name}] pressed (next ~{d:.1f}s)")

def space_watcher():
    global space_held
    while running:
        if paused:
            if space_held: send_scan(SCAN["space"], up=True); space_held = False
            time.sleep(0.25); continue
        if focused() and not space_held:
            send_scan(SCAN["space"]); space_held = True
        elif not focused() and space_held:
            send_scan(SCAN["space"], up=True); space_held = False
        time.sleep(0.25)

def screenshot_timer(interval):
    while running:
        time.sleep(interval)
        if not running or paused: continue
        save_screenshot(tag="auto")

def stuck_log_timer(interval):
    """Passively log frame diff % every N seconds for threshold tuning. No unstuck triggered.
    Uses percentage of pixels that changed beyond a noise floor (per-pixel threshold of 10/255)."""
    prev = None
    while running:
        time.sleep(interval)
        if not running: continue
        frame = capture_window(target_hwnd)
        if frame is None or frame.mean() < 5:
            continue
        if prev is not None and frame.shape == prev.shape:
            # Per-pixel max channel difference
            diff_per_pixel = np.max(np.abs(frame.astype(np.int16) - prev.astype(np.int16)), axis=2)
            # Count pixels that changed more than noise floor (10 out of 255)
            changed = np.count_nonzero(diff_per_pixel > 10)
            total = diff_per_pixel.shape[0] * diff_per_pixel.shape[1]
            pct = changed / total * 100.0
            print(f"  [{ts()}] [STUCK-LOG] {pct:.2f}% pixels changed  ({changed}/{total})")
            if pct > 30.0:
                try:
                    os.makedirs("stuck-logs", exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"stuck-logs/log_prev_{stamp}.png", prev)
                    cv2.imwrite(f"stuck-logs/log_curr_{stamp}.png", frame)
                    print(f"  [{ts()}] [STUCK-LOG] dumped prev+curr to stuck-logs/ ({pct:.2f}%)")
                except Exception as e:
                    print(f"  [{ts()}] [STUCK-LOG] dump failed: {e}")
        else:
            print(f"  [{ts()}] [STUCK-LOG] (first frame captured, {frame.shape[1]}x{frame.shape[0]})")
        prev = frame

# ── Detection Viewer (dedicated GUI thread) ───────────────────────────────

def detection_viewer():
    """Dedicated thread that repaints the detection window at ~30 fps.
    Reads from the shared _latest_vis_frame; the main_loop writes to it."""
    global _latest_vis_frame
    WIN_NAME = "Metin2 Bot — Detections"
    shown = False
    while running:
        with _vis_frame_lock:
            frame = _latest_vis_frame
        if frame is not None:
            cv2.imshow(WIN_NAME, frame)
            shown = True
        key = cv2.waitKey(33) & 0xFF  # ~30 fps, keeps window responsive
        if key == 27:  # ESC closes the viewer
            break
    if shown:
        cv2.destroyWindow(WIN_NAME)

# ── Inventory Biolog Routine ──────────────────────────────────────────────

# All coordinates measured on 1536x780, stored as ratios
_BIOLOG_BTN     = (870 / 1536, 320 / 780)      # button to click after F7
# Grid: 5 cols × 3 rows. Centers of corner cells:
#   top-left cell center:     (1220, 250)
#   bottom-right cell center: (1350, 310)
# Col spacing: (1350-1220)/4 = 32.5px,  Row spacing: (310-250)/2 = 30px
_BIOLOG_GRID_ORIGIN = (1220 / 1536, 250 / 780)  # center of cell [0,0]
_BIOLOG_COL_STEP    = 32.5 / 1536                # x step between column centers
_BIOLOG_ROW_STEP    = 30.0 / 780                 # y step between row centers
_BIOLOG_COLS    = 5
_BIOLOG_ROWS    = 3
_BIOLOG_CLOSE   = (790 / 1536, 280 / 780)       # close button

def run_biolog(hwnd):
    """Press F7, click biolog button, right-click 15 grid squares (with confirm between each), click close, ESC×2."""
    cr = wt.RECT(); u32.GetClientRect(hwnd, ctypes.byref(cr))
    cw, ch = cr.right - cr.left, cr.bottom - cr.top
    if cw <= 0 or ch <= 0:
        return

    def _fast_click(rx, ry, right=False):
        x, y = int(rx * cw), int(ry * ch)
        move_to(hwnd, x, y)
        time.sleep(0.02)
        if right:
            right_click()
        else:
            left_click()
        time.sleep(0.1)

    def _drag_click(rx, ry):
        """Drag cursor onto target before clicking (needed for game UI buttons)."""
        x, y = int(rx * cw), int(ry * ch)
        # Offset in relative terms (~2-4% of screen width left, ~1-2.5% up)
        ox = int(x - cw * random.uniform(0.02, 0.04))
        oy = int(y - ch * random.uniform(0.01, 0.025))
        move_to(hwnd, ox, oy)
        time.sleep(random.uniform(0.1, 0.2))
        human_move_to(hwnd, x, y)
        time.sleep(random.uniform(0.08, 0.15))
        left_click()
        time.sleep(random.uniform(0.08, 0.15))

    # 1. Press F7 to open biolog window
    print(f"  [{ts()}] [BIOLOG] pressing F7")
    press_key(SCAN["f7"])
    time.sleep(random.uniform(0.3, 0.5))

    # 2. Click the biolog button
    print(f"  [{ts()}] [BIOLOG] clicking biolog button")
    _fast_click(*_BIOLOG_BTN)
    time.sleep(random.uniform(0.3, 0.5))

    # 3. Right-click 15 grid squares (5 cols × 3 rows), confirm after each
    x0, y0 = _BIOLOG_GRID_ORIGIN

    print(f"  [{ts()}] [BIOLOG] right-clicking {_BIOLOG_COLS}x{_BIOLOG_ROWS} grid")
    for row in range(_BIOLOG_ROWS):
        for col in range(_BIOLOG_COLS):
            cx = x0 + col * _BIOLOG_COL_STEP
            cy = y0 + row * _BIOLOG_ROW_STEP
            _fast_click(cx, cy, right=True)
            press_key(SCAN["esc"])

    # 4. Close shopping window
    press_key(SCAN["esc"])
    time.sleep(0.1)

    # 4. Click close button (drag onto it — game needs mouse movement)
    print(f"  [{ts()}] [BIOLOG] clicking close")
    _drag_click(*_BIOLOG_CLOSE)
    time.sleep(0.1)

    # 5. Close biolog window
    press_key(SCAN["esc"])
    time.sleep(0.1)

    # 6. Close inventory
    press_key(SCAN["esc"])

    print(f"  [{ts()}] [BIOLOG] done")

# ── Serialized Main Loop ──────────────────────────────────────────────────

def main_loop(model, conf, captcha_dir, header_tmpl, interval,
              enable_captcha, enable_respawn, biolog_interval,
              enable_stuck, enable_buffs, enable_metin, enable_enemy,
              stuck_thresh, stuck_consec,
              save_detections_every, show_detections):
    """Single-threaded loop: capture once, then captcha → respawn → biolog → stuck → buffs → metin → enemy.
    If nothing detected, click a random position (idle roam)."""
    global boulder_count, _latest_vis_frame

    prev_frame = None
    low_count = 0
    enemy_count = 0
    idle_count = 0
    tick = 0
    last_biolog_time = 0.0  # epoch; runs on first eligible tick

    # Detection colors: BGR
    COLOR_BOULDER = (0, 255, 0)     # green
    COLOR_ENEMY   = (0, 165, 255)   # orange
    COLOR_OTHER   = (200, 200, 200) # gray

    while running:
        time.sleep(interval)
        if not running or paused:
            continue
        if not focused():
            continue

        tick += 1

        # ── 1. Capture a single frame ──────────────────────────────────
        frame = capture_window(target_hwnd)
        if frame is None or frame.mean() < 5:
            continue

        # ── 2. CAPTCHA check (highest priority) ───────────────────────
        if enable_captcha and header_tmpl is not None:
            region = find_captcha_region(frame, header_tmpl)
            if region is not None:
                print(f"  [{ts()}] [LOOP] captcha detected — solving")
                run_captcha(captcha_dir)
                prev_frame = None   # reset stuck tracking after captcha
                low_count = 0
                continue            # restart loop

        # ── 2b. RESPAWN check ─────────────────────────────────────────
        if enable_respawn and _respawn_btn_tmpl is not None:
            if do_respawn(frame):
                prev_frame = None   # reset stuck tracking after respawn
                low_count = 0
                continue            # restart loop immediately

        # ── 2c. BIOLOG check (time-gated) ────────────────────────────
        if biolog_interval is not None:
            now = time.time()
            if now - last_biolog_time >= biolog_interval:
                print(f"  [{ts()}] [BIOLOG] {biolog_interval:.0f}s elapsed — running biolog routine")
                run_biolog(target_hwnd)
                last_biolog_time = now
                prev_frame = None
                low_count = 0
                continue            # restart loop after biolog

        # ── 3. STUCK check ─────────────────────────────────────────────
        if enable_stuck:
            pct = frame_diff_pct(prev_frame, frame)
            if prev_frame is not None:
                if pct < stuck_thresh:
                    low_count += 1
                    print(f"  [{ts()}] [STUCK] {pct:.2f}% < {stuck_thresh}% ({low_count}/{stuck_consec})")
                    if low_count >= stuck_consec:
                        print(f"  [{ts()}] [STUCK] {stuck_consec} consecutive low-diff frames — stuck!")
                        save_stuck_visualization(prev_frame, frame, pct, stuck_thresh)
                        prev_frame = run_unstuck(stuck_thresh, frame)
                        low_count = 0
                        continue    # restart loop after unstuck
                else:
                    if low_count > 0:
                        print(f"  [{ts()}] [STUCK] reset ({pct:.2f}% >= {stuck_thresh}%)")
                    low_count = 0
            prev_frame = frame

        # ── 3b. BUFFS check ────────────────────────────────────────────
        if enable_buffs and _buff_icon_tmpl is not None:
            if find_buff_icon(frame, _buff_icon_tmpl):
                do_buffs()
                prev_frame = None
                low_count = 0
                continue            # restart loop after buffs

        # ── 4. YOLO detection (metin + enemy) ─────────────────────────
        if (enable_metin or enable_enemy) and model is not None:
            results = model.predict(source=frame, conf=conf, verbose=False, device="cpu")
            best_boulder = None
            best_enemy = None

            # Build annotated frame for visualization
            vis_frame = frame.copy() if (save_detections_every > 0 or show_detections) else None

            if results and results[0].boxes is not None:
                r = results[0]
                for box in r.boxes:
                    cls_name = r.names[int(box.cls[0])]
                    c = float(box.conf[0])
                    coords = tuple(map(int, box.xyxy[0]))

                    if cls_name.lower() == "boulder":
                        if best_boulder is None or c > best_boulder[0]:
                            best_boulder = (c, *coords)
                    elif cls_name == "Enemy":
                        if best_enemy is None or c > best_enemy[0]:
                            best_enemy = (c, *coords)

                    # Draw all detections on visualization frame
                    if vis_frame is not None:
                        x1b, y1b, x2b, y2b = coords
                        if cls_name.lower() == "boulder":
                            color = COLOR_BOULDER
                        elif cls_name == "Enemy":
                            color = COLOR_ENEMY
                        else:
                            color = COLOR_OTHER
                        cv2.rectangle(vis_frame, (x1b, y1b), (x2b, y2b), color, 2)
                        label = f"{cls_name} {c:.0%}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(vis_frame, (x1b, y1b - lh - 6), (x1b + lw, y1b), color, -1)
                        cv2.putText(vis_frame, label, (x1b, y1b - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Add status bar to visualization
            if vis_frame is not None:
                status = f"tick={tick}  metin={'ON' if enable_metin else 'off'}  enemy={'ON' if enable_enemy else 'off'}"
                if best_boulder:
                    status += f"  | BOULDER {best_boulder[0]:.0%}"
                if best_enemy:
                    status += f"  | ENEMY {best_enemy[0]:.0%}"
                if not best_boulder and not best_enemy:
                    status += "  | no targets (idle roam)"
                cv2.putText(vis_frame, status, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(vis_frame, status, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            # Save annotated detection screenshot every N ticks
            if save_detections_every > 0 and vis_frame is not None and tick % save_detections_every == 0:
                os.makedirs("detections", exist_ok=True)
                path = f"detections/det_{datetime.now():%Y%m%d_%H%M%S}_{tick}.png"
                cv2.imwrite(path, vis_frame)
                print(f"  [{ts()}] [DETECT] saved: {path}")

            # Update live detection window (viewer thread reads this)
            if show_detections and vis_frame is not None:
                with _vis_frame_lock:
                    _latest_vis_frame = vis_frame

            # 4a. Metin/boulder takes priority — Shift+Right-click
            if enable_metin and best_boulder is not None:
                c, x1, y1, x2, y2 = best_boulder
                bx, by = (x1+x2)//2, (y1+y2)//2
                print(f"  [{ts()}] [BOULDER] conf={c:.0%}")
                move_to(target_hwnd, bx + random.randint(-5, -2), by + random.randint(-3, 3))
                time.sleep(random.uniform(0.03, 0.06))
                move_to(target_hwnd, bx, by); time.sleep(0.05)
                send_scan(SCAN["lshift"]); time.sleep(0.05)
                right_click(); time.sleep(0.05)
                send_scan(SCAN["lshift"], up=True)
                boulder_count += 1
                print(f"  [{ts()}] [BOULDER] #{boulder_count} at ({bx},{by})")
                continue    # handled — restart loop

            # 4b. Enemy fallback — left-click only
            if enable_enemy and best_enemy is not None:
                c, x1, y1, x2, y2 = best_enemy
                ex, ey = (x1+x2)//2, (y1+y2)//2
                move_to(target_hwnd, ex + random.randint(-5, -2), ey + random.randint(-3, 3))
                time.sleep(random.uniform(0.03, 0.06))
                move_to(target_hwnd, ex, ey); time.sleep(0.05)
                left_click()
                enemy_count += 1
                print(f"  [{ts()}] [ENEMY] #{enemy_count} conf={c:.0%} at ({ex},{ey})")
                continue    # handled — restart loop

            # 4c. Nothing detected — click a random position to roam
            cr = wt.RECT(); u32.GetClientRect(target_hwnd, ctypes.byref(cr))
            cw, ch = cr.right - cr.left, cr.bottom - cr.top
            if cw > 0 and ch > 0:
                rx = random.randint(int(cw * 0.25), int(cw * 0.75))
                ry = random.randint(int(cw * 0.25), int(ch * 0.75))
                move_to(target_hwnd, rx + random.randint(-5, -2), ry + random.randint(-3, 3))
                time.sleep(random.uniform(0.03, 0.06))
                move_to(target_hwnd, rx, ry); time.sleep(0.05)
                left_click()
                idle_count += 1
                print(f"  [{ts()}] [IDLE] #{idle_count} random click at ({rx},{ry})")

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global target_hwnd, running, space_held

    fkeys = list(VK_FKEYS.keys())
    p = argparse.ArgumentParser(description="Metin2 Bot — serialized main loop")
    p.add_argument("--loop-interval", type=float, default=3.0,
                   help="Seconds between main-loop ticks (default: 3)")
    p.add_argument("--captcha", action="store_true", help="Enable captcha detection+solving in main loop")
    p.add_argument("--respawn", action="store_true", help="Enable respawn detection (click respawn + Ctrl+H)")
    p.add_argument("--biolog", type=float, nargs="?", const=300.0, default=None, metavar="SECS",
                   help="Enable inventory biolog routine every N seconds (default: 300 = 5 min)")
    p.add_argument("--stuck", action="store_true", help="Enable stuck detection in main loop")
    p.add_argument("--buffs", action="store_true", help="Enable buff detection (re-apply buffs when spell icon inactive)")
    p.add_argument("--buff-icon", default="spells/non-active-spell.png", help="Buff icon template image")
    p.add_argument("--metin", action="store_true", help="Enable boulder/metin detection in main loop")
    p.add_argument("--enemy", action="store_true", help="Enable enemy detection in main loop")
    p.add_argument("--save-detections", type=int, default=0, metavar="N",
                   help="Save annotated YOLO frame every N loop ticks to detections/ (0=off)")
    p.add_argument("--show-detections", action="store_true",
                   help="Show live OpenCV window with YOLO detections")
    p.add_argument("--space", action="store_true", help="Hold spacebar while focused")
    p.add_argument("--key1", type=float, nargs="?", const=3.0, default=None,
                   help="Enable '1' key loop; optionally set mean interval in seconds (default: 3)")
    p.add_argument("--keyz", type=float, nargs="?", const=1.0, default=None,
                   help="Enable 'Z' key loop; optionally set mean interval in seconds (default: 1)")
    p.add_argument("--model", default="runs/detect/metin2_detect/boulder_v1/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--window", default="nothyr")
    p.add_argument("--pause-key", default="F6", choices=fkeys)
    p.add_argument("--save-every", type=float, default=0, help="Auto-screenshot interval in seconds (0=off)")
    p.add_argument("--save-key", default="F9", choices=fkeys, help="Manual screenshot hotkey")
    p.add_argument("--stuck-thresh", type=float, default=3.0,
                   help="Stuck threshold: %% of pixels changed (default: 3.0)")
    p.add_argument("--stuck-consec", type=int, default=5,
                   help="Consecutive low-diff frames before unstuck (default: 5)")
    p.add_argument("--stuck-log", type=float, default=0, help="Log frame diff %% every N seconds for threshold tuning (0=off)")
    p.add_argument("--stuck-key", default="F11", choices=fkeys, help="Manual stuck check hotkey")
    p.add_argument("--captcha-key", default="F10", choices=fkeys, help="Captcha solver hotkey")
    p.add_argument("--captcha-data", default="capcha-data", help="Captcha template root dir")
    p.add_argument("--captcha-header", default="capcha-data/captcha_header.png", help="Captcha header template image")
    p.add_argument("--respawn-button", default="respawn/respawn-button.png", help="Respawn button template image")
    args = p.parse_args()

    # Validate no duplicate hotkeys
    keys_used = [args.pause_key, args.save_key, args.stuck_key, args.captcha_key]
    if len(keys_used) != len(set(keys_used)):
        print(f"[{ts()}] Error: duplicate hotkeys assigned"); sys.exit(1)

    matches = find_window(args.window)
    if not matches: print(f"[{ts()}] Window not found."); sys.exit(1)
    if len(matches) > 1:
        for i, (_, t) in enumerate(matches): print(f"  [{i}] {t}")
        idx = int(input("Pick: ").strip() or "0")
    else: idx = 0
    target_hwnd, title = matches[idx]

    # Load YOLO model (needed for metin + enemy detection)
    model = None
    if args.metin or args.enemy:
        print(f"[{ts()}] Loading model: {args.model}")
        model = YOLO(args.model)

    # Pre-initialize OCR (downloads models on first run, then cached)
    try:
        init_ocr()
    except Exception as e:
        print(f"[{ts()}] [WARN] OCR init failed: {e} — captcha will fall back to default class")

    # Load captcha header template
    global _captcha_header_tmpl
    header_tmpl = None
    if os.path.isfile(args.captcha_header):
        header_tmpl = cv2.imread(args.captcha_header)
        if header_tmpl is not None:
            _captcha_header_tmpl = header_tmpl
            print(f"[{ts()}] Captcha header loaded: {args.captcha_header}")
    if header_tmpl is None and args.captcha:
        print(f"[{ts()}] [WARN] Captcha header not found: {args.captcha_header} — captcha check disabled")
        args.captcha = False

    # Load respawn button template
    global _respawn_btn_tmpl
    if args.respawn:
        if os.path.isfile(args.respawn_button):
            _respawn_btn_tmpl = cv2.imread(args.respawn_button)
            if _respawn_btn_tmpl is not None:
                print(f"[{ts()}] Respawn button loaded: {args.respawn_button}")
            else:
                print(f"[{ts()}] [WARN] Failed to read respawn button: {args.respawn_button} — respawn disabled")
                args.respawn = False
        else:
            print(f"[{ts()}] [WARN] Respawn button not found: {args.respawn_button} — respawn disabled")
            args.respawn = False

    # Load buff icon template
    global _buff_icon_tmpl
    if args.buffs:
        if os.path.isfile(args.buff_icon):
            _buff_icon_tmpl = cv2.imread(args.buff_icon)
            if _buff_icon_tmpl is not None:
                print(f"[{ts()}] Buff icon loaded: {args.buff_icon}")
            else:
                print(f"[{ts()}] [WARN] Failed to read buff icon: {args.buff_icon} — buffs disabled")
                args.buffs = False
        else:
            print(f"[{ts()}] [WARN] Buff icon not found: {args.buff_icon} — buffs disabled")
            args.buffs = False

    checks = [name for name, enabled in [
        ("captcha", args.captcha), ("respawn", args.respawn),
        (f"biolog({args.biolog}s)" if args.biolog else "biolog", args.biolog is not None),
        ("stuck", args.stuck), ("buffs", args.buffs),
        ("metin", args.metin), ("enemy", args.enemy),
    ] if enabled]
    extras = [f for f, on in [
        ("space", args.space),
        (f"key1={args.key1}s", args.key1 is not None),
        (f"keyz={args.keyz}s", args.keyz is not None),
        (f"save-every={args.save_every}s", args.save_every > 0),
        (f"save-detections=every {args.save_detections} ticks", args.save_detections > 0),
        ("show-detections", args.show_detections),
        (f"stuck-log={args.stuck_log}s", args.stuck_log > 0),
    ] if on]

    print(f"[{ts()}] Window: \"{title}\"")
    print(f"[{ts()}] Main loop: every {args.loop_interval}s  checks: {' → '.join(checks) or 'none'}")
    if extras:
        print(f"[{ts()}] Extras: {', '.join(extras)}")
    print(f"[{ts()}] YOLO conf={args.conf}  stuck_thresh={args.stuck_thresh}%  stuck_consec={args.stuck_consec}")
    print(f"[{ts()}] Hotkeys: pause={args.pause_key} screenshot={args.save_key} captcha={args.captcha_key} stuck={args.stuck_key}")
    print(f"[{ts()}] Ctrl+C to stop\n")

    def start(fn, a=()): threading.Thread(target=fn, args=a, daemon=True).start()

    # Pause hotkey (always on)
    start(hotkey_listener, (VK_FKEYS[args.pause_key], args.pause_key, pause_callback))

    # Manual screenshot hotkey (always on)
    start(hotkey_listener, (VK_FKEYS[args.save_key], args.save_key,
                            lambda k: save_screenshot("manual")))

    # Captcha solver hotkey (always on)
    start(hotkey_listener, (VK_FKEYS[args.captcha_key], args.captcha_key,
                            lambda k: run_captcha(args.captcha_data)))

    # Manual stuck check hotkey (always on) — forces unstuck immediately
    stuck_state = {"prev": None}
    def manual_stuck(k):
        frame = capture_window(target_hwnd)
        if frame is None: return
        pct = frame_diff_pct(stuck_state["prev"], frame)
        print(f"  [{ts()}] [{k}] diff={pct:.2f}%")
        if pct < args.stuck_thresh:
            save_stuck_visualization(stuck_state["prev"], frame, pct, args.stuck_thresh)
            stuck_state["prev"] = run_unstuck(args.stuck_thresh, frame)
        else:
            print(f"  [{ts()}] [{k}] not stuck")
            stuck_state["prev"] = frame
    start(hotkey_listener, (VK_FKEYS[args.stuck_key], args.stuck_key, manual_stuck))

    # Optional helper threads (these don't capture frames themselves)
    if args.space: start(space_watcher)
    if args.key1 is not None:  start(key_loop, ("1", SCAN["1"], args.key1))
    if args.keyz is not None:  start(key_loop, ("Z", SCAN["z"], args.keyz))
    if args.save_every > 0:  start(screenshot_timer, (args.save_every,))
    if args.stuck_log > 0: start(stuck_log_timer, (args.stuck_log,))

    # Detection viewer (dedicated GUI thread for responsive window)
    if args.show_detections:
        start(detection_viewer)

    # ── The serialized main loop (single thread for captcha/respawn/stuck/buffs/metin/enemy) ──
    if args.captcha or args.respawn or args.stuck or args.buffs or args.metin or args.enemy or args.biolog is not None:
        start(main_loop, (model, args.conf, args.captcha_data, header_tmpl,
                          args.loop_interval,
                          args.captcha, args.respawn, args.biolog,
                          args.stuck, args.buffs, args.metin, args.enemy,
                          args.stuck_thresh, args.stuck_consec,
                          args.save_detections, args.show_detections))

    try:
        # Event.wait() is interruptible by Ctrl+C on Windows, unlike time.sleep()
        # when other threads hold the GIL (ctypes/YOLO inference)
        _shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        _shutdown_event.set()
        if space_held: send_scan(SCAN["space"], up=True)
        cv2.destroyAllWindows()
        print(f"\n[{ts()}] Stopped.")

if __name__ == "__main__":
    main()