"""
main.py – Entry Point & Core Loop  (MediaPipe Tasks API)
=========================================================
Uses mediapipe.tasks.python.vision.HandLandmarker (compatible with
mediapipe >= 0.10.14, including 0.10.33+).

Requires hand_landmarker.task in the same directory.
Download once with:
    python -c "import urllib.request; urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        'hand_landmarker.task')"

Controls
--------
  Q or ESC  – quit gracefully.
  PyAutoGUI FAILSAFE: move OS cursor to top-left corner (0,0) to abort.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gestures import MouseController, GestureDetector

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX:     int   = 0
FRAME_W:          int   = 640
FRAME_H:          int   = 480
MODEL_PATH:       str   = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

EMA_ALPHA:        float = 0.30
DEAD_ZONE_PX:     float = 3.5
ZONE_RATIO:       float = 0.65
PINCH_THRESH:     float = 0.055
CLICK_COOLDOWN_S: float = 0.40

# ─── Landmark drawing connections (21-point model) ────────────────────────────
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (5,9),(9,10),(10,11),(11,12),  # middle
    (9,13),(13,14),(14,15),(15,16),# ring
    (13,17),(17,18),(18,19),(19,20),# pinky
    (0,17),                         # palm base
]

# ─── HUD colours (BGR) ────────────────────────────────────────────────────────
C_GREEN  = (50,  220,  50)
C_RED    = (40,   40, 220)
C_ORANGE = (20,  165, 255)
C_YELLOW = (10,  230, 230)
C_WHITE  = (240, 240, 240)
C_DARK   = (20,   20,  20)


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _text(frame, txt: str, pos: tuple, color=C_WHITE, scale=0.6, thick=2):
    x, y = pos
    cv2.putText(frame, txt, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, C_DARK, thick+1, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _draw_skeleton(frame, landmarks, fw: int, fh: int):
    """Draw hand skeleton from raw NormalizedLandmark list."""
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], C_GREEN, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, C_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 4, C_GREEN,  1, cv2.LINE_AA)


def _draw_hud(frame, result: dict, fps: float, zone_rect: tuple):
    gesture   = result["gesture"]
    screen_xy = result["screen_xy"]
    clutch    = result["clutch"]
    x1, y1, x2, y2 = zone_rect

    zone_col = C_RED if clutch else C_GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), zone_col, 2)
    _text(frame, "Active Zone", (x1+4, y1-8), zone_col, 0.45, 1)

    badge_col = {
        GestureDetector.IDLE:  C_RED,
        GestureDetector.CLICK: C_YELLOW,
        GestureDetector.MOVE:  C_GREEN,
    }.get(gesture, C_WHITE)

    cv2.rectangle(frame, (8, 8), (240, 38), C_DARK, -1)
    cv2.rectangle(frame, (8, 8), (240, 38), badge_col, 1)
    _text(frame, f"Gesture: {gesture}", (14, 30), badge_col, 0.60, 1)

    if screen_xy:
        _text(frame, f"Screen: {screen_xy[0]},{screen_xy[1]}", (14, 58), C_WHITE, 0.50, 1)
    else:
        _text(frame, "Screen: -- IDLE --", (14, 58), C_RED, 0.50, 1)

    fps_txt = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _text(frame, fps_txt, (frame.shape[1] - tw - 12, 30), C_ORANGE, 0.55, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Validate model file ────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Download it with:")
        print("  python -c \"import urllib.request; urllib.request.urlretrieve("
              "'https://storage.googleapis.com/mediapipe-models/hand_landmarker"
              "/hand_landmarker/float16/latest/hand_landmarker.task',"
              " 'hand_landmarker.task')\"")
        sys.exit(1)

    # ── Open webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened at {fw}×{fh}")

    # ── MediaPipe HandLandmarker (Tasks API) ───────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,   # synchronous per-frame
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    # ── Mouse controller ───────────────────────────────────────────────────
    controller = MouseController(
        frame_w=fw, frame_h=fh,
        ema_alpha=EMA_ALPHA,
        dead_zone_px=DEAD_ZONE_PX,
        zone_ratio=ZONE_RATIO,
        pinch_thresh=PINCH_THRESH,
        click_cooldown_s=CLICK_COOLDOWN_S,
    )
    zone_rect = controller.mapper.zone_rect

    # ── FPS tracker ────────────────────────────────────────────────────────
    fps_buf: list[float] = []
    fps_win  = 15
    prev_t   = time.monotonic()

    print("[INFO] Started. Press 'H' for Help/Guide. Press 'Q' to quit.")

    # ── v0.2 GUI Window Setup ──────────────────────────────────────────────
    WIN_NAME = "Hand Gesture Overlay [H: Guide, Q: Quit]"
    
    # 1. Make window resizable and movable
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    
    # 2. Force window to stay "Always on Top" (Picture-in-Picture style)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
    
    # 3. Start it small like a mini-player (Width, Height)
    cv2.resizeWindow(WIN_NAME, 320, 240) 

    show_guide = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # FPS
        now = time.monotonic()
        fps_buf.append(1.0 / max(now - prev_t, 1e-9))
        prev_t = now
        if len(fps_buf) > fps_win:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        # ── MediaPipe inference ──────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_result = landmarker.detect(mp_image)

        # ── Process gestures ─────────────────────────────────────────────
        if mp_result.hand_landmarks:
            lm_list = mp_result.hand_landmarks[0]   # first hand

            # Draw skeleton
            _draw_skeleton(frame, lm_list, fw, fh)

            result = controller.process(lm_list, fw, fh)
        else:
            result = {"gesture": "NO HAND", "screen_xy": None, "clutch": True}

        _draw_hud(frame, result, fps, zone_rect)

        # ── v0.2 Draw the Help Guide Overlay ─────────────────────────────
        if show_guide:
            # Create a dark semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, fh), (20, 20, 20), -1) 
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
            
            # Active Gestures
            _text(frame, "--- ACTIVE GESTURES ---", (20, 50), C_YELLOW, 0.7, 2)
            _text(frame, "[Index Extended] : Move Mouse", (20, 90), C_WHITE, 0.6, 1)
            _text(frame, "[Pinch] : Left Click", (20, 130), C_WHITE, 0.6, 1)
            _text(frame, "[Fist]  : Clutch / Pause", (20, 170), C_WHITE, 0.6, 1)
            
            # Future Gestures (Names only)
            _text(frame, "--- COMING SOON ---", (20, 240), C_ORANGE, 0.7, 2)
            _text(frame, "- Scroll (Two Fingers)", (20, 280), (150, 150, 150), 0.6, 1)
            _text(frame, "- Right Click", (20, 320), (150, 150, 150), 0.6, 1)
            _text(frame, "- Drag & Drop", (20, 360), (150, 150, 150), 0.6, 1)
            _text(frame, "- Volume Control", (20, 400), (150, 150, 150), 0.6, 1)

        cv2.imshow(WIN_NAME, frame)

        # ── Keyboard Controls ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("h"), ord("H")):
            show_guide = not show_guide # Toggle the guide on and off

    # ── Cleanup ───────────────────────────────────────────────────────────
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()