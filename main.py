"""
main.py – Core Loop & Neon Magic Visualizer  (MediaPipe Tasks API)
====================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
from collections import deque
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gestures import MouseController, GestureDetector

CAMERA_INDEX:     int   = 0
FRAME_W:          int   = 1280
FRAME_H:          int   = 720
MODEL_PATH:       str   = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

EMA_ALPHA:        float = 0.30
DEAD_ZONE_PX:     float = 3.5
ZONE_RATIO:       float = 0.65
PINCH_THRESH:     float = 0.055
CLICK_COOLDOWN_S: float = 0.40
SCROLL_SPEED:     float = 1.0

C_WHITE   = (240, 240, 240)
C_HUD_BG  = (15,  15,  20)
C_GREEN   = (80, 230, 120)
C_RED     = (80,  80, 230)
C_CYAN    = (255, 220,  50)
C_YELLOW  = (50,  230, 240)
C_ORANGE  = (50,  180, 255)
C_DIM     = (120, 120, 130)

_PALM_IDX   = 0
_FINGERTIPS = [4, 8, 12, 16, 20]
_ADJACENT_PAIRS = [(4, 8), (8, 12), (12, 16), (16, 20)]

def _neon_color_cycle(t: float) -> tuple[int, int, int]:
    h = int((t * 30) % 180)  
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def _fade_color(color: tuple, alpha: float) -> tuple[int, int, int]:
    return (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))

class HandTrailState:
    def __init__(self, max_trail: int = 14):
        self.trails: dict[int, deque] = {tip: deque(maxlen=max_trail) for tip in _FINGERTIPS}
        self.prev_index_pos: Optional[tuple[int, int]] = None

class NeonVisualizer:
    def __init__(self, max_hands: int = 2, max_trail: int = 14):
        self.hand_states: list[HandTrailState] = [HandTrailState(max_trail) for _ in range(max_hands)]
        self.max_trail = max_trail

    def draw(self, frame: np.ndarray, all_hand_landmarks: list, fw: int, fh: int) -> np.ndarray:
        canvas = np.zeros_like(frame)
        t = time.time()
        neon_color = _neon_color_cycle(t)

        for hand_idx, landmarks in enumerate(all_hand_landmarks):
            if hand_idx >= len(self.hand_states): break
            state = self.hand_states[hand_idx]
            self._draw_hand(canvas, landmarks, fw, fh, neon_color, state)

        glow = cv2.GaussianBlur(canvas, (0, 0), sigmaX=9, sigmaY=9)
        canvas = cv2.add(canvas, glow)
        return cv2.add(frame, canvas)

    def _draw_hand(self, canvas: np.ndarray, landmarks, fw: int, fh: int, neon_color: tuple, state: HandTrailState) -> None:
        pts = {i: (int(lm.x * fw), int(lm.y * fh)) for i, lm in enumerate(landmarks)}
        palm = pts[_PALM_IDX]

        def _is_ext(tip_i, pip_i):
            return math.hypot(landmarks[tip_i].x - landmarks[0].x, landmarks[tip_i].y - landmarks[0].y) > \
                   math.hypot(landmarks[pip_i].x - landmarks[0].x, landmarks[pip_i].y - landmarks[0].y)

        ext_tips = []
        if _is_ext(4, 3): ext_tips.append(4)
        if _is_ext(8, 6): ext_tips.append(8)
        if _is_ext(12, 10): ext_tips.append(12)
        if _is_ext(16, 14): ext_tips.append(16)
        if _is_ext(20, 18): ext_tips.append(20)

        cur_idx = pts[8]
        speed = 0.0
        if state.prev_index_pos is not None:
            dx = cur_idx[0] - state.prev_index_pos[0]
            dy = cur_idx[1] - state.prev_index_pos[1]
            speed = math.hypot(dx, dy)
        state.prev_index_pos = cur_idx

        core_thick  = max(2, min(int(speed * 0.12 + 1.5), 5))
        glow_thick  = max(4, core_thick + 4)

        for tip_id in ext_tips:
            cv2.line(canvas, palm, pts[tip_id], _fade_color(neon_color, 0.35), glow_thick, cv2.LINE_AA)
            cv2.line(canvas, palm, pts[tip_id], neon_color, core_thick, cv2.LINE_AA)

        web_thick = max(1, core_thick - 1)
        for a, b in _ADJACENT_PAIRS:
            if a in ext_tips and b in ext_tips:
                cv2.line(canvas, pts[a], pts[b], _fade_color(neon_color, 0.5), web_thick, cv2.LINE_AA)

        for tip_id in _FINGERTIPS:
            state.trails[tip_id].append(pts[tip_id])
            trail = list(state.trails[tip_id])
            n = len(trail)
            for i in range(1, n):
                alpha = i / n
                thickness = max(1, int(alpha * 5))
                fc = _fade_color(neon_color, alpha * 0.7)
                cv2.line(canvas, trail[i - 1], trail[i], fc, thickness, cv2.LINE_AA)
            cv2.circle(canvas, pts[tip_id], core_thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, palm, core_thick, neon_color, -1, cv2.LINE_AA)


def _rounded_rect_overlay(frame: np.ndarray, x: int, y: int, w: int, h: int, color: tuple = C_HUD_BG, alpha: float = 0.70, border_color: tuple | None = None) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if border_color:
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)

def _put_text(frame: np.ndarray, text: str, x: int, y: int, color: tuple = C_WHITE, scale: float = 0.50, thickness: int = 1) -> None:
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def _gesture_badge_color(gesture: str) -> tuple:
    return {
        GestureDetector.IDLE:        C_RED,
        GestureDetector.LEFT_CLICK:  C_YELLOW,
        GestureDetector.RIGHT_CLICK: C_ORANGE,
        GestureDetector.SCROLL:      C_CYAN,
        GestureDetector.MOVE:        C_GREEN,
    }.get(gesture, C_WHITE)

def _draw_hud(frame: np.ndarray, result: dict, fps: float, zone_rect: tuple, num_hands: int) -> None:
    fh, fw = frame.shape[:2]
    gesture   = result["gesture"]
    screen_xy = result["screen_xy"]
    clutch    = result["clutch"]
    scroll_d  = result.get("scroll_delta", 0)

    x1, y1, x2, y2 = zone_rect
    zone_col = C_RED if clutch else (60, 60, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), zone_col, 1, cv2.LINE_AA)

    badge_col = _gesture_badge_color(gesture)
    pill_w = 260
    _rounded_rect_overlay(frame, 10, 10, pill_w, 58, alpha=0.75, border_color=(50, 50, 55))
    _put_text(frame, f"Gesture", 20, 32, C_DIM, 0.38, 1)
    _put_text(frame, gesture, 100, 32, badge_col, 0.50, 1)
    if screen_xy:
        _put_text(frame, f"Cursor  {screen_xy[0]}, {screen_xy[1]}", 20, 56, C_DIM, 0.36, 1)
    elif clutch:
        _put_text(frame, "Clutch engaged — reposition", 20, 56, C_RED, 0.36, 1)
    else:
        _put_text(frame, "No cursor data", 20, 56, C_DIM, 0.36, 1)

    if gesture == GestureDetector.SCROLL and scroll_d != 0:
        arrow = "▲" if scroll_d > 0 else "▼"
        _put_text(frame, f"SCROLL {arrow} {abs(scroll_d)}", 20, 86, C_CYAN, 0.42, 1)

    info_w = 150
    _rounded_rect_overlay(frame, fw - info_w - 10, 10, info_w, 58, alpha=0.75, border_color=(50, 50, 55))
    _put_text(frame, f"FPS  {fps:.0f}", fw - info_w, 32, C_ORANGE, 0.42, 1)
    _put_text(frame, f"Hands  {num_hands}", fw - info_w, 56, C_DIM, 0.38, 1)

def _draw_guide(frame: np.ndarray) -> None:
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (10, 10, 14), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cx = fw // 2 - 200
    _put_text(frame, "GESTURE GUIDE", cx, 60, C_YELLOW, 0.85, 2)
    _put_text(frame, "─" * 40, cx, 85, (60, 60, 65), 0.45, 1)

    entries = [
        ("Index Extended", "Move Cursor", C_GREEN),
        ("Index + Thumb Pinch", "Left Click", C_YELLOW),
        ("Middle + Thumb Pinch", "Right Click", C_ORANGE),
        ("Index + Middle Up", "Scroll", C_CYAN),
        ("Fist (all folded)", "Clutch / Pause", C_RED),
    ]
    y = 120
    for label, action, col in entries:
        _put_text(frame, f"•  {label}", cx, y, C_WHITE, 0.52, 1)
        _put_text(frame, action, cx + 300, y, col, 0.52, 1)
        y += 40

    _put_text(frame, "Press H to close  |  Q / ESC to quit", cx, y + 30, C_DIM, 0.42, 1)

def run_loop(cap: cv2.VideoCapture | None = None, controller: MouseController | None = None, external: bool = False) -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    own_cap = False
    if cap is None:
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
        own_cap = True

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6, min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    if controller is None:
        controller = MouseController(frame_w=fw, frame_h=fh, ema_alpha=EMA_ALPHA, dead_zone_px=DEAD_ZONE_PX, zone_ratio=ZONE_RATIO, pinch_thresh=PINCH_THRESH, click_cooldown_s=CLICK_COOLDOWN_S, scroll_speed=SCROLL_SPEED)
    zone_rect = controller.mapper.zone_rect

    neon = NeonVisualizer(max_hands=2)

    fps_buf: list[float] = []
    fps_win = 15
    prev_t  = time.monotonic()

    WIN_NAME = "HG Control — Lite Edition"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    show_guide = False

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)

        now = time.monotonic()
        dt = max(now - prev_t, 1e-9)
        fps_buf.append(1.0 / dt)
        prev_t = now
        if len(fps_buf) > fps_win: fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = landmarker.detect(mp_image)

        num_hands = len(mp_result.hand_landmarks) if mp_result.hand_landmarks else 0

        if num_hands > 0:
            frame = neon.draw(frame, mp_result.hand_landmarks, fw, fh)
            result = controller.process(mp_result.hand_landmarks[0], fw, fh)
        else:
            result = {"gesture": "NO HAND", "screen_xy": None, "clutch": True, "scroll_delta": 0}

        zone_rect = controller.mapper.zone_rect
        _draw_hud(frame, result, fps, zone_rect, num_hands)

        if show_guide: _draw_guide(frame)
        cv2.imshow(WIN_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27): break
        elif key in (ord("h"), ord("H")): show_guide = not show_guide

    landmarker.close()
    if own_cap: cap.release()
    cv2.destroyAllWindows()

def main() -> None:
    run_loop()

if __name__ == "__main__":
    main()