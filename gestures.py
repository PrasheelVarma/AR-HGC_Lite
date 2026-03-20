"""
gestures.py – Mathematical Engine & OS Input Layer  (MediaPipe Tasks API)
==========================================================================
Compatible with mediapipe >= 0.10.14  (Tasks-based HandLandmarker)

Responsibilities
----------------
1. EMA Smoothing        – tames webcam jitter while staying responsive.
2. Dead-zone guard      – suppresses drift when hand is held still.
3. Spatial Mapping      – active zone → full screen (trackpad-style).
4. Gesture Recognition  – pinch=click, fist=idle/clutch.
5. Debounce             – time-gated click prevents repeat-fire.
6. OS Mouse Control     – PyAutoGUI moves/clicks the system cursor.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pyautogui

# ── PyAutoGUI safety ────────────────────────────────────────────────────────
pyautogui.PAUSE = 0   # Remove built-in 0.1 s inter-call sleep
# FAILSAFE remains True: move cursor to top-left corner to abort.

SCREEN_W, SCREEN_H = pyautogui.size()


# ═══════════════════════════════════════════════════════════════════════════
# 1. EMA Filter
# ═══════════════════════════════════════════════════════════════════════════

class EMAFilter:
    """
    Exponential Moving Average filter for (x, y) cursor coordinates.

    alpha       – blend factor: lower = smoother, higher = more responsive.
    dead_zone_px – minimum pixel-distance change before committing a new pos.
    """

    def __init__(self, alpha: float = 0.30, dead_zone_px: float = 3.5) -> None:
        self.alpha = alpha
        self.dead_zone_px = dead_zone_px
        self._s: Optional[np.ndarray] = None

    def update(self, x: float, y: float) -> tuple[int, int]:
        v = np.array([x, y], dtype=float)
        if self._s is None:
            self._s = v.copy()
            return int(v[0]), int(v[1])
        blended = self.alpha * v + (1.0 - self.alpha) * self._s
        if np.linalg.norm(blended - self._s) < self.dead_zone_px:
            return int(self._s[0]), int(self._s[1])
        self._s = blended
        return int(blended[0]), int(blended[1])

    def reset(self) -> None:
        self._s = None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Spatial Mapper
# ═══════════════════════════════════════════════════════════════════════════

class SpatialMapper:
    """
    Maps a central sub-region of the camera frame to the full screen.
    zone_ratio – fraction of frame width/height used as the active zone.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        zone_ratio: float = 0.65,
        screen_w: int = SCREEN_W,
        screen_h: int = SCREEN_H,
    ) -> None:
        self.sw, self.sh = screen_w, screen_h
        mx = frame_w * (1 - zone_ratio) / 2
        my = frame_h * (1 - zone_ratio) / 2
        self.x1, self.y1 = mx, my
        self.x2, self.y2 = frame_w - mx, frame_h - my
        self.zw = self.x2 - self.x1
        self.zh = self.y2 - self.y1

    def map(self, cx: float, cy: float) -> tuple[float, float]:
        cx = max(self.x1, min(cx, self.x2))
        cy = max(self.y1, min(cy, self.y2))
        nx = 1.0 - (cx - self.x1) / self.zw   # flip x for mirror UX
        ny = (cy - self.y1) / self.zh
        return nx * self.sw, ny * self.sh

    @property
    def zone_rect(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Gesture Detector  (works with mediapipe Tasks NormalizedLandmark list)
# ═══════════════════════════════════════════════════════════════════════════

# MediaPipe hand landmark indices
_IDX_TIP   = 8    # Index fingertip
_THUMB_TIP = 4    # Thumb tip
_FINGER_TIPS_PIPS = [
    (8,  6),   # index  tip, pip
    (12, 10),  # middle tip, pip
    (16, 14),  # ring   tip, pip
    (20, 18),  # pinky  tip, pip
]


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _folded(lm, tip_i: int, pip_i: int) -> bool:
    """True if finger tip is below its PIP joint (image-space y grows down)."""
    return lm[tip_i].y > lm[pip_i].y


class GestureDetector:
    MOVE  = "MOVE"
    CLICK = "LEFT_CLICK"
    IDLE  = "IDLE"

    def __init__(
        self,
        pinch_thresh: float = 0.055,
        click_cooldown_s: float = 0.40,
    ) -> None:
        self.pinch_thresh = pinch_thresh
        self.cooldown = click_cooldown_s
        self._last_click: float = 0.0
        self._click_fired: bool = False

    def detect(self, landmarks) -> str:
        lm = landmarks

        # IDLE – all four fingers folded (fist)
        if all(_folded(lm, tip, pip) for tip, pip in _FINGER_TIPS_PIPS):
            self._click_fired = False
            return self.IDLE

        # CLICK – index–thumb pinch
        if _dist(lm[_THUMB_TIP], lm[_IDX_TIP]) < self.pinch_thresh:
            now = time.monotonic()
            if not self._click_fired or (now - self._last_click) >= self.cooldown:
                self._last_click = now
                self._click_fired = True
                return self.CLICK
            return self.MOVE   # pinch held but in cooldown
        else:
            self._click_fired = False

        return self.MOVE


# ═══════════════════════════════════════════════════════════════════════════
# 4. MouseController  – high-level façade used by main.py
# ═══════════════════════════════════════════════════════════════════════════

class MouseController:
    """
    Wires SpatialMapper + EMAFilter + GestureDetector + PyAutoGUI together.

    Call process(landmarks, frame_w, frame_h) once per frame.
    Returns a dict with 'gesture', 'screen_xy', 'clutch' for the HUD.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        ema_alpha: float = 0.30,
        dead_zone_px: float = 3.5,
        zone_ratio: float = 0.65,
        pinch_thresh: float = 0.055,
        click_cooldown_s: float = 0.40,
    ) -> None:
        self.mapper   = SpatialMapper(frame_w, frame_h, zone_ratio)
        self.smoother = EMAFilter(ema_alpha, dead_zone_px)
        self.detector = GestureDetector(pinch_thresh, click_cooldown_s)
        self._clutch  = False

    def process(self, landmarks, frame_w: int, frame_h: int) -> dict:
        gesture = self.detector.detect(landmarks)

        if gesture == GestureDetector.IDLE:
            if not self._clutch:
                self._clutch = True
                self.smoother.reset()
            return {"gesture": gesture, "screen_xy": None, "clutch": True}
        self._clutch = False

        # Index fingertip (landmark 8) drives the cursor
        tip = landmarks[_IDX_TIP]
        sx_r, sy_r = self.mapper.map(tip.x * frame_w, tip.y * frame_h)
        sx, sy = self.smoother.update(sx_r, sy_r)
        sx = max(0, min(sx, SCREEN_W - 1))
        sy = max(0, min(sy, SCREEN_H - 1))

        if gesture == GestureDetector.CLICK:
            pyautogui.click(sx, sy)
        else:
            pyautogui.moveTo(sx, sy)

        return {"gesture": gesture, "screen_xy": (sx, sy), "clutch": False}
