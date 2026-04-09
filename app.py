"""
app.py – Spatial Computing Dashboard  (CustomTkinter)
======================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
from collections import deque

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gestures import MouseController, GestureDetector

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
CAM_W, CAM_H = 1280, 720
DISPLAY_W, DISPLAY_H = 800, 450       

_FINGERTIPS = [4, 8, 12, 16, 20]
_ADJACENT = [(4, 8), (8, 12), (12, 16), (16, 20)]

def _neon_color(t: float) -> tuple[int, int, int]:
    h = int((t * 30) % 180)
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def _fade(color, a):
    return tuple(int(c * a) for c in color)

class _HandTrails:
    def __init__(self, maxlen=14):
        self.trails = {t: deque(maxlen=maxlen) for t in _FINGERTIPS}
        self.prev = None

class HandControllerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HG Control — Lite Dashboard")
        self.geometry("1280x720")
        self.minsize(1100, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.configure(fg_color="#0d0d12")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model not found: {MODEL_PATH}")
            sys.exit(1)
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self.controller = MouseController(frame_w=self.fw, frame_h=self.fh)
        self._hand_trails = [_HandTrails(), _HandTrails()]
        self._fps_buf: list[float] = []
        self._prev_t = time.monotonic()

        self._build_ui()
        self._update_frame()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, fg_color="#111118", corner_radius=0, border_width=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.configure(width=320)

        title_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=24, pady=(28, 4))
        ctk.CTkLabel(title_frame, text="HG Control", font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"), text_color="#e0e0f0").pack(anchor="w")
        ctk.CTkLabel(title_frame, text="Lite Edition  •  Spatial Computing", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#5a5a72").pack(anchor="w", pady=(2, 0))

        ctk.CTkFrame(sidebar, height=1, fg_color="#222230").pack(fill="x", padx=20, pady=(18, 18))

        self.smooth_slider = self._add_slider(sidebar, "Cursor Smoothing", 0.05, 0.80, 0.30)
        self.zone_slider   = self._add_slider(sidebar, "Active Trackpad Zone", 0.30, 0.95, 0.65)
        self.scroll_slider = self._add_slider(sidebar, "Scroll Speed", 0.20, 5.00, 1.00)

        ctk.CTkFrame(sidebar, height=1, fg_color="#222230").pack(fill="x", padx=20, pady=(22, 14))

        card = ctk.CTkFrame(sidebar, fg_color="#16161f", corner_radius=12)
        card.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(card, text="Gesture Reference", font=ctk.CTkFont(size=13, weight="bold"), text_color="#9090a8").pack(anchor="w", padx=16, pady=(14, 8))

        gestures_info = [
            ("🟢", "Index Extended", "Move"),
            ("🟡", "Thumb + Index Pinch", "Left Click"),
            ("🟠", "Thumb + Middle Pinch", "Right Click"),
            ("🔵", "Index + Middle Up", "Scroll"),
            ("🔴", "Fist (all folded)", "Clutch"),
        ]
        for icon, label, action in gestures_info:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(row, text=icon, width=24, font=ctk.CTkFont(size=13)).pack(side="left")
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12), text_color="#b0b0c0").pack(side="left", padx=(6, 0))
            ctk.CTkLabel(row, text=action, font=ctk.CTkFont(size=11, weight="bold"), text_color="#6a6a82").pack(side="right")

        ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

        self.status_label = ctk.CTkLabel(sidebar, text="● Initializing...", font=ctk.CTkFont(size=12), text_color="#4a4a60")
        self.status_label.pack(side="bottom", anchor="w", padx=24, pady=(0, 20))

        right_panel = ctk.CTkFrame(self, fg_color="#0d0d12", corner_radius=0)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)

        cam_container = ctk.CTkFrame(right_panel, fg_color="#111118", corner_radius=14, border_width=1, border_color="#1e1e2a")
        cam_container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        cam_container.grid_columnconfigure(0, weight=1)
        cam_container.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(cam_container, text="")
        self.video_label.grid(row=0, column=0, padx=8, pady=8)

    def _add_slider(self, parent, label_text: str, from_: float, to_: float, default: float) -> ctk.CTkSlider:
        wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        wrapper.pack(fill="x", padx=24, pady=(0, 14))

        top = ctk.CTkFrame(wrapper, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=label_text, font=ctk.CTkFont(size=12), text_color="#8888a0").pack(side="left")
        val_label = ctk.CTkLabel(top, text=f"{default:.2f}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#c0c0d8")
        val_label.pack(side="right")

        slider = ctk.CTkSlider(
            wrapper, from_=from_, to=to_, number_of_steps=100,
            fg_color="#1e1e28", progress_color="#3a3a55",
            button_color="#6060a0", button_hover_color="#7878c0",
            height=14, command=lambda v, lbl=val_label: self._on_slider(v, lbl),
        )
        slider.set(default)
        slider.pack(fill="x", pady=(6, 0))
        slider._value_label = val_label  
        return slider

    def _on_slider(self, value, label: ctk.CTkLabel):
        label.configure(text=f"{float(value):.2f}")
        self._apply_settings()

    def _apply_settings(self):
        self.controller.smoother.alpha = float(self.smooth_slider.get())
        new_zone = float(self.zone_slider.get())
        self.controller.mapper.set_zone_ratio(new_zone)
        self.controller.scroll_speed = float(self.scroll_slider.get())

    def _draw_neon(self, frame: np.ndarray, all_lm: list) -> np.ndarray:
        canvas = np.zeros_like(frame)
        nc = _neon_color(time.time())

        for hi, lm in enumerate(all_lm):
            if hi >= len(self._hand_trails): break
            st = self._hand_trails[hi]
            pts = {i: (int(l.x * self.fw), int(l.y * self.fh)) for i, l in enumerate(lm)}
            palm = pts[0]

            def _is_ext(tip_i, pip_i):
                return math.hypot(lm[tip_i].x - lm[0].x, lm[tip_i].y - lm[0].y) > \
                       math.hypot(lm[pip_i].x - lm[0].x, lm[pip_i].y - lm[0].y)

            ext_tips = []
            if _is_ext(4, 3): ext_tips.append(4)
            if _is_ext(8, 6): ext_tips.append(8)
            if _is_ext(12, 10): ext_tips.append(12)
            if _is_ext(16, 14): ext_tips.append(16)
            if _is_ext(20, 18): ext_tips.append(20)

            cur = pts[8]
            spd = 0.0
            if st.prev:
                spd = math.hypot(cur[0] - st.prev[0], cur[1] - st.prev[1])
            st.prev = cur

            ct = max(2, min(int(spd * 0.12 + 1.5), 5))
            gt = max(4, ct + 4)

            for t in ext_tips:
                cv2.line(canvas, palm, pts[t], _fade(nc, 0.35), gt, cv2.LINE_AA)
                cv2.line(canvas, palm, pts[t], nc, ct, cv2.LINE_AA)
            for a, b in _ADJACENT:
                if a in ext_tips and b in ext_tips:
                    cv2.line(canvas, pts[a], pts[b], _fade(nc, 0.5), max(1, ct - 1), cv2.LINE_AA)

            for t in _FINGERTIPS:
                st.trails[t].append(pts[t])
                trail = list(st.trails[t])
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    cv2.line(canvas, trail[i - 1], trail[i], _fade(nc, alpha * 0.7), max(1, int(alpha * 5)), cv2.LINE_AA)
                cv2.circle(canvas, pts[t], ct + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, palm, ct, nc, -1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(canvas, (0, 0), sigmaX=9, sigmaY=9)
        canvas = cv2.add(canvas, glow)
        return cv2.add(frame, canvas)

    def _draw_mini_hud(self, frame: np.ndarray, gesture: str, fps: float, nh: int):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 36), (w, h), (10, 10, 14), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        col_map = {
            GestureDetector.IDLE: (80, 80, 230),
            GestureDetector.LEFT_CLICK: (50, 230, 240),
            GestureDetector.RIGHT_CLICK: (50, 180, 255),
            GestureDetector.SCROLL: (255, 220, 50),
            GestureDetector.MOVE: (80, 230, 120),
        }
        gc = col_map.get(gesture, (180, 180, 180))
        cv2.putText(frame, gesture, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {fps:.0f}  |  Hands {nh}", (w - 180, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 140), 1, cv2.LINE_AA)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            dt = max(now - self._prev_t, 1e-9)
            self._fps_buf.append(1.0 / dt)
            self._prev_t = now
            if len(self._fps_buf) > 15: self._fps_buf.pop(0)
            fps = sum(self._fps_buf) / len(self._fps_buf)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = self.landmarker.detect(mp_img)

            nh = len(mp_result.hand_landmarks) if mp_result.hand_landmarks else 0

            if nh > 0:
                frame = self._draw_neon(frame, mp_result.hand_landmarks)
                result = self.controller.process(mp_result.hand_landmarks[0], self.fw, self.fh)
                gesture = result["gesture"]
            else:
                gesture = "NO HAND"

            z = self.controller.mapper.zone_rect
            zone_col = (80, 80, 230) if gesture == GestureDetector.IDLE else (40, 40, 50)
            cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), zone_col, 1, cv2.LINE_AA)

            self._draw_mini_hud(frame, gesture, fps, nh)

            self.status_label.configure(
                text=f"● {gesture}  |  FPS {fps:.0f}  |  Hands {nh}",
                text_color="#50d090" if nh > 0 else "#884444",
            )

            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_display)
            pil_img = pil_img.resize((DISPLAY_W, DISPLAY_H), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(DISPLAY_W, DISPLAY_H))
            self.video_label.configure(image=ctk_img)
            self.video_label._ctk_img = ctk_img  

        self.after(12, self._update_frame)

    def _on_close(self):
        self.cap.release()
        self.landmarker.close()
        self.destroy()

if __name__ == "__main__":
    app = HandControllerApp()
    app.mainloop()