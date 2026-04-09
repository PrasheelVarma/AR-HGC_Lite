"""
app.py – The v0.2 "Hyper App" GUI
=================================
Modern UI using CustomTkinter. Features live settings, embedded camera,
and gesture guides.
"""

import cv2
import time
import customtkinter as ctk
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os

from gestures import MouseController

# --- Initialization ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HandControllerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hand Gesture Controller v0.2")
        self.geometry("1000x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Engine Variables
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fw, self.fh = 640, 480
        
        # Load MediaPipe
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        
        # Load our Math Engine
        self.controller = MouseController(frame_w=self.fw, frame_h=self.fh)

        self.build_ui()
        self.update_frame() # Start the camera loop

    def build_ui(self):
        # --- LEFT PANEL: Settings ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="Control Panel", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 10))

        # Smoothing Slider
        ctk.CTkLabel(self.sidebar, text="Cursor Smoothing (Anti-Jitter)").pack(pady=(20, 0), padx=20, anchor="w")
        self.smooth_slider = ctk.CTkSlider(self.sidebar, from_=0.05, to=0.8, command=self.update_settings)
        self.smooth_slider.set(0.3)
        self.smooth_slider.pack(pady=10, padx=20, fill="x")

        # Trackpad Size Slider
        ctk.CTkLabel(self.sidebar, text="Trackpad Zone Size").pack(pady=(20, 0), padx=20, anchor="w")
        self.zone_slider = ctk.CTkSlider(self.sidebar, from_=0.3, to=0.9, command=self.update_settings)
        self.zone_slider.set(0.65)
        self.zone_slider.pack(pady=10, padx=20, fill="x")

        # Gestures Guide List
        ctk.CTkLabel(self.sidebar, text="Active Gestures", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(40, 10), anchor="w", padx=20)
        ctk.CTkLabel(self.sidebar, text="🟢 Index Finger: Move Mouse", anchor="w").pack(padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="🟡 Pinch: Left Click", anchor="w").pack(padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="🔴 Fist: Pause / Clutch", anchor="w").pack(padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="Coming Soon...", font=ctk.CTkFont(size=16, weight="bold"), text_color="gray").pack(pady=(40, 10), anchor="w", padx=20)
        ctk.CTkLabel(self.sidebar, text="- Scroll (Two Fingers)", text_color="gray", anchor="w").pack(padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="- Right Click (Middle Finger)", text_color="gray", anchor="w").pack(padx=20, fill="x")

        # --- RIGHT PANEL: Camera Feed ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(expand=True)

    def update_settings(self, value=None):
        # Update our gestures.py math live from the UI sliders!
        self.controller.smoother.alpha = float(self.smooth_slider.get())
        
        # Update the Trackpad Zone live
        new_ratio = float(self.zone_slider.get())
        mx = self.fw * (1 - new_ratio) / 2
        my = self.fh * (1 - new_ratio) / 2
        self.controller.mapper.x1, self.controller.mapper.y1 = mx, my
        self.controller.mapper.x2, self.controller.mapper.y2 = self.fw - mx, self.fh - my
        self.controller.mapper.zw = self.controller.mapper.x2 - self.controller.mapper.x1
        self.controller.mapper.zh = self.controller.mapper.y2 - self.controller.mapper.y1

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Flip the frame for the mirror effect!
            frame = cv2.flip(frame, 1)

            # 2. Process AI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            mp_result = self.landmarker.detect(mp_image)

            # 3. Process Mouse Math
            if mp_result.hand_landmarks:
                lm_list = mp_result.hand_landmarks[0]
                
                # Draw simple skeleton
                pts = [(int(lm.x * self.fw), int(lm.y * self.fh)) for lm in lm_list]
                for p in pts:
                    cv2.circle(rgb_frame, p, 4, (0, 255, 0), -1)

                self.controller.process(lm_list, self.fw, self.fh)

            # 4. Draw the Trackpad Box
            z = self.controller.mapper
            cv2.rectangle(rgb_frame, (int(z.x1), int(z.y1)), (int(z.x2), int(z.y2)), (255, 0, 0), 2)

            # 5. Push to UI
            img = Image.fromarray(rgb_frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(self.fw, self.fh))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

        # Loop this function extremely fast
        self.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.landmarker.close()
        self.destroy()

if __name__ == "__main__":
    app = HandControllerApp()
    app.mainloop()