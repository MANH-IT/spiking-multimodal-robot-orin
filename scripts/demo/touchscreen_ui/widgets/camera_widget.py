# -*- coding: utf-8 -*-
"""Khung camera RGB-D realtime"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COLORS

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class CameraWidget(QWidget):
    def __init__(self, width=512, height=320, camera_index=0, parent=None):
        super().__init__(parent)
        self.display_width = width
        self.display_height = height
        self.camera_index = camera_index
        self.cap = None
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {COLORS['text_dark']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(self.display_width, self.display_height)
        self.video_label.setStyleSheet(f"background: {COLORS['text_dark']}; color: white;")
        self.video_label.setText("📷 Đang kết nối camera...")
        layout.addWidget(self.video_label)

        if CV2_AVAILABLE:
            self.start_camera()
        else:
            self.video_label.setText("⚠️ Cần cài opencv-python")

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(33)  # ~30 FPS
            else:
                self.show_placeholder()
        except Exception as e:
            self.video_label.setText(f"Lỗi camera: {str(e)[:30]}")

    def show_placeholder(self):
        # Tạo ảnh placeholder
        img = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        img[:] = (0, 80, 40)  # Xanh đậm
        h, w = img.shape[:2]
        cv2.putText(img, "Camera Demo", (w//4, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(img, "UTT Robot", (w//3, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        self.display_image(img)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.display_width, self.display_height))
                self.display_image(frame)
            else:
                self.show_placeholder()
        else:
            self.show_placeholder()

    def display_image(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
            bytes_per_line = w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def stop_camera(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
