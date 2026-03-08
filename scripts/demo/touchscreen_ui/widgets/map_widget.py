# -*- coding: utf-8 -*-
"""Khung bản đồ tòa nhà - vị trí hiện tại, điểm đến"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COLORS, FONTS


class SimpleMapWidget(QWidget):
    def __init__(self, current_room="A.101", destination=None, parent=None):
        super().__init__(parent)
        self.current_room = current_room
        self.destination = destination
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {COLORS['background']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Sơ đồ tòa nhà")
        title.setFont(QFont("Arial", FONTS["title"]))
        title.setStyleSheet(f"color: {COLORS['text_dark']}; font-weight: bold;")
        layout.addWidget(title)

        # Map placeholder - sơ đồ đơn giản
        self.map_label = QLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setMinimumHeight(120)
        self.map_label.setStyleSheet(f"""
            background: white;
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
        """)
        self.map_label.setText("🗺️ Bản đồ\n\n📍 Vị trí: " + self.current_room +
                              (f"\n🎯 Đến: {self.destination}" if self.destination else ""))
        self.map_label.setFont(QFont("Arial", FONTS["small"]))
        layout.addWidget(self.map_label)

    def update_location(self, current, destination=None):
        self.current_room = current
        self.destination = destination
        self._update_display()

    def highlight_room(self, room: str):
        """Highlight phòng đích trên bản đồ"""
        self.destination = room
        self._update_display(highlight=room)

    def _update_display(self, highlight=None):
        room = highlight or self.destination
        text = "🗺️ Sơ đồ tòa nhà\n\n📍 Vị trí: " + self.current_room
        if room:
            text += f"\n🎯 Đến: {room}"
            if highlight:
                text += " ✨"
        self.map_label.setText(text)
