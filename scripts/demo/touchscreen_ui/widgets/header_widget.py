# -*- coding: utf-8 -*-
"""Header: Logo, tên robot, pin %, thời gian"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from datetime import datetime
import sys
from pathlib import Path

# Add parent for config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COLORS, FONTS, SCREEN_WIDTH


class HeaderWidget(QWidget):
    def __init__(self, robot_name="Robot UTT", battery=85, parent=None):
        super().__init__(parent)
        self.robot_name = robot_name
        self.battery = battery
        self.setup_ui()

    def setup_ui(self):
        self.setFixedHeight(70)
        self.setStyleSheet(f"""
            QWidget {{
                background: {COLORS['primary']};
                border-radius: 0;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(20)

        # Logo (placeholder - text hoặc icon)
        logo_label = QLabel("UTT")
        logo_label.setStyleSheet(f"""
            color: {COLORS['text_light']};
            font-size: 24px;
            font-weight: bold;
        """)
        logo_label.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(logo_label)

        # Tên robot
        name_label = QLabel(self.robot_name)
        name_label.setStyleSheet(f"""
            color: {COLORS['text_light']};
            font-size: {FONTS['header']}px;
        """)
        name_label.setFont(QFont("Arial", FONTS["header"]))
        layout.addWidget(name_label)
        layout.addStretch()

        # Pin
        self.battery_label = QLabel(f"🔋 {self.battery}%")
        self.battery_label.setStyleSheet(f"color: {COLORS['text_light']}; font-size: {FONTS['body']}px;")
        self.battery_label.setFont(QFont("Arial", FONTS["body"]))
        layout.addWidget(self.battery_label)

        # Thời gian
        self.time_label = QLabel()
        self.time_label.setStyleSheet(f"color: {COLORS['text_light']}; font-size: {FONTS['body']}px;")
        self.time_label.setFont(QFont("Arial", FONTS["body"]))
        layout.addWidget(self.time_label)

        # Timer cập nhật thời gian
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()

    def update_time(self):
        self.time_label.setText(datetime.now().strftime("%H:%M %d/%m"))

    def set_battery(self, value):
        self.battery = max(0, min(100, value))
        self.battery_label.setText(f"🔋 {self.battery}%")
