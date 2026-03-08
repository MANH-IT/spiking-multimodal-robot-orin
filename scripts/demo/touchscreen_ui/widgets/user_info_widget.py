# -*- coding: utf-8 -*-
"""Hiển thị thông tin người dùng (MSSV, tên, khoa)"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COLORS, FONTS, SAMPLE_USERS


class UserInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_user = SAMPLE_USERS["default"]
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {COLORS['text_dark']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.name_label = QLabel("👤 " + self.current_user["name"])
        self.name_label.setFont(QFont("Arial", FONTS["body"]))
        self.name_label.setStyleSheet(f"color: {COLORS['text_light']}; font-weight: bold;")
        layout.addWidget(self.name_label)

        self.mssv_label = QLabel("MSSV: " + self.current_user["mssv"])
        self.mssv_label.setFont(QFont("Arial", FONTS["small"]))
        self.mssv_label.setStyleSheet(f"color: {COLORS['text_light']};")
        layout.addWidget(self.mssv_label)

        self.faculty_label = QLabel("Khoa: " + self.current_user["faculty"])
        self.faculty_label.setFont(QFont("Arial", FONTS["small"]))
        self.faculty_label.setStyleSheet(f"color: {COLORS['text_light']};")
        layout.addWidget(self.faculty_label)

    def set_user(self, user_data):
        self.current_user = user_data
        self.name_label.setText("👤 " + user_data["name"])
        self.mssv_label.setText("MSSV: " + user_data["mssv"])
        self.faculty_label.setText("Khoa: " + user_data["faculty"])
