# -*- coding: utf-8 -*-
"""
Bàn phím ảo đơn giản cho màn hình cảm ứng
Hiển thị khi nhấn vào ô input (Windows touch keyboard hoặc custom)
"""

from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QVBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COLORS, FONTS


class VirtualKeyboard(QWidget):
    """Bàn phím ảo - các phím cơ bản"""
    key_pressed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setFixedHeight(200)
        self.setStyleSheet(f"""
            QWidget {{ background: {COLORS['background']}; }}
            QPushButton {{
                background: {COLORS['secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                font-size: {FONTS['body']}px;
            }}
            QPushButton:hover {{ background: {COLORS['chat_user_bg']}; }}
        """)

        layout = QVBoxLayout(self)
        grid = QGridLayout()

        # Hàng 1: số
        for i, c in enumerate("1234567890"):
            btn = self._make_key_btn(c)
            grid.addWidget(btn, 0, i)

        # Hàng 2: qwerty...
        row2 = "qwertyuiop"
        for i, c in enumerate(row2):
            btn = self._make_key_btn(c)
            grid.addWidget(btn, 1, i)

        # Hàng 3: asdf...
        row3 = "asdfghjkl"
        for i, c in enumerate(row3):
            btn = self._make_key_btn(c)
            grid.addWidget(btn, 2, i)

        # Hàng 4: zxcv...
        row4 = "zxcvbnm"
        for i, c in enumerate(row4):
            btn = self._make_key_btn(c)
            grid.addWidget(btn, 3, i)

        layout.addLayout(grid)

    def _make_key_btn(self, char):
        btn = QPushButton(char)
        btn.setFixedSize(45, 40)
        btn.setFont(QFont("Arial", FONTS["body"]))
        btn.clicked.connect(lambda _, c=char: self.key_pressed.emit(c))
        return btn
