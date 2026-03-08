#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chạy giao diện màn hình cảm ứng - Robot Đa phương thức ĐH Giao thông Vận tải

Usage:
    python scripts/demo/touchscreen_ui/run_touchscreen.py

Hoặc từ project root:
    python -m scripts.demo.touchscreen_ui.run_touchscreen
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Import from touchscreen_ui
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_window import MainWindow


def main(fullscreen=False):
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Robot UTT - Màn hình cảm ứng")

    # Font mặc định lớn hơn
    font = QFont("Arial", 12)
    app.setFont(font)

    window = MainWindow()

    # Simulate face recognition sau 2 giây (demo)
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(2000, window.simulate_face_recognition)

    if fullscreen:
        window.showFullScreen()
    else:
        window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fullscreen", action="store_true")
    args = p.parse_args()
    main(fullscreen=args.fullscreen)
