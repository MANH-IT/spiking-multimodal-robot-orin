# -*- coding: utf-8 -*-
"""
🤖 Robot AI — ĐH Giao thông Vận tải
=====================================
Giao diện Premium — NCKH 2026
"""

from __future__ import annotations
import re, sys, time, random, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFrame, QGridLayout,
    QGroupBox, QProgressBar, QTabWidget, QSpinBox, QScrollArea,
    QMessageBox, QGraphicsView, QGraphicsScene, QSlider, QStackedWidget,
    QSizePolicy, QDesktopWidget, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QTimer, QSize, QRectF, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QBrush, QPen, QLinearGradient,
    QRadialGradient, QPainterPath, QFontDatabase, QIcon, QPixmap,
)

# ── Import building data ────────────────────────────────────
try:
    from data.building_15floors import (
        BUILDING_DATA, ELEVATORS, get_floor, get_room_info,
        get_floor_info, get_direction, search_rooms, best_elevator,
        get_all_floor_names, FLOOR_DISPLAY_NAMES,
    )
    HAS_BUILDING = True
except ImportError:
    HAS_BUILDING = False
    FLOOR_DISPLAY_NAMES = {}


# ============================================================
# 🎨 DESIGN SYSTEM — MODERN DARK GLASSMORPHISM
# ============================================================

C = {
    "bg":           "#0B1120",
    "bg2":          "#111827",
    "surface":      "#1E293B",
    "surface2":     "#334155",
    "glass":        "rgba(30,41,59,0.85)",
    "glass_border": "rgba(71,85,105,0.5)",
    "primary":      "#3B82F6",
    "primary_h":    "#60A5FA",
    "primary_glow": "rgba(59,130,246,0.3)",
    "secondary":    "#8B5CF6",
    "secondary_h":  "#A78BFA",
    "emerald":      "#10B981",
    "emerald_h":    "#34D399",
    "amber":        "#F59E0B",
    "rose":         "#F43F5E",
    "cyan":         "#06B6D4",
    "text":         "#F1F5F9",
    "text2":        "#94A3B8",
    "text3":        "#64748B",
    "border":       "#334155",
    "glow_blue":    "rgba(59,130,246,0.15)",
}

STYLE = f"""
QMainWindow {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 {C["bg"]}, stop:1 {C["bg2"]});
}}
QWidget {{
    color: {C["text"]};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
}}

/* ── Cards ── */
QGroupBox {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 16px;
    margin-top: 12px;
    padding: 20px 16px 16px 16px;
    font-weight: 600;
    font-size: 13px;
    color: {C["text2"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 20px;
    padding: 0 8px;
    color: {C["text2"]};
}}

/* ── Buttons ── */
QPushButton {{
    background-color: {C["surface2"]};
    color: {C["text"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    min-height: 40px;
}}
QPushButton:hover {{
    background-color: {C["primary"]};
    border-color: {C["primary"]};
}}
QPushButton:pressed {{
    background-color: {C["primary_h"]};
}}
QPushButton#primary {{
    background-color: {C["primary"]};
    border: none;
}}
QPushButton#primary:hover {{
    background-color: {C["primary_h"]};
}}
QPushButton#success {{
    background-color: {C["emerald"]};
    border: none;
}}
QPushButton#success:hover {{
    background-color: {C["emerald_h"]};
}}
QPushButton#danger {{
    background-color: {C["rose"]};
    border: none;
}}

/* ── Input ── */
QLineEdit {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 12px 16px;
    color: {C["text"]};
    font-size: 15px;
    min-height: 44px;
    selection-background-color: {C["primary"]};
}}
QLineEdit:focus {{
    border-color: {C["primary"]};
    background-color: {C["surface2"]};
}}

QTextEdit {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 14px;
    color: {C["text"]};
    font-size: 14px;
    selection-background-color: {C["primary"]};
}}

/* ── Tabs ── */
QTabWidget::pane {{
    border: 1px solid {C["border"]};
    border-radius: 12px;
    background-color: {C["surface"]};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {C["bg2"]};
    color: {C["text3"]};
    padding: 12px 28px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 3px;
    font-weight: 600;
    font-size: 13px;
    min-height: 36px;
    border: 1px solid {C["border"]};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background-color: {C["surface"]};
    color: {C["primary"]};
    border-bottom: 2px solid {C["primary"]};
}}
QTabBar::tab:hover:!selected {{
    background-color: {C["surface2"]};
    color: {C["text"]};
}}

/* ── Progress ── */
QProgressBar {{
    background-color: {C["bg2"]};
    border: none;
    border-radius: 6px;
    text-align: center;
    color: {C["text"]};
    font-size: 11px;
    font-weight: bold;
    min-height: 12px;
    max-height: 12px;
}}
QProgressBar::chunk {{
    background-color: {C["primary"]};
    border-radius: 6px;
}}

/* ── SpinBox ── */
QSpinBox {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 8px 12px;
    color: {C["text"]};
    font-size: 16px;
    font-weight: bold;
    min-height: 36px;
}}

/* ── ScrollBar ── */
QScrollBar:vertical {{
    background: {C["bg2"]};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C["surface2"]};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C["text3"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


# ============================================================
# 🤖 ROBOT FACE — ANIMATED
# ============================================================

class RobotFaceWidget(QWidget):
    MOODS = {
        "happy":     {"eye_scale": 1.0,  "mouth": "smile",  "color": "#3B82F6"},
        "thinking":  {"eye_scale": 0.7,  "mouth": "flat",   "color": "#F59E0B"},
        "listening": {"eye_scale": 1.2,  "mouth": "open",   "color": "#10B981"},
        "error":     {"eye_scale": 0.5,  "mouth": "frown",  "color": "#F43F5E"},
        "idle":      {"eye_scale": 1.0,  "mouth": "flat",   "color": "#64748B"},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mood = "happy"
        self.blink = False
        self.pulse = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)
        self._bt = QTimer(self)
        self._bt.timeout.connect(self._do_blink)
        self._bt.start(3500)
        self.setFixedSize(160, 160)
        self.setCursor(Qt.PointingHandCursor)

    def set_mood(self, m):
        self.mood = m if m in self.MOODS else "idle"
        self.update()

    def _tick(self):
        self.pulse = (self.pulse + 0.025) % 1.0
        self.update()

    def _do_blink(self):
        self.blink = True
        self.update()
        QTimer.singleShot(120, self._end_blink)

    def _end_blink(self):
        self.blink = False
        self.update()

    def mousePressEvent(self, e):
        ms = [m for m in self.MOODS if m != self.mood]
        self.set_mood(random.choice(ms))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 8
        mi = self.MOODS[self.mood]
        color = QColor(mi["color"])

        # Outer glow
        gr = int(r * (1.15 + 0.06 * math.sin(self.pulse * 2 * math.pi)))
        g = QRadialGradient(cx, cy, gr)
        g.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 40))
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(g))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - gr, cy - gr, gr * 2, gr * 2)

        # Face circle
        fg = QRadialGradient(cx, cy - r // 3, int(r * 1.5))
        fg.setColorAt(0, QColor(C["surface2"]))
        fg.setColorAt(1, QColor(C["bg"]))
        p.setBrush(QBrush(fg))
        p.setPen(QPen(color, 3))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # Eyes
        es = mi["eye_scale"]
        ew = int(r * 0.2 * es)
        eh = int(r * 0.28 * es) if not self.blink else 3
        ey = cy - r // 5
        p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        for ex in [cx - r // 3, cx + r // 3]:
            p.drawRoundedRect(ex - ew, ey - eh, ew * 2, eh * 2, ew, ew)
            p.setBrush(QBrush(QColor(255, 255, 255, 180)))
            hl = max(ew // 3, 2)
            p.drawEllipse(ex - hl + 3, ey - hl - 3, hl * 2, hl * 2)
            p.setBrush(QBrush(color))

        # Mouth
        my = cy + r // 3
        mw = r // 2
        p.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        mt = mi["mouth"]
        if mt == "smile":
            path = QPainterPath()
            path.moveTo(cx - mw, my)
            path.quadTo(cx, my + r // 3, cx + mw, my)
            p.drawPath(path)
        elif mt == "frown":
            path = QPainterPath()
            path.moveTo(cx - mw, my + r // 5)
            path.quadTo(cx, my - r // 5, cx + mw, my + r // 5)
            p.drawPath(path)
        elif mt == "open":
            p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
            p.drawEllipse(cx - mw // 2, my - 8, mw, 22)
        else:
            p.drawLine(cx - mw, my, cx + mw, my)
        p.end()


# ============================================================
# 📹 CAMERA WIDGET
# ============================================================

class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 240)
        self.frame_count = 0
        self.detections = [
            {"name": "Người", "conf": 0.95, "x": 0.12, "y": 0.15, "w": 0.22, "h": 0.55},
            {"name": "Cửa",  "conf": 0.88, "x": 0.58, "y": 0.08, "w": 0.18, "h": 0.65},
            {"name": "Bảng", "conf": 0.76, "x": 0.38, "y": 0.03, "w": 0.12, "h": 0.15},
        ]
        self.scan_pos = 0
        self._t = QTimer(self)
        self._t.timeout.connect(self._tick)
        self._t.start(33)

    def _tick(self):
        self.frame_count += 1
        self.scan_pos = (self.scan_pos + 2) % max(self.height(), 1)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # BG with subtle gradient
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor(12, 18, 30))
        bg.setColorAt(1, QColor(8, 12, 24))
        p.fillRect(0, 0, w, h, QBrush(bg))

        # Grid
        p.setPen(QPen(QColor(C["border"]), 1, Qt.DotLine))
        for x in range(0, w, 60):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, 60):
            p.drawLine(0, y, w, y)

        # Scan line
        g = QLinearGradient(0, self.scan_pos - 20, 0, self.scan_pos + 20)
        g.setColorAt(0, QColor(59, 130, 246, 0))
        g.setColorAt(0.5, QColor(59, 130, 246, 50))
        g.setColorAt(1, QColor(59, 130, 246, 0))
        p.fillRect(0, self.scan_pos - 20, w, 40, QBrush(g))

        # Detections
        for d in self.detections:
            bx, by = int(w * d["x"]), int(h * d["y"])
            bw, bh = int(w * d["w"]), int(h * d["h"])
            c = d["conf"]
            color = QColor(C["emerald"]) if c > 0.9 else QColor(C["amber"]) if c > 0.7 else QColor(C["rose"])

            # Corner brackets
            p.setPen(QPen(color, 2))
            p.setBrush(Qt.NoBrush)
            cl = min(bw, bh) // 4
            for cx_, cy_, dx, dy in [(bx, by, 1, 1), (bx + bw, by, -1, 1),
                                      (bx, by + bh, 1, -1), (bx + bw, by + bh, -1, -1)]:
                p.drawLine(cx_, cy_, cx_ + cl * dx, cy_)
                p.drawLine(cx_, cy_, cx_, cy_ + cl * dy)

            # Label pill
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 200)))
            lbl_w = max(len(d["name"]) * 9 + 40, 70)
            p.drawRoundedRect(bx, by - 24, lbl_w, 22, 6, 6)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(QRectF(bx + 4, by - 24, lbl_w - 8, 22), Qt.AlignCenter,
                       f"{d['name']} {c:.0%}")

        # Top-left info
        p.setPen(QColor(C["text2"]))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(10, 18, f"📹 CAMERA  ▪  Frame #{self.frame_count}")
        # Bottom-right
        p.drawText(QRectF(0, h - 22, w - 10, 20), Qt.AlignRight, "DepthAwareSNN • RGB-D")

        # Border radius clip
        p.setPen(QPen(QColor(C["border"]), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(0, 0, w - 1, h - 1, 12, 12)
        p.end()


# ============================================================
# 💬 CHAT WIDGET
# ============================================================

class ChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Chat display
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C["bg2"]};
                border: 1px solid {C["border"]};
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.6;
            }}
        """)
        layout.addWidget(self.display, stretch=1)

        # Input row
        row = QHBoxLayout()
        row.setSpacing(8)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Nhập tin nhắn... (vd: 'Chỉ đường đến phòng 302')")
        self.input.returnPressed.connect(self._on_send)
        row.addWidget(self.input, stretch=1)

        self.btn_send = QPushButton("Gửi")
        self.btn_send.setObjectName("primary")
        self.btn_send.setFixedSize(80, 48)
        self.btn_send.clicked.connect(self._on_send)
        row.addWidget(self.btn_send)

        self.btn_mic = QPushButton("🎤")
        self.btn_mic.setObjectName("success")
        self.btn_mic.setFixedSize(48, 48)
        self.btn_mic.setFont(QFont("Segoe UI", 18))
        self.btn_mic.setStyleSheet(f"""
            QPushButton {{
                background-color: {C["emerald"]};
                border: none;
                border-radius: 24px;
                font-size: 20px;
            }}
            QPushButton:hover {{ background-color: {C["emerald_h"]}; }}
        """)
        self.btn_mic.clicked.connect(self._on_mic)
        row.addWidget(self.btn_mic)

        layout.addLayout(row)
        self.on_user_message = None

    def _on_send(self):
        t = self.input.text().strip()
        if t:
            self.add_message("user", t)
            self.input.clear()
            if self.on_user_message:
                self.on_user_message(t)

    def _on_mic(self):
        self.add_message("system", "🎤 Chức năng giọng nói đang phát triển...")

    def set_input_text(self, t):
        self.input.setText(t)
        self._on_send()

    def add_message(self, role, text):
        ts = time.strftime("%H:%M")
        text_html = text.replace("\n", "<br>")
        if role == "user":
            html = (f'<div style="text-align:right;margin:8px 0;">'
                    f'<span style="background:{C["primary"]};color:white;'
                    f'padding:8px 14px;border-radius:12px 12px 2px 12px;'
                    f'display:inline-block;font-size:14px;">{text_html}</span>'
                    f'<br><span style="color:{C["text3"]};font-size:10px;">{ts}</span></div>')
        elif role == "robot":
            html = (f'<div style="margin:8px 0;">'
                    f'<span style="color:{C["primary"]};font-size:11px;font-weight:bold;">🤖 Robot</span><br>'
                    f'<span style="background:{C["surface"]};color:{C["text"]};'
                    f'padding:8px 14px;border-radius:2px 12px 12px 12px;'
                    f'display:inline-block;font-size:14px;border:1px solid {C["border"]};">'
                    f'{text_html}</span>'
                    f'<br><span style="color:{C["text3"]};font-size:10px;">{ts}</span></div>')
        else:
            html = (f'<div style="text-align:center;margin:6px 0;">'
                    f'<span style="color:{C["text3"]};font-size:12px;">{text_html}</span></div>')
        self.display.append(html)
        sb = self.display.verticalScrollBar()
        sb.setValue(sb.maximum())


# ============================================================
# 🗺️ BUILDING MAP
# ============================================================

class FloorMapPainter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.floor = 1
        self.highlight_room = None
        self.on_room_click = None
        self.room_rects = {}
        self.setCursor(Qt.PointingHandCursor)

    def set_floor(self, f):
        self.floor = f
        self.highlight_room = None
        self.room_rects = {}
        self.update()

    def mousePressEvent(self, e):
        for rn, rect in self.room_rects.items():
            if rect.contains(e.pos().x(), e.pos().y()):
                self.highlight_room = rn
                self.update()
                if self.on_room_click:
                    self.on_room_click(rn)
                return

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(C["bg2"]))

        if not HAS_BUILDING:
            p.setPen(QColor(C["text3"]))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "Không có dữ liệu tòa nhà")
            p.end()
            return

        fi = get_floor_info(self.floor)
        if not fi or not fi.rooms:
            p.setPen(QColor(C["text3"]))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, f"Tầng {self.floor}: Không có phòng")
            p.end()
            return

        rooms = list(fi.rooms.values())
        n = len(rooms)
        if n == 0:
            p.end()
            return

        # Grid layout
        cols = min(max(3, int(math.ceil(math.sqrt(n * 1.5)))), 6)
        rows_ = math.ceil(n / cols)

        margin = 12
        gap = 6
        cw = (w - margin * 2 - gap * (cols - 1)) / cols
        ch = (h - margin * 2 - gap * (rows_ - 1)) / rows_
        cw = min(cw, 220)
        ch = min(ch, 90)

        self.room_rects = {}
        for i, room in enumerate(rooms):
            row = i // cols
            col = i % cols

            x = margin + col * (cw + gap)
            y = margin + row * (ch + gap)
            rect = QRectF(x, y, cw, ch)
            self.room_rects[room.number] = rect

            # Room color by function
            func = room.function
            is_highlight = room.number == self.highlight_room

            if is_highlight:
                bg_color = QColor(C["primary"])
                border_color = QColor(C["primary_h"])
                text_color = QColor("white")
            elif func == "meeting":
                bg_color = QColor("#1E3A5F")
                border_color = QColor(C["cyan"])
                text_color = QColor(C["cyan"])
            elif func == "office":
                bg_color = QColor("#1A2E40")
                border_color = QColor(C["amber"])
                text_color = QColor(C["amber"])
            elif func == "research":
                bg_color = QColor("#1E2940")
                border_color = QColor(C["secondary"])
                text_color = QColor(C["secondary"])
            elif func in ("server", "security"):
                bg_color = QColor("#2A1520")
                border_color = QColor(C["rose"])
                text_color = QColor(C["rose"])
            else:
                bg_color = QColor(C["surface"])
                border_color = QColor(C["border"])
                text_color = QColor(C["text2"])

            # Draw room
            p.setPen(QPen(border_color, 2 if is_highlight else 1))
            p.setBrush(QBrush(bg_color))
            p.drawRoundedRect(rect, 8, 8)

            # Room number
            dn = room.display_number or str(room.number)
            p.setPen(text_color)
            p.setFont(QFont("Segoe UI", 11, QFont.Bold))
            p.drawText(QRectF(x + 6, y + 4, cw - 12, 20), Qt.AlignLeft, f"P.{dn}")

            # Room name (truncate if needed)
            name = room.name
            if len(name) > 28:
                name = name[:26] + "..."
            p.setFont(QFont("Segoe UI", 9))
            fc = QColor("white") if is_highlight else QColor(C["text2"])
            p.setPen(fc)
            p.drawText(QRectF(x + 6, y + 22, cw - 12, ch - 28),
                       Qt.AlignLeft | Qt.TextWordWrap, name)

        # Legend
        ly = h - 28
        legend = [("Họp", C["cyan"]), ("Văn phòng", C["amber"]),
                  ("Bộ môn", C["text2"]), ("Nghiên cứu", C["secondary"])]
        lx = margin
        p.setFont(QFont("Segoe UI", 9))
        for name, clr in legend:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(clr)))
            p.drawRoundedRect(lx, ly + 4, 10, 10, 3, 3)
            p.setPen(QColor(C["text3"]))
            p.drawText(lx + 14, ly + 14, name)
            lx += len(name) * 8 + 30

        p.end()


class BuildingMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_floor = 1
        self.selected_room = None
        self.on_direction_request = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header
        h1 = QHBoxLayout()
        h1.setSpacing(10)
        h1.addWidget(QLabel("🏢"))
        self.floor_spin = QSpinBox()
        self.floor_spin.setRange(1, 15)
        self.floor_spin.setValue(1)
        self.floor_spin.setFixedSize(80, 44)
        self.floor_spin.valueChanged.connect(self._change_floor)
        h1.addWidget(self.floor_spin)

        # Quick buttons
        for label, fl in [("T1", 1), ("T3", 3), ("T4", 4), ("T9", 9),
                          ("12A", 13), ("T14", 14), ("T15", 15)]:
            b = QPushButton(label)
            b.setFixedSize(52, 40)
            b.clicked.connect(lambda _, x=fl: self.floor_spin.setValue(x))
            h1.addWidget(b)

        h1.addStretch()

        # Search
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Tìm phòng...")
        self.search.setFixedWidth(200)
        self.search.returnPressed.connect(self._search)
        h1.addWidget(self.search)

        btn_s = QPushButton("Tìm")
        btn_s.setObjectName("primary")
        btn_s.setFixedSize(60, 40)
        btn_s.clicked.connect(self._search)
        h1.addWidget(btn_s)

        layout.addLayout(h1)

        # Floor title
        self.title = QLabel("TẦNG 1: HÀNH CHÍNH")
        self.title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.title.setStyleSheet(f"color:{C['primary']};padding:4px 0;")
        layout.addWidget(self.title)

        # Map
        self.map = FloorMapPainter()
        self.map.setMinimumHeight(300)
        self.map.on_room_click = self._on_room
        layout.addWidget(self.map, stretch=1)

        # Info + Direction
        bot = QHBoxLayout()
        bot.setSpacing(10)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(100)
        self.info.setPlaceholderText("👆 Chạm vào phòng để xem thông tin...")
        bot.addWidget(self.info, stretch=2)

        self.btn_dir = QPushButton("🚶 Chỉ đường")
        self.btn_dir.setObjectName("success")
        self.btn_dir.setFixedSize(160, 60)
        self.btn_dir.setEnabled(False)
        self.btn_dir.clicked.connect(self._dir_click)
        bot.addWidget(self.btn_dir)
        layout.addLayout(bot)
        self._change_floor(1)

    def _change_floor(self, f):
        self.current_floor = f
        if HAS_BUILDING:
            fi = get_floor_info(f)
            if fi:
                self.title.setText(fi.display_name)
                self.title.setStyleSheet(f"color:{fi.color};padding:4px 0;font-size:16px;font-weight:bold;")
        self.map.set_floor(f)
        self.info.clear()
        self.btn_dir.setEnabled(False)

    def _on_room(self, rn):
        self.selected_room = rn
        self.btn_dir.setEnabled(True)
        if HAS_BUILDING:
            ri = get_room_info(rn)
            fi = get_floor_info(get_floor(rn))
            if ri and fi:
                dn = ri.display_number or str(rn)
                html = f"<b style='font-size:14px;color:{C['primary']};'>P.{dn}: {ri.name}</b><br>"
                html += f"<span style='color:{C['text2']};'>{fi.display_name}</span><br>"
                if ri.department:
                    html += f"🏛️ <b>{ri.department}</b><br>"
                if ri.manager:
                    html += f"👤 {ri.manager}<br>"
                if ri.phone:
                    html += f"📞 {ri.phone}<br>"
                if ri.email:
                    html += f"📧 {ri.email}<br>"
                self.info.setHtml(html)

    def _dir_click(self):
        if self.selected_room and self.on_direction_request:
            self.on_direction_request(self.selected_room)

    def _search(self):
        t = self.search.text().strip()
        if re.match(r'12a\d{2}', t.lower()):
            rn = 1300 + int(t[3:5])
            self.floor_spin.setValue(13)
            self._on_room(rn)
            self.map.highlight_room = rn
            self.map.update()
        elif t.isdigit():
            rn = int(t)
            fl = get_floor(rn)
            self.floor_spin.setValue(fl)
            self._on_room(rn)
            self.map.highlight_room = rn
            self.map.update()
        elif HAS_BUILDING:
            res = search_rooms(t)
            if res:
                html = f"<b>🔍 Tìm thấy {len(res)} phòng:</b><br>"
                for r in res[:8]:
                    dn = r.display_number or str(r.number)
                    html += f"• P.{dn}: {r.name}<br>"
                self.info.setHtml(html)
            else:
                self.info.setText(f"Không tìm thấy '{t}'")


# ============================================================
# 📊 STATUS BAR — COMPACT
# ============================================================

class StatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            background-color: {C["surface"]};
            border-radius: 10px;
            border: 1px solid {C["border"]};
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(24)

        self.items = {}
        defaults = [
            ("🤖", "Robot", "Online"),
            ("💻", "Device", "Jetson AGX"),
            ("📡", "Status", "Sẵn sàng"),
            ("⏱️", "FPS", "30"),
            ("🕒", "Time", time.strftime("%H:%M")),
        ]
        for icon, key, val in defaults:
            lbl = QLabel(f"{icon} <b>{val}</b>")
            lbl.setStyleSheet(f"font-size:12px;color:{C['text2']};")
            lay.addWidget(lbl)
            self.items[key] = lbl
        lay.addStretch()

    def update_item(self, key, val, icon=""):
        if key in self.items:
            ic = icon or ""
            self.items[key].setText(f"{ic} <b>{val}</b>")


# ============================================================
# 🏠 MAIN WINDOW — PREMIUM LAYOUT
# ============================================================

class RobotTouchMainWindow(QMainWindow):
    def __init__(self, fullscreen=False):
        super().__init__()
        self.setWindowTitle("🤖 Robot AI — ĐH Giao thông Vận tải")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(STYLE)
        self._center()
        self._build_ui()
        self._timers()
        self._welcome()
        if fullscreen:
            self.showFullScreen()

    def _center(self):
        qr = self.frameGeometry()
        qr.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(qr.topLeft())

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)
        root.setSpacing(10)
        root.setContentsMargins(16, 12, 16, 10)

        # ═══ HEADER ═══
        hdr = QHBoxLayout()

        # Logo
        logo = QLabel("🤖")
        logo.setFont(QFont("Segoe UI", 28))
        hdr.addWidget(logo)

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        t1 = QLabel("ROBOT AI")
        t1.setFont(QFont("Segoe UI", 18, QFont.Bold))
        t1.setStyleSheet(f"color:{C['text']};")
        title_col.addWidget(t1)
        t2 = QLabel("Trường Đại học Giao thông Vận tải")
        t2.setFont(QFont("Segoe UI", 11))
        t2.setStyleSheet(f"color:{C['text2']};")
        title_col.addWidget(t2)
        hdr.addLayout(title_col)
        hdr.addStretch()

        # Time
        self.time_lbl = QLabel()
        self.time_lbl.setFont(QFont("Segoe UI", 14))
        self.time_lbl.setStyleSheet(f"color:{C['text2']};")
        hdr.addWidget(self.time_lbl)

        # Fullscreen
        fs_btn = QPushButton("⛶")
        fs_btn.setFixedSize(40, 40)
        fs_btn.clicked.connect(self._toggle_fs)
        hdr.addWidget(fs_btn)

        root.addLayout(hdr)

        # ═══ MAIN CONTENT — 2 COLUMN ═══
        body = QHBoxLayout()
        body.setSpacing(12)

        # ──── LEFT (sidebar) ────
        left = QVBoxLayout()
        left.setSpacing(10)

        # Robot face card
        face_card = QGroupBox("ROBOT")
        fl = QVBoxLayout(face_card)
        fl.setAlignment(Qt.AlignCenter)
        self.face = RobotFaceWidget()
        fl.addWidget(self.face, alignment=Qt.AlignCenter)
        self.mood_label = QLabel("HAPPY")
        self.mood_label.setAlignment(Qt.AlignCenter)
        self.mood_label.setStyleSheet(f"color:{C['primary']};font-weight:bold;font-size:11px;")
        fl.addWidget(self.mood_label)
        left.addWidget(face_card)

        # Quick actions
        qa_card = QGroupBox("⚡ TÁC VỤ NHANH")
        ql = QVBoxLayout(qa_card)
        ql.setSpacing(6)

        actions = [
            ("🚶 Chỉ đường P.302",    "Chỉ đường đến phòng 302"),
            ("🔍 Tìm phòng AI",       "Tìm phòng AI"),
            ("🏢 Tầng 3 — CNTT",      "Tầng 3 có gì?"),
            ("🏢 Tầng 9 — Công trình", "Tầng 9 có gì?"),
            ("🛗 Thang máy",           "Thang máy nào lên tầng 15?"),
            ("📋 Phòng họp",           "Tìm phòng họp"),
        ]
        for label, cmd in actions:
            b = QPushButton(label)
            b.setFixedHeight(42)
            b.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 14px;
                    font-size: 13px;
                }}
            """)
            b.clicked.connect(lambda _, c=cmd: self._quick(c))
            ql.addWidget(b)
        left.addWidget(qa_card)

        # System info compact
        info_card = QGroupBox("ℹ️ HỆ THỐNG")
        il = QVBoxLayout(info_card)
        il.setSpacing(4)
        sys_items = [
            ("👁️ Vision", "DepthAwareSNN"),
            ("🗣️ NLP", "SpikingLM"),
            ("🔗 Fusion", "CrossModalAttn"),
            ("🏢 Tòa nhà", "15 tầng"),
        ]
        for k, v in sys_items:
            r = QHBoxLayout()
            lk = QLabel(k)
            lk.setStyleSheet(f"color:{C['text3']};font-size:11px;")
            lv = QLabel(v)
            lv.setStyleSheet(f"color:{C['text']};font-size:12px;font-weight:bold;")
            r.addWidget(lk)
            r.addStretch()
            r.addWidget(lv)
            il.addLayout(r)

        # Performance bars
        il.addWidget(QLabel(""))
        self.perf = {}
        for name, val, color in [("CPU", 32, C["primary"]), ("GPU", 45, C["emerald"]),
                                  ("RAM", 60, C["amber"])]:
            r = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{C['text3']};font-size:11px;min-width:30px;")
            r.addWidget(lbl)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(val)
            bar.setFormat(f"{val}%")
            bar.setStyleSheet(f"QProgressBar::chunk{{background:{color};border-radius:6px;}}")
            r.addWidget(bar)
            il.addLayout(r)
            self.perf[name] = bar

        left.addWidget(info_card)
        left.addStretch()
        body.addLayout(left, stretch=2)

        # ──── CENTER (main content) ────
        center = QVBoxLayout()
        center.setSpacing(8)

        self.tabs = QTabWidget()

        # Tab 1: Camera + Chat
        tab1 = QWidget()
        t1 = QVBoxLayout(tab1)
        t1.setSpacing(8)
        t1.setContentsMargins(8, 8, 8, 8)

        self.camera = CameraWidget()
        self.camera.setFixedHeight(260)
        t1.addWidget(self.camera)

        self.chat = ChatWidget()
        self.chat.on_user_message = self._on_msg
        t1.addWidget(self.chat, stretch=1)

        self.tabs.addTab(tab1, "📹 Camera & Chat")

        # Tab 2: Map
        tab2 = QWidget()
        t2 = QVBoxLayout(tab2)
        t2.setContentsMargins(8, 8, 8, 8)
        self.bmap = BuildingMapWidget()
        self.bmap.on_direction_request = self._dir_from_map
        t2.addWidget(self.bmap)
        self.tabs.addTab(tab2, "🗺️ Bản đồ tòa nhà")

        center.addWidget(self.tabs)
        body.addLayout(center, stretch=5)

        root.addLayout(body)

        # ═══ STATUS BAR ═══
        self.status = StatusBar()
        root.addWidget(self.status)

    def _timers(self):
        t1 = QTimer(self)
        t1.timeout.connect(self._tick_time)
        t1.start(1000)
        self._tick_time()

        t2 = QTimer(self)
        t2.timeout.connect(self._tick_perf)
        t2.start(3000)

    def _tick_time(self):
        self.time_lbl.setText(f"🕒 {time.strftime('%H:%M:%S  •  %d/%m/%Y')}")
        self.status.update_item("Time", time.strftime("%H:%M"), "🕒")

    def _tick_perf(self):
        cpu = random.randint(20, 50)
        gpu = random.randint(30, 60)
        ram = random.randint(50, 75)
        self.perf["CPU"].setValue(cpu)
        self.perf["CPU"].setFormat(f"{cpu}%")
        self.perf["GPU"].setValue(gpu)
        self.perf["GPU"].setFormat(f"{gpu}%")
        self.perf["RAM"].setValue(ram)
        self.perf["RAM"].setFormat(f"{ram}%")
        self.status.update_item("FPS", str(random.randint(28, 32)), "⏱️")

    def _welcome(self):
        self.camera.detections = [
            {"name": "Người", "conf": 0.95, "x": 0.12, "y": 0.15, "w": 0.22, "h": 0.55},
            {"name": "Cửa",  "conf": 0.88, "x": 0.58, "y": 0.08, "w": 0.18, "h": 0.65},
        ]
        self.chat.add_message("robot",
            "Xin chào! Tôi là Robot AI của Trường ĐH Giao thông Vận tải. 👋\n\n"
            "Tôi có thể giúp bạn:\n"
            "• 🚶 Chỉ đường trong tòa nhà 15 tầng\n"
            "• 🏢 Xem thông tin các tầng, khoa, bộ môn\n"
            "• 🔍 Tìm kiếm phòng theo từ khóa\n"
            "• 🗺️ Xem bản đồ tòa nhà ở tab thứ 2\n\n"
            "Hãy thử: 'Chỉ đường đến phòng 302' hoặc 'Tìm phòng AI'"
        )

    def _toggle_fs(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ── Message handling ──

    def _on_msg(self, text):
        self.face.set_mood("thinking")
        self.mood_label.setText("THINKING")
        QTimer.singleShot(400, lambda: self._respond(text))

    def _respond(self, text):
        self.face.set_mood("happy")
        self.mood_label.setText("HAPPY")
        tl = text.lower()

        if "chỉ đường" in tl or "đường đến" in tl or "dẫn đến" in tl:
            self._do_direction(tl)
        elif "tầng" in tl and ("có gì" in tl or "thông tin" in tl):
            self._do_floor(tl)
        elif "tìm" in tl or "ở đâu" in tl:
            self._do_search(tl)
        elif "thang máy" in tl:
            self._do_elevator()
        elif "chào" in tl or "hello" in tl:
            self.chat.add_message("robot", "Xin chào! 😊 Tôi có thể giúp gì cho bạn?")
        elif "cảm ơn" in tl:
            self.chat.add_message("robot", "Không có gì! Chúc bạn một ngày tốt lành! 😊")
        else:
            self.chat.add_message("robot",
                "Tôi chưa hiểu ý bạn. Bạn có thể thử:\n"
                "• 'Chỉ đường đến phòng 302'\n"
                "• 'Tầng 3 có gì?'\n"
                "• 'Tìm phòng AI'\n"
                "• 'Thang máy nào lên tầng 15?'")

    def _do_direction(self, text):
        rm = re.search(r'12a\s*(\d{1,2})', text, re.IGNORECASE)
        rm2 = re.search(r'phòng\s*(\d{3,4})', text)
        if rm:
            rn = 1300 + int(rm.group(1))
        elif rm2:
            rn = int(rm2.group(1))
        else:
            self.chat.add_message("robot", "Bạn muốn đến phòng nào? (vd: 302, 1501, 12A05)")
            return
        if HAS_BUILDING:
            self.chat.add_message("robot", get_direction(1, rn))
            self.tabs.setCurrentIndex(1)
            fl = get_floor(rn)
            self.bmap.floor_spin.setValue(fl)
            self.bmap.map.highlight_room = rn
            self.bmap.map.update()

    def _do_floor(self, text):
        m = re.search(r'tầng\s*(12a|\d{1,2})', text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            fl = 13 if raw.lower() == "12a" else int(raw)
            if HAS_BUILDING:
                fi = get_floor_info(fl)
                if fi:
                    r = f"🏢 {fi.display_name}\n\n📋 DANH SÁCH PHÒNG:\n"
                    for rn, ri in fi.rooms.items():
                        dn = ri.display_number or str(rn)
                        r += f"  • P.{dn}: {ri.name}\n"
                    self.chat.add_message("robot", r)
                    self.tabs.setCurrentIndex(1)
                    self.bmap.floor_spin.setValue(fl)
                else:
                    self.chat.add_message("robot", f"Không có tầng {raw}")

    def _do_search(self, text):
        kw = text
        for w in ["tìm", "ở đâu", "phòng", "khoa", "bộ môn", "?"]:
            kw = kw.replace(w, "")
        kw = kw.strip()
        if HAS_BUILDING and kw:
            res = search_rooms(kw)
            if res:
                r = f"🔍 Tìm thấy {len(res)} phòng:\n\n"
                for rm in res[:10]:
                    dn = rm.display_number or str(rm.number)
                    fl = FLOOR_DISPLAY_NAMES.get(get_floor(rm.number), "?")
                    r += f"  • P.{dn}: {rm.name} (Tầng {fl})\n"
                self.chat.add_message("robot", r)
            else:
                self.chat.add_message("robot", f"Không tìm thấy '{kw}'")
        else:
            self.chat.add_message("robot", "Bạn muốn tìm gì?\n• 'Tìm CNTT'\n• 'Tìm phòng họp'\n• 'Tìm AI'")

    def _do_elevator(self):
        r = "🛗 THANG MÁY TÒA NHÀ:\n\n"
        if HAS_BUILDING:
            for eid, e in ELEVATORS.items():
                r += f"  • Thang {eid}: {e['note']}\n"
        r += "\n💡 Tầng 12A-15 nên dùng thang A."
        self.chat.add_message("robot", r)

    def _dir_from_map(self, rn):
        self.tabs.setCurrentIndex(0)
        self.chat.set_input_text(f"Chỉ đường đến phòng {rn}")

    def _quick(self, cmd):
        self.tabs.setCurrentIndex(0)
        self.chat.set_input_text(cmd)


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robot AI — ĐH GTVT")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    w = RobotTouchMainWindow(fullscreen=args.fullscreen)
    w.show()

    print("=" * 50)
    print("🤖 Robot AI — ĐH Giao thông Vận tải")
    print("✅ Giao diện đã sẵn sàng!")
    print("=" * 50)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()