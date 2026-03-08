# -*- coding: utf-8 -*-
"""
Cửa sổ chính - Giao diện màn hình cảm ứng Robot UTT
Bố cục: Header | Camera(40%) + Chat + Map | Quick buttons
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QDialog, QLabel, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys
from pathlib import Path

# Add parent for imports
PARENT = Path(__file__).resolve().parent
sys.path.insert(0, str(PARENT))

from config import (
    COLORS, FONTS, SCREEN_WIDTH, SCREEN_HEIGHT,
    SAMPLE_USERS, SAMPLE_QA, QUICK_SUGGESTIONS
)
from widgets import (
    HeaderWidget, CameraWidget, ChatWidget,
    SimpleMapWidget, UserInfoWidget
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Đa phương thức - ĐH Giao thông Vận tải")
        self.setFixedSize(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setStyleSheet(f"background: {COLORS['background']};")

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 10, 15, 15)
        main_layout.setSpacing(12)

        # === HEADER ===
        self.header = HeaderWidget(robot_name="Robot UTT", battery=85)
        main_layout.addWidget(self.header)

        # === CONTENT: Camera (40%) + Chat + Map ===
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # Left: Camera + User info (40% width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # Camera - 40% màn hình ~ 512x320
        cam_width = int(SCREEN_WIDTH * 0.4)
        cam_height = int(320)
        self.camera_widget = CameraWidget(width=cam_width, height=cam_height)
        left_layout.addWidget(self.camera_widget)

        # User info (nhận diện khuôn mặt)
        self.user_info = UserInfoWidget()
        left_layout.addWidget(self.user_info)

        content_layout.addWidget(left_panel)

        # Right: Chat + Map
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        # Chat (enable ASR + TTS)
        self.chat_widget = ChatWidget(enable_asr=True, enable_tts=True)
        self.chat_widget.message_sent.connect(self.on_user_message)
        right_layout.addWidget(self.chat_widget, 3)

        # Map
        self.map_widget = SimpleMapWidget(current_room="Sảnh A", destination=None)
        right_layout.addWidget(self.map_widget, 1)

        content_layout.addWidget(right_panel, 1)

        main_layout.addLayout(content_layout)

        # === QUICK SUGGESTIONS - 6 nút ===
        quick_layout = QHBoxLayout()
        quick_layout.setSpacing(10)

        for i, text in enumerate(QUICK_SUGGESTIONS):
            btn = QPushButton(text)
            btn.setMinimumHeight(50)
            btn.setFont(QFont("Arial", FONTS["button"]))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {COLORS['primary']};
                    color: {COLORS['text_light']};
                    border: none;
                    border-radius: 8px;
                    padding: 8px;
                }}
                QPushButton:hover {{
                    background: {COLORS['primary_light']};
                }}
                QPushButton:pressed {{
                    background: {COLORS['accent']};
                }}
            """)
            btn.clicked.connect(lambda checked, t=text: self.on_quick_suggestion(t))
            quick_layout.addWidget(btn)

        main_layout.addLayout(quick_layout)

        # Nút hành động: Dẫn đường, Chi tiết, Hỏi khác
        action_layout = QHBoxLayout()
        action_layout.setSpacing(15)

        nav_btn = QPushButton("🗺️ Dẫn đường")
        detail_btn = QPushButton("📋 Xem chi tiết")
        other_btn = QPushButton("❓ Hỏi câu khác")

        for btn in [nav_btn, detail_btn, other_btn]:
            btn.setMinimumHeight(45)
            btn.setFont(QFont("Arial", FONTS["button"]))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {COLORS['accent']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                }}
                QPushButton:hover {{ background: {COLORS['primary_light']}; }}
            """)

        nav_btn.clicked.connect(self.on_navigate)
        detail_btn.clicked.connect(self.on_view_detail)
        other_btn.clicked.connect(self.on_ask_other)

        action_layout.addWidget(nav_btn)
        action_layout.addWidget(detail_btn)
        action_layout.addWidget(other_btn)

        main_layout.addLayout(action_layout)

        # State
        self.last_question = None
        self.last_answer_data = None
        self.nlu = None
        try:
            from nlp_system.inference import NLUEngine
            self.nlu = NLUEngine()
        except ImportError:
            pass

    def on_user_message(self, text):
        if text and text != "[voice_input]":
            self.process_question(text)

    def on_quick_suggestion(self, text):
        self.chat_widget.add_message(text, is_user=True)
        self.process_question(text)

    def process_question(self, question):
        """Xử lý câu hỏi - NLU thật hoặc SAMPLE_QA"""
        answer_data = None

        if self.nlu:
            result = self.nlu.understand(question)
            if result["intent"] != "unknown":
                answer_data = {
                    "answer": result["response"],
                    "image": None,
                    "room": result.get("room"),
                }

        # Fallback: SAMPLE_QA
        if not answer_data:
            for q, data in SAMPLE_QA.items():
                if q.lower() in question.lower() or question.lower() in q.lower():
                    answer_data = data
                    break

        if not answer_data:
            # Câu trả lời mặc định
            answer_data = {
                "answer": "Xin lỗi, tôi chưa có thông tin chi tiết. Vui lòng liên hệ phòng Công tác Sinh viên (A.101) hoặc truy cập utt.edu.vn để biết thêm.",
                "image": None,
                "room": None,
            }

        self.last_question = question
        self.last_answer_data = answer_data

        # Hiển thị câu trả lời + TTS
        answer_text = answer_data["answer"]
        self.chat_widget.add_robot_response(answer_text, play_tts=True)

        # Cập nhật bản đồ và highlight phòng
        if answer_data.get("room"):
            self.map_widget.update_location("Sảnh A", answer_data["room"])
            self.map_widget.highlight_room(answer_data["room"])

    def on_navigate(self):
        if self.last_answer_data and self.last_answer_data.get("room"):
            room = self.last_answer_data["room"]
            self.map_widget.update_location("Sảnh A", room)
            self.chat_widget.add_robot_response(f"Đang dẫn đường đến phòng {room}...")
        else:
            self.chat_widget.add_robot_response("Vui lòng hỏi về địa điểm cụ thể trước khi dẫn đường.")

    def on_view_detail(self):
        if self.last_answer_data:
            self._show_detail_dialog()
        else:
            self.chat_widget.add_robot_response("Chưa có thông tin chi tiết. Hãy đặt câu hỏi trước.")

    def _show_detail_dialog(self):
        """Hiển thị dialog chi tiết với animation slide-in"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Chi tiết")
        dialog.setFixedSize(600, 400)
        dialog.setStyleSheet(f"""
            QDialog {{
                background: {COLORS['background']};
                border: 2px solid {COLORS['primary']};
                border-radius: 12px;
            }}
        """)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        label = QLabel(self.last_answer_data["answer"])
        label.setWordWrap(True)
        label.setFont(QFont("Arial", FONTS["body"]))
        scroll.setWidget(label)
        layout.addWidget(scroll)
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['primary']};
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: {FONTS['button']}px;
            }}
        """)
        layout.addWidget(close_btn)
        # Hiển thị dialog (animation qua style transition)
        dialog.exec_()

    def on_ask_other(self):
        self.chat_widget.add_robot_response("Bạn muốn hỏi gì tiếp theo? Tôi sẵn sàng hỗ trợ!")

    def simulate_face_recognition(self):
        """Mô phỏng nhận diện khuôn mặt - set user mẫu"""
        self.user_info.set_user(SAMPLE_USERS["20241234"])
