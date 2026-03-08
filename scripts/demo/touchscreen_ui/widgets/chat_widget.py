# -*- coding: utf-8 -*-
"""Khung hội thoại - lịch sử chat giữa người và robot"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel,
    QHBoxLayout, QLineEdit, QPushButton, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COLORS, FONTS

# Add project root for nlp_system
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ChatBubble(QFrame):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['chat_user_bg'] if is_user else COLORS['chat_robot_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setFont(QFont("Arial", FONTS["body"]))
        label.setStyleSheet(f"color: {COLORS['text_dark']};")
        layout.addWidget(label)


class RecordingWorker(QThread):
    """Worker ghi âm trong thread riêng"""
    finished = pyqtSignal(str, bool)  # path, success

    def __init__(self, duration=5):
        super().__init__()
        self.duration = duration

    def run(self):
        try:
            from voice_utils import record_audio
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from voice_utils import record_audio as rec
            path = rec(self.duration)
            self.finished.emit(path or "", bool(path))
        except Exception:
            self.finished.emit("", False)


class ChatWidget(QWidget):
    message_sent = pyqtSignal(str)
    voice_text_received = pyqtSignal(str)  # Text từ ASR

    def __init__(self, parent=None, enable_asr=False, enable_tts=False):
        super().__init__(parent)
        self.enable_asr = enable_asr
        self.enable_tts = enable_tts
        self.is_recording = False
        self._asr = None
        self._tts = None
        self._recording_worker = None
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

        # Scroll area cho chat
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.scroll.setWidget(self.chat_container)

        layout.addWidget(self.scroll)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Nhập câu hỏi hoặc nhấn mic...")
        self.input_field.setFont(QFont("Arial", FONTS["body"]))
        self.input_field.setMinimumHeight(45)
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                padding: 10px;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                background: white;
                font-size: {FONTS['body']}px;
            }}
        """)
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setFixedSize(50, 45)
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 20px;
            }}
            QPushButton:hover {{ background: {COLORS['primary_light']}; }}
            QPushButton:pressed {{ background: {COLORS['accent']}; }}
        """)
        self.mic_btn.setToolTip("Nhấn để ghi âm (giữ 5 giây)")
        self.mic_btn.clicked.connect(self.on_mic_click)
        input_layout.addWidget(self.mic_btn)

        layout.addLayout(input_layout)

        # Init ASR/TTS nếu enable
        if self.enable_asr:
            try:
                from nlp_system.inference import ASREngine
                import os
                provider = "fpt" if os.getenv("FPT_AI_KEY") else "mock"
                self._asr = ASREngine(provider=provider)
            except ImportError:
                self._asr = None
        if self.enable_tts:
            try:
                from nlp_system.inference import TTSEngine
                import os
                provider = "fpt" if os.getenv("FPT_AI_KEY") else "mock"
                self._tts = TTSEngine(provider=provider)
            except ImportError:
                self._tts = None

        # Thêm tin nhắn chào mừng
        self.add_message("Xin chào! Tôi là Robot hỗ trợ UTT. Bạn cần hỗ trợ gì?", is_user=False)

    def add_message(self, text, is_user=True):
        bubble = ChatBubble(text, is_user=is_user)
        bubble.setMaximumWidth(400)
        align = Qt.AlignRight if is_user else Qt.AlignLeft
        self.chat_layout.addWidget(bubble, 0, align)

        # Scroll xuống cuối
        QTimer.singleShot(100, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

    def send_message(self):
        text = self.input_field.text().strip()
        if text:
            self.add_message(text, is_user=True)
            self.input_field.clear()
            self.message_sent.emit(text)

    def on_mic_click(self):
        if self.is_recording:
            return
        self._start_recording()

    def _start_recording(self):
        """Bắt đầu ghi âm"""
        try:
            from voice_utils import record_audio
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            try:
                from voice_utils import record_audio
            except ImportError:
                self.add_message("[Cần cài pyaudio: pip install pyaudio]", is_user=True)
                return

        self.is_recording = True
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background: #cc0000;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 20px;
            }}
        """)
        self.mic_btn.setText("⏺")

        self._recording_worker = RecordingWorker(duration=5)
        self._recording_worker.finished.connect(self._on_recording_finished)
        self._recording_worker.start()

    def _on_recording_finished(self, audio_path: str, success: bool):
        self.is_recording = False
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 20px;
            }}
            QPushButton:hover {{ background: {COLORS['primary_light']}; }}
        """)
        self.mic_btn.setText("🎤")

        if not success or not audio_path:
            self.add_message("[Không ghi được âm thanh. Thử lại.]", is_user=True)
            return

        # ASR
        text = ""
        if self._asr:
            text, _ = self._asr.transcribe(audio_path)
        else:
            try:
                from nlp_system.inference import ASREngine
                asr = ASREngine(provider="mock")
                text, _ = asr.transcribe(audio_path)
            except ImportError:
                text = "Phòng A.205 ở đâu?"  # Demo

        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass

        if text:
            self.add_message(text, is_user=True)
            self.message_sent.emit(text)
        else:
            self.add_message("[Không nhận dạng được. Nói rõ hơn nhé.]", is_user=True)

    def add_robot_response(self, text, play_tts=False):
        self.add_message(text, is_user=False)
        if play_tts and self._tts:
            self._play_tts(text)

    def _play_tts(self, text: str):
        """Phát TTS (async để không block UI)"""
        def _do_tts():
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    path = f.name
                out = self._tts.synthesize(text, path)
                if out:
                    from voice_utils import play_audio
                    play_audio(out)
            except Exception:
                pass
        QTimer.singleShot(100, _do_tts)