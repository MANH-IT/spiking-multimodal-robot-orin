# -*- coding: utf-8 -*-
"""
Voice utilities - ASR (ghi âm), TTS (phát âm thanh)
Graceful fallback khi thiếu PyAudio/pygame
"""

import tempfile
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple

# Optional: PyAudio cho ghi âm
try:
    import pyaudio
    import wave
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Optional: pygame cho phát âm thanh
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def record_audio(duration_sec: float = 5, sample_rate: int = 16000) -> Optional[str]:
    """
    Ghi âm từ microphone.

    Returns:
        Đường dẫn file wav tạm, hoặc None nếu lỗi
    """
    if not HAS_PYAUDIO:
        return None

    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024,
        )

        frames = []
        for _ in range(int(sample_rate / 1024 * duration_sec)):
            data = stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

        return tmp_path
    except Exception:
        return None


def play_audio(file_path: str) -> bool:
    """
    Phát file âm thanh (mp3, wav).

    Returns:
        True nếu phát thành công
    """
    if not Path(file_path).exists():
        return False

    if HAS_PYGAME:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            return True
        except Exception:
            pass

    # Fallback: dùng trình phát mặc định của hệ thống
    try:
        if os.name == "nt":
            os.startfile(file_path)
        elif os.name == "posix":
            subprocess.Popen(
                ["xdg-open", file_path] if "DISPLAY" in os.environ else ["aplay", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception:
        return False
