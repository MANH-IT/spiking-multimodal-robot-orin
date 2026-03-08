# -*- coding: utf-8 -*-
"""
Adapters để tích hợp NLP inference vào demo (nlp_webapp, touchscreen)
API tương thích với MockASR, MockNLU, MockTTS
"""

import os
from typing import Optional

try:
    from .asr import ASREngine
    from .tts import TTSEngine
    from .nlu import NLUEngine
    from .pipeline import NLPPipeline
    HAS_NLP = True
except ImportError:
    HAS_NLP = False


def get_asr(provider: str = "fpt"):
    """Lấy ASR - real hoặc mock"""
    if HAS_NLP:
        key = os.getenv("FPT_AI_KEY") if provider == "fpt" else os.getenv("VIETTEL_AI_KEY")
        return ASREngine(provider=provider, api_key=key or None)
    return None


def get_tts(provider: str = "fpt"):
    """Lấy TTS - real hoặc mock"""
    if HAS_NLP:
        key = os.getenv("FPT_AI_KEY") if provider == "fpt" else os.getenv("VIETTEL_AI_KEY")
        return TTSEngine(provider=provider, api_key=key or None)
    return None


def get_nlu():
    """Lấy NLU engine"""
    if HAS_NLP:
        return NLUEngine()
    return None


def get_pipeline(asr_provider: str = "fpt", tts_provider: str = "fpt"):
    """Lấy pipeline hoàn chỉnh"""
    if HAS_NLP:
        return NLPPipeline(
            asr_provider=asr_provider,
            tts_provider=tts_provider,
        )
    return None


class ASRAdapter:
    """Adapter ASR - tương thích MockASR.transcribe()"""

    def __init__(self):
        self.engine = get_asr()
        self.name = "Real ASR" if self.engine else "Mock ASR"
        self.supported_languages = ["vi", "en"]

    def transcribe(self, audio_file, language="vi"):
        if self.engine:
            text, _ = self.engine.transcribe(audio_file, language)
            return text
        return "Xin chào, tôi muốn tìm một cái cốc màu đỏ"

    def transcribe_text(self, text):
        return text


class NLUAdapter:
    """Adapter NLU - tương thích MockNLU.understand()"""

    def __init__(self):
        self.engine = get_nlu()
        self.name = "Real NLU" if self.engine else "Mock NLU"
        self.intents = ["tim_phong", "tra_cuu_hoc_phi", "lich_thi", "thong_bao"]

    def understand(self, text, language="vi"):
        if self.engine:
            result = self.engine.understand(text)
            return {
                "intent": result["intent"],
                "entities": [{"type": k, "value": v} for k, v in result["params"].items()],
                "confidence": result.get("confidence", 0.8),
                "language": language,
                "response": result["response"],
            }
        return {
            "intent": "unknown",
            "entities": [],
            "confidence": 0.5,
            "language": language,
        }


class TTSAdapter:
    """Adapter TTS - tương thích MockTTS.synthesize()"""

    def __init__(self):
        self.engine = get_tts()
        self.name = "Real TTS" if self.engine else "Mock TTS"
        self.supported_languages = ["vi", "en"]

    def synthesize(self, text, language="vi"):
        if self.engine:
            path = self.engine.synthesize(text)
            return {
                "text": text,
                "language": language,
                "audio_file": path,
                "duration": len(text) * 0.05,
            }
        return {
            "text": text,
            "language": language,
            "audio_file": None,
            "duration": len(text) * 0.1,
        }
