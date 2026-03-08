# -*- coding: utf-8 -*-
"""
Pipeline NLP hoàn chỉnh: ASR -> NLU -> TTS
Thay thế mock trong nlp_system
"""

import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from .asr import ASREngine
from .tts import TTSEngine
from .nlu import NLUEngine


class NLPPipeline:
    """
    Pipeline: Voice/Text -> ASR (nếu voice) -> NLU -> Response + TTS
    Fallback mock khi mất mạng
    """

    def __init__(
        self,
        asr_provider: str = "fpt",
        tts_provider: str = "fpt",
        asr_key: Optional[str] = None,
        tts_key: Optional[str] = None,
    ):
        self.asr = ASREngine(provider=asr_provider, api_key=asr_key)
        self.tts = TTSEngine(provider=tts_provider, api_key=tts_key)
        self.nlu = NLUEngine()

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Xử lý câu hỏi dạng text.

        Returns:
            {
                "intent": str,
                "params": dict,
                "response": str,
                "room": str | None,
                "audio_path": str | None
            }
        """
        nlu_result = self.nlu.understand(text)
        return nlu_result

    def process_voice(
        self,
        audio_path: str,
        play_audio: bool = False,
    ) -> Dict[str, Any]:
        """
        Xử lý câu hỏi dạng giọng nói.

        Returns:
            {
                "user_text": str,
                "intent": str,
                "params": dict,
                "response": str,
                "room": str | None,
                "audio_path": str | None
            }
        """
        text, confidence = self.asr.transcribe(audio_path)
        if not text:
            return {
                "user_text": "",
                "intent": "unknown",
                "params": {},
                "response": "Xin lỗi, tôi không nghe rõ. Bạn có thể nhắc lại không?",
                "room": None,
                "audio_path": None,
            }

        nlu_result = self.nlu.understand(text)
        response_text = nlu_result["response"]

        # TTS response
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path_out = f.name
        audio_path_out = self.tts.synthesize(response_text, audio_path_out)

        return {
            "user_text": text,
            "intent": nlu_result["intent"],
            "params": nlu_result["params"],
            "response": response_text,
            "room": nlu_result.get("room"),
            "audio_path": audio_path_out,
        }

    def get_response_audio(self, text: str) -> Optional[str]:
        """Chuyển text thành audio (cho TTS phản hồi)"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        return self.tts.synthesize(text, path)
