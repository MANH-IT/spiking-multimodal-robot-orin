# -*- coding: utf-8 -*-
"""
ASR (Automatic Speech Recognition) - Nhận dạng giọng nói tiếng Việt

Hỗ trợ: FPT.AI, Viettel AI
Fallback: Mock khi mất mạng
Cache: Giảm gọi API
"""

import os
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).resolve().parents[1] / "resources" / "cache" / "asr"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ASREngine:
    """ASR Engine - FPT.AI / Viettel AI với fallback mock"""

    # API endpoints
    FPT_ASR_URL = "https://api.fpt.ai/hmi/asr/v1"
    VIETTEL_ASR_URL = "https://viettelgroup.ai/voice/api/asr/v1/short-audio"

    def __init__(
        self,
        provider: str = "fpt",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Args:
            provider: "fpt" | "viettel" | "mock"
            api_key: API key (hoặc từ env FPT_AI_KEY / VIETTEL_AI_KEY)
            use_cache: Bật cache kết quả
            cache_ttl: Thời gian cache (giây)
        """
        self.provider = provider.lower()
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

        if api_key:
            self.api_key = api_key
        else:
            if self.provider == "fpt":
                self.api_key = os.getenv("FPT_AI_KEY", "")
            else:
                self.api_key = os.getenv("VIETTEL_AI_KEY", "")

        if not self.api_key and self.provider != "mock":
            logger.warning("API key không có. Sẽ fallback sang mock khi gọi API.")

    def transcribe(
        self,
        audio_path: str,
        language: str = "vi",
    ) -> Tuple[str, float]:
        """
        Chuyển audio thành text.

        Args:
            audio_path: Đường dẫn file audio (wav, mp3, flac)
            language: vi | en

        Returns:
            (text, confidence) - confidence 0-1
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"File không tồn tại: {audio_path}")
            return "", 0.0

        # Check cache
        if self.use_cache:
            cached = self._get_from_cache(audio_path)
            if cached:
                return cached["text"], cached["confidence"]

        # Call API
        try:
            if self.provider == "fpt":
                text, conf = self._transcribe_fpt(audio_path)
            elif self.provider == "viettel":
                text, conf = self._transcribe_viettel(audio_path)
            else:
                text, conf = self._transcribe_mock(audio_path)
        except Exception as e:
            logger.warning(f"ASR API lỗi ({e}), fallback mock")
            text, conf = self._transcribe_mock(audio_path)

        # Save to cache
        if self.use_cache and text:
            self._save_to_cache(audio_path, text, conf)

        return text, conf

    def _transcribe_fpt(self, audio_path: Path) -> Tuple[str, float]:
        """FPT.AI Speech-to-Text API - api.fpt.ai/hmi/asr/v1"""
        if not self.api_key:
            return self._transcribe_mock(audio_path)

        url = self.FPT_ASR_URL
        headers = {"api-key": self.api_key}

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/wav")}
                response = requests.post(
                    url, headers=headers, files=files, timeout=30
                )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("hypotheses", [{}])[0].get("utterance", "")
                return text.strip(), 0.9
            elif isinstance(result, dict):
                text = result.get("text", result.get("result", ""))
                return str(text).strip(), 0.9
            return "", 0.0
        except requests.RequestException as e:
            raise RuntimeError(f"FPT ASR: {e}")

    def _transcribe_viettel(self, audio_path: Path) -> Tuple[str, float]:
        """Viettel AI Speech-to-Text API"""
        if not self.api_key:
            return self._transcribe_mock(audio_path)

        url = self.VIETTEL_ASR_URL
        headers = {
            "token": self.api_key,
            "Content-Type": "application/json",
        }

        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "audio": audio_b64,
            "sample_rate": 16000,
            "format": "wav",
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            text = result.get("text", result.get("transcript", ""))
            return str(text).strip(), 0.9
        except requests.RequestException as e:
            raise RuntimeError(f"Viettel ASR: {e}")

    def _transcribe_mock(self, audio_path: Path) -> Tuple[str, float]:
        """Mock ASR - trả về câu mẫu khi không có API"""
        mock_texts = [
            "Phòng A.205 ở đâu?",
            "Học phí ngành CNTT bao nhiêu?",
            "Lịch thi học kỳ 1 khi nào?",
            "Có thông báo mới không?",
            "Xin chào, tôi cần hỗ trợ.",
        ]
        # Dùng hash của file để chọn mock (deterministic)
        h = hashlib.md5(audio_path.read_bytes()).hexdigest()
        idx = int(h[:8], 16) % len(mock_texts)
        return mock_texts[idx], 0.7

    def _cache_key(self, audio_path: Path) -> str:
        return hashlib.md5(
            f"{audio_path}{audio_path.stat().st_mtime}".encode()
        ).hexdigest()

    def _get_from_cache(self, audio_path: Path) -> Optional[dict]:
        key = self._cache_key(audio_path)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return data
            except Exception:
                pass
        return None

    def _save_to_cache(self, audio_path: Path, text: str, confidence: float):
        key = self._cache_key(audio_path)
        cache_file = CACHE_DIR / f"{key}.json"
        cache_file.write_text(
            json.dumps({"text": text, "confidence": confidence}, ensure_ascii=False),
            encoding="utf-8",
        )
