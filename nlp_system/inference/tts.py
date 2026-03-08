# -*- coding: utf-8 -*-
"""
TTS (Text-to-Speech) - Tổng hợp giọng nói tiếng Việt

Hỗ trợ: FPT.AI (giọng nữ miền Bắc), Viettel AI
Fallback: Mock
Cache: Giảm gọi API
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "resources" / "cache" / "tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TTSEngine:
    """TTS Engine - FPT.AI / Viettel AI, giọng nữ miền Bắc"""

    FPT_TTS_URL = "https://api.fpt.ai/hmi/tts/v5"
    VIETTEL_TTS_URL = "https://viettelgroup.ai/voice/api/tts/v1/speech"

    # Giọng nữ miền Bắc
    VOICES = {
        "fpt": "banmai",  # Nữ Bắc - FPT
        "viettel": "hn-quynhanh",  # Nữ Hà Nội
    }

    def __init__(
        self,
        provider: str = "fpt",
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            provider: "fpt" | "viettel" | "mock"
            api_key: API key
            voice: Tên giọng (mặc định: nữ Bắc)
            use_cache: Cache file audio
        """
        self.provider = provider.lower()
        self.use_cache = use_cache

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv(
                "FPT_AI_KEY" if self.provider == "fpt" else "VIETTEL_AI_KEY",
                "",
            )

        self.voice = voice or self.VOICES.get(self.provider, "banmai")

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speed: float = 0,
    ) -> Optional[str]:
        """
        Chuyển text thành file audio.

        Args:
            text: Văn bản cần đọc
            output_path: Đường dẫn lưu file (mp3)
            speed: Tốc độ -1 đến 1 (0 = bình thường)

        Returns:
            Đường dẫn file audio hoặc None
        """
        if not text.strip():
            return None

        # Check cache
        if self.use_cache:
            cached_path = self._get_cached_path(text)
            if cached_path and cached_path.exists():
                if output_path:
                    import shutil

                    shutil.copy(cached_path, output_path)
                    return output_path
                return str(cached_path)

        # Generate
        try:
            if self.provider == "fpt":
                path = self._synthesize_fpt(text, output_path, speed)
            elif self.provider == "viettel":
                path = self._synthesize_viettel(text, output_path, speed)
            else:
                path = self._synthesize_mock(text, output_path)
        except Exception as e:
            logger.warning(f"TTS API lỗi ({e}), fallback mock")
            path = self._synthesize_mock(text, output_path)

        # Save to cache
        if self.use_cache and path:
            self._save_to_cache(text, path)

        return path

    def _synthesize_fpt(
        self,
        text: str,
        output_path: Optional[str],
        speed: float,
    ) -> Optional[str]:
        """FPT.AI TTS v5 - giọng banmai (nữ Bắc)"""
        if not self.api_key:
            return self._synthesize_mock(text, output_path)

        url = self.FPT_TTS_URL
        headers = {"api-key": self.api_key}

        # FPT TTS v5: GET với params hoặc POST
        params = {"text": text, "voice": self.voice}
        if speed != 0:
            params["speed"] = speed

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                response = requests.post(
                    url, headers=headers, json=params, timeout=30
                )
            response.raise_for_status()
            audio_content = response.content

            # FPT có thể trả JSON {async_id} hoặc binary trực tiếp
            if response.headers.get("Content-Type", "").startswith("application/json"):
                import base64
                try:
                    data = response.json()
                    audio_content = base64.b64decode(
                        data.get("audio", data.get("data", ""))
                    )
                except Exception:
                    pass

            if not output_path:
                output_path = str(CACHE_DIR / f"{hashlib.md5(text.encode()).hexdigest()}.mp3")

            Path(output_path).write_bytes(audio_content)
            return output_path
        except requests.RequestException as e:
            raise RuntimeError(f"FPT TTS: {e}")

    def _synthesize_viettel(
        self,
        text: str,
        output_path: Optional[str],
        speed: float,
    ) -> Optional[str]:
        """Viettel AI TTS"""
        if not self.api_key:
            return self._synthesize_mock(text, output_path)

        url = self.VIETTEL_TTS_URL
        headers = {
            "token": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice": self.voice,
            "speed": speed,
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            audio_content = response.content

            if not output_path:
                output_path = str(CACHE_DIR / f"{hashlib.md5(text.encode()).hexdigest()}.mp3")

            Path(output_path).write_bytes(audio_content)
            return output_path
        except requests.RequestException as e:
            raise RuntimeError(f"Viettel TTS: {e}")

    def _synthesize_mock(
        self,
        text: str,
        output_path: Optional[str],
    ) -> Optional[str]:
        """Mock TTS - tạo file rỗng hoặc placeholder"""
        if not output_path:
            output_path = str(CACHE_DIR / f"mock_{hashlib.md5(text.encode()).hexdigest()}.txt")

        if output_path.endswith(".mp3"):
            # Tạo file wav đơn giản (silence) - hoặc chỉ ghi text
            Path(output_path).with_suffix(".txt").write_text(text, encoding="utf-8")
            return output_path.replace(".mp3", ".txt")
        Path(output_path).write_text(text, encoding="utf-8")
        return output_path

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{text}{self.voice}".encode()).hexdigest()

    def _get_cached_path(self, text: str) -> Optional[Path]:
        key = self._cache_key(text)
        for ext in [".mp3", ".wav", ".txt"]:
            p = CACHE_DIR / f"{key}{ext}"
            if p.exists():
                return p
        return None

    def _save_to_cache(self, text: str, path: str):
        key = self._cache_key(text)
        dest = CACHE_DIR / f"{key}{Path(path).suffix}"
        if Path(path).exists() and str(dest) != path:
            import shutil

            shutil.copy(path, dest)
