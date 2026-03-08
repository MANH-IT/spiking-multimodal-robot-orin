#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ROBOT UTT - GIAO DIỆN ĐẦY ĐỦ CHO ĐH GIAO THÔNG VẬN TẢI
============================================================
Phiên bản tích hợp đầy đủ:
  - Màn hình cảm ứng (Touchscreen UI)
  - Nhận dạng giọng nói (ASR) - FPT.AI / Viettel AI / VOSK
  - Tổng hợp giọng nói (TTS) - FPT.AI / Google TTS / pyttsx3
  - Hiểu ngôn ngữ tiếng Việt (NLU)
  - Điều khiển robot, chỉ đường trong tòa nhà 15 tầng

Usage:
    python scripts/run_robot_utt.py
    python scripts/run_robot_utt.py --fullscreen
    python scripts/run_robot_utt.py --no-voice (chạy không voice)
"""

import os
import sys
import json
import time
import wave
import threading
import queue
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# CẤU HÌNH API
# ============================================================

# Có thể set bằng environment variables hoặc file config
CONFIG = {
    "fpt_ai_key": os.environ.get("FPT_AI_KEY", ""),
    "viettel_ai_key": os.environ.get("VIETTEL_AI_KEY", ""),
    "google_tts_key": os.environ.get("GOOGLE_TTS_KEY", ""),
    "use_mock": True,  # Mặc định dùng mock nếu không có key
    "mic_device": 0,   # Index mic
    "sample_rate": 16000,
    "language": "vi",
}

# ============================================================
# 1. ASR MODULE - NHẬN DẠNG GIỌNG NÓI
# ============================================================

class ASRModule:
    """Nhận dạng giọng nói tiếng Việt"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = self._detect_provider()
        self.engine = self._init_engine()
        
    def _detect_provider(self) -> str:
        """Phát hiện provider khả dụng"""
        if self.config.get("fpt_ai_key") and not self.config.get("use_mock"):
            return "fpt"
        elif self.config.get("viettel_ai_key") and not self.config.get("use_mock"):
            return "viettel"
        else:
            # Thử import VOSK (offline)
            try:
                import vosk
                # Kiểm tra model
                model_path = PROJECT_ROOT / "models" / "vosk-model-small-vi-0.3"
                if model_path.exists():
                    return "vosk"
            except ImportError:
                pass
            return "mock"
    
    def _init_engine(self):
        """Khởi tạo engine ASR"""
        if self.provider == "fpt":
            return self._init_fpt()
        elif self.provider == "viettel":
            return self._init_viettel()
        elif self.provider == "vosk":
            return self._init_vosk()
        else:
            return self._init_mock()
    
    def _init_fpt(self):
        """Khởi tạo FPT.AI Speech-to-Text"""
        try:
            import requests
            class FPTASR:
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.url = "https://api.fpt.ai/hmi/asr/general"
                    
                def transcribe(self, audio_path):
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    
                    headers = {
                        'api-key': self.api_key,
                        'Content-Type': 'audio/wav'
                    }
                    
                    try:
                        response = requests.post(self.url, headers=headers, data=audio_data, timeout=10)
                        if response.status_code == 200:
                            result = response.json()
                            return result.get('hypotheses', [{}])[0].get('utterance', '')
                    except Exception as e:
                        print(f"FPT ASR error: {e}")
                    return ""
            return FPTASR(self.config["fpt_ai_key"])
        except ImportError:
            print("⚠️ requests not installed. Install: pip install requests")
            return self._init_mock()
    
    def _init_viettel(self):
        """Khởi tạo Viettel AI Speech-to-Text"""
        try:
            import requests
            class ViettelASR:
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.url = "https://viettelai.vn/asm/api/asr"
                    
                def transcribe(self, audio_path):
                    with open(audio_path, 'rb') as f:
                        files = {'file': f}
                        headers = {'api-key': self.api_key}
                        
                        try:
                            response = requests.post(self.url, files=files, headers=headers, timeout=10)
                            if response.status_code == 200:
                                result = response.json()
                                return result.get('text', '')
                        except Exception as e:
                            print(f"Viettel ASR error: {e}")
                    return ""
            return ViettelASR(self.config["viettel_ai_key"])
        except ImportError:
            return self._init_mock()
    
    def _init_vosk(self):
        """Khởi tạo VOSK offline"""
        try:
            from vosk import Model, KaldiRecognizer
            model_path = PROJECT_ROOT / "models" / "vosk-model-small-vi-0.3"
            
            if not model_path.exists():
                print("⚠️ VOSK model not found. Download from:")
                print("   https://alphacephei.com/vosk/models/vosk-model-small-vi-0.3.zip")
                return self._init_mock()
            
            class VoskASR:
                def __init__(self, model_path):
                    self.model = Model(str(model_path))
                    
                def transcribe(self, audio_path):
                    wf = wave.open(audio_path, "rb")
                    rec = KaldiRecognizer(self.model, wf.getframerate())
                    
                    results = []
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            res = json.loads(rec.Result())
                            results.append(res.get('text', ''))
                    
                    res = json.loads(rec.FinalResult())
                    results.append(res.get('text', ''))
                    
                    return ' '.join(results)
            
            return VoskASR(model_path)
        except ImportError:
            print("⚠️ VOSK not installed. Install: pip install vosk")
            return self._init_mock()
    
    def _init_mock(self):
        """Mock ASR cho testing"""
        class MockASR:
            def transcribe(self, audio_path):
                print(f"🎤 Mock ASR processing: {audio_path}")
                # Giả lập nhận dạng
                return "chỉ đường đến phòng 1501"
        return MockASR()
    
    def transcribe(self, audio_path: str) -> str:
        """Chuyển audio thành text"""
        try:
            return self.engine.transcribe(audio_path)
        except Exception as e:
            print(f"ASR error: {e}")
            return ""


# ============================================================
# 2. TTS MODULE - TỔNG HỢP GIỌNG NÓI
# ============================================================

class TTSModule:
    """Tổng hợp giọng nói tiếng Việt"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = self._detect_provider()
        self.engine = self._init_engine()
        
    def _detect_provider(self) -> str:
        """Phát hiện provider khả dụng"""
        if self.config.get("fpt_ai_key") and not self.config.get("use_mock"):
            return "fpt"
        elif self.config.get("google_tts_key") and not self.config.get("use_mock"):
            return "google"
        else:
            try:
                import pyttsx3
                return "pyttsx"
            except ImportError:
                return "mock"
    
    def _init_engine(self):
        """Khởi tạo engine TTS"""
        if self.provider == "fpt":
            return self._init_fpt()
        elif self.provider == "google":
            return self._init_google()
        elif self.provider == "pyttsx":
            return self._init_pyttsx()
        else:
            return self._init_mock()
    
    def _init_fpt(self):
        """Khởi tạo FPT.AI Text-to-Speech"""
        try:
            import requests
            import pygame
            
            class FPTTTS:
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.url = "https://api.fpt.ai/hmi/tts/v5"
                    self.voice = "banmai"  # banmai, thuminh, giahuy, ...
                    
                def speak(self, text):
                    headers = {
                        'api-key': self.api_key,
                        'voice': self.voice,
                        'Content-Type': 'application/json'
                    }
                    
                    try:
                        response = requests.post(
                            self.url, 
                            headers=headers, 
                            data=json.dumps(text),
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            audio_url = result.get('async', '')
                            
                            if audio_url:
                                # Download và play
                                audio_response = requests.get(audio_url, timeout=10)
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                                temp_file.write(audio_response.content)
                                temp_file.close()
                                
                                pygame.mixer.init()
                                pygame.mixer.music.load(temp_file.name)
                                pygame.mixer.music.play()
                                while pygame.mixer.music.get_busy():
                                    time.sleep(0.1)
                                pygame.mixer.music.stop()
                                
                                os.unlink(temp_file.name)
                                return True
                    except Exception as e:
                        print(f"FPT TTS error: {e}")
                    return False
            
            import pygame
            return FPTTTS(self.config["fpt_ai_key"])
        except ImportError:
            print("⚠️ requests/pygame not installed")
            return self._init_mock()
    
    def _init_google(self):
        """Khởi tạo Google TTS"""
        try:
            from gtts import gTTS
            import pygame
            
            class GoogleTTS:
                def speak(self, text, lang='vi'):
                    try:
                        tts = gTTS(text=text, lang=lang, slow=False)
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        tts.save(temp_file.name)
                        
                        pygame.mixer.init()
                        pygame.mixer.music.load(temp_file.name)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        pygame.mixer.music.stop()
                        
                        os.unlink(temp_file.name)
                        return True
                    except Exception as e:
                        print(f"Google TTS error: {e}")
                        return False
            return GoogleTTS()
        except ImportError:
            print("⚠️ gTTS not installed. Install: pip install gtts pygame")
            return self._init_mock()
    
    def _init_pyttsx(self):
        """Khởi tạo pyttsx3 (offline)"""
        try:
            import pyttsx3
            
            class PyttsxTTS:
                def __init__(self):
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', 150)  # Tốc độ nói
                    self.engine.setProperty('volume', 0.9)  # Âm lượng
                    
                    # Tìm giọng tiếng Việt
                    voices = self.engine.getProperty('voices')
                    for voice in voices:
                        if 'vi' in voice.id.lower() or 'vietnamese' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                            
                def speak(self, text):
                    self.engine.say(text)
                    self.engine.runAndWait()
                    return True
                    
            return PyttsxTTS()
        except ImportError:
            print("⚠️ pyttsx3 not installed. Install: pip install pyttsx3")
            return self._init_mock()
    
    def _init_mock(self):
        """Mock TTS cho testing"""
        class MockTTS:
            def speak(self, text):
                print(f"🔊 [MOCK TTS]: {text}")
                return True
        return MockTTS()
    
    def speak(self, text: str):
        """Phát âm thanh từ text"""
        try:
            return self.engine.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
            return False


# ============================================================
# 3. NLU MODULE - HIỂU NGÔN NGỮ TỰ NHIÊN
# ============================================================

@dataclass
class IntentResult:
    """Kết quả phân tích ý định"""
    intent: str  # direction, floor_info, search, elevator, greeting, etc.
    entities: Dict[str, Any]
    confidence: float
    raw_text: str


class NLUModule:
    """Hiểu ngôn ngữ tự nhiên cho robot"""
    
    def __init__(self):
        # Từ khóa cho các intent
        self.intent_patterns = {
            "direction": [
                "chỉ đường", "đường đến", "dẫn đến", "đến phòng", 
                "cách đi", "làm sao lên", "làm sao xuống"
            ],
            "floor_info": [
                "tầng có gì", "thông tin tầng", "phòng tầng",
                "các phòng tầng", "khoa nào ở tầng"
            ],
            "search": [
                "tìm", "ở đâu", "địa chỉ", "phòng nào", "vị trí"
            ],
            "elevator": [
                "thang máy", "thang nào", "lên tầng", "xuống tầng"
            ],
            "greeting": [
                "xin chào", "chào", "hello", "hi", "chào robot"
            ],
            "thanks": [
                "cảm ơn", "thanks", "thank you"
            ],
            "help": [
                "giúp", "hỗ trợ", "có thể làm gì", "help"
            ]
        }
        
        # Pattern cho entities
        self.entity_patterns = {
            "room_number": r'phòng\s*(\d{3,4})',
            "room_12a": r'12a\s*(\d{1,2})',
            "floor_number": r'tầng\s*(\d{1,2})',
            "floor_12a": r'tầng\s*12a',
            "keyword": r'tìm\s+(.+?)(?:\?|$)',
        }
        
        # Import thư viện xử lý tiếng Việt nếu có
        self._init_vietnamese_processor()
        
    def _init_vietnamese_processor(self):
        """Khởi tạo bộ xử lý tiếng Việt"""
        try:
            from pyvi import ViTokenizer
            self.tokenizer = ViTokenizer
            self.has_pyvi = True
        except ImportError:
            self.has_pyvi = False
            print("⚠️ pyvi not installed. Install: pip install pyvi")
    
    def preprocess(self, text: str) -> str:
        """Tiền xử lý văn bản tiếng Việt"""
        text = text.lower().strip()
        
        # Xóa dấu câu
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize nếu có pyvi
        if self.has_pyvi:
            text = self.tokenizer.tokenize(text)
        
        return text
    
    def extract_intent(self, text: str) -> tuple:
        """Trích xuất intent chính"""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent, 0.9
        
        return "unknown", 0.3
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Trích xuất thực thể từ câu"""
        import re
        entities = {}
        
        # Tìm số phòng
        room_match = re.search(r'phòng\s*(\d{3,4})', text, re.IGNORECASE)
        if room_match:
            entities["room_number"] = int(room_match.group(1))
        
        # Tìm phòng 12A
        room_12a = re.search(r'12a\s*(\d{1,2})', text, re.IGNORECASE)
        if room_12a:
            room_num = 1300 + int(room_12a.group(1))
            entities["room_number"] = room_num
            entities["display_room"] = f"12A{room_12a.group(1)}"
        
        # Tìm số tầng
        floor_match = re.search(r'tầng\s*(\d{1,2})', text, re.IGNORECASE)
        if floor_match:
            entities["floor"] = int(floor_match.group(1))
        
        # Tìm tầng 12A
        if "tầng 12a" in text.lower():
            entities["floor"] = 13
            entities["display_floor"] = "12A"
        
        # Tìm từ khóa tìm kiếm
        keyword_match = re.search(r'tìm\s+(.+?)(?:\?|$)', text, re.IGNORECASE)
        if keyword_match:
            entities["keyword"] = keyword_match.group(1).strip()
        
        return entities
    
    def parse(self, text: str) -> IntentResult:
        """Phân tích câu nói"""
        processed = self.preprocess(text)
        intent, confidence = self.extract_intent(processed)
        entities = self.extract_entities(text)
        
        return IntentResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            raw_text=text
        )


# ============================================================
# 4. MICROPHONE RECORDER - GHI ÂM TỪ MIC
# ============================================================

class MicrophoneRecorder:
    """Ghi âm từ microphone"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self._init_mic()
        
    def _init_mic(self):
        """Khởi tạo microphone"""
        try:
            import pyaudio
            
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()
            
            # Liệt kê các thiết bị
            print("🎤 Available microphones:")
            for i in range(self.audio.get_device_count()):
                dev = self.audio.get_device_info_by_index(i)
                if dev['maxInputChannels'] > 0:
                    print(f"   [{i}] {dev['name']}")
            
            self.CHUNK = 1024
            self.FORMAT = pyaudio.paInt16
            self.CHANNELS = 1
            self.RATE = self.config.get("sample_rate", 16000)
            
            self.has_pyaudio = True
        except ImportError:
            print("⚠️ pyaudio not installed. Install: pip install pyaudio")
            self.has_pyaudio = False
    
    def start_recording(self):
        """Bắt đầu ghi âm"""
        if not self.has_pyaudio:
            return False
            
        self.is_recording = True
        self.audio_queue = queue.Queue()
        
        def record_thread():
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.config.get("mic_device", 0),
                frames_per_buffer=self.CHUNK
            )
            
            frames = []
            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK)
                    frames.append(data)
                except:
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Lưu vào file tạm
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            self.audio_queue.put(temp_file.name)
        
        threading.Thread(target=record_thread, daemon=True).start()
        return True
    
    def stop_recording(self) -> Optional[str]:
        """Dừng ghi âm và trả về đường dẫn file"""
        self.is_recording = False
        try:
            return self.audio_queue.get(timeout=5)
        except queue.Empty:
            return None


# ============================================================
# 5. ROBOT CONTROLLER - ĐIỀU KHIỂN ROBOT
# ============================================================

class RobotController:
    """Điều khiển robot dựa trên intent"""
    
    def __init__(self):
        # Load dữ liệu tòa nhà
        try:
            from data.building_15floors import (
                get_direction, get_floor_info, get_room_info,
                search_rooms, FLOOR_DISPLAY_NAMES
            )
            self.has_building = True
        except ImportError:
            self.has_building = False
            print("⚠️ Building data not found")
    
    def process(self, intent_result: IntentResult) -> Dict[str, Any]:
        """Xử lý intent và trả về response"""
        
        intent = intent_result.intent
        entities = intent_result.entities
        
        response = {
            "text": "",
            "action": None,
            "data": None,
            "tts": True
        }
        
        if intent == "direction":
            response = self._handle_direction(entities)
        elif intent == "floor_info":
            response = self._handle_floor_info(entities)
        elif intent == "search":
            response = self._handle_search(entities)
        elif intent == "elevator":
            response = self._handle_elevator()
        elif intent == "greeting":
            response["text"] = "Xin chào! Tôi là robot của ĐH Giao thông Vận tải. Tôi có thể giúp gì cho bạn?"
        elif intent == "thanks":
            response["text"] = "Không có gì! Chúc bạn một ngày tốt lành!"
        elif intent == "help":
            response["text"] = self._get_help_text()
        else:
            response["text"] = "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể thử: 'Chỉ đường đến phòng 1501', 'Tầng 3 có gì?', hoặc 'Tìm phòng AI'."
        
        return response
    
    def _handle_direction(self, entities: Dict) -> Dict[str, Any]:
        """Xử lý yêu cầu chỉ đường"""
        response = {"text": "", "action": "navigate", "data": None, "tts": True}
        
        if "room_number" in entities:
            room = entities["room_number"]
            
            if self.has_building:
                direction_text = get_direction(1, room)
                response["text"] = direction_text
                response["data"] = {"room": room, "floor": room // 100}
            else:
                response["text"] = f"Đang chỉ đường đến phòng {room}"
        else:
            response["text"] = "Bạn muốn đến phòng nào? Vui lòng nói số phòng (ví dụ: 1501 hoặc 12A05)"
        
        return response
    
    def _handle_floor_info(self, entities: Dict) -> Dict[str, Any]:
        """Xử lý yêu cầu thông tin tầng"""
        response = {"text": "", "action": "show_floor", "data": None, "tts": True}
        
        if "floor" in entities:
            floor = entities["floor"]
            
            if self.has_building:
                fi = get_floor_info(floor)
                if fi:
                    text = f"Tầng {entities.get('display_floor', floor)}: {fi.name}\n\nCác phòng:\n"
                    for rn, ri in list(fi.rooms.items())[:10]:  # Giới hạn 10 phòng cho TTS
                        dn = ri.display_number or str(rn)
                        text += f"Phòng {dn}: {ri.name}\n"
                    
                    if len(fi.rooms) > 10:
                        text += f"...và {len(fi.rooms) - 10} phòng khác."
                    
                    response["text"] = text
                    response["data"] = {"floor": floor}
                else:
                    response["text"] = f"Không tìm thấy thông tin tầng {floor}"
            else:
                response["text"] = f"Tầng {floor}"
        else:
            response["text"] = "Bạn muốn xem thông tin tầng nào?"
        
        return response
    
    def _handle_search(self, entities: Dict) -> Dict[str, Any]:
        """Xử lý tìm kiếm"""
        response = {"text": "", "action": "search", "data": None, "tts": True}
        
        if "keyword" in entities:
            keyword = entities["keyword"]
            
            if self.has_building:
                results = search_rooms(keyword)
                if results:
                    text = f"Tìm thấy {len(results)} phòng:\n"
                    for r in results[:5]:
                        dn = r.display_number or str(r.number)
                        text += f"Phòng {dn}: {r.name}\n"
                    
                    if len(results) > 5:
                        text += f"...và {len(results) - 5} kết quả khác."
                    
                    response["text"] = text
                    response["data"] = {"results": [r.number for r in results[:5]]}
                else:
                    response["text"] = f"Không tìm thấy '{keyword}'"
            else:
                response["text"] = f"Đang tìm '{keyword}'"
        else:
            response["text"] = "Bạn muốn tìm gì? Ví dụ: 'Tìm phòng AI' hoặc 'Tìm khoa CNTT'"
        
        return response
    
    def _handle_elevator(self) -> Dict[str, Any]:
        """Xử lý thông tin thang máy"""
        text = "Tòa nhà có 2 thang máy:\n"
        text += "- Thang A: Tầng 1-15 (thang chính)\n"
        text += "- Thang B: Tầng 1-12A (thang phụ)\n"
        text += "Tầng 12A và 15 nên dùng thang A."
        
        return {"text": text, "action": "info", "data": None, "tts": True}
    
    def _get_help_text(self) -> str:
        """Lấy text hướng dẫn"""
        return (
            "Tôi có thể giúp bạn:\n"
            "• Chỉ đường: 'Chỉ đường đến phòng 1501'\n"
            "• Thông tin tầng: 'Tầng 3 có gì?'\n"
            "• Tìm kiếm: 'Tìm phòng AI'\n"
            "• Thang máy: 'Thang máy nào lên tầng 15?'\n"
            "• Xem bản đồ: Mở tab Bản đồ để xem chi tiết."
        )


# ============================================================
# 6. MAIN UI - TÍCH HỢP TẤT CẢ
# ============================================================

class RobotUTTApp:
    """Ứng dụng robot đầy đủ"""
    
    def __init__(self, fullscreen: bool = False, no_voice: bool = False):
        self.fullscreen = fullscreen
        self.no_voice = no_voice
        self.config = CONFIG
        
        # Khởi tạo các module
        print("=" * 60)
        print("🤖 ROBOT UTT - ĐH GIAO THÔNG VẬN TẢI")
        print("=" * 60)
        
        if not no_voice:
            print("🎤 Khởi tạo ASR...")
            self.asr = ASRModule(self.config)
            print(f"   Provider: {self.asr.provider}")
            
            print("🔊 Khởi tạo TTS...")
            self.tts = TTSModule(self.config)
            print(f"   Provider: {self.tts.provider}")
        else:
            self.asr = None
            self.tts = None
            print("🔇 Voice disabled")
        
        print("🧠 Khởi tạo NLU...")
        self.nlu = NLUModule()
        
        print("🎮 Khởi tạo Robot Controller...")
        self.controller = RobotController()
        
        if not no_voice:
            print("🎤 Khởi tạo Microphone...")
            self.mic = MicrophoneRecorder(self.config)
        
        print("✅ Khởi tạo hoàn tất!")
        print("=" * 60)
        
        # Queue cho voice processing
        self.voice_queue = queue.Queue()
        self.is_listening = False
        
        # Chạy UI
        self._run_ui()
    
    def _run_ui(self):
        """Chạy giao diện chính"""
        # Import UI từ file touchscreen
        touchscreen_dir = PROJECT_ROOT / "scripts" / "demo" / "touchscreen_ui"
        sys.path.insert(0, str(touchscreen_dir))
        
        try:
            # Thử import UI mới đã phát triển
            from run_robot_ui_touch import RobotTouchMainWindow
            self._run_custom_ui()
        except ImportError:
            # Fallback to old UI
            self._run_legacy_ui()
    
    def _run_custom_ui(self):
        """Chạy UI tùy chỉnh với tích hợp voice"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        
        app = QApplication(sys.argv)
        
        # Import UI
        try:
            from run_robot_ui_touch import RobotTouchMainWindow
        except ImportError:
            print("❌ Không tìm thấy file run_robot_ui_touch.py")
            return
        
        # Tạo window
        window = RobotTouchMainWindow(fullscreen=self.fullscreen)
        
        # Thêm các phương thức xử lý voice
        if not self.no_voice:
            self._enhance_window_with_voice(window)
        
        window.show()
        sys.exit(app.exec_())
    
    def _enhance_window_with_voice(self, window):
        """Thêm chức năng voice vào window"""
        
        def on_mic_click():
            """Xử lý khi nhấn nút mic"""
            if not self.is_listening:
                self._start_listening(window)
            else:
                self._stop_listening(window)
        
        # Thay thế nút mic trong chat
        if hasattr(window, 'chat') and hasattr(window.chat, 'btn_mic'):
            window.chat.btn_mic.clicked.disconnect()
            window.chat.btn_mic.clicked.connect(on_mic_click)
        
        # Thêm callback xử lý voice
        window.process_voice_result = self._process_voice_result
        
        # Thêm timer xử lý queue
        self.voice_timer = QTimer()
        self.voice_timer.timeout.connect(lambda: self._check_voice_queue(window))
        self.voice_timer.start(100)
    
    def _start_listening(self, window):
        """Bắt đầu lắng nghe"""
        self.is_listening = True
        window.chat.btn_mic.setText("🔴")
        window.chat.btn_mic.setStyleSheet("background-color: #F55E5E;")
        window.chat.add_message("system", "🎤 Đang nghe... (nói gì đó)")
        
        # Bắt đầu ghi âm
        self.mic.start_recording()
    
    def _stop_listening(self, window):
        """Dừng lắng nghe và xử lý"""
        self.is_listening = False
        window.chat.btn_mic.setText("🎤")
        window.chat.btn_mic.setStyleSheet("")
        
        # Dừng ghi âm và xử lý
        audio_file = self.mic.stop_recording()
        if audio_file:
            window.chat.add_message("system", "⏳ Đang xử lý giọng nói...")
            
            # Xử lý trong thread riêng
            threading.Thread(
                target=self._process_audio,
                args=(audio_file,),
                daemon=True
            ).start()
    
    def _process_audio(self, audio_file):
        """Xử lý audio trong thread riêng"""
        try:
            # ASR
            text = self.asr.transcribe(audio_file)
            
            if text:
                self.voice_queue.put(("result", text))
            else:
                self.voice_queue.put(("error", "Không nhận dạng được giọng nói"))
        except Exception as e:
            self.voice_queue.put(("error", str(e)))
        finally:
            # Xóa file tạm
            try:
                os.unlink(audio_file)
            except:
                pass
    
    def _check_voice_queue(self, window):
        """Kiểm tra queue voice result"""
        try:
            msg_type, data = self.voice_queue.get_nowait()
            
            if msg_type == "result":
                # Hiển thị kết quả
                window.chat.add_message("user", f"🎤 {data}")
                # Xử lý như tin nhắn thường
                window._on_user_message(data)
            elif msg_type == "error":
                window.chat.add_message("system", f"❌ Lỗi: {data}")
        except queue.Empty:
            pass
    
    def _process_voice_result(self, text: str, window):
        """Xử lý kết quả voice và phản hồi bằng TTS"""
        # NLU
        intent_result = self.nlu.parse(text)
        
        # Controller
        response = self.controller.process(intent_result)
        
        # Hiển thị và nói
        if response["text"]:
            window.chat.add_message("robot", response["text"])
            
            if self.tts and response["tts"]:
                threading.Thread(
                    target=self.tts.speak,
                    args=(response["text"],),
                    daemon=True
                ).start()
        
        # Thực hiện action
        if response["action"] == "show_floor" and response["data"]:
            floor = response["data"].get("floor")
            if floor and hasattr(window, 'building_map'):
                window.tabs.setCurrentIndex(1)
                window.building_map.floor_spin.setValue(floor)
    
    def _run_legacy_ui(self):
        """Chạy UI cũ (fallback)"""
        try:
            from run_touchscreen import main as run_ui
            run_ui(fullscreen=self.fullscreen)
        except ImportError as e:
            print(f"❌ Không thể chạy UI: {e}")
            print("   Hãy chắc chắn đã có file run_touchscreen.py")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Robot UTT - ĐH Giao thông Vận tải")
    parser.add_argument("--fullscreen", "-f", action="store_true",
                       help="Chạy ở chế độ toàn màn hình")
    parser.add_argument("--no-voice", "-nv", action="store_true",
                       help="Chạy không voice (chỉ text)")
    parser.add_argument("--config", "-c", type=str,
                       help="Đường dẫn file config JSON")
    
    args = parser.parse_args()
    
    # Load config từ file nếu có
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                CONFIG.update(user_config)
        except Exception as e:
            print(f"⚠️ Không thể đọc config: {e}")
    
    # Chạy app
    app = RobotUTTApp(
        fullscreen=args.fullscreen,
        no_voice=args.no_voice
    )


if __name__ == "__main__":
    main()