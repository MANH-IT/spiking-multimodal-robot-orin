# -*- coding: utf-8 -*-
"""
NLU (Natural Language Understanding) - Hiểu ý định

Chủ đề: tìm phòng, học phí, lịch thi, thông báo
Pattern matching + entity extraction
"""

import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Intent patterns - từ khóa và regex
INTENT_PATTERNS = {
    "tim_phong": {
        "keywords": [
            "phòng", "phong", "ở đâu", "o dau", "đường đi", "duong di",
            "đi đến", "di den", "tìm phòng", "tim phong", "phòng nào",
        ],
        "entity_patterns": {
            "phong": r"(?:phòng|phong)\s*([A-Z]?\d+(?:\.\d+)?)",
            "phong_alt": r"([A-Z]\.\d{3})",
        },
    },
    "tra_cuu_hoc_phi": {
        "keywords": [
            "học phí", "hoc phi", "đóng tiền", "dong tien", "đóng muộn",
            "bao nhiêu", "bao nhieu", "ngành", "nganh", "CNTT", "công nghệ",
        ],
        "entity_patterns": {
            "nganh": r"(?:ngành|nganh)\s*(\w+)",
            "nganh_cn": r"(CNTT|Công nghệ thông tin|Khoa học máy tính)",
        },
    },
    "lich_thi": {
        "keywords": [
            "lịch thi", "lich thi", "thi khi nào", "thi khi nao",
            "học kỳ", "hoc ky", "lịch học", "lich hoc", "thời khóa biểu",
        ],
        "entity_patterns": {
            "hoc_ky": r"(?:học kỳ|hoc ky)\s*(\d)",
            "mon": r"(?:môn|mon)\s*(\w+)",
        },
    },
    "thong_bao": {
        "keywords": [
            "thông báo", "thong bao", "tin mới", "tin moi", "có gì mới",
            "co gi moi", "announcement",
        ],
        "entity_patterns": {},
    },
    "dang_ky_mon_hoc": {
        "keywords": [
            "đăng ký môn", "dang ky mon", "đăng ký học", "dang ky hoc",
            "đăng ký học phần", "đăng ký tín chỉ",
        ],
        "entity_patterns": {},
    },
    "lien_he_phong_ban": {
        "keywords": [
            "liên hệ", "lien he", "phòng ban", "phong ban", "contact",
        ],
        "entity_patterns": {},
    },
}

# Câu trả lời mẫu theo intent
INTENT_RESPONSES = {
    "tim_phong": {
        "default": "Phòng {phong} nằm ở tầng {tang}, khu nhà A. Từ sảnh chính đi thẳng 50m, rẽ trái.",
        "A.205": "Phòng A.205 ở tầng 2, khu nhà A. Đi từ cầu thang chính lên tầng 2, phòng đầu tiên bên trái.",
        "A.101": "Phòng A.101 ở tầng 1, khu nhà A - Phòng Công tác Sinh viên. Ngay cửa vào chính.",
        "A.102": "Phòng A.102 ở tầng 1, khu nhà A - Phòng Đào tạo.",
        "B.201": "Phòng B.201 ở tầng 2, khu nhà B.",
    },
    "tra_cuu_hoc_phi": {
        "default": "Học phí tùy theo ngành và bậc đào tạo. Liên hệ phòng Kế toán A.105 để biết chi tiết.",
        "CNTT": "Học phí ngành CNTT khoảng 25-30 triệu/học kỳ (tham khảo). Chi tiết tại phòng A.105.",
        "công nghệ": "Học phí các ngành công nghệ dao động 22-28 triệu/học kỳ. Xem bảng học phí tại utt.edu.vn.",
        "đóng muộn": "Nộp tại phòng A.101, phí phạt 50.000đ/tuần. Mang theo CMND và thẻ sinh viên. Giờ làm việc: 8h-11h30, 13h30-16h30.",
    },
    "lich_thi": {
        "default": "Lịch thi được công bố trên cổng thông tin sinh viên 2 tuần trước kỳ thi. Kiểm tra tại phòng Đào tạo A.102.",
        "học kỳ 1": "Lịch thi học kỳ 1 thường vào cuối tháng 12. Xem chi tiết trên website trường.",
    },
    "thong_bao": {
        "default": "Các thông báo mới nhất xem tại bảng tin sảnh A hoặc website utt.edu.vn. Bạn quan tâm thông báo nào?",
    },
    "dang_ky_mon_hoc": {
        "default": "Đăng ký môn học qua cổng thông tin sinh viên hoặc đến phòng Đào tạo A.102. Thời gian đăng ký: tuần 1-2 của học kỳ.",
    },
    "lien_he_phong_ban": {
        "default": "Phòng Đào tạo: A.102 | Phòng CTSV: A.101 | Phòng Kế toán: A.105. Hotline: 024.3767.xxxx. Xem chi tiết tại utt.edu.vn.",
    },
}


class NLUEngine:
    """NLU Engine - Intent + Entity extraction"""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[str, Dict] = {}

    def understand(self, text: str) -> Dict[str, Any]:
        """
        Phân tích câu hỏi -> intent + entities.

        Args:
            text: Câu hỏi của user

        Returns:
            {
                "intent": str,
                "params": dict,
                "confidence": float,
                "response": str,
                "room": str | None
            }
        """
        text = text.strip()
        if not text:
            return self._fallback_response()

        # Check cache
        if self.use_cache and text in self._cache:
            return self._cache[text]

        text_lower = text.lower()
        text_norm = self._normalize(text_lower)

        best_intent = "unknown"
        best_params: Dict[str, str] = {}
        best_score = 0.0

        for intent, config in INTENT_PATTERNS.items():
            score = 0
            params = {}

            # Keyword matching
            for kw in config["keywords"]:
                if kw in text_lower or kw in text_norm:
                    score += 2
                    break

            # Entity extraction
            for entity_name, pattern in config["entity_patterns"].items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    params[entity_name] = match.group(1).strip()
                    score += 3

            if score > best_score:
                best_score = score
                best_intent = intent
                best_params = params

        # Extract phòng từ pattern chung (A.xxx, B.xxx)
        if "phong" not in best_params:
            phong_match = re.search(r"([A-Z]\.\d{3})", text, re.IGNORECASE)
            if phong_match:
                best_params["phong"] = phong_match.group(1).upper()

        # Build response
        result = self._build_response(best_intent, best_params, text)
        result["confidence"] = min(1.0, best_score / 5.0) if best_score > 0 else 0.5

        if self.use_cache:
            self._cache[text] = result

        return result

    def _normalize(self, text: str) -> str:
        """Chuẩn hóa tiếng Việt không dấu"""
        replacements = {
            "áàảãạăắằẳẵặâấầẩẫậ": "a",
            "éèẻẽẹêếềểễệ": "e",
            "íìỉĩị": "i",
            "óòỏõọôốồổỗộơớờởỡợ": "o",
            "úùủũụưứừửữự": "u",
            "ýỳỷỹỵ": "y",
            "đ": "d",
        }
        result = text
        for chars, rep in replacements.items():
            for c in chars:
                result = result.replace(c, rep)
        return result

    def _build_response(
        self,
        intent: str,
        params: Dict[str, str],
        original_text: str,
    ) -> Dict[str, Any]:
        """Tạo câu trả lời từ intent và params"""
        responses = INTENT_RESPONSES.get(intent, {})
        response_text = responses.get("default", "Xin lỗi, tôi chưa hiểu rõ. Vui lòng hỏi lại.")

        # Custom response theo entity
        phong = params.get("phong", "")
        if phong and phong in responses:
            response_text = responses[phong]

        nganh = params.get("nganh", params.get("nganh_cn", ""))
        if nganh and nganh in responses:
            response_text = responses[nganh]

        # Học phí đóng muộn
        if intent == "tra_cuu_hoc_phi" and ("đóng muộn" in original_text.lower() or "dong muon" in original_text.lower()):
            response_text = responses.get("đóng muộn", response_text)

        # Format placeholders - tránh trùng key với params
        room = phong if intent == "tim_phong" else None
        format_vars = dict(params)
        format_vars["phong"] = phong or "?"
        format_vars["tang"] = phong[0] if phong else "?"
        try:
            response_text = response_text.format(**format_vars)
        except KeyError:
            pass  # Giữ nguyên nếu format lỗi

        return {
            "intent": intent,
            "params": params,
            "response": response_text,
            "room": room,
        }

    def _fallback_response(self) -> Dict[str, Any]:
        return {
            "intent": "unknown",
            "params": {},
            "response": "Xin lỗi, tôi không nghe rõ. Bạn có thể nhắc lại không?",
            "room": None,
            "confidence": 0.0,
        }
