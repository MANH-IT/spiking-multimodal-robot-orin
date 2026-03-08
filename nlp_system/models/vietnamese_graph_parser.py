# -*- coding: utf-8 -*-
"""
Vietnamese Graph Parser - NCKH 2026
=====================================
Implementation đầy đủ của VietnameseGraphParser với:
- Tone-aware embedding (6 thanh điệu tiếng Việt)
- Dependency graph construction (dùng underthesea)
- Named Entity Recognition
- Adaptive grammar via DynamicGrammarAdaptor
- Context-Aware language switching

Tham chiếu đề cương NLP mục 3.2.1 và 3.2.2
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- underthesea ---
try:
    from underthesea import (
        word_tokenize,
        pos_tag,
        ner,
        dependency_parse,
        sent_tokenize,
    )
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    print("[GraphParser] WARNING: underthesea not found. pip install underthesea")

# ============================================================
# Data Classes
# ============================================================

@dataclass
class ParseToken:
    """Đại diện cho 1 token trong parse tree."""
    index: int
    form: str           # dạng gốc
    lemma: str          # dạng cơ sở
    pos: str            # Part-of-speech
    tone_id: int        # Thanh điệu (0-5)
    head: int           # Index của head node (dependency)
    dep_rel: str        # Quan hệ dependency


@dataclass
class ParseResult:
    """Kết quả phân tích cú pháp của 1 câu."""
    text: str
    tokens: List[ParseToken]
    entities: List[Dict[str, Any]]   # NER results
    intent_hint: str = ""            # Gợi ý intent từ cú pháp
    language: str = "vi"
    raw_pos: List[Tuple] = field(default_factory=list)
    raw_dep: List[Tuple] = field(default_factory=list)


# ============================================================
# Tone Detection
# ============================================================

_TONE_CHARS: Dict[int, str] = {
    0: 'aăâeêioôơuưy',                     # bằng (level)
    1: 'àằầèềìòồờùừỳ',                     # huyền (grave)
    2: 'ảẳẩẻểỉỏổởủửỷ',                     # hỏi (hook above)
    3: 'ãẵẫẽễĩõỗỡũữỹ',                     # ngã (tilde)
    4: 'áắấéếíóốớúứý',                      # sắc (acute)
    5: 'ạặậẹệịọộợụựỵ',                      # nặng (dot below)
}

_CHAR_TO_TONE: Dict[str, int] = {}
for _tid, _chars in _TONE_CHARS.items():
    for _c in _chars:
        _CHAR_TO_TONE[_c] = _tid

TONE_NAMES = ['bằng', 'huyền', 'hỏi', 'ngã', 'sắc', 'nặng']
NUM_TONES = 6


def detect_tone(syllable: str) -> int:
    """Phát hiện thanh điệu của âm tiết tiếng Việt (0-5)."""
    for ch in syllable.lower():
        if ch in _CHAR_TO_TONE:
            return _CHAR_TO_TONE[ch]
    return 0  # bằng (mặc định)


def is_vietnamese(text: str) -> bool:
    """Kiểm tra nhanh xem text có ký tự tiếng Việt không."""
    vi_chars = set('àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ')
    return any(c in vi_chars for c in text.lower())


# ============================================================
# Tone Embedding Layer
# ============================================================

class ToneEmbeddingLayer(nn.Module):
    """
    Embedding cho 6 thanh điệu tiếng Việt.
    Được concatenate với word embedding để tạo tone-aware representation.
    """

    def __init__(self, tone_embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(NUM_TONES, tone_embed_dim)
        nn.init.normal_(self.embed.weight, std=0.1)

    def forward(self, tone_ids: torch.Tensor) -> torch.Tensor:
        """tone_ids: (B, seq_len) → (B, seq_len, tone_embed_dim)"""
        return self.embed(tone_ids)


# ============================================================
# Vietnamese Graph Parser (Main Class)
# ============================================================

class VietnameseGraphParser(nn.Module):
    """
    Graph-based parser đặc thù cho tiếng Việt.

    Components:
    - ToneEmbeddingLayer: encode 6 thanh điệu
    - underthesea backend: word_tokenize, pos_tag, ner, dependency_parse
    - DynamicGrammarAdaptor: context-aware rule selection
    - Learnable projection: raw parse features → dense vector

    Args:
        word_embed_dim: Chiều word embedding đầu vào (để concat với tone)
        tone_embed_dim: Chiều tone embedding
        hidden_dim: Chiều output projection
    """

    def __init__(
        self,
        word_embed_dim: int = 256,
        tone_embed_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Tone-aware embedding
        self.tone_embedding = ToneEmbeddingLayer(tone_embed_dim)

        # Projection: (word_embed + tone_embed) → hidden
        self.graph_proj = nn.Sequential(
            nn.Linear(word_embed_dim + tone_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Dependency relation embedding (HEAD, NSUBJ, OBJ, ...)
        self.dep_embed = nn.Embedding(64, 32)

        # Grammar adaptor
        self.grammar_adaptor = DynamicGrammarAdaptor()

        # POS tag map (underthesea)
        self.pos_map = {
            'N': 'NOUN', 'V': 'VERB', 'A': 'ADJ', 'R': 'ADV',
            'P': 'PRON', 'M': 'NUM', 'E': 'PREP', 'C': 'CONJ',
            'I': 'INTJ', 'X': 'OTHER', 'Np': 'PROPN', 'Nc': 'NOUN',
            'Nu': 'NUM',  'Cc': 'CONJ', 'T': 'PART',
        }

    # ----------------------------------------------------------
    # Core parsing (dùng underthesea)
    # ----------------------------------------------------------

    def parse_vietnamese(self, sentence: str) -> ParseResult:
        """
        Phân tích đầy đủ 1 câu tiếng Việt.

        Returns ParseResult với tokens, entities, dependency tree.
        """
        # ── Tokenize ──────────────────────────────────────────
        if HAS_UNDERTHESEA:
            words = word_tokenize(sentence, format='text').split()
        else:
            words = sentence.split()

        # ── POS Tagging ────────────────────────────────────────
        raw_pos: List[Tuple] = []
        if HAS_UNDERTHESEA:
            try:
                raw_pos = pos_tag(sentence)
            except Exception:
                raw_pos = [(w, 'X') for w in words]
        else:
            raw_pos = [(w, 'X') for w in words]

        # ── NER ────────────────────────────────────────────────
        entities: List[Dict] = []
        if HAS_UNDERTHESEA:
            try:
                ner_result = ner(sentence)
                for word, pos, chunk, entity in ner_result:
                    if entity != 'O':
                        entities.append({
                            'word': word,
                            'entity': entity,
                            'pos': pos,
                        })
            except Exception:
                pass

        # ── Dependency parsing ─────────────────────────────────
        raw_dep: List[Tuple] = []
        if HAS_UNDERTHESEA:
            try:
                raw_dep = dependency_parse(sentence)
            except Exception:
                raw_dep = []

        # ── Build ParseToken list ──────────────────────────────
        tokens = []
        for i, (word, pos) in enumerate(raw_pos):
            # Tìm dep relation nếu có
            head_idx, dep_rel = 0, 'ROOT'
            if i < len(raw_dep):
                try:
                    _, _, head_idx, dep_rel = raw_dep[i]
                except (ValueError, IndexError):
                    pass

            tokens.append(ParseToken(
                index=i,
                form=word,
                lemma=word.lower(),
                pos=self.pos_map.get(pos, 'X'),
                tone_id=detect_tone(word),
                head=int(head_idx) if head_idx else 0,
                dep_rel=str(dep_rel) if dep_rel else 'ROOT',
            ))

        # ── Infer intent hint từ cú pháp ──────────────────────
        intent_hint = self._infer_intent_hint(tokens, entities)

        return ParseResult(
            text=sentence,
            tokens=tokens,
            entities=entities,
            intent_hint=intent_hint,
            language='vi',
            raw_pos=raw_pos,
            raw_dep=raw_dep,
        )

    def parse_multilingual(self, sentence: str, lang: str = 'auto') -> ParseResult:
        """Parse cho nhiều ngôn ngữ."""
        if lang == 'auto':
            lang = 'vi' if is_vietnamese(sentence) else 'en'

        if lang == 'vi':
            return self.parse_vietnamese(sentence)

        # Tiếng Anh / Trung: simple tokenization
        words = re.findall(r'[\w\u4e00-\u9fff]+|[^\w\s]', sentence.lower())
        tokens = [
            ParseToken(i, w, w, 'X', 0, 0, 'ROOT')
            for i, w in enumerate(words)
        ]
        return ParseResult(
            text=sentence,
            tokens=tokens,
            entities=[],
            language=lang,
        )

    # ----------------------------------------------------------
    # Intent hint from syntax
    # ----------------------------------------------------------

    def _infer_intent_hint(
        self,
        tokens: List[ParseToken],
        entities: List[Dict],
    ) -> str:
        """
        Suy luận gợi ý intent từ cú pháp (rule-based, nhanh).
        Kết hợp với SNN model để đưa ra quyết định cuối.
        """
        text = ' '.join(t.form.lower() for t in tokens)
        verbs = [t.form.lower() for t in tokens if t.pos == 'VERB']
        nouns = [t.form.lower() for t in tokens if t.pos in ('NOUN', 'PROPN')]

        # Sân bay context
        airport_words = {'gate', 'cổng', 'boarding', 'check-in', 'chuyến', 'bay', 'flight'}
        if any(w in text for w in airport_words):
            if any(w in text for w in ('ở đâu', 'đâu', 'hướng', 'đi')):
                return 'ask_directions'
            if any(w in text for w in ('check-in', 'thủ tục', 'làm thủ tục')):
                return 'check_in_request'
            return 'check_flight_gate'

        # Lớp học context
        classroom_words = {'giải thích', 'giảng', 'bài tập', 'điểm', 'thi', 'môn', 'thầy', 'cô'}
        if any(w in text for w in classroom_words):
            if any(w in text for w in ('giải thích', 'giảng', 'nghĩa')):
                return 'ask_explanation'
            if any(w in text for w in ('bài tập', 'giải', 'làm')):
                return 'solve_problem'
            if any(w in text for w in ('điểm', 'điểm số')):
                return 'ask_grade'
            return 'ask_homework'

        # Gia đình context
        home_words = {'bật', 'tắt', 'mở', 'đèn', 'tivi', 'máy lạnh', 'nhắc', 'báo thức'}
        if any(w in text for w in home_words):
            if any(w in text for w in ('nhắc', 'báo thức', 'hẹn giờ')):
                return 'set_reminder'
            return 'control_device'

        return ''

    # ----------------------------------------------------------
    # Feature extraction (cho integration với SNN model)
    # ----------------------------------------------------------

    def extract_features(
        self,
        parse_result: ParseResult,
        word_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Trích xuất feature vector từ parse result.

        Args:
            parse_result: Kết quả từ parse_vietnamese()
            word_embeddings: (seq_len, embed_dim) embedding từ SNN model

        Returns:
            (seq_len, hidden_dim) parse-aware features
        """
        seq_len = len(parse_result.tokens)

        # Tone IDs
        tone_ids = torch.tensor(
            [t.tone_id for t in parse_result.tokens],
            dtype=torch.long,
        )  # (seq_len,)
        tone_feats = self.tone_embedding(tone_ids)  # (seq_len, tone_dim)

        if word_embeddings is not None:
            # Concat word embed + tone embed → project
            combined = torch.cat([word_embeddings[:seq_len], tone_feats], dim=-1)
            return self.graph_proj(combined)
        else:
            # Standalone: dùng tone feats + dep relation embed
            dep_ids = torch.tensor(
                [hash(t.dep_rel) % 64 for t in parse_result.tokens],
                dtype=torch.long,
            )
            dep_feats = self.dep_embed(dep_ids)  # (seq_len, 32)
            # Pad tone_feats đến hidden_dim
            padded = F.pad(tone_feats, (0, self.graph_proj[0].out_features - tone_feats.shape[-1]))
            return padded

    def to_dict(self, result: ParseResult) -> dict:
        """Chuyển ParseResult → dict để logging / debug."""
        return {
            'text': result.text,
            'language': result.language,
            'intent_hint': result.intent_hint,
            'tokens': [
                {
                    'form': t.form,
                    'pos': t.pos,
                    'tone': TONE_NAMES[t.tone_id],
                    'dep_rel': t.dep_rel,
                }
                for t in result.tokens
            ],
            'entities': result.entities,
        }


# ============================================================
# Dynamic Grammar Adaptor
# ============================================================

class DynamicGrammarAdaptor:
    """
    Context-aware grammar rule selection và adaptation.

    Ngữ cảnh hỗ trợ: airport, classroom, home.
    Ngôn ngữ: vi, en, zh
    """

    # Grammar rules theo context
    CONTEXT_RULES: Dict[str, Dict] = {
        'airport': {
            'question_patterns': [
                r'(chuyến|flight|航班)\s+(\w+)',
                r'(cổng|gate|登机口)\s+(\w+)',
                r'(check-in|làm thủ tục)',
            ],
            'entity_types': ['flight_number', 'destination', 'gate'],
            'primary_lang': 'vi',
            'fallback_langs': ['en', 'zh'],
        },
        'classroom': {
            'question_patterns': [
                r'(bài|câu|problem)\s+(\d+)',
                r'(môn|subject)\s+(\w+)',
                r'(điểm|grade|score)',
            ],
            'entity_types': ['lesson_number', 'subject', 'grade'],
            'primary_lang': 'vi',
            'fallback_langs': ['en'],
        },
        'home': {
            'question_patterns': [
                r'(bật|tắt|mở|turn on|turn off)\s+(\w+)',
                r'(nhắc|remind|remind me)',
                r'(thời tiết|weather|天气)',
            ],
            'entity_types': ['device', 'location', 'time'],
            'primary_lang': 'vi',
            'fallback_langs': ['en', 'zh'],
        },
    }

    def __init__(self):
        self._interaction_history: List[Dict] = []
        self._grammar_weights: Dict[str, float] = {
            ctx: 1.0 for ctx in self.CONTEXT_RULES
        }

    def detect_context(self, text: str) -> str:
        """Tự động detect ngữ cảnh từ text."""
        text_lower = text.lower()

        airport_kw = {'chuyến', 'bay', 'gate', 'cổng', 'check-in', 'flight',
                      'sân bay', 'airport', 'boarding', '航班', 'hành lý'}
        classroom_kw = {'thầy', 'cô', 'bài', 'môn', 'học', 'thi', 'điểm',
                        'teacher', 'homework', 'grade', 'class', 'subject'}
        home_kw = {'bật', 'tắt', 'đèn', 'tivi', 'nhắc', 'thời tiết',
                   'máy lạnh', 'turn on', 'weather', 'reminder', '打开', '天气'}

        scores = {
            'airport': sum(1 for kw in airport_kw if kw in text_lower),
            'classroom': sum(1 for kw in classroom_kw if kw in text_lower),
            'home': sum(1 for kw in home_kw if kw in text_lower),
        }

        best = max(scores, key=lambda k: scores[k] * self._grammar_weights[k])
        return best if scores[best] > 0 else 'home'

    def adapt_grammar(
        self,
        sentence: str,
        context: Optional[str] = None,
        lang: str = 'vi',
    ) -> Dict[str, Any]:
        """
        Điều chỉnh ngữ pháp dựa trên ngữ cảnh.

        Returns:
            {
                'context': str,
                'lang': str,
                'matched_patterns': list,
                'extracted_entities': dict,
                'confidence': float,
            }
        """
        if context is None:
            context = self.detect_context(sentence)

        rules = self.CONTEXT_RULES.get(context, self.CONTEXT_RULES['home'])
        matched = []
        extracted = {}

        for pattern in rules['question_patterns']:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                matched.append(pattern)
                # Lấy groups nếu có
                if m.lastindex:
                    for i in range(1, m.lastindex + 1):
                        extracted[f'entity_{i}'] = m.group(i)

        confidence = min(1.0, len(matched) / max(1, len(rules['question_patterns'])))

        result = {
            'context': context,
            'lang': lang,
            'matched_patterns': matched,
            'extracted_entities': extracted,
            'confidence': confidence,
        }

        # Online learning: ghi lại interaction
        self._interaction_history.append({
            'sentence': sentence,
            'result': result,
        })
        self._update_grammar_weights(context, confidence)

        return result

    def _update_grammar_weights(self, context: str, confidence: float):
        """Online learning: tăng weight cho context được dùng nhiều."""
        for ctx in self._grammar_weights:
            if ctx == context:
                self._grammar_weights[ctx] = min(2.0,
                    self._grammar_weights[ctx] * 1.01)
            else:
                self._grammar_weights[ctx] = max(0.5,
                    self._grammar_weights[ctx] * 0.999)

    def get_stats(self) -> dict:
        return {
            'total_interactions': len(self._interaction_history),
            'context_weights': dict(self._grammar_weights),
        }


# ============================================================
# Context-Aware Language Switch
# ============================================================

class ContextAwareSwitch:
    """
    Tự động chuyển ngôn ngữ dựa trên ngữ cảnh thực tế.
    Tham chiếu đề cương mục 3.4.2
    """

    LOCATION_LANG = {
        'airport':    ['en', 'zh', 'vi'],   # Sân bay: ưu tiên Anh
        'classroom':  ['vi', 'en'],          # Lớp: ưu tiên Việt
        'home':       ['vi'],                # Gia đình: Việt
        'hospital':   ['vi', 'en'],
    }

    def __init__(self):
        self._user_profiles: Dict[str, Dict] = {}

    def determine_language(self, context: Dict[str, Any]) -> str:
        """
        Xác định ngôn ngữ phù hợp nhất.

        context = {
            'location': 'airport' | 'classroom' | 'home',
            'user_id': str (optional),
            'task': str (optional),
            'detected_lang': str (optional, từ langdetect),
        }
        """
        # Ưu tiên 1: ngôn ngữ đã detect được
        if 'detected_lang' in context and context['detected_lang']:
            return context['detected_lang']

        # Ưu tiên 2: user preference
        user_id = context.get('user_id')
        if user_id and user_id in self._user_profiles:
            pref = self._user_profiles[user_id].get('preferred_lang')
            if pref:
                return pref

        # Ưu tiên 3: location-based rule
        location = context.get('location', 'home')
        langs = self.LOCATION_LANG.get(location, ['vi'])
        return langs[0]

    def update_user_preference(self, user_id: str, lang: str):
        self._user_profiles[user_id] = {'preferred_lang': lang}

    def detect_language_from_text(self, text: str) -> str:
        """Detect ngôn ngữ từ text (dùng langdetect nếu có)."""
        try:
            from langdetect import detect
            detected = detect(text)
            # Map langdetect codes → our codes
            lang_map = {'vi': 'vi', 'en': 'en', 'zh-cn': 'zh',
                        'zh-tw': 'zh', 'zh': 'zh'}
            return lang_map.get(detected, 'vi')
        except Exception:
            # Fallback: kiểm tra ký tự
            if is_vietnamese(text):
                return 'vi'
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                return 'zh'
            return 'en'


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("=== VietnameseGraphParser Test ===\n")
    parser = VietnameseGraphParser()
    adaptor = DynamicGrammarAdaptor()
    lang_switch = ContextAwareSwitch()

    test_sentences = [
        ("Phòng A.205 ở đâu ạ?", 'vi'),
        ("Chuyến bay VN123 đến Hà Nội ở gate nào?", 'vi'),
        ("Thầy ơi, giải thích lại định luật Newton được không?", 'vi'),
        ("Bật đèn phòng khách lên giúp tôi.", 'vi'),
        ("Where is the boarding gate for flight VN456?", 'en'),
    ]

    for text, lang in test_sentences:
        print(f"Input [{lang}]: {text}")
        result = parser.parse_vietnamese(text) if lang == 'vi' \
                 else parser.parse_multilingual(text, lang)
        print(f"  Intent hint : {result.intent_hint or '(none)'}")
        print(f"  Entities    : {result.entities[:2]}")
        print(f"  POS tags    : {[(t.form, t.pos) for t in result.tokens[:5]]}")

        ctx = adaptor.detect_context(text)
        grammar = adaptor.adapt_grammar(text, ctx, lang)
        print(f"  Context     : {grammar['context']}")
        print(f"  Confidence  : {grammar['confidence']:.2f}")

        detected_lang = lang_switch.detect_language_from_text(text)
        print(f"  Detected lang: {detected_lang}")
        print()
