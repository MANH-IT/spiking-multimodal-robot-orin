# -*- coding: utf-8 -*-
"""
Cross-Lingual Adapter - NCKH 2026
====================================
Adapter đa ngôn ngữ Việt–Anh–Trung với:
- Multilingual sentence embeddings (sentence-transformers)
- Automatic language detection (langdetect)
- Language-specific fine-tuning adapters (Bottleneck adapter layers)
- Cross-lingual alignment layer

Tham chiếu đề cương NLP mục 3.4.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple

# --- sentence-transformers (multilingual) ---
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[CrossLingual] WARNING: sentence_transformers not found.")
    print("  pip install sentence-transformers")

# --- langdetect ---
try:
    from langdetect import detect as langdetect_detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("[CrossLingual] WARNING: langdetect not found. pip install langdetect")

SUPPORTED_LANGS = ['vi', 'en', 'zh']
MULTILINGUAL_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBED_DIM = 384   # Output dim của MiniLM-L12


# ============================================================
# Bottleneck Adapter Layer
# ============================================================

class BottleneckAdapter(nn.Module):
    """
    Lightweight adapter layer (Houlsby et al., 2019) cho language-specific tuning.
    Kiến trúc: Down-projection → GELU → Up-projection, với residual.

    Args:
        input_dim: Chiều embedding đầu vào
        bottleneck_dim: Chiều bottleneck (thường input_dim // 8)
    """

    def __init__(self, input_dim: int = EMBED_DIM, bottleneck_dim: int = 48):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

        # Khởi tạo gần zero để ban đầu không ảnh hưởng đến base embedding
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim)"""
        adapted = self.up(self.act(self.down(x)))
        return self.norm(x + adapted)   # Residual connection


# ============================================================
# Cross-Lingual Alignment Layer
# ============================================================

class CrossLingualAlignment(nn.Module):
    """
    Align feature vectors từ nhiều ngôn ngữ vào 1 chung space.
    Dùng attention-based pooling + projection.

    Args:
        embed_dim: Chiều embedding đầu vào
        out_dim: Chiều output (aligned space)
    """

    def __init__(self, embed_dim: int = EMBED_DIM, out_dim: int = 256):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: list of tensors, mỗi tensor (B, embed_dim)
        Returns: (B, out_dim) aligned representation
        """
        if len(features) == 1:
            return self.proj(features[0])

        # Stack → (B, num_langs, embed_dim)
        stacked = torch.stack(features, dim=1)
        # Self-attention across languages
        attn_out, _ = self.attn(stacked, stacked, stacked)
        # Mean pool
        pooled = attn_out.mean(dim=1)  # (B, embed_dim)
        return self.proj(pooled)


# ============================================================
# CrossLingualAdapter
# ============================================================

class CrossLingualAdapter(nn.Module):
    """
    Adapter đa ngôn ngữ cho robot.

    Workflow:
        text (vi/en/zh)
            ↓ language detection
            ↓ multilingual embedding (sentence-transformers)
            ↓ language-specific BottleneckAdapter
            ↓ CrossLingualAlignment
            → aligned feature vector (256-dim)

    Args:
        target_languages: Danh sách ngôn ngữ hỗ trợ
        embed_dim: Chiều embedding của base model
        out_dim: Chiều output sau alignment
        use_gpu: Có dùng GPU không
    """

    def __init__(
        self,
        target_languages: List[str] = None,
        embed_dim: int = EMBED_DIM,
        out_dim: int = 256,
        use_gpu: bool = True,
    ):
        super().__init__()
        if target_languages is None:
            target_languages = SUPPORTED_LANGS
        self.target_languages = target_languages
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Language-specific adapters
        self.adapters = nn.ModuleDict({
            lang: BottleneckAdapter(embed_dim) for lang in target_languages
        })

        # Cross-lingual alignment
        self.alignment_layer = CrossLingualAlignment(embed_dim, out_dim)

        # Classification head (intent)
        self.lang_classifier = nn.Linear(out_dim, len(target_languages))

        # Base multilingual model (không train, dùng để extract features)
        self._base_model: Optional[SentenceTransformer] = None
        self._device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

    def _load_base_model(self):
        """Lazy load sentence-transformers model."""
        if self._base_model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "Cần cài: pip install sentence-transformers"
                )
            print(f"[CrossLingual] Loading {MULTILINGUAL_MODEL}...")
            self._base_model = SentenceTransformer(
                MULTILINGUAL_MODEL,
                device=self._device,
            )
            # Freeze base model
            for param in self._base_model.parameters():
                param.requires_grad = False
            print("[CrossLingual] Base model loaded.")

    def detect_language(self, text: str) -> str:
        """Tự động detect ngôn ngữ từ text."""
        if HAS_LANGDETECT:
            try:
                detected = langdetect_detect(text)
                _MAP = {
                    'vi': 'vi', 'en': 'en',
                    'zh-cn': 'zh', 'zh-tw': 'zh', 'zh': 'zh',
                }
                lang = _MAP.get(detected, 'vi')
                if lang not in self.target_languages:
                    lang = 'vi'
                return lang
            except Exception:
                pass
        # Fallback: ký tự đặc trưng
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return 'zh'
        vi_chars = set('àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặ')
        if any(c in vi_chars for c in text.lower()):
            return 'vi'
        return 'en'

    def encode_text(
        self,
        text: Union[str, List[str]],
        lang: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode text → embedding tensor dùng sentence-transformers.

        Returns: (B, embed_dim) hoặc (embed_dim,) nếu input là str
        """
        self._load_base_model()
        single = isinstance(text, str)
        texts = [text] if single else text

        embeddings = self._base_model.encode(
            texts,
            convert_to_tensor=True,
            device=self._device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # (B, embed_dim)

        if single:
            return embeddings[0]
        return embeddings

    def forward(
        self,
        texts: List[str],
        langs: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Xử lý batch texts đa ngôn ngữ.

        Args:
            texts: List các câu văn bản
            langs: List ngôn ngữ tương ứng (auto-detect nếu None)

        Returns:
            {
                'aligned_features': (B, out_dim) — dùng cho downstream tasks,
                'lang_logits': (B, num_langs) — phân loại ngôn ngữ,
                'detected_langs': List[str],
                'adapted_features': (B, embed_dim) — trước alignment,
            }
        """
        # ── 1. Language detection ──────────────────────────────
        if langs is None:
            detected_langs = [self.detect_language(t) for t in texts]
        else:
            detected_langs = langs

        # ── 2. Base embedding ──────────────────────────────────
        base_embeds = self.encode_text(texts)   # (B, embed_dim)

        # ── 3. Language-specific adaptation ───────────────────
        adapted_list = []
        for i, lang in enumerate(detected_langs):
            adapter = self.adapters.get(lang, self.adapters['vi'])
            adapted = adapter(base_embeds[i])   # (embed_dim,)
            adapted_list.append(adapted)

        adapted_embeds = torch.stack(adapted_list)  # (B, embed_dim)

        # ── 4. Cross-lingual alignment ─────────────────────────
        # Align theo từng ngôn ngữ rồi average
        lang_groups: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
        for i, lang in enumerate(detected_langs):
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append((i, adapted_embeds[i]))

        aligned = self.alignment_layer([adapted_embeds])  # (B, out_dim)

        # ── 5. Language classifier ─────────────────────────────
        lang_logits = self.lang_classifier(aligned)   # (B, num_langs)

        return {
            'aligned_features': aligned,
            'lang_logits': lang_logits,
            'detected_langs': detected_langs,
            'adapted_features': adapted_embeds,
        }

    def process_multilingual(
        self,
        text: Union[str, List[str]],
        languages: Optional[List[str]] = None,
    ) -> Dict:
        """API đơn giản: text → aligned features."""
        texts = [text] if isinstance(text, str) else text
        if languages is None:
            languages = [self.detect_language(t) for t in texts]

        with torch.no_grad():
            out = self.forward(texts, languages)

        return {
            'features': out['aligned_features'].cpu().numpy(),
            'detected_langs': out['detected_langs'],
        }

    def embed_single(self, text: str, lang: Optional[str] = None) -> torch.Tensor:
        """Encode 1 câu → aligned feature vector."""
        if lang is None:
            lang = self.detect_language(text)
        with torch.no_grad():
            out = self.forward([text], [lang])
        return out['aligned_features'][0]  # (out_dim,)


# ============================================================
# Simplified version (không cần sentence-transformers)
# ============================================================

class SimpleCrossLingualAdapter(nn.Module):
    """
    Phiên bản đơn giản, không phụ thuộc sentence-transformers.
    Dùng character-level features để phân biệt ngôn ngữ.
    Phù hợp cho testing và khi chưa cài đủ dependency.
    """

    def __init__(self, embed_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Char embedding
        self.char_embed = nn.Embedding(256, 32)

        # Encoder
        self.encoder = nn.GRU(32, embed_dim // 2, batch_first=True,
                               bidirectional=True)

        # Adapters
        self.adapters = nn.ModuleDict({
            lang: BottleneckAdapter(embed_dim, embed_dim // 8)
            for lang in SUPPORTED_LANGS
        })

        self.proj = nn.Linear(embed_dim, out_dim)

    def _text_to_tensor(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Encode text as byte indices."""
        encoded = [min(b, 255) for b in text.encode('utf-8')][:max_len]
        padded = encoded + [0] * (max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    def forward(self, texts: List[str], langs: List[str]) -> torch.Tensor:
        tensors = torch.stack([self._text_to_tensor(t) for t in texts])
        embedded = self.char_embed(tensors)          # (B, L, 32)
        _, hidden = self.encoder(embedded)           # hidden: (2, B, H/2)
        feat = torch.cat([hidden[0], hidden[1]], -1) # (B, H)

        adapted_list = []
        for i, lang in enumerate(langs):
            adapter = self.adapters.get(lang, self.adapters['vi'])
            adapted_list.append(adapter(feat[i]))
        adapted = torch.stack(adapted_list)

        return self.proj(adapted)  # (B, out_dim)

    def process_multilingual(self, text, languages=None):
        texts = [text] if isinstance(text, str) else text
        if languages is None:
            languages = ['vi'] * len(texts)
        with torch.no_grad():
            out = self.forward(texts, languages)
        return {'features': out.cpu().numpy(), 'detected_langs': languages}


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    print("=== CrossLingualAdapter Test ===\n")

    # Test SimpleCrossLingualAdapter (không cần sentence-transformers)
    simple_adapter = SimpleCrossLingualAdapter(embed_dim=256, out_dim=256)
    test_texts = [
        "Phòng A.205 ở đâu ạ?",
        "Where is gate B2?",
        "飞往河内的航班在哪个登机口？",
    ]
    langs = ['vi', 'en', 'zh']

    print("=== SimpleCrossLingualAdapter ===")
    out = simple_adapter(test_texts, langs)
    print(f"Output shape: {out.shape}")

    # Language detection test
    if HAS_SENTENCE_TRANSFORMERS:
        print("\n=== Full CrossLingualAdapter ===")
        adapter = CrossLingualAdapter()
        for text, lang in zip(test_texts, langs):
            detected = adapter.detect_language(text)
            print(f"[{lang}] '{text[:40]}...' → detected: {detected}")
    else:
        print("\n[INFO] sentence-transformers không có, test SimpleCrossLingualAdapter only")

    print("\n[Done]")
