"""
Multimodal Fusion Transformer — Full Model
==========================================
Thay thế rule-based MultimodalBridge bằng mô hình học sâu.

Pipeline:
    [SNN Vision Output (T=8)]   [Text Input (tiếng Việt)]
           ↓                            ↓
    TemporalSpikeAggregator      TextEncoder (PhoBERT / SpikeEmb)
           ↓                            ↓
              TemporalCrossAttentionFusion (3 layers, 4 heads)
                            ↓
                   Fused Representation (B, d_model)
                            ↓
              ┌─────────────┴─────────────┐
         ActionHead                  ConfidenceHead
     (action_type probs)          (confidence scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from multimodal_fusion.temporal_cross_attention import TemporalCrossAttentionFusion


# ─────────────────────────────────────────────────────────────────────────────
# Intent & Action class definitions (đồng nhất với toàn bộ hệ thống)
# ─────────────────────────────────────────────────────────────────────────────

# Intent classes từ training_data.json (NLU tiếng Việt)
NLU_INTENTS = ["thong_tin_truong", "tuyen_sinh", "dao_tao", "nghien_cuu", "khac"]

# Robot action types (mapping sang hardware controller)
ROBOT_ACTIONS = [
    "idle",            # 0 — đứng yên
    "follow",          # 1 — theo dõi người
    "navigate_around", # 2 — tránh vật cản
    "approach_and_greet",# 3 — tiến lại chào
    "search",          # 4 — quay tìm người
    "respond",         # 5 — trả lời (RAG mode, dừng robot)
]

NUM_ACTIONS = len(ROBOT_ACTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Text Encoder — adapter nhẹ sau PhoBERT CLS token
# ─────────────────────────────────────────────────────────────────────────────

class TextProjector(nn.Module):
    """
    Project PhoBERT embedding (768-dim CLS) → d_model tokens.
    Hoặc expand single vector thành Nt tokens bằng learned upsampler.
    """
    def __init__(self, text_dim: int = 768, d_model: int = 256, n_tokens: int = 16):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Linear(text_dim, d_model * n_tokens)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embedding: (B, text_dim) — CLS token từ PhoBERT
        Returns:
            tokens: (B, n_tokens, d_model)
        """
        B = cls_embedding.shape[0]
        x = self.proj(cls_embedding)               # (B, d_model * n_tokens)
        x = x.view(B, self.n_tokens, -1)           # (B, n_tokens, d_model)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Spike Text Projector — dùng khi KHÔNG có PhoBERT (edge case)
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedTextEncoder(nn.Module):
    """
    Fallback encoder: vocab nhỏ (tiếng Việt keyword-based) → d_model.
    Dùng khi PhoBERT không load được trên Jetson edge.
    """
    VOCAB = {
        "<pad>": 0, "<unk>": 1,
        "theo": 2, "tránh": 3, "chào": 4, "hỏi": 5, "tìm": 6,
        "phòng": 7, "tầng": 8, "thư viện": 9, "căn tin": 10,
        "thông tin": 11, "tuyển sinh": 12, "học phí": 13,
        "ngành": 14, "nghiên cứu": 15, "đề tài": 16,
        "follow": 17, "avoid": 18, "hello": 19, "stop": 20,
    }

    def __init__(self, d_model: int = 256, n_tokens: int = 16):
        super().__init__()
        self.n_tokens = n_tokens
        vocab_size = len(self.VOCAB)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def encode_text(self, text: str) -> torch.Tensor:
        """String → token indices (simple keyword matching)."""
        text_lower = text.lower()
        indices = []
        for kw, idx in self.VOCAB.items():
            if kw in text_lower and kw not in ("<pad>", "<unk>"):
                indices.append(idx)
        # Pad / truncate tới n_tokens
        if len(indices) == 0:
            indices = [1]  # <unk>
        indices = (indices * self.n_tokens)[:self.n_tokens]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args: token_ids (B, n_tokens)
        Returns: (B, n_tokens, d_model)
        """
        x = self.embedding(token_ids) + self.pos_embed[:, :token_ids.shape[1], :]
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Action Head
# ─────────────────────────────────────────────────────────────────────────────

class ActionHead(nn.Module):
    """
    MLP phân loại action từ fused representation.
    Output: phân phối xác suất qua NUM_ACTIONS classes.
    """
    def __init__(self, d_model: int = 256, num_actions: int = NUM_ACTIONS, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args: fused (B, d_model)
        Returns: logits (B, num_actions)
        """
        return self.net(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-Gating Head
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceHead(nn.Module):
    """
    Dự đoán confidence tổng thể của quyết định (0–1).
    Nếu confidence < threshold → yêu cầu người dùng xác nhận.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns: (B, 1) confidence scores."""
        return self.net(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Main: SpikingFusionTransformer
# ─────────────────────────────────────────────────────────────────────────────

class SpikingFusionTransformer(nn.Module):
    """
    Full Multimodal Fusion Transformer.
    
    Thay thế class MultimodalBridge (rule-based) bằng mô hình học sâu.
    Compatible với output của SNNVisionWrapper và RealNLPProcessor.
    
    Args:
        vision_channels: số channel output của SNN (default: 64)
        text_dim:        dim của PhoBERT CLS token (768) hoặc SpikeEmb
        d_model:         chiều model nội bộ (default: 256)
        num_heads:       số attention heads (default: 4)
        num_layers:      số cross-attention blocks (default: 3)
        T:               số timesteps SNN (default: 8)
        n_text_tokens:   số token text cho cross-attention (default: 16)
        confidence_thr:  ngưỡng để yêu cầu xác nhận từ người dùng
        use_phobert:     True = PhoBERT CLS input, False = rule-based keyword vocab
    """

    def __init__(
        self,
        vision_channels: int = 64,
        text_dim: int = 768,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        T: int = 8,
        spatial_size: int = 8,
        n_text_tokens: int = 16,
        confidence_thr: float = 0.6,
        use_phobert: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.confidence_thr = confidence_thr
        self.use_phobert = use_phobert
        self.d_model = d_model

        # Text encoder
        if use_phobert:
            self.text_encoder = TextProjector(text_dim, d_model, n_text_tokens)
        else:
            self.text_encoder = RuleBasedTextEncoder(d_model, n_text_tokens)

        # Cross-attention fusion
        self.fusion = TemporalCrossAttentionFusion(
            vision_channels=vision_channels,
            spatial_size=spatial_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            T=T,
            dropout=dropout,
        )

        # Heads
        self.action_head = ActionHead(d_model, NUM_ACTIONS, dropout)
        self.confidence_head = ConfidenceHead(d_model)

    def forward(
        self,
        vision_spikes: torch.Tensor,                # (T, B, C, H, W)
        text_input: torch.Tensor,                   # (B, text_dim) float nếu phobert, (B, n_tok) Long nếu rule
        vision_only: bool = False,                  # Nếu True, bỏ qua text (chỉ dùng vision)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
            action_logits:  (B, NUM_ACTIONS)
            action_probs:   (B, NUM_ACTIONS)
            action_idx:     (B,)
            confidence:     (B, 1)
            needs_confirm:  (B,) bool — True nếu confidence < threshold
            attention_info: dict
        """
        B = vision_spikes.shape[1]
        # Encode text → tokens
        if vision_only:
            text_tokens = torch.zeros(B, 16, self.d_model, device=vision_spikes.device)
        elif self.use_phobert:
            # TextProjector: input là float CLS embedding (B, text_dim)
            text_tokens = self.text_encoder(text_input.float())
        else:
            # RuleBasedTextEncoder: input là Long token ids (B, n_tok)
            text_tokens = self.text_encoder(text_input.long())

        # Fusion
        fused, attn_info = self.fusion(vision_spikes, text_tokens)

        # Heads
        action_logits = self.action_head(fused)          # (B, NUM_ACTIONS)
        action_probs  = F.softmax(action_logits, dim=-1)
        action_idx    = action_probs.argmax(dim=-1)      # (B,)
        confidence    = self.confidence_head(fused)      # (B, 1)
        needs_confirm = (confidence.squeeze(-1) < self.confidence_thr)  # (B,) bool

        return {
            "action_logits":  action_logits,
            "action_probs":   action_probs,
            "action_idx":     action_idx,
            "confidence":     confidence,
            "needs_confirm":  needs_confirm,
            "attention_info": attn_info,
        }

    def predict(
        self,
        vision_spikes: torch.Tensor,
        text_input: torch.Tensor,
    ) -> Dict:
        """Inference wrapper (no_grad)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(vision_spikes, text_input)
        idx = out["action_idx"].item()
        conf = out["confidence"].item()
        return {
            "action_type":    ROBOT_ACTIONS[idx],
            "action_idx":     idx,
            "confidence":     conf,
            "needs_confirm":  bool(out["needs_confirm"].item()),
            "action_probs":   {ROBOT_ACTIONS[i]: float(p)
                               for i, p in enumerate(out["action_probs"][0])},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_fusion_model(config: dict = None) -> SpikingFusionTransformer:
    """
    Khởi tạo mô hình từ config dict (hoặc default).
    
    Example config:
        {
            'vision_channels': 64,
            'text_dim': 768,
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 3,
            'T': 8,
            'confidence_thr': 0.6,
            'use_phobert': True,
        }
    """
    default = {
        "vision_channels": 64,
        "text_dim": 768,
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 3,
        "T": 8,
        "spatial_size": 8,
        "n_text_tokens": 16,
        "confidence_thr": 0.6,
        "use_phobert": True,
        "dropout": 0.1,
    }
    if config:
        default.update(config)
    return SpikingFusionTransformer(**default)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, C, H, W = 2, 8, 64, 16, 16

    model = build_fusion_model({"use_phobert": False})
    model.eval()

    # Mock inputs
    vision_spikes = torch.randn(T, B, C, H, W)
    token_ids = torch.randint(0, 20, (B, 16))  # Rule-based encoder

    with torch.no_grad():
        out = model.forward(vision_spikes, token_ids)

    print("✅ SpikingFusionTransformer forward pass OK")
    print(f"   action_probs shape : {out['action_probs'].shape}")
    print(f"   confidence shape   : {out['confidence'].shape}")
    print(f"   action_idx         : {out['action_idx'].tolist()}")
    print(f"   needs_confirm      : {out['needs_confirm'].tolist()}")
    print(f"\nActions: {ROBOT_ACTIONS}")
    for i, p in enumerate(out['action_probs'][0]):
        print(f"   {ROBOT_ACTIONS[i]:25s}: {p:.3f}")
