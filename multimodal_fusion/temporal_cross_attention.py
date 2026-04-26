"""
Temporal Cross-Attention Module
================================
Cross-attention giữa Vision Spike Tokens (T=8) và Text Tokens (PhoBERT).
Cho phép mô hình học mối quan hệ ngữ nghĩa giữa những gì robot nhìn thấy
và những gì người dùng nói — thay vì mapping cứng theo rule-based logic.

Architecture:
    Vision spikes:  (T, B, Nv, d_model)  ← spatial spike tokens theo thời gian
    Text tokens:    (B, Nt, d_model)      ← từ PhoBERT hoặc spike embedding
    Output:         (B, d_model)          ← biểu diễn fused để đưa vào action head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scaled Dot-Product Attention (base)
# ─────────────────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """Attention chuẩn với optional mask."""
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, heads, Nq, d_k)
        k: torch.Tensor,  # (B, heads, Nk, d_k)
        v: torch.Tensor,  # (B, heads, Nk, d_v)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, heads, Nq, Nk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # (B, heads, Nq, d_v)
        return out, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head Cross-Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: Query từ một modality, Key/Value từ modality kia.
    Hỗ trợ cả vision→text và text→vision direction.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,        # (B, Nq, d_model)
        key_value: torch.Tensor,    # (B, Nk, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nq, _ = query.shape

        # Project + reshape thành multi-head
        def reshape(x):
            # .contiguous() cần thiết vì tensor có thể non-contiguous sau permute
            return x.contiguous().view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = reshape(self.W_q(query))      # (B, heads, Nq, d_k)
        k = reshape(self.W_k(key_value))  # (B, heads, Nk, d_k)
        v = reshape(self.W_v(key_value))  # (B, heads, Nk, d_k)

        # Attention
        attn_out, attn_weights = self.attention(q, k, v, mask)  # (B, heads, Nq, d_k)

        # Concat heads → (B, Nq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, -1)
        out = self.W_o(attn_out)

        # Residual + LayerNorm
        out = self.norm(query + self.dropout(out))
        return out, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. Temporal Spike Aggregator
# Gộp T timesteps của SNN thành dãy spatial tokens cho cross-attention
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSpikeAggregator(nn.Module):
    """
    Chuyển đổi output SNN (T, B, C, H, W) → (B, N_tokens, d_model).
    
    2 strategy:
      - 'mean':   average pooling qua T → (B, C, H, W) → flatten → project
      - 'learned': dùng learnable temporal weights trước khi pool
    """
    def __init__(
        self,
        vision_channels: int = 64,
        spatial_size: int = 8,        # Sau pooling spatial xuống còn spatial_size×spatial_size
        d_model: int = 256,
        T: int = 8,
        strategy: str = "learned",
    ):
        super().__init__()
        self.T = T
        self.strategy = strategy
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))

        # Learnable temporal weight (attention qua T steps)
        if strategy == "learned":
            self.temporal_attn = nn.Parameter(torch.ones(T) / T)  # softmax will normalize

        # Project từ (C * H * W) → d_model
        n_tokens = spatial_size * spatial_size
        self.token_proj = nn.Linear(vision_channels, d_model)

        # Optional: learnable positional encoding cho spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

    def forward(self, vision_spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_spikes: (T, B, C, H, W) — output từ SNN
                           hoặc (T, B, C, 1, H, W) — sẽ được squeeze tự động
        Returns:
            tokens: (B, N_tokens, d_model)  — sẵn sàng cho cross-attention
        """
        # Squeeze depth dim nếu có
        if vision_spikes.dim() == 6:
            vision_spikes = vision_spikes.squeeze(3)

        # Đảm bảo contiguous sau permute() từ DataLoader hoặc các phép transform
        vision_spikes = vision_spikes.contiguous()
        T, B, C, H, W = vision_spikes.shape

        # Spatial pooling: (T*B, C, H, W) → pool → (T*B, C, sp, sp)
        flat = vision_spikes.reshape(T * B, C, H, W)
        pooled = self.spatial_pool(flat)   # (T*B, C, sp, sp)
        sp = pooled.shape[-1]
        # → (T, B, C, sp*sp) → permute → (T, B, sp*sp, C)
        pooled = pooled.reshape(T, B, C, sp * sp).permute(0, 1, 3, 2).contiguous()

        # Temporal aggregation
        if self.strategy == "learned":
            w = F.softmax(self.temporal_attn, dim=0)  # (T,)
            # weighted sum qua T: (B, sp*sp, C)
            tokens = (pooled * w.view(T, 1, 1, 1)).sum(dim=0)
        else:
            tokens = pooled.mean(dim=0)  # (B, sp*sp, C)

        # Project → (B, N_tokens, d_model)
        tokens = self.token_proj(tokens) + self.pos_embed[:, :tokens.shape[1], :]
        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bidirectional Temporal Cross-Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class BiModalCrossAttentionBlock(nn.Module):
    """
    Một block cross-attention 2 chiều:
      - Text attends to Vision  (text truy vấn vision để tìm context thị giác)
      - Vision attends to Text  (vision tokens được điều chỉnh bởi ngữ nghĩa câu)
    
    Sau đó concat + project thành biểu diễn fused duy nhất.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Text → Vision cross-attention
        self.text_to_vision = MultiHeadCrossAttention(d_model, num_heads, dropout)
        # Vision → Text cross-attention
        self.vision_to_text = MultiHeadCrossAttention(d_model, num_heads, dropout)

        # Feed-forward sau fusion
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,    # (B, Nt, d_model)
        vision_tokens: torch.Tensor,  # (B, Nv, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:          (B, d_model)  — biểu diễn kết hợp cuối cùng
            t_cross_attn:   attention weights text→vision
            v_cross_attn:   attention weights vision→text
        """
        # 1. Text attends to Vision
        t_updated, t_attn = self.text_to_vision(text_tokens, vision_tokens)
        # 2. Vision attends to Text
        v_updated, v_attn = self.vision_to_text(vision_tokens, text_tokens)

        # 3. Pool mỗi modality thành 1 vector
        t_vec = t_updated.mean(dim=1)  # (B, d_model)
        v_vec = v_updated.mean(dim=1)  # (B, d_model)

        # 4. Concat + FFN
        concat = torch.cat([t_vec, v_vec], dim=-1)  # (B, d_model*2)
        fused = self.ffn(concat)                     # (B, d_model)
        fused = self.norm(fused + (t_vec + v_vec) / 2)  # residual

        return fused, t_attn, v_attn


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Temporal Cross-Attention Stack
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCrossAttentionFusion(nn.Module):
    """
    Stack nhiều BiModalCrossAttentionBlock để học biểu diễn fused sâu hơn.
    
    Input:
        vision_spikes:  (T, B, C, H, W) — raw SNN output
        text_tokens:    (B, Nt, d_model) — từ PhoBERT hoặc SpikeEmbedding
    Output:
        fused_repr:     (B, d_model)
        attention_info: dict chứa attention weights từ mỗi layer
    """
    def __init__(
        self,
        vision_channels: int = 64,
        spatial_size: int = 8,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        T: int = 8,
        dropout: float = 0.1,
        temporal_strategy: str = "learned",
    ):
        super().__init__()
        self.d_model = d_model

        # Aggregator: SNN output → spatial tokens
        self.spike_aggregator = TemporalSpikeAggregator(
            vision_channels, spatial_size, d_model, T, temporal_strategy
        )

        # Stack cross-attention blocks
        self.blocks = nn.ModuleList([
            BiModalCrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Dropout cuối
        self.final_dropout = nn.Dropout(dropout)

    def forward(
        self,
        vision_spikes: torch.Tensor,   # (T, B, C, H, W)
        text_tokens: torch.Tensor,     # (B, Nt, d_model)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            fused: (B, d_model)
            attention_info: {'layer_0': {'text_to_vision': ..., 'vision_to_text': ...}, ...}
        """
        # Aggregate SNN spikes → visual tokens (B, Nv, d_model)
        vision_tokens = self.spike_aggregator(vision_spikes)

        attn_info = {}
        fused = None

        for i, block in enumerate(self.blocks):
            # Sau block đầu tiên, dùng fused từ block trước thay cho text raw
            if fused is not None:
                # Inject fused context vào text tokens qua residual
                text_tokens = text_tokens + fused.unsqueeze(1)

            fused, t_attn, v_attn = block(text_tokens, vision_tokens)
            attn_info[f"layer_{i}"] = {
                "text_to_vision": t_attn.detach(),
                "vision_to_text": v_attn.detach(),
            }

        fused = self.final_dropout(fused)
        return fused, attn_info


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, C, H, W = 2, 8, 64, 16, 16
    Nt, d = 32, 256

    vision = torch.randn(T, B, C, H, W)
    text   = torch.randn(B, Nt, d)

    fusion = TemporalCrossAttentionFusion(
        vision_channels=C, spatial_size=8, d_model=d,
        num_heads=4, num_layers=3, T=T
    )
    fused, attn = fusion(vision, text)
    print(f"✅ Fused output: {fused.shape}")  # Expected: (2, 256)
    print(f"   Attention layers: {list(attn.keys())}")
