# -*- coding: utf-8 -*-
"""
Spiking Language Model - NCKH 2026
====================================
Mô hình ngôn ngữ dựa trên Spiking Neural Network (SNN) thực sự.
Dùng snntorch.Leaky (Leaky Integrate-and-Fire neurons) thay vì LSTM.

Kiến trúc:
    Embedding → [LIF Layer 1] → [LIF Layer 2] → Intent Head + Entity Head

Encoding thời gian: Rate Coding (lặp lại T timestep)
Hàm surrogate gradient: fast_sigmoid (cho backprop qua spike)

Tham chiếu đề cương: SpikingLanguageModel (mục 3.3.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

# --- snntorch ---
try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False
    print("[SpikingLM] WARNING: snntorch not found. pip install snntorch")
    print("[SpikingLM] Falling back to LIF-simulated (ALIF) neurons.")


# ============================================================
# Fallback: Approximate LIF (nếu chưa cài snntorch)
# ============================================================

class ApproxLIFCell(nn.Module):
    """
    Approximate Leaky Integrate-and-Fire (fallback khi không có snntorch).
    Membrane: v[t] = beta * v[t-1] + I[t]
    Spike:    s[t] = heaviside(v[t] - threshold)  ← surrogate: sigmoid
    """
    def __init__(self, beta: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def init_mem(self) -> torch.Tensor:
        return torch.zeros(1)  # sẽ broadcast

    def forward(
        self, current: torch.Tensor, mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.beta * mem + current
        # Surrogate gradient: straight-through sigmoid
        spike = torch.sigmoid((mem - self.threshold) * 10.0)
        # Reset: subtract threshold where spiked
        mem = mem - spike.detach() * self.threshold
        return spike, mem


# ============================================================
# Spike Encoding
# ============================================================

class RateCoder(nn.Module):
    """
    Rate Coding: chuyển embedding vector → spike train.
    Normalize giá trị về [0,1] rồi sample Bernoulli T lần.
    """
    def __init__(self, num_steps: int = 25):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq, embed_dim) hoặc (B, embed_dim)
        returns: (num_steps, B, ...) spike train
        """
        x_norm = torch.sigmoid(x)  # chuẩn hoá về (0,1)
        # Bernoulli sampling T lần
        spikes = torch.stack([
            torch.bernoulli(x_norm) for _ in range(self.num_steps)
        ])  # (T, B, seq, embed)
        return spikes


# ============================================================
# Spiking Language Model
# ============================================================

class SpikingLanguageModel(nn.Module):
    """
    Mô hình ngôn ngữ SNN cho phân loại intent (NLU).

    Args:
        vocab_size: Kích thước từ điển (mặc định 50000 theo đề cương)
        embed_dim: Chiều embedding
        hidden_dim: Chiều hidden của LIF layers
        output_dim: Số intent classes
        num_steps: Số time-step mô phỏng SNN (rate coding)
        beta: Decay factor của membrane potential
        use_snntorch: Ép dùng snntorch kể cả khi import thất bại

    Input:
        x: (B, seq_len) token IDs
        lang_ids: (B,) language IDs (tùy chọn, cho multilingual)

    Output:
        logits: (B, output_dim) intent logits
    """

    def __init__(
        self,
        vocab_size: int = 50_000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 14,    # 14 intents trong VMMD
        num_steps: int = 25,
        beta: float = 0.9,
        dropout: float = 0.3,
        use_snntorch: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.use_real_snn = use_snntorch and HAS_SNNTORCH

        # ── Embedding ──────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # ── Positional projection ──────────────────────────────
        # Gộp chuỗi theo chiều seq (mean pooling) trước khi vào SNN
        # Có thể thay bằng multi-head attention sau này
        self.seq_proj = nn.Linear(embed_dim, hidden_dim)

        # ── LIF Layers ─────────────────────────────────────────
        if self.use_real_snn:
            # snntorch Leaky neurons — LIF thực sự
            spike_grad = surrogate.fast_sigmoid(slope=25)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                   init_hidden=False)
            self.lif2 = snn.Leaky(beta=beta * 0.95, spike_grad=spike_grad,
                                   init_hidden=False)
        else:
            # Fallback approximate LIF
            self.lif1 = ApproxLIFCell(beta=beta)
            self.lif2 = ApproxLIFCell(beta=beta * 0.95)

        # Feed-forward giữa các LIF layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── Output Heads ───────────────────────────────────────
        # Intent classification head
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Context classification head (airport / classroom / home)
        self.context_head = nn.Linear(hidden_dim, 3)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization cho stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            x: (B, seq_len) token IDs
            attention_mask: (B, seq_len) 1 = real token, 0 = pad

        Returns:
            {
                'intent_logits': (B, output_dim),
                'context_logits': (B, 3),
                'spike_rate': scalar — tỷ lệ spike trung bình (metric năng lượng),
            }
        """
        B, L = x.shape

        # ── 1. Embedding ───────────────────────────────────────
        embed = self.embedding(x)             # (B, L, E)
        embed = self.embed_dropout(embed)

        # Masked mean pooling (bỏ PAD tokens)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (embed * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = embed.mean(dim=1)        # (B, E)

        # Project → hidden_dim
        h = self.seq_proj(pooled)             # (B, H)

        # ── 2. SNN Temporal Simulation ─────────────────────────
        spike_accumulator = torch.zeros_like(h)

        if self.use_real_snn:
            # snntorch: khởi tạo membrane mỗi forward
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
        else:
            mem1 = torch.zeros_like(h)
            mem2 = torch.zeros_like(h)

        total_spikes = 0.0

        for _ in range(self.num_steps):
            # Layer 1: linear → LIF
            cur1 = self.fc1(h)
            cur1 = self.bn1(cur1)

            if self.use_real_snn:
                spk1, mem1 = self.lif1(cur1, mem1)
            else:
                spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2: linear → LIF
            cur2 = self.fc2(self.dropout(spk1))
            cur2 = self.bn2(cur2)

            if self.use_real_snn:
                spk2, mem2 = self.lif2(cur2, mem2)
            else:
                spk2, mem2 = self.lif2(cur2, mem2)

            spike_accumulator = spike_accumulator + spk2
            total_spikes += spk2.mean().item()

        # Rate-coded output: tổng spike / T
        out = spike_accumulator / self.num_steps   # (B, H)
        spike_rate = total_spikes / self.num_steps

        # ── 3. Output Heads ────────────────────────────────────
        intent_logits = self.intent_head(out)      # (B, output_dim)
        context_logits = self.context_head(out)    # (B, 3)

        return {
            'intent_logits': intent_logits,
            'context_logits': context_logits,
            'spike_rate': spike_rate,
        }

    # ----------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------

    def predict_intent(
        self, x: torch.Tensor,
        intent_map: Optional[dict] = None
    ) -> list[str]:
        """Predict intent labels từ token IDs."""
        with torch.no_grad():
            out = self.forward(x)
        preds = out['intent_logits'].argmax(dim=-1).tolist()
        if intent_map:
            idx2intent = {v: k for k, v in intent_map.items()}
            return [idx2intent.get(p, 'unknown') for p in preds]
        return preds

    def energy_metric(self, x: torch.Tensor) -> dict:
        """
        Ước tính năng lượng tiêu thụ (SNN metrics).
        Spike rate thấp → tiêu thụ ít hơn ANN.
        """
        with torch.no_grad():
            out = self.forward(x)
        return {
            'spike_rate': out['spike_rate'],
            'estimated_ops': out['spike_rate'] * self.hidden_dim * self.num_steps,
            'sparsity': 1.0 - out['spike_rate'],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        mode = "snntorch LIF" if self.use_real_snn else "Approximate LIF"
        return (f"SpikingLanguageModel("
                f"vocab={self.vocab_size}, "
                f"hidden={self.hidden_dim}, "
                f"intents={self.output_dim}, "
                f"T={self.num_steps}, "
                f"mode={mode}, "
                f"params={self.count_parameters():,})")


# ============================================================
# EnergyEfficientNLP - Power Manager tích hợp
# ============================================================

class EnergyEfficientNLP:
    """
    Quản lý năng lượng và power mode cho NLP inference trên Jetson.
    Mục tiêu đề cương: < 15W ở chế độ NLP-only.
    """

    MODES = {
        'ECO':         {'num_steps': 10, 'hidden_scale': 0.5},
        'BALANCED':    {'num_steps': 20, 'hidden_scale': 0.75},
        'PERFORMANCE': {'num_steps': 25, 'hidden_scale': 1.0},
    }

    def __init__(self, model: Optional[SpikingLanguageModel] = None):
        self.model = model
        self.mode = 'BALANCED'

    def set_power_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Mode phải là: {list(self.MODES.keys())}")
        self.mode = mode
        cfg = self.MODES[mode]
        if self.model is not None:
            self.model.num_steps = cfg['num_steps']
        print(f"[EnergyNLP] Mode → {mode} "
              f"(T={cfg['num_steps']}, scale={cfg['hidden_scale']})")

    def estimate_power(self, spike_rate: float) -> float:
        """
        Ước tính công suất (W) dựa trên spike rate.
        SNN tiết kiệm vì chỉ xử lý khi có spike.
        """
        base_power = {'ECO': 5.0, 'BALANCED': 10.0, 'PERFORMANCE': 15.0}
        return base_power[self.mode] * (0.3 + 0.7 * spike_rate)

    def get_mode_config(self) -> dict:
        return self.MODES[self.mode]


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    print("=== SpikingLanguageModel Quick Test ===")
    model = SpikingLanguageModel(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        output_dim=14,
        num_steps=5,
    )
    print(model)
    print(f"Parameters: {model.count_parameters():,}")

    # Dummy batch
    B, L = 4, 32
    x = torch.randint(0, 1000, (B, L))
    mask = torch.ones(B, L)
    mask[0, 20:] = 0   # Simulate padding

    out = model(x, mask)
    print(f"\nIntent logits shape: {out['intent_logits'].shape}")
    print(f"Context logits shape: {out['context_logits'].shape}")
    print(f"Spike rate: {out['spike_rate']:.4f}")

    energy = model.energy_metric(x)
    print(f"\nEnergy metrics: {energy}")

    # Power manager
    nlp_manager = EnergyEfficientNLP(model)
    nlp_manager.set_power_mode('ECO')
    print(f"\nEstimated power: {nlp_manager.estimate_power(out['spike_rate']):.2f} W")
