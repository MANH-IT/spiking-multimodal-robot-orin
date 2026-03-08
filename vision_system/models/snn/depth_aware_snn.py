# -*- coding: utf-8 -*-
"""
DepthAwareSNN - Upgraded with snntorch LIF neurons - NCKH 2026
================================================================
Kiến trúc nhận diện đối tượng 3D từ RGB-D sequence dùng SNN thực sự.

Luồng xử lý:
  RGB  (B,T,3,H,W) ──→ SpikingConvEncoder (LIF) ──→ |
                                                       ├→ DepthAwareAttentionFusion
  Depth(B,T,1,H,W) ──→ SpikingConvEncoder (LIF) ──→ |
                                                       ↓
                                              MultiScaleSTDFE (optional)
                                                       ↓
                                              SpikingTemporalRNN (LIF-GRU)
                                                       ↓
                                              SpikingDetection3D
                                                  ↓         ↓
                                              bbox_3d    class_logits

Chỉ tiêu đề cương: < 20ms latency, > 85% mAP@3D, < 35W trên Jetson AGX Orin
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_stdfe import MultiScaleSTDFE

# ── snntorch ──────────────────────────────────────────────────
try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    snn = None
    HAS_SNNTORCH = False
    print("[DepthAwareSNN] WARNING: snntorch not found. pip install snntorch")


# ============================================================
# Surrogate gradient helper
# ============================================================

def _get_spike_grad():
    if HAS_SNNTORCH:
        return surrogate.fast_sigmoid(slope=25)
    return None


# ============================================================
# Spiking Conv Encoder (thay ReLU bằng LIF thực)
# ============================================================

class SpikingConvEncoder(nn.Module):
    """
    Convolutional encoder với snntorch.Leaky (LIF) neurons thay ReLU.

    Xử lý 1 frame: (B, C_in, H, W) → (B, C_out, H, W) + spike stats

    Kiến trúc:
        Conv2d → BN → LIF1 → Conv2d → BN → LIF2
        (optionally thêm residual connection)

    Args:
        in_channels: RGB=3, Depth=1
        out_channels: Output feature channels (mặc định 64)
        beta: Membrane decay constant cho LIF
        use_snn: True = LIF thực, False = ReLU fallback
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        beta: float = 0.9,
        use_snn: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_snn = use_snn and HAS_SNNTORCH

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # Layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Layer 3 (deeper)
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        if self.use_snn:
            spike_grad = _get_spike_grad()
            self.lif1 = snn.Leaky(beta=beta,       spike_grad=spike_grad)
            self.lif2 = snn.Leaky(beta=beta * 0.95, spike_grad=spike_grad)
            self.lif3 = snn.Leaky(beta=beta * 0.90, spike_grad=spike_grad)
        else:
            self.act = nn.ReLU(inplace=True)

        # Residual projection nếu in_channels ≠ out_channels
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        self._spike_rate = 0.0

    def init_hidden(self, batch_size: int, device: torch.device):
        """Khởi tạo membrane potential cho mỗi forward pass."""
        if self.use_snn:
            return (
                self.lif1.init_leaky(),
                self.lif2.init_leaky(),
                self.lif3.init_leaky(),
            )
        return None

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], float]:
        """
        x: (B, C_in, H, W) — 1 frame
        Returns: (features, hidden_state, spike_rate)
        """
        residual = self.residual(x)

        if self.use_snn and hidden is not None:
            mem1, mem2, mem3 = hidden

            # Layer 1
            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3 + residual
            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            out = spk3 + residual

            spike_rate = (spk1.mean() + spk2.mean() + spk3.mean()).item() / 3
            self._spike_rate = spike_rate

            return out, (mem1, mem2, mem3), spike_rate

        else:
            # ANN fallback
            h = self.act(self.bn1(self.conv1(x)))
            h = self.act(self.bn2(self.conv2(h)))
            h = self.act(self.bn3(self.conv3(h)))
            return h + residual, None, 0.0


# ============================================================
# Depth-Aware Attention Fusion
# ============================================================

class DepthAwareAttentionFusion(nn.Module):
    """
    Fusion RGB + Depth với depth-weighting.

    Depth gần (giá trị nhỏ) → weight cao hơn (vật thể quan trọng gần camera).
    Dùng spatial attention để tập trung vào vùng có vật thể.

    Cải tiến so với SpikingAttentionFusion cũ:
    - Thêm depth-weighting map
    - Spatial attention (không chỉ channel attention)
    - Normalize depth weight theo scene
    """

    def __init__(self, channels: int):
        super().__init__()

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

        # Spatial attention từ depth
        self.depth_spatial = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,    # (B, C, H, W)
        depth_feat: torch.Tensor,  # (B, C, H, W)
        depth_raw: Optional[torch.Tensor] = None,  # (B, 1, H, W) raw depth
    ) -> torch.Tensor:
        B, C, H, W = rgb_feat.shape

        # Concat và project
        concat = torch.cat([rgb_feat, depth_feat], dim=1)   # (B, 2C, H, W)
        fused  = self.fusion_proj(concat)                    # (B, C, H, W)

        # Channel attention
        attn_w = self.channel_attn(concat).view(B, C, 1, 1)
        fused  = fused * attn_w

        # Depth spatial attention (nếu có raw depth)
        if depth_raw is not None:
            # Invert: gần = weight cao
            depth_norm = 1.0 / (1.0 + depth_raw)
            spatial_w  = self.depth_spatial(depth_norm)    # (B, 1, H, W)
            fused = fused * spatial_w

        return fused


# ============================================================
# Spiking Temporal Processor (LIF-GRU)
# ============================================================

class SpikingTemporalRNN(nn.Module):
    """
    Recurrent SNN cho temporal processing.

    Dùng snntorch.RLeaky (Recurrent LIF) thay cho LSTM thường.
    Nếu không có snntorch → fallback GRU.

    Args:
        in_dim: Chiều input feature
        hidden_dim: Chiều hidden state
        beta: Membrane decay
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        beta: float = 0.85,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_snn    = HAS_SNNTORCH

        if self.use_snn:
            # snntorch.RLeaky = Recurrent LIF
            spike_grad    = _get_spike_grad()
            self.rlif     = snn.RLeaky(
                beta=beta,
                spike_grad=spike_grad,
                linear_features=hidden_dim,
                init_hidden=False,
            )
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            # Fallback: GRU tốt hơn LSTM cho SNN simulation
            self.gru = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        Returns: (B, T, hidden_dim) temporal features
        """
        B, T, _ = x.shape

        if self.use_snn:
            spk = torch.zeros(B, self.hidden_dim, device=x.device)
            mem = torch.zeros(B, self.hidden_dim, device=x.device)

            proj = self.input_proj(x)    # (B, T, hidden_dim)
            outputs = []
            for t in range(T):
                # RLeaky: combine input + recurrent
                combined = proj[:, t] + spk
                spk, mem = self.rlif(combined, spk, mem)
                outputs.append(spk)
            return torch.stack(outputs, dim=1)   # (B, T, H)
        else:
            out, _ = self.gru(x)
            return out


# ============================================================
# Spiking Detection Head
# ============================================================

class SpikingDetection3D(nn.Module):
    """
    3D Detection head: feature → (bbox_3d, class_logits)

    Output:
        bbox_3d: (B, 6) = [cx, cy, cz, w, h, d]
        logits:  (B, num_classes)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        mid_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(mid_dim, mid_dim // 2),
            nn.GELU(),
            nn.Linear(mid_dim // 2, 6),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(mid_dim, mid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, in_dim)"""
        h = self.shared(x)
        bbox   = self.bbox_head(h)   # (B, 6)
        logits = self.cls_head(h)    # (B, num_classes)
        return bbox, logits


# ============================================================
# DepthAwareSNN — Main Model
# ============================================================

class DepthAwareSNN(nn.Module):
    """
    SNN-based 3D Object Detection từ RGB-D sequences.

    Cải thiện so với version cũ:
    ✅ SpikingConvEncoder với snntorch.Leaky (LIF) thực sự
    ✅ DepthAwareAttentionFusion với depth-weighting spatial attention
    ✅ SpikingTemporalRNN với snntorch.RLeaky (Recurrent LIF)
    ✅ SpikingDetection3D với LayerNorm + GELU
    ✅ Stateful forward (membrane potentials được carry qua frames)
    ✅ Spike rate tracking (energy metric)

    Input:
        rgb_seq:   (B, T, 3, H, W)
        depth_seq: (B, T, 1, H, W)

    Output:
        bbox_3d:  (B, 6)   — center + size trong 3D space
        logits:   (B, num_classes) — class scores
    """

    def __init__(
        self,
        num_classes: int = 253,      # 253 HILO object classes
        backbone_channels: int = 64,
        temporal_hidden_dim: int = 256,
        use_multiscale: bool = True,
        use_snn_backbone: bool = True,  # True = snntorch LIF
        beta: float = 0.9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_snn_backbone = use_snn_backbone and HAS_SNNTORCH
        self.use_multiscale   = use_multiscale
        self.backbone_channels = backbone_channels

        # ── Encoders ──────────────────────────────────────────
        self.rgb_encoder = SpikingConvEncoder(
            in_channels=3,
            out_channels=backbone_channels,
            beta=beta,
            use_snn=use_snn_backbone,
        )
        self.depth_encoder = SpikingConvEncoder(
            in_channels=1,
            out_channels=backbone_channels,
            beta=beta,
            use_snn=use_snn_backbone,
        )

        # ── Fusion ────────────────────────────────────────────
        self.fusion = DepthAwareAttentionFusion(channels=backbone_channels)

        # ── Spatial Pooling / Multi-Scale ─────────────────────
        if use_multiscale:
            self.ms_stdfe = MultiScaleSTDFE(in_channels=backbone_channels)
            # MultiScaleSTDFE output dim (xem multiscale_stdfe.py)
            temporal_in_dim = backbone_channels * 3   # 3 scales
        else:
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
            temporal_in_dim = backbone_channels

        # ── Temporal ──────────────────────────────────────────
        self.temporal = SpikingTemporalRNN(
            in_dim=temporal_in_dim,
            hidden_dim=temporal_hidden_dim,
            beta=beta * 0.95,
        )

        # ── Detection Head ────────────────────────────────────
        self.detector = SpikingDetection3D(
            in_dim=temporal_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Spike rate tracking
        self._total_spike_rate: float = 0.0
        self._forward_count: int = 0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        rgb_seq: torch.Tensor,    # (B, T, 3, H, W)
        depth_seq: torch.Tensor,  # (B, T, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T, _, H, W = rgb_seq.shape
        device = rgb_seq.device

        # Khởi tạo hidden states cho LIF neurons
        rgb_hidden   = self.rgb_encoder.init_hidden(B, device)
        depth_hidden = self.depth_encoder.init_hidden(B, device)

        fused_frames: List[torch.Tensor] = []
        total_spike = 0.0

        # ── Per-frame processing ───────────────────────────────
        for t in range(T):
            rgb_t   = rgb_seq[:, t]      # (B, 3, H, W)
            depth_t = depth_seq[:, t]    # (B, 1, H, W)

            # Encode
            rgb_feat, rgb_hidden, s_rgb = \
                self.rgb_encoder(rgb_t, rgb_hidden)
            depth_feat, depth_hidden, s_dep = \
                self.depth_encoder(depth_t, depth_hidden)

            total_spike += (s_rgb + s_dep) / 2

            # Fuse (pass raw depth for spatial weighting)
            fused = self.fusion(rgb_feat, depth_feat, depth_t)
            fused_frames.append(fused)

        fused_seq = torch.stack(fused_frames, dim=1)   # (B, T, C, H, W)

        # ── Spatial/Multi-scale ───────────────────────────────
        if self.use_multiscale:
            try:
                feat_seq = self.ms_stdfe(fused_seq)       # (B, T, F)
            except Exception:
                # Fallback nếu ms_stdfe có lỗi
                feat_seq = fused_seq.flatten(2).mean(-1, keepdim=True)
                feat_seq = feat_seq.expand(-1, -1, self.backbone_channels)
        else:
            feats = []
            for t in range(T):
                pooled = self.spatial_pool(fused_seq[:, t]).flatten(1)
                feats.append(pooled)
            feat_seq = torch.stack(feats, dim=1)   # (B, T, C)

        # ── Temporal ──────────────────────────────────────────
        temporal_out = self.temporal(feat_seq)   # (B, T, hidden)
        last_hidden  = temporal_out[:, -1, :]    # (B, hidden)

        # ── Detection ─────────────────────────────────────────
        bbox_3d, logits = self.detector(last_hidden)

        # Track spike rate
        self._total_spike_rate += total_spike / T
        self._forward_count += 1

        return bbox_3d, logits

    def get_avg_spike_rate(self) -> float:
        """Trả về spike rate trung bình kể từ lần đầu forward."""
        if self._forward_count == 0:
            return 0.0
        return self._total_spike_rate / self._forward_count

    def reset_spike_stats(self):
        self._total_spike_rate = 0.0
        self._forward_count = 0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        snn_mode = "snntorch LIF" if self.use_snn_backbone else "ReLU (ANN)"
        ms_mode  = "MultiScaleSTDFE" if self.use_multiscale else "Global Pool"
        return (
            f"DepthAwareSNN(\n"
            f"  backbone    = {snn_mode}\n"
            f"  spatial     = {ms_mode}\n"
            f"  temporal    = {'RLeaky' if HAS_SNNTORCH else 'GRU'}\n"
            f"  classes     = {self.detector.cls_head[-1].out_features}\n"
            f"  params      = {self.count_parameters():,}\n"
            f")"
        )


# ============================================================
# Backward-compat aliases
# ============================================================

# Giữ các tên cũ để tương thích với depth_aware_train.py
SpikingConvLayers     = SpikingConvEncoder
SpikingAttentionFusion = DepthAwareAttentionFusion
SpikingLSTM           = SpikingTemporalRNN
SpikingDetection3D    = SpikingDetection3D


__all__ = [
    'DepthAwareSNN',
    'SpikingConvEncoder',
    'SpikingConvLayers',          # backward compat
    'DepthAwareAttentionFusion',
    'SpikingAttentionFusion',     # backward compat
    'SpikingTemporalRNN',
    'SpikingLSTM',                # backward compat
    'SpikingDetection3D',
]


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    print("=== DepthAwareSNN Test ===\n")

    model = DepthAwareSNN(
        num_classes=253,        # HILO 253 classes
        backbone_channels=64,
        temporal_hidden_dim=256,
        use_multiscale=False,   # Tắt multi-scale để test nhanh
        use_snn_backbone=True,
    )
    print(model)
    print()

    # Simulate 1 batch: B=2, T=4 frames, 224×224
    B, T = 2, 4
    rgb   = torch.randn(B, T, 3, 224, 224)
    depth = torch.rand(B, T, 1, 224, 224)   # depth [0,1] meters

    print(f"Input RGB  : {rgb.shape}")
    print(f"Input Depth: {depth.shape}")

    bbox, logits = model(rgb, depth)
    print(f"\nOutput bbox_3d: {bbox.shape}")    # (2, 6)
    print(f"Output logits : {logits.shape}")   # (2, 253)
    print(f"Avg spike rate: {model.get_avg_spike_rate():.4f}")

    # Memory
    params = model.count_parameters()
    print(f"\nParameters: {params:,} ({params * 4 / 1e6:.1f} MB float32)")
