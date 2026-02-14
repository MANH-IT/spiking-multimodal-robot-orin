"""
Skeleton kiến trúc DepthAwareSNN cho nhận diện đối tượng 3D từ RGB-D.

Mục tiêu:
- Bám theo đề cương: nhánh RGB, nhánh Depth, fusion, temporal processing,
  detection head 3D.
- Dùng PyTorch làm baseline (sau này có thể thay thế module bởi SNNTorch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from .multiscale_stdfe import MultiScaleSTDFE

try:
    import snntorch as snn  # type: ignore

    HAS_SNNTORCH = True
except ImportError:  # pragma: no cover - optional dependency
    snn = None
    HAS_SNNTORCH = False


class SpikingConvLayers(nn.Module):
    """
    Encoder convolution cho nhánh RGB/Depth.

    - Mặc định dùng Conv + ReLU (ANN baseline, dễ debug).
    - Nếu `use_snn=True` và cài đặt `snntorch`, thay ReLU bằng
      snntorch.Leaky (skeleton cho SNN thực sau này).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        use_snn: bool = False,
    ):
        super().__init__()

        if use_snn and HAS_SNNTORCH:
            act = snn.Leaky(beta=0.9)
        else:
            act = nn.ReLU(inplace=True)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            act,
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            act.__class__()
            if isinstance(act, nn.ReLU)
            else snn.Leaky(beta=0.9)
            if use_snn and HAS_SNNTORCH
            else nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return self.net(x)


class SpikingAttentionFusion(nn.Module):
    """
    Fusion đơn giản giữa feature RGB và Depth.
    Hiện tại dùng attention dạng channel-wise.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.attn_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        w_rgb = self.attn_rgb(rgb_feat)
        w_depth = self.attn_depth(depth_feat)
        fused = w_rgb * rgb_feat + w_depth * depth_feat
        return fused


class SpikingLSTM(nn.Module):
    """
    Placeholder cho temporal processing.
    Dùng LSTM tiêu chuẩn trên feature pooled theo không gian.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        return out


class SpikingDetection3D(nn.Module):
    """
    Đầu ra 3D detection head (skeleton).
    Hiện tại: dự đoán center (x,y,z), size (w,h,d) và class logits.
    """

    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256)
        self.act = nn.ReLU(inplace=True)
        self.bbox_head = nn.Linear(256, 6)  # cx, cy, cz, w, h, d
        self.cls_head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, F)
        h = self.act(self.fc(x))
        bbox = self.bbox_head(h)
        logits = self.cls_head(h)
        return bbox, logits


class DepthAwareSNN(nn.Module):
    """
    Kiến trúc SNN nhận diện đối tượng 3D từ RGB-D (skeleton).

    Định dạng input dự kiến:
    - RGB: (B, T, 3, H, W)
    - Depth: (B, T, 1, H, W)
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone_channels: int = 64,
        temporal_hidden_dim: int = 256,
        use_multiscale: bool = False,
        use_snn_backbone: bool = False,
    ):
        super().__init__()

        # RGB branch
        self.rgb_encoder = SpikingConvLayers(
            in_channels=3,
            hidden_channels=backbone_channels,
            use_snn=use_snn_backbone,
        )

        # Depth branch
        self.depth_encoder = SpikingConvLayers(
            in_channels=1,
            hidden_channels=backbone_channels,
            use_snn=use_snn_backbone,
        )

        # Fusion layer
        self.fusion_layer = SpikingAttentionFusion(channels=backbone_channels)

        # Multi-scale spatiotemporal extractor (tuỳ chọn)
        self.use_multiscale = use_multiscale
        if self.use_multiscale:
            # Sử dụng MultiScaleSTDFE trên feature fused
            self.ms_stdfe = MultiScaleSTDFE(in_channels=backbone_channels)
            temporal_in_dim = self.ms_stdfe.spike_encoder.fc[0].out_features
        else:
            # Global pooling trước khi vào temporal
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
            temporal_in_dim = backbone_channels

        # Temporal processing
        self.temporal_processor = SpikingLSTM(
            in_dim=temporal_in_dim,
            hidden_dim=temporal_hidden_dim,
        )

        # 3D detection head
        self.detection_head = SpikingDetection3D(
            in_dim=temporal_hidden_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        rgb_seq: torch.Tensor,
        depth_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param rgb_seq: (B, T, 3, H, W)
        :param depth_seq: (B, T, 1, H, W)
        :return:
            bbox_3d: (B, 6)
            logits: (B, num_classes)
        """
        b, t, _, h, w = rgb_seq.shape

        # Encode từng frame
        fused_frames = []
        for ti in range(t):
            rgb = rgb_seq[:, ti]  # (B, 3, H, W)
            depth = depth_seq[:, ti]  # (B, 1, H, W)

            rgb_f = self.rgb_encoder(rgb)
            depth_f = self.depth_encoder(depth)
            fused = self.fusion_layer(rgb_f, depth_f)  # (B, C, H, W)
            fused_frames.append(fused)

        fused_seq = torch.stack(fused_frames, dim=1)  # (B, T, C, H, W)

        if self.use_multiscale:
            # (B, T, F_ms)
            feat_seq = self.ms_stdfe(fused_seq)
        else:
            feats = []
            for ti in range(t):
                pooled = self.spatial_pool(fused_seq[:, ti]).flatten(1)  # (B, C)
                feats.append(pooled)
            feat_seq = torch.stack(feats, dim=1)  # (B, T, C)

        # Temporal processing
        temporal_out = self.temporal_processor(feat_seq)  # (B, T, H)

        # Lấy frame cuối cùng làm tóm tắt (có thể thay bằng attention sau này)
        last = temporal_out[:, -1, :]  # (B, H)

        bbox_3d, logits = self.detection_head(last)
        return bbox_3d, logits


__all__ = [
    "DepthAwareSNN",
    "SpikingConvLayers",
    "SpikingAttentionFusion",
    "SpikingLSTM",
    "SpikingDetection3D",
]

