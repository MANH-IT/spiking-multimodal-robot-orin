"""
Multi-Scale Spatiotemporal Deep Feature Extraction (skeleton).

Ý tưởng:
- Xử lý multi-scale (1x, 1/2x, 1/4x) cho RGB-D hoặc feature map đã trích xuất.
- Thêm hook cho depth-aware pyramid và temporal gradients.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingTemporalGrad(nn.Module):
    """
    Placeholder cho temporal gradient trên chuỗi frame.
    Hiện tại: đơn giản là x[t] - x[t-1].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        diff = x[:, 1:] - x[:, :-1]
        # Pad để giữ nguyên T
        first = torch.zeros_like(diff[:, :1])
        return torch.cat([first, diff], dim=1)


class AdaptiveEncoder(nn.Module):
    """
    Placeholder cho adaptive spike encoding.
    Hiện tại: MLP đơn giản trên feature vector.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MultiScaleSTDFE(nn.Module):
    """
    Trích xuất đặc trưng đa tỉ lệ không–thời gian (skeleton).
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        self.scales = [1, 2, 4]

        #Shared conv cho mỗi scale (có thể tách riêng sau)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.temporal_gradient = SpikingTemporalGrad()

        # Sau pool không gian, concat multi-scale -> encoder
        self.feature_dim = base_channels * len(self.scales)
        self.spike_encoder = AdaptiveEncoder(in_dim=self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C, H, W)
        :return: (B, T, F_encoded)
        """
        b, t, c, h, w = x.shape

        # Tính temporal gradients
        x_grad = self.temporal_gradient(x)  # (B, T, C, H, W)

        # Gộp thông tin gốc + gradient (simple sum)
        x_combined = x + x_grad

        multi_scale_feats: List[torch.Tensor] = []

        for s in self.scales:
            if s == 1:
                x_s = x_combined
            else:
                # Downsample theo scale s (1/2, 1/4)
                x_s = F.interpolate(
                    x_combined.view(b * t, c, h, w),
                    scale_factor=1.0 / s,
                    mode="bilinear",
                    align_corners=False,
                ).view(b, t, c, h // s, w // s)

            # Conv từng frame
            feat = self.conv(
                x_s.view(b * t, c, x_s.shape[-2], x_s.shape[-1])
            ).view(b, t, -1, x_s.shape[-2], x_s.shape[-1])

            # Global average pool không gian -> (B, T, C)
            pooled = feat.mean(dim=[-2, -1])
            multi_scale_feats.append(pooled)

        # Concat theo channel -> (B, T, C * num_scales)
        concat = torch.cat(multi_scale_feats, dim=-1)

        # Encoder cuối
        encoded = self.spike_encoder(concat)  # (B, T, F_encoded)
        return encoded


__all__ = [
    "MultiScaleSTDFE",
    "SpikingTemporalGrad",
    "AdaptiveEncoder",
]

