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
    Tính temporal gradient trên chuỗi frame.

    Hiện tại: đơn giản là x[t] - x[t-1].
    Frame đầu tiên được pad với zeros để giữ nguyên số lượng frames.

    Args:
        x: Input tensor với shape (B, T, C, H, W)

    Returns:
        Temporal gradient với shape (B, T, C, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        diff = x[:, 1:] - x[:, :-1]
        # Pad để giữ nguyên T
        first = torch.zeros_like(diff[:, :1])
        return torch.cat([first, diff], dim=1)


class AdaptiveEncoder(nn.Module):
    """
    Adaptive encoder cho spike encoding.

    Hiện tại: MLP đơn giản trên feature vector.
    Có thể được mở rộng thành SNN encoding sau này.

    Args:
        in_dim: Số chiều input feature
        hidden_dim: Số chiều hidden layer (default: 256)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MultiScaleSTDFE(nn.Module):
    """
    Multi-Scale Spatiotemporal Deep Feature Extraction.

    Trích xuất đặc trưng đa tỉ lệ không–thời gian từ RGB-D sequences.
    Sử dụng temporal gradients và multi-scale pooling.

    Args:
        in_channels: Số channels của input feature
        base_channels: Số channels cho conv layers (default: 64)
    """

    def __init__(self, in_channels: int, base_channels: int = 64) -> None:
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
        # Note: feature_dim sẽ được tính động trong forward nếu cần
        self.base_channels = base_channels
        self.expected_scales = len(self.scales)
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
            try:
                if s == 1:
                    x_s = x_combined
                else:
                    # Downsample theo scale s (1/2, 1/4)
                    new_h, new_w = max(1, h // s), max(1, w // s)
                    if new_h < 1 or new_w < 1:
                        # Skip scale if too small
                        continue
                    x_s = F.interpolate(
                        x_combined.view(b * t, c, h, w),
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    ).view(b, t, c, new_h, new_w)

                # Conv từng frame
                if x_s.numel() > 0 and x_s.shape[-2] > 0 and x_s.shape[-1] > 0:
                    feat = self.conv(
                        x_s.view(b * t, c, x_s.shape[-2], x_s.shape[-1])
                    ).view(b, t, -1, x_s.shape[-2], x_s.shape[-1])

                    # Global average pool không gian -> (B, T, C)
                    pooled = feat.mean(dim=[-2, -1])
                    multi_scale_feats.append(pooled)
            except Exception as e:
                # Skip this scale if there's an error
                continue

        # Ensure at least one scale is processed (fallback to original if needed)
        if len(multi_scale_feats) == 0:
            # Fallback: use original input without multiscale
            feat = self.conv(x.view(b * t, c, h, w)).view(b, t, -1, h, w)
            pooled = feat.mean(dim=[-2, -1])
            multi_scale_feats.append(pooled)

        # Concat theo channel -> (B, T, C * num_scales)
        concat = torch.cat(multi_scale_feats, dim=-1)  # (B, T, actual_feature_dim)
        
        # Adjust encoder input size if needed (shouldn't happen in normal case)
        actual_feature_dim = concat.shape[-1]
        if actual_feature_dim != self.feature_dim:
            # Create a temporary encoder with correct size, or use projection
            # For now, use a simple linear projection if dimensions don't match
            if not hasattr(self, '_dynamic_proj') or self._dynamic_proj.in_features != actual_feature_dim:
                from torch.nn import Linear
                self._dynamic_proj = Linear(actual_feature_dim, self.feature_dim).to(concat.device)
            concat = self._dynamic_proj(concat)

        # Encoder cuối
        encoded = self.spike_encoder(concat)  # (B, T, F_encoded)
        return encoded


__all__ = [
    "MultiScaleSTDFE",
    "SpikingTemporalGrad",
    "AdaptiveEncoder",
]

