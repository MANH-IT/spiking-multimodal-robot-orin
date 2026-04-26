"""
ThreeDSpikingCNN — Fixed Version
==================================
Các vấn đề đã fix so với phiên bản cũ:

1. ✅ Input size chuẩn hóa: nhận BATT_SIZE=128 (configurable), không hardcode 224
2. ✅ Anchor boxes đúng với input size (dùng `anchor_size` relative)
3. ✅ NMS (Non-Maximum Suppression) tích hợp sẵn vào forward()
4. ✅ Multi-anchor support: 3 anchor scales thay vì 1
5. ✅ return_feats trả về đúng format cho Temporal Cross-Attention
6. ✅ Confidence threshold configurable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

try:
    from torchvision.ops import nms, batched_nms
    _HAS_TORCHVISION_NMS = True
except ImportError:
    _HAS_TORCHVISION_NMS = False

from vision_system.models.snn.lif_neuron import ParametricLIF


# ─────────────────────────────────────────────────────────────────────────────
# Config mặc định
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE   = 128        # Chuẩn toàn hệ thống
NUM_CLASSES  = 2          # person, obstacle
T_DEFAULT    = 8          # Timesteps SNN

# Anchor sizes (relative, fraction of input_size)
# 3 scales: nhỏ (0.2), trung (0.4), lớn (0.6)
DEFAULT_ANCHORS = [
    (0.20, 0.20),  # ~26×26 px tại 128
    (0.40, 0.40),  # ~51×51 px
    (0.60, 0.60),  # ~77×77 px
]
NUM_ANCHORS = len(DEFAULT_ANCHORS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spiking Conv Block (không đổi, ổn định)
# ─────────────────────────────────────────────────────────────────────────────

class SpikingConv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, pool_size=(1, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn   = nn.BatchNorm3d(out_ch)
        self.lif  = ParametricLIF()
        self.pool = nn.MaxPool3d(pool_size) if pool_size else nn.Identity()

    def forward(self, x: torch.Tensor, mem=None):
        """
        x: (T, B, C, H, W) hoặc (T, B, C, D, H, W)
        """
        if x.dim() == 6:
            T, B, C, D, H, W = x.shape
        else:
            T, B, C, H, W = x.shape
            D = 1

        if mem is None:
            mem = self.lif.init_mem(B, self.conv.out_channels, x.device)

        outputs = []
        for t in range(T):
            inp = x[t] if x[t].dim() == 5 else x[t].unsqueeze(2)
            cur = self.bn(self.conv(inp))
            spk, mem = self.lif(cur, mem)
            spk = self.pool(spk)
            outputs.append(spk)

        return torch.stack(outputs, dim=0), mem  # (T, B, C', D', H', W')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Detection Head với Multi-Anchor
# ─────────────────────────────────────────────────────────────────────────────

class SpikeDetectionHead(nn.Module):
    """
    Detection head dùng cho SNN output.
    Predict: [tx, ty, tw, th, conf, cls...] per anchor per grid cell.

    Input:  (B, in_ch, Hf, Wf)
    Output: boxes (B,N,4), conf (B,N,1), cls (B,N,num_classes)
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = NUM_CLASSES,
        anchors: List[Tuple[float, float]] = None,
        input_size: int = INPUT_SIZE,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors or DEFAULT_ANCHORS
        self.num_anchors = len(self.anchors)
        self.input_size = input_size

        # Mỗi anchor predict: 4 (box) + 1 (conf) + num_classes
        out_per_anchor = 5 + num_classes
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels * 2, self.num_anchors * out_per_anchor, 1),
        )
        self.out_per_anchor = out_per_anchor

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B, in_channels, Hf, Wf) — SNN feature map
        Returns:
            boxes: (B, Hf*Wf*num_anchors, 4) — normalized [x1,y1,x2,y2]
            conf:  (B, Hf*Wf*num_anchors, 1) — objectness
            cls:   (B, Hf*Wf*num_anchors, num_classes) — class probs
        """
        B, _, Hf, Wf = feat.shape
        N = Hf * Wf * self.num_anchors

        raw = self.head(feat)  # (B, num_anchors * out_per_anchor, Hf, Wf)
        # → (B, Hf, Wf, num_anchors, out_per_anchor)
        raw = raw.permute(0, 2, 3, 1).reshape(B, Hf, Wf, self.num_anchors, self.out_per_anchor)

        # Grid offsets
        grid_y, grid_x = torch.meshgrid(
            torch.arange(Hf, dtype=torch.float32, device=feat.device),
            torch.arange(Wf, dtype=torch.float32, device=feat.device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(-1) / Wf  # (Hf, Wf, 1)
        grid_y = grid_y.unsqueeze(-1) / Hf

        # Decode box
        tx = raw[..., 0]
        ty = raw[..., 1]
        tw = raw[..., 2]
        th = raw[..., 3]

        cx = grid_x + torch.sigmoid(tx) / Wf
        cy = grid_y + torch.sigmoid(ty) / Hf

        # Anchor-based w, h
        anchor_w = torch.tensor([a[0] for a in self.anchors], device=feat.device)
        anchor_h = torch.tensor([a[1] for a in self.anchors], device=feat.device)
        w = anchor_w * torch.exp(tw.clamp(-4, 4))
        h = anchor_h * torch.exp(th.clamp(-4, 4))

        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, Hf, Wf, A, 4)
        boxes = boxes.reshape(B, N, 4)

        conf = torch.sigmoid(raw[..., 4]).reshape(B, N, 1)
        cls  = F.softmax(raw[..., 5:], dim=-1).reshape(B, N, self.num_classes)

        return boxes, conf, cls


# ─────────────────────────────────────────────────────────────────────────────
# 3. NMS helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_nms(
    boxes: torch.Tensor,   # (N, 4)  x1y1x2y2 normalized
    conf:  torch.Tensor,   # (N,)
    cls_ids: torch.Tensor, # (N,)
    conf_thr: float = 0.35,
    iou_thr:  float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Áp dụng NMS cho một batch item.
    Returns: boxes, conf, cls_ids sau NMS (có thể empty)
    """
    mask = conf > conf_thr
    if mask.sum() == 0:
        empty = torch.zeros(0, device=boxes.device)
        return boxes[:0], empty, empty.long()

    b, c, ci = boxes[mask], conf[mask], cls_ids[mask]

    if _HAS_TORCHVISION_NMS:
        # Batched NMS (per class)
        keep = batched_nms(b, c, ci, iou_threshold=iou_thr)
    else:
        # Fallback: greedy IOU NMS
        keep = _greedy_nms(b, c, iou_thr)

    return b[keep], c[keep], ci[keep]


def _greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float):
    """Greedy NMS fallback (không cần torchvision)."""
    _, order = scores.sort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        # Tính IOU
        ix1, iy1, ix2, iy2 = boxes[i]
        rx1 = boxes[rest, 0].clamp(min=ix1)
        ry1 = boxes[rest, 1].clamp(min=iy1)
        rx2 = boxes[rest, 2].clamp(max=ix2)
        ry2 = boxes[rest, 3].clamp(max=iy2)
        inter = (rx2 - rx1).clamp(0) * (ry2 - ry1).clamp(0)
        area_i = (ix2 - ix1) * (iy2 - iy1)
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ThreeDSpikingCNN — Upgraded
# ─────────────────────────────────────────────────────────────────────────────

class ThreeDSpikingCNN(nn.Module):
    """
    SNN Object Detector — Fixed Version.

    Inputs:
        x: (T, B, 4, H, W)  — T timesteps, 4 channels (RGB+Depth)
            H=W=input_size (default 128)

    Outputs (detection mode):
        boxes: (B, N, 4)  — normalized [x1,y1,x2,y2]
        conf:  (B, N, 1)  — objectness confidence
        cls:   (B, N, num_classes)  — class probabilities

    Output (feature mode, return_feats=True):
        (T, B, 64, Hf, Wf)  — spike feature map → dùng cho Fusion Transformer
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        T: int = T_DEFAULT,
        input_size: int = INPUT_SIZE,
        anchors: List[Tuple[float, float]] = None,
        conf_threshold: float = 0.35,
        nms_iou_threshold: float = 0.45,
    ):
        super().__init__()
        self.T = T
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Backbone: 3× SpikingConv3D + MaxPool
        # 128 → 64 → 32 → 16 (spatial)
        self.conv1 = SpikingConv3DBlock(4,  16, pool_size=(1, 2, 2))
        self.conv2 = SpikingConv3DBlock(16, 32, pool_size=(1, 2, 2))
        self.conv3 = SpikingConv3DBlock(32, 64, pool_size=(1, 2, 2))
        # Feature map: (T, B, 64, 16, 16) for input 128×128

        # Detection head (multi-anchor)
        self.det_head = SpikeDetectionHead(
            in_channels=64,
            num_classes=num_classes,
            anchors=anchors or DEFAULT_ANCHORS,
            input_size=input_size,
        )

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward qua 3 spiking conv blocks."""
        spk1, _ = self.conv1(x)
        spk2, _ = self.conv2(spk1)
        spk3, _ = self.conv3(spk2)
        return spk3  # (T, B, 64, Hf, Wf)

    def forward(
        self,
        x: torch.Tensor,                      # (T, B, 4, H, W)
        return_feats: bool = False,
        apply_nms_: bool = True,
    ):
        """
        Args:
            x:            (T, B, 4, H, W)
            return_feats: Nếu True → trả về spike features cho Fusion Transformer
            apply_nms_:   Nếu True → áp dụng NMS vào detection output
        """
        # Đảm bảo input contiguous (sau DataLoader permute)
        x = x.contiguous()

        # Backbone
        feat_seq = self._backbone(x)  # (T, B, 64, Hf, Wf)

        if return_feats:
            # Trả về spike feature sequence → dùng cho Fusion Transformer
            # Squeeze depth dim nếu có (T, B, C, 1, H, W) -> (T, B, C, H, W)
            if feat_seq.dim() == 6:
                feat_seq = feat_seq.squeeze(3)
            elif feat_seq.dim() == 5:
                # Nếu là (T, B, C, H, W) thì ok
                pass
            return feat_seq  # (T, B, 64, Hf, Wf)

        # Aggregate T timesteps → 1 feature map
        feat = feat_seq.mean(dim=0)  # (B, 64, Hf, Wf)
        if feat.dim() == 5:
            feat = feat.squeeze(2)   # Squeeze depth dim nếu còn

        # Detection head (multi-anchor)
        boxes, conf, cls = self.det_head(feat)
        # boxes: (B, N, 4), conf: (B, N, 1), cls: (B, N, num_classes)

        # NMS per batch item
        if apply_nms_:
            boxes, conf, cls = self._batch_nms(boxes, conf, cls)

        return boxes, conf, cls

    def _batch_nms(
        self,
        boxes: torch.Tensor,  # (B, N, 4)
        conf:  torch.Tensor,  # (B, N, 1)
        cls:   torch.Tensor,  # (B, N, num_classes)
    ):
        """Áp dụng NMS cho toàn bộ batch — giữ nguyên shape (B, N', ...)."""
        B = boxes.shape[0]
        out_b, out_c, out_cls = [], [], []

        for b in range(B):
            conf_b = conf[b, :, 0]
            cls_ids = cls[b].argmax(dim=-1)

            b_boxes, b_conf, b_cls = apply_nms(
                boxes[b], conf_b, cls_ids,
                conf_thr=self.conf_threshold,
                iou_thr=self.nms_iou_threshold,
            )

            out_b.append(b_boxes)
            out_c.append(b_conf.unsqueeze(-1))
            out_cls.append(F.one_hot(b_cls.long(), self.num_classes).float() if b_cls.numel() > 0
                          else torch.zeros(0, self.num_classes, device=boxes.device))

        # Pad thành batch (dùng box (0,0,0,0) cho padding)
        max_det = max(b.shape[0] for b in out_b) if any(b.shape[0] > 0 for b in out_b) else 1
        max_det = max(max_det, 1)

        boxes_out = torch.zeros(B, max_det, 4, device=boxes.device)
        conf_out  = torch.zeros(B, max_det, 1, device=boxes.device)
        cls_out   = torch.zeros(B, max_det, self.num_classes, device=boxes.device)

        for b in range(B):
            n = out_b[b].shape[0]
            if n > 0:
                boxes_out[b, :n] = out_b[b]
                conf_out[b, :n]  = out_c[b]
                cls_out[b, :n]   = out_cls[b]

        return boxes_out, conf_out, cls_out


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T, B, C, H = 8, 2, 4, 128
    x = torch.randn(T, B, C, H, H)
    model = ThreeDSpikingCNN(num_classes=2, T=T, input_size=H)
    model.eval()

    with torch.no_grad():
        # Detection mode
        boxes, conf, cls = model(x, apply_nms_=True)
        print(f"✅ Detection: boxes={boxes.shape}, conf={conf.shape}, cls={cls.shape}")

        # Feature mode (for Fusion Transformer)
        feats = model(x, return_feats=True)
        print(f"✅ Features:  {feats.shape}  → (T, B, 64, Hf, Wf)")

    # Check NMS filtering
    total_anchors = 16 * 16 * 3  # Hf*Wf*num_anchors
    print(f"\n   Grid: 16×16, {3} anchors → {total_anchors} raw predictions per image")
    print(f"   After NMS: {boxes.shape[1]} detections per image")
