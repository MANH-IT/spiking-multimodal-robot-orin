# -*- coding: utf-8 -*-
"""
Vision-NLP Bridge - NCKH 2026
================================
Cầu nối chính giữa Vision System (DepthAwareSNN) và NLP System.

Workflow:
    RGB-D frames + Speech text
        ↓ Parallel processing
    Vision features + NLP features
        ↓ CrossModalAttention
    Fused multimodal representation
        ↓ RobotDecisionMaker
    Action command

Tham chiếu đề cương tích hợp mục IV
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Data Classes
# ============================================================

@dataclass
class VisionResult:
    """Kết quả từ Vision System."""
    bbox_3d: torch.Tensor          # (N, 6): cx,cy,cz,w,h,d
    class_logits: torch.Tensor     # (N, num_classes)
    class_names: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    depth_map: Optional[torch.Tensor] = None
    processing_time_ms: float = 0.0

    @property
    def top_object(self) -> Optional[str]:
        if not self.class_names:
            return None
        scores = torch.softmax(self.class_logits, dim=-1)
        best = scores.max(dim=-1).values.argmax().item()
        return self.class_names[int(best)] if self.class_names else None


@dataclass
class NLPResult:
    """Kết quả từ NLP System."""
    intent: str
    context: str                    # airport / classroom / home
    entities: Dict[str, str]
    language: str                   # vi / en / zh
    confidence: float
    features: Optional[torch.Tensor] = None   # (hidden_dim,)
    response_hint: str = ""
    processing_time_ms: float = 0.0


@dataclass
class FusedResult:
    """Kết quả sau Multimodal Fusion."""
    action: str
    target_object: Optional[str]
    target_location: Optional[str]
    speech_response: str
    confidence: float
    vision: Optional[VisionResult] = None
    nlp: Optional[NLPResult] = None
    fused_features: Optional[torch.Tensor] = None
    total_latency_ms: float = 0.0


# ============================================================
# Cross-Modal Attention
# ============================================================

class CrossModalAttention(nn.Module):
    """
    Multi-head attention giữa Vision features và NLP features.
    Vision features → Query, NLP features → Key/Value.

    Args:
        vision_dim: Chiều output của DepthAwareSNN
        nlp_dim: Chiều output của SpikingLanguageModel
        out_dim: Chiều fused output
        num_heads: Số attention heads
    """

    def __init__(
        self,
        vision_dim: int = 256,
        nlp_dim: int = 512,
        out_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project về cùng chiều trước attention
        self.vision_proj = nn.Linear(vision_dim, out_dim)
        self.nlp_proj    = nn.Linear(nlp_dim,    out_dim)

        # Cross attention: vision attends to NLP
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self attention sau fusion
        self.self_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
        )

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(
        self,
        vision_feat: torch.Tensor,    # (B, vision_dim) hoặc (B, N, vision_dim)
        nlp_feat: torch.Tensor,       # (B, nlp_dim)    hoặc (B, M, nlp_dim)
    ) -> torch.Tensor:
        """
        Returns: (B, out_dim) fused representation
        """
        # Ensure 3D (batch, seq, dim)
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)   # (B, 1, V)
        if nlp_feat.dim() == 2:
            nlp_feat = nlp_feat.unsqueeze(1)         # (B, 1, N)

        # Project
        v = self.vision_proj(vision_feat)   # (B, 1, out_dim)
        n = self.nlp_proj(nlp_feat)         # (B, 1, out_dim)

        # Cross attention: v attends to n
        attn_out, _ = self.cross_attn(query=v, key=n, value=n)
        v = self.norm1(v + attn_out)

        # Self attention
        self_out, _ = self.self_attn(query=v, key=v, value=v)
        v = self.norm2(v + self_out)

        # FFN
        ffn_out = self.ffn(v)
        v = self.norm3(v + ffn_out)

        # Pool
        return v.mean(dim=1)   # (B, out_dim)


# ============================================================
# Vision-NLP Bridge
# ============================================================

class VisionNLPBridge(nn.Module):
    """
    Module kết nối Vision System và NLP System.

    Có thể hoạt động ở 2 chế độ:
    1. With models: nhận raw tensors, chạy qua cả 2 models
    2. Feature-only: nhận pre-computed features, chỉ chạy fusion

    Args:
        vision_model: DepthAwareSNN instance (optional)
        nlp_model: SpikingLanguageModel instance (optional)
        vision_dim: Chiều feature của vision model
        nlp_dim: Chiều feature của NLP model
        fused_dim: Chiều fused representation
    """

    def __init__(
        self,
        vision_model=None,
        nlp_model=None,
        vision_dim: int = 256,
        nlp_dim: int = 512,
        fused_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()

        self.vision_model = vision_model
        self.nlp_model    = nlp_model

        # Cross-modal attention fusion
        self.cross_attention = CrossModalAttention(
            vision_dim=vision_dim,
            nlp_dim=nlp_dim,
            out_dim=fused_dim,
            num_heads=num_heads,
        )

        # Action classifier sau fusion
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim // 2, len(ACTION_CLASSES)),
        )

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def fuse_features(
        self,
        vision_feat: torch.Tensor,
        nlp_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fusion vision + NLP features."""
        return self.cross_attention(vision_feat, nlp_feat)

    def forward(
        self,
        vision_feat: torch.Tensor,
        nlp_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused = self.fuse_features(vision_feat, nlp_feat)
        action_logits = self.action_head(fused)
        confidence = self.confidence_head(fused)
        return {
            'fused': fused,
            'action_logits': action_logits,
            'confidence': confidence,
        }

    def process(
        self,
        rgb_seq: Optional[torch.Tensor],
        depth_seq: Optional[torch.Tensor],
        speech_text: str,
        vision_result: Optional[VisionResult] = None,
        nlp_result: Optional[NLPResult] = None,
        device: str = 'cpu',
    ) -> FusedResult:
        """
        Xử lý đồng thời Vision + NLP và trả về FusedResult.

        Có thể truyền vào pre-computed results để skip inference.
        """
        t_start = time.time()

        # ── Parallel inference ─────────────────────────────────
        def run_vision():
            if vision_result is not None:
                return vision_result
            if self.vision_model is not None and rgb_seq is not None:
                t0 = time.time()
                with torch.no_grad():
                    bbox_3d, logits = self.vision_model(
                        rgb_seq.to(device), depth_seq.to(device)
                    )
                return VisionResult(
                    bbox_3d=bbox_3d,
                    class_logits=logits,
                    processing_time_ms=(time.time() - t0) * 1000,
                )
            return VisionResult(
                bbox_3d=torch.zeros(1, 6),
                class_logits=torch.zeros(1, 10),
            )

        def run_nlp():
            if nlp_result is not None:
                return nlp_result
            # Dùng NLU rule-based làm fallback nhanh
            from nlp_system.inference.nlu import NLUEngine
            nlu = NLUEngine()
            t0 = time.time()
            result = nlu.understand(speech_text)
            return NLPResult(
                intent=result['intent'],
                context='home',
                entities=result.get('params', {}),
                language='vi',
                confidence=result.get('confidence', 0.5),
                response_hint=result.get('response', ''),
                processing_time_ms=(time.time() - t0) * 1000,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            v_future = executor.submit(run_vision)
            n_future = executor.submit(run_nlp)
            vis_res  = v_future.result()
            nlp_res  = n_future.result()

        # ── Feature fusion ──────────────────────────────────────
        # Dùng dummy features nếu không có model features
        vision_feat = (
            nlp_res.features.unsqueeze(0)
            if nlp_res.features is not None
            else torch.zeros(1, 512)
        )
        nlp_feat = (
            nlp_res.features.unsqueeze(0)
            if nlp_res.features is not None
            else torch.zeros(1, 512)
        )

        with torch.no_grad():
            fused = self.fuse_features(
                vis_res.class_logits.mean(0).unsqueeze(0),
                torch.zeros(1, self.cross_attention.nlp_proj.in_features),
            )

        # ── Decision ────────────────────────────────────────────
        total_ms = (time.time() - t_start) * 1000
        decision = rule_based_decision(vis_res, nlp_res)

        return FusedResult(
            action=decision['action'],
            target_object=decision.get('target_object'),
            target_location=decision.get('target_location'),
            speech_response=decision.get('response', nlp_res.response_hint),
            confidence=nlp_res.confidence,
            vision=vis_res,
            nlp=nlp_res,
            fused_features=fused,
            total_latency_ms=total_ms,
        )


# ============================================================
# Action Classes
# ============================================================

ACTION_CLASSES = [
    'navigate',          # Dẫn đường
    'fetch_object',      # Lấy vật thể
    'explain',           # Giải thích / thông tin
    'control_device',    # Điều khiển thiết bị
    'set_reminder',      # Đặt nhắc nhở
    'play_media',        # Phát nhạc / video
    'weather_query',     # Truy vấn thời tiết
    'idle',              # Chờ
]

ACTION_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
IDX_ACTION = {i: a for a, i in ACTION_IDX.items()}


def rule_based_decision(
    vis_res: VisionResult,
    nlp_res: NLPResult,
) -> Dict[str, Any]:
    """
    Rule-based decision maker (fallback nhanh trước khi có trained model).

    Intent + Context + Objects → Action
    """
    intent = nlp_res.intent
    context = nlp_res.context
    entities = nlp_res.entities
    top_obj = vis_res.top_object

    # Airport scenarios
    if context == 'airport' or intent in (
        'ask_directions', 'check_flight_gate', 'check_flight_time',
        'check_in_request', 'report_lost_item'
    ):
        if intent == 'ask_directions':
            return {
                'action': 'navigate',
                'target_object': top_obj,
                'target_location': entities.get('place', entities.get('entity_1')),
                'response': f"Tôi sẽ dẫn bạn đến {entities.get('place', 'điểm đến')}.",
            }
        if intent in ('check_flight_gate', 'check_flight_time'):
            flight = entities.get('flight', entities.get('entity_2', ''))
            return {
                'action': 'explain',
                'target_object': None,
                'response': f"Thông tin chuyến bay {flight} đang được tra cứu.",
            }

    # Home scenarios
    if context == 'home' or intent in ('control_device', 'set_reminder', 'ask_weather', 'play_music'):
        if intent == 'control_device':
            device = entities.get('device', entities.get('entity_2', top_obj or 'thiết bị'))
            return {
                'action': 'control_device',
                'target_object': device,
                'response': f"Đang điều khiển {device}.",
            }
        if intent == 'set_reminder':
            return {'action': 'set_reminder', 'response': nlp_res.response_hint}
        if intent == 'play_music':
            return {'action': 'play_media', 'response': 'Đang phát nhạc...'}
        if intent == 'ask_weather':
            return {'action': 'weather_query', 'response': 'Đang kiểm tra thời tiết...'}

    # Classroom scenarios
    if context == 'classroom' or intent in ('ask_explanation', 'solve_problem', 'ask_grade'):
        return {
            'action': 'explain',
            'target_object': top_obj,
            'response': nlp_res.response_hint or 'Tôi sẽ giải thích cho bạn.',
        }

    # Fallback
    return {
        'action': 'idle',
        'target_object': None,
        'response': nlp_res.response_hint or 'Xin lỗi, tôi chưa hiểu yêu cầu.',
    }


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    print("=== VisionNLPBridge Test ===\n")

    bridge = VisionNLPBridge(
        vision_dim=10,    # DepthAwareSNN output classes
        nlp_dim=512,
        fused_dim=256,
    )
    print(f"Bridge created. Params: {sum(p.numel() for p in bridge.parameters()):,}")

    # Giả lập vision result
    vis = VisionResult(
        bbox_3d=torch.randn(1, 6),
        class_logits=torch.randn(1, 10),
        class_names=['cup', 'phone', 'bag'],
        confidence=[0.85],
    )

    # Giả lập NLP result
    nlp = NLPResult(
        intent='ask_directions',
        context='airport',
        entities={'place': 'cổng B2'},
        language='vi',
        confidence=0.92,
        response_hint='Tôi sẽ dẫn bạn đến cổng B2.',
    )

    # Decision
    decision = rule_based_decision(vis, nlp)
    print(f"Intent   : {nlp.intent}")
    print(f"Context  : {nlp.context}")
    print(f"Action   : {decision['action']}")
    print(f"Location : {decision.get('target_location')}")
    print(f"Response : {decision.get('response')}")

    print("\n[Cross-modal fusion test]")
    v_feat = torch.randn(2, 10)    # B=2, vision_dim=10
    n_feat = torch.randn(2, 512)   # B=2, nlp_dim=512

    # Manual cross-attention test
    cm = CrossModalAttention(vision_dim=10, nlp_dim=512, out_dim=64, num_heads=4)
    fused = cm(v_feat, n_feat)
    print(f"Fused shape: {fused.shape}")   # (2, 64)
    print("\n[Done]")
