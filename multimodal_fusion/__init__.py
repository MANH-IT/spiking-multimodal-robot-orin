# multimodal_fusion/__init__.py
"""
Multimodal Fusion Module - NCKH 2026
=====================================
Tích hợp Vision System + NLP System.
"""

from multimodal_fusion.bridges.vision_nlp_bridge import (
    VisionNLPBridge,
    CrossModalAttention,
    VisionResult,
    NLPResult,
    FusedResult,
    rule_based_decision,
    ACTION_CLASSES,
)
from multimodal_fusion.decision.robot_decision import (
    RobotDecisionMaker,
    RobotAction,
)

__all__ = [
    'VisionNLPBridge',
    'CrossModalAttention',
    'VisionResult',
    'NLPResult',
    'FusedResult',
    'RobotDecisionMaker',
    'RobotAction',
    'rule_based_decision',
    'ACTION_CLASSES',
]
