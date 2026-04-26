# Task: [A] Connect real SNN features to Fusion Transformer
# File: d:\robot_eeec\multimodal_fusion\learned_bridge.py

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

from multimodal_fusion.spiking_fusion import (
    build_fusion_model, SpikingFusionTransformer,
    ROBOT_ACTIONS, RuleBasedTextEncoder,
)

# ── Fallback: import bridge cũ nếu cần ───────────────────────────────────────
try:
    from multimodal_fusion.bridge import MultimodalBridge as _OldBridge
    _HAS_OLD_BRIDGE = True
except ImportError:
    _HAS_OLD_BRIDGE = False


# ─────────────────────────────────────────────────────────────────────────────
# LearnedMultimodalBridge
# ─────────────────────────────────────────────────────────────────────────────

class LearnedMultimodalBridge:
    """
    Multimodal Bridge dùng SpikingFusionTransformer.
    
    Interface giống MultimodalBridge cũ:
        result = bridge.process_frame_with_query(rgbd_frames, user_query)
    """

    def __init__(
        self,
        vision_model=None,
        nlp_processor=None,
        model_path: str = "models/fusion/best_fusion.pth",
        config: dict = None,
        class_names: List[str] = None,
        device: str = "auto",
        fallback_to_rules: bool = True,
    ):
        self.vision_model    = vision_model
        self.nlp_processor   = nlp_processor
        self.class_names     = class_names or ["person", "obstacle"]
        self.fallback_rules  = fallback_to_rules
        self.history         = []

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        cfg = config or {}
        if 'use_phobert' not in cfg:
            cfg['use_phobert'] = False
        self.model: SpikingFusionTransformer = build_fusion_model(cfg)
        self.model.to(self.device)

        # Text encoder (offline keyword-based, không phụ thuộc PhoBERT)
        self.text_enc = RuleBasedTextEncoder(d_model=256, n_tokens=16)

        # Load weights nếu có
        abs_path = os.path.join(root, model_path)
        if os.path.exists(abs_path):
            try:
                self.model.load_state_dict(
                    torch.load(abs_path, map_location=self.device)
                )
                print(f"✅ FusionTransformer loaded from {abs_path}")
            except Exception as e:
                print(f"⚠️ Cannot load fusion weights: {e} — using random init")
        else:
            print(f"⚠️ Fusion model not found at {abs_path}. Run train_fusion.py first.")
            print(f"   Using random initialized model (degraded performance).")

        self.model.eval()

        # Fallback rule-based bridge
        self._rule_bridge = None
        if fallback_to_rules and _HAS_OLD_BRIDGE:
            self._rule_bridge = _OldBridge(vision_model, nlp_processor, class_names=self.class_names)

    # ── Vision Processing ────────────────────────────────────────────────────

    def _run_vision(self, rgbd_frames: List[np.ndarray]) -> Dict:
        """
        Chạy SNN vision và trả về cả detections VÀ raw spike features.
        """
        if not self.vision_model or not rgbd_frames:
            return {"detections": [], "spikes": None}
        
        try:
            # 1. Chạy lấy detections
            boxes, conf, cls = self.vision_model.process_frame_sequence(rgbd_frames)
            detections = []
            for i in range(len(boxes[0])):
                score = float(conf[0][i][0])
                if score > 0.5:
                    bbox = boxes[0][i].tolist()
                    detections.append({
                        "class":      self.class_names[int(cls[0][i].argmax())],
                        "confidence": score,
                        "bbox":       bbox,
                        "center":     [
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                        ],
                    })
            
            # 2. Lấy raw spike features thật (Option A)
            # ThreeDSpikingCNN trả về (T, B, C, Hf, Wf) khi return_raw_spikes=True
            vision_spikes = self.vision_model.process_frame_sequence(rgbd_frames, return_raw_spikes=True)
            
            return {
                "detections": detections,
                "spikes":     vision_spikes
            }
        except Exception as e:
            print(f"[VisionBridge] Error: {e}")
            return {"detections": [], "spikes": None}

    # ── Text Processing ──────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        """Text → token_ids tensor (B=1, n_tokens)."""
        token_ids = self.text_enc.encode_text(text).unsqueeze(0).to(self.device)
        return token_ids

    # ── Main interface ───────────────────────────────────────────────────────

    def process_frame_with_query(
        self,
        rgbd_frames: List[np.ndarray],
        user_query: Optional[str] = None,
    ) -> Dict:
        """
        Main API: Fusion Vision + NLP
        """
        result = {
            "detections": [],
            "intent":     None,
            "action":     None,
            "response":   None,
            "vision_spikes": None,
        }

        # 1. Vision (Lấy cả detections và spikes thật)
        vision_out = self._run_vision(rgbd_frames)
        detections = vision_out["detections"]
        vision_spikes = vision_out["spikes"] # (T, B, C, Hf, Wf)
        
        result["detections"] = detections
        result["vision_spikes"] = vision_spikes

        # 2. NLP intent
        if user_query and self.nlp_processor:
            try:
                nlp_out = self.nlp_processor.process(user_query)
                result["intent"] = nlp_out.get("intent")
            except Exception as e:
                print(f"[NLPBridge] Error: {e}")

        # 3. Fusion inference
        if user_query:
            try:
                # Encode text
                token_ids = self._encode_text(user_query)

                # Nếu không lấy được spikes thật, dùng mock làm fallback
                if vision_spikes is None:
                    # Mock vision spikes từ detections
                    T, C, HW = 8, 64, 16
                    vision_spikes = torch.rand(T, 1, C, HW, HW, device=self.device) * 0.1
                    for det in detections:
                        if det["class"] == "person":
                            vision_spikes[:, :, :32, 4:12, 4:12] += 0.5
                    print("⚠️ Using mock spikes (Vision model failed to return real spikes)")

                # Forward qua Fusion Transformer
                pred = self.model.predict(vision_spikes, token_ids)

                # Lấy target từ detections
                persons   = [d for d in detections if d["class"] == "person"]
                obstacles = [d for d in detections if d["class"] == "obstacle"]
                target = None
                if pred["action_type"] in ("follow", "approach_and_greet") and persons:
                    target = persons[0]["center"]
                elif pred["action_type"] == "navigate_around" and obstacles:
                    target = obstacles[0]["center"]

                # Xây dựng action dict
                action = {
                    "type":         pred["action_type"],
                    "target":       target,
                    "confidence":   pred["confidence"],
                    "needs_confirm": pred["needs_confirm"],
                    "action_probs": pred["action_probs"],
                    "response":     self._get_response(pred["action_type"], pred),
                    "parameters":   {},
                }
                result["action"] = action
                result["response"] = action["response"]

            except Exception as e:
                print(f"[FusionBridge] Inference error: {e}")
                # Fallback sang rule-based nếu có
                if self._rule_bridge:
                    return self._rule_bridge.process_frame_with_query(
                        rgbd_frames, user_query
                    )

        self.history.append(result)
        if len(self.history) > 30:
            self.history.pop(0)

        return result

    def _get_response(self, action_type: str, pred: dict) -> str:
        """Tạo response text cho action (tiếng Việt)."""
        conf = pred["confidence"]
        if pred["needs_confirm"]:
            return f"Tôi không chắc chắn lắm ({conf:.0%}). Bạn muốn tôi '{action_type}' không?"

        responses = {
            "idle":              "Đang sẵn sàng. Hãy cho tôi biết bạn cần gì.",
            "follow":            "Đang đi theo bạn. Hãy dẫn đường nhé!",
            "navigate_around":   "Phát hiện vật cản. Đang tìm đường tránh.",
            "approach_and_greet":"Chào bạn! Rất vui được gặp bạn.",
            "search":            "Đang tìm kiếm người xung quanh...",
            "respond":           "Tôi đang tìm câu trả lời cho bạn...",
        }
        return responses.get(action_type, "Đã nhận lệnh.")
