import torch
import torch.nn as nn

class VisionNLPBridge:
    def __init__(self, vision_model=None):
        self.vision_model = vision_model
    
    def to(self, device):
        if self.vision_model:
            self.vision_model.to(device)
        return self

    def process(self, rgb_seq, depth_seq, speech_text, device="cpu"):
        # Mock processing result
        class Result:
            def __init__(self):
                self.speech_response = f"Tôi đã nhận được yêu cầu: '{speech_text}'"
                self.action = "idle"
                self.target_object = "none"
                self.confidence = 0.9
                self.total_latency_ms = 150
        
        return Result()
