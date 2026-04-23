import torch
import torch.nn as nn
import snntorch as snn
from vision_system.models.snn.lif_neuron import ParametricLIF

class SpikingFusionNetwork(nn.Module):
    """
    Hệ thống hội tụ đa phương thức dựa trên SNN.
    Kết hợp luồng xung từ Vision (3D SNN) và NLP (Spiking NLU).
    """
    def __init__(self, vision_dim=64, nlp_dim=128, fusion_hidden=256, num_outputs=5, T=20):
        super().__init__()
        self.T = T
        
        # Lớp xử lý đặc trưng Vision (Pooling spatial dims)
        self.vision_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vision_fc = nn.Linear(vision_dim, fusion_hidden // 2)
        
        # Lớp xử lý đặc trưng NLP
        self.nlp_fc = nn.Linear(nlp_dim, fusion_hidden // 2)
        
        # Lớp hội tụ (Fusion Layer)
        self.fusion_lif = ParametricLIF(beta_init=0.9, threshold_init=0.5)
        self.out_fc = nn.Linear(fusion_hidden, num_outputs)
        
    def forward(self, vision_spikes, nlp_spikes):
        """
        Args:
            vision_spikes: (T, B, 64, H, W) hoặc (T, B, 64, 1, H, W)
            nlp_spikes: (T, B, dim) luồng xung từ NLU
            
        Returns:
            fused_spikes: (T, B, num_outputs)
        """
        # Squeeze depth dim nếu có (kết quả từ 3D SNN trả về (T, B, C, 1, H, W))
        if vision_spikes.dim() == 6:
            vision_spikes = vision_spikes.squeeze(3)
            
        T, B, C, H, W = vision_spikes.shape
        
        # 1. Tiền xử lý Vision Spikes
        # (T*B, C, H, W) để pooling nhanh
        v_flat = vision_spikes.view(-1, C, H, W)
        v_pooled = self.vision_pool(v_flat).view(T, B, C)
        v_feats = self.vision_fc(v_pooled) # (T, B, fusion_hidden//2)
        
        # 2. Tiền xử lý NLP Spikes
        # Giả định nlp_spikes đã có dạng (T, B, dim)
        n_feats = self.nlp_fc(nlp_spikes) # (T, B, fusion_hidden//2)
        
        # 3. Concatenate (Hội tụ đặc trưng)
        fused_input = torch.cat([v_feats, n_feats], dim=-1) # (T, B, fusion_hidden)
        
        # 4. Truyền qua lớp Spiking Fusion
        hidden_dim = fused_input.shape[-1]
        mem = self.fusion_lif.init_mem(B, hidden_dim, vision_spikes.device)
        fused_outputs = []
        
        for t in range(T):
            # Cổng hội tụ: Chỉ cho phép vision/nlp tương tác qua LIF neuron
            # Điều này tạo ra "Cross-modal coincidence detection"
            spk, mem = self.fusion_lif(fused_input[t], mem)
            # Map sang số lượng output (intent/action)
            out = self.out_fc(spk)
            fused_outputs.append(out)
            
        return torch.stack(fused_outputs, dim=0) # (T, B, num_outputs)

    def predict(self, vision_spikes, nlp_spikes):
        """Trả về intent/action cuối cùng sau khi tích lũy xung"""
        fused_spikes = self.forward(vision_spikes, nlp_spikes)
        # Tích lũy xung qua thời gian
        sum_spikes = fused_spikes.sum(dim=0)
        return torch.argmax(sum_spikes, dim=-1)
