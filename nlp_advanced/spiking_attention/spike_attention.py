import torch
import torch.nn as nn
import snntorch as snn

class SpikingAttention(nn.Module):
    def __init__(self, dim, T=20):
        super().__init__()
        self.T = T
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.lif = snn.Leaky(beta=0.9)
    
    def forward(self, spikes):
        # spikes: (T, B, seq_len, dim)
        T, B, L, D = spikes.shape
        Q = self.q_proj(spikes)  # (T,B,L,D)
        K = self.k_proj(spikes)
        V = self.v_proj(spikes)
        
        # Accumulate over time (spike-driven attention)
        attn_scores = torch.zeros(B, L, L, device=spikes.device)
        for t in range(T):
            attn_scores += torch.einsum('bld,bmd->blm', Q[t], K[t])
        attn = torch.softmax(attn_scores / (D**0.5), dim=-1)
        out = torch.einsum('blm,bmd->bld', attn, V.mean(dim=0))
        return out
