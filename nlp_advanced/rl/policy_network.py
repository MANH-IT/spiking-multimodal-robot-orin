import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SpikingPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, T=20):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
    
    def forward(self, x):
        # x: (T, B, input_dim)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []
        for t in range(self.T):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
        out = torch.stack(spk_rec, dim=0).mean(dim=0)  # (B, output_dim)
        return out
