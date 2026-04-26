import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class ParametricLIF(nn.Module):
    def __init__(self, beta_init=0.9, threshold_init=0.5, learn_beta=True, learn_threshold=True):
        super().__init__()
        # Dùng trực tiếp learn_beta/threshold của snntorch.Leaky
        # Điều này đảm bảo parameters được quản lý đúng bởi Module.to(device)
        self.lif = snn.Leaky(
            beta=beta_init, 
            threshold=threshold_init, 
            learn_beta=learn_beta, 
            learn_threshold=learn_threshold,
            spike_grad=surrogate.fast_sigmoid()
        )
    def forward(self, input, mem):
        return self.lif(input, mem)
    def init_mem(self, batch_size, dim, device):
        return torch.zeros(batch_size, dim, device=device)
