import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class ParametricLIF(nn.Module):
    def __init__(self, beta_init=0.9, threshold_init=0.5, learn_beta=True, learn_threshold=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init)) if learn_beta else beta_init
        self.threshold = nn.Parameter(torch.tensor(threshold_init)) if learn_threshold else threshold_init
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=surrogate.fast_sigmoid())
    def forward(self, input, mem):
        return self.lif(input, mem)
    def init_mem(self, batch_size, dim, device):
        return torch.zeros(batch_size, dim, device=device)
