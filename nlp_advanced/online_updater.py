import torch
import torch.nn as nn

class OnlineGrammarUpdater(nn.Module):
    """
    Cập nhật trực tuyến các trọng số của biaffine parser dựa trên phản hồi.
    Sử dụng một SNN nhẹ để điều chỉnh điểm số cạnh.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.snn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_delta(self, head_vec, dep_vec, reward):
        # head_vec, dep_vec: vector biểu diễn của head và dependent
        combined = torch.cat([head_vec, dep_vec], dim=-1)
        delta = self.snn(combined) * reward
        return delta
