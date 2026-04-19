import torch
import torch.nn as nn
import torch.nn.functional as F

class BiaffineDependencyParser(nn.Module):
    def __init__(self, word_dim=300, lstm_dim=256, hidden_dim=128, num_labels=20):
        super().__init__()
        self.lstm = nn.LSTM(word_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.mlp_head = nn.Linear(lstm_dim*2, hidden_dim)
        self.mlp_dep = nn.Linear(lstm_dim*2, hidden_dim)
        self.biaffine = nn.Bilinear(hidden_dim, hidden_dim, num_labels, bias=False)
        self.linear = nn.Linear(hidden_dim*2, num_labels)
    
    def forward(self, word_embeds, mask=None):
        lstm_out, _ = self.lstm(word_embeds)
        head = torch.tanh(self.mlp_head(lstm_out))
        dep = torch.tanh(self.mlp_dep(lstm_out))
        # Broadcast cho Bilinear
        B, T, H = head.size()
        head_exp = head.unsqueeze(2).expand(B, T, T, H)
        dep_exp = dep.unsqueeze(1).expand(B, T, T, H)
        # (B, T, T, num_labels)
        score = self.biaffine(head_exp, dep_exp)
        edge_scores = score.max(dim=-1).values  # (B, T, T)
        return edge_scores, score
