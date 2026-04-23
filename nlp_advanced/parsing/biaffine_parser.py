"""
Biaffine Parser cho Dependency Parsing - ĐÃ SỬA LỖI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiaffineDependencyParser(nn.Module):
    def __init__(self, word_dim=128, lstm_dim=256, hidden_dim=128, num_labels=20):
        super().__init__()
        self.lstm = nn.LSTM(word_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.mlp_head = nn.Linear(lstm_dim * 2, hidden_dim)
        self.mlp_dep = nn.Linear(lstm_dim * 2, hidden_dim)
        # Sửa lỗi biaffine: input1_dim, input2_dim, num_labels
        self.biaffine = nn.Bilinear(hidden_dim, hidden_dim, num_labels, bias=False)
        self.linear = nn.Linear(hidden_dim * 2, num_labels)
        
    def forward(self, word_embeds, mask=None):
        # word_embeds: (B, T, D)
        lstm_out, _ = self.lstm(word_embeds)  # (B, T, lstm_dim*2)
        head = torch.tanh(self.mlp_head(lstm_out))  # (B, T, hidden_dim)
        dep = torch.tanh(self.mlp_dep(lstm_out))    # (B, T, hidden_dim)
        
        # Biaffine score: (B, T, T, num_labels)
        # Cách đúng: head.unsqueeze(2) (B, T, 1, H) và dep.unsqueeze(1) (B, 1, T, H)
        biaffine_score = self.biaffine(head.unsqueeze(2), dep.unsqueeze(1))  # (B, T, T, num_labels)
        
        # Linear score
        head_dep_cat = torch.cat([head.unsqueeze(2).expand(-1, -1, dep.size(1), -1),
                                  dep.unsqueeze(1).expand(-1, head.size(1), -1, -1)], dim=-1)
        linear_score = self.linear(head_dep_cat)  # (B, T, T, num_labels)
        
        # Kết hợp
        score = biaffine_score + linear_score  # (B, T, T, num_labels)
        
        # Edge scores (max over labels)
        edge_scores = score.max(dim=-1).values  # (B, T, T)
        
        return edge_scores, score
