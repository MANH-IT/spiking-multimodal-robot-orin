import torch
import torch.nn as nn
import snntorch as snn

class SpikeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, T=20, max_len=100):
        super().__init__()
        self.T = T
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.lif = snn.Leaky(beta=0.9)
        # positional encoding (dạng xung)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim))
    
    def forward(self, x):
        # x: (B, seq_len)
        emb = self.embed(x)  # (B, seq_len, embed_dim)
        emb = self.proj(emb) + self.pos_embed[:, :x.shape[1], :]
        spikes = []
        mem = torch.zeros(emb.shape[0], emb.shape[1], emb.shape[2], device=x.device)
        for t in range(self.T):
            spk, mem = self.lif(emb, mem)
            spikes.append(spk)
        return torch.stack(spikes, dim=0)  # (T, B, seq_len, hidden_dim)
