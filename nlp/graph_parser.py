"""
Graph-based Dependency Parser cho tiếng Việt
Phân tích cú pháp dựa trên đồ thị cho câu tiếng Việt
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from underthesea import word_tokenize, pos_tag
from typing import List, Dict, Tuple

class GraphDependencyParser(nn.Module):
    """Graph-based parser using GNN and attention"""
    
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, num_relations=15):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(20, 32)  # 20 POS tags
        
        # Graph convolution layers
        self.gcn1 = nn.Linear(embed_dim + 32, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention for edge prediction
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Dependency relation classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_relations)
        )
        
        # POS tag mapping
        self.pos_map = {
            'N': 0, 'Np': 1, 'V': 2, 'A': 3, 'R': 4,
            'P': 5, 'C': 6, 'L': 7, 'M': 8, 'E': 9,
            'T': 10, 'X': 11, 'CH': 12, 'CL': 13, 'FW': 14
        }
        
        self.relations = [
            'root', 'nsubj', 'dobj', 'iobj', 'amod', 'nmod',
            'case', 'det', 'aux', 'cc', 'conj', 'mark',
            'advmod', 'neg', 'dep'
        ]
    
    def encode_sentence(self, tokens: List[str], pos_tags: List[Tuple]) -> torch.Tensor:
        """Encode sentence thành vector representations"""
        # Token to indices (simplified - in practice use proper vocab)
        token_ids = torch.tensor([hash(t) % 10000 for t in tokens[:50]])
        
        # POS to indices
        pos_ids = []
        for word, tag in pos_tags[:50]:
            pos_ids.append(self.pos_map.get(tag, 14))
        
        # Pad
        max_len = 50
        if len(token_ids) < max_len:
            token_ids = torch.cat([token_ids, torch.zeros(max_len - len(token_ids))])
            pos_ids = pos_ids + [14] * (max_len - len(pos_ids))
        
        token_ids = token_ids.long()
        pos_ids = torch.tensor(pos_ids[:max_len]).long()
        
        # Embeddings
        token_emb = self.embedding(token_ids)
        pos_emb = self.pos_embedding(pos_ids)
        
        # Concatenate
        return torch.cat([token_emb, pos_emb], dim=-1).unsqueeze(0)
    
    def build_dependency_graph(self, sentence: str) -> nx.DiGraph:
        """Xây dựng dependency graph cho câu"""
        # Tokenize
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(sentence)
        
        # Encode
        features = self.encode_sentence(tokens, pos_tags)
        
        # Graph convolution
        h = torch.relu(self.gcn1(features))
        h = torch.relu(self.gcn2(h))
        
        # Self-attention
        h_attn, _ = self.attention(h, h, h)
        
        # Predict edges
        num_tokens = len(tokens)
        G = nx.DiGraph()
        
        # Add nodes
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            G.add_node(i, token=token, pos=pos[1])
        
        # Predict dependencies
        for i in range(min(num_tokens, 50)):
            for j in range(min(num_tokens, 50)):
                if i != j:
                    edge_feat = torch.cat([h_attn[0, i], h_attn[0, j]])
                    rel_logits = self.edge_classifier(edge_feat)
                    rel_idx = rel_logits.argmax().item()
                    
                    if rel_idx > 0:  # Non-zero relation
                        G.add_edge(i, j, relation=self.relations[rel_idx])
        
        return G, tokens, pos_tags
    
    def parse(self, sentence: str) -> Dict:
        """Parse sentence và trả về dependency tree"""
        G, tokens, pos_tags = self.build_dependency_graph(sentence)
        
        # Find root (node with no incoming edges)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        return {
            'sentence': sentence,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'dependency_graph': G,
            'root': roots[0] if roots else None,
            'edges': [(u, v, G.edges[u, v].get('relation', 'dep')) 
                      for u, v in G.edges()]
        }
    
    def get_dependency_path(self, sentence: str, word1: str, word2: str) -> List:
        """Tìm đường đi dependency giữa 2 từ"""
        result = self.parse(sentence)
        G = result['dependency_graph']
        tokens = result['tokens']
        
        try:
            idx1 = tokens.index(word1)
            idx2 = tokens.index(word2)
            path = nx.shortest_path(G, idx1, idx2)
            return [tokens[i] for i in path]
        except (ValueError, nx.NetworkXNoPath):
            return []


# Simple test
if __name__ == "__main__":
    parser = GraphDependencyParser()
    
    test_sentences = [
        "Trường Đại học Giao thông Vận tải có nhiều ngành đào tạo",
        "Sinh viên học tập và nghiên cứu khoa học",
        "Tuyển sinh năm 2024 bắt đầu từ tháng 3"
    ]
    
    for sent in test_sentences:
        print(f"\n📝 Câu: {sent}")
        result = parser.parse(sent)
        print(f"   Tokens: {result['tokens']}")
        print(f"   Root: {result['root']}")
        print(f"   Edges: {result['edges'][:5]}...")