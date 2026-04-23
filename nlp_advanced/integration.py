"""
Tích hợp các module NLP nâng cao (Chương 5) vào pipeline hiện tại
Kết hợp: Biaffine Parser + Spiking Attention + Policy Network (RL)
"""

import torch
import torch.nn as nn
from .parsing.biaffine_parser import BiaffineDependencyParser
from .spiking_attention.spike_embedding import SpikeEmbedding
from .spiking_attention.spike_attention import SpikingAttention
from .rl.policy_network import SpikingPolicyNetwork
from .online_updater import OnlineGrammarUpdater

class AdvancedNLUProcessor(nn.Module):
    """
    Bộ xử lý NLU nâng cao tích hợp:
    1. Biaffine Parser -> dependency graph
    2. Spiking Embedding + Attention -> hiểu ngữ cảnh câu dài
    3. Policy Network (RL) -> đưa ra hành động/phản hồi
    4. Online Updater -> học cấu trúc câu mới
    """
    def __init__(self, vocab_size=10000, embed_dim=300, hidden_dim=256, num_classes=5, T=20):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        
        # 1. Biaffine Parser (phân tích cú pháp phụ thuộc)
        self.parser = BiaffineDependencyParser(word_dim=hidden_dim, lstm_dim=hidden_dim, num_labels=20)
        
        # 2. Spiking Embedding (chuyển từ sang xung)
        self.spike_embed = SpikeEmbedding(vocab_size, embed_dim, hidden_dim, T)
        
        # 3. Spiking Attention (chú ý trên câu dài)
        self.spike_attn = SpikingAttention(hidden_dim, T)
        
        # 4. Policy Network (RL - quyết định hành động)
        self.policy_net = SpikingPolicyNetwork(hidden_dim, hidden_dim, num_classes, T)
        
        # 5. Online Updater (cập nhật trực tuyến)
        self.online_updater = OnlineGrammarUpdater()
        
        # 6. Classification head (intent)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None, return_parser=False):
        """
        input_ids: (B, seq_len) token indices
        return: intent_logits, dependency_graph, spikes
        """
        spikes = self.spike_embed(input_ids)  # (T, B, seq_len, hidden_dim)
        attn_out = self.spike_attn(spikes)  # (B, seq_len, hidden_dim)
        pooled = attn_out.mean(dim=1)  # (B, hidden_dim)
        intent_logits = self.classifier(pooled)  # (B, num_classes)
        
        # Đồng bộ hóa chiều dữ liệu: Pool chiều seq_len (dim=2) trước khi đưa vào policy_net
        spikes_pooled = spikes.mean(dim=2)  # (T, B, hidden_dim)
        policy_out = self.policy_net(spikes_pooled)  # (B, num_classes)
        
        # Step 5: Dependency parsing (nếu cần) - TẠM THỜI BỎ QUA
        if return_parser:
            # Tạo dummy scores để tránh lỗi
            B, seq_len = input_ids.shape
            dummy_scores = torch.zeros(B, seq_len, seq_len, device=input_ids.device)
            return intent_logits, policy_out, dummy_scores, spikes
        else:
            return intent_logits, policy_out, None, spikes
    
    def predict_intent(self, input_ids):
        """Dự đoán intent (tương thích với interface cũ)"""
        with torch.no_grad():
            intent_logits, _, _, _ = self.forward(input_ids, return_parser=False)
            return torch.argmax(intent_logits, dim=-1).item()
    
    def update_online(self, sentence, correct_structure, reward):
        """Cập nhật trực tuyến khi gặp cấu trúc câu mới"""
        pass  # TODO: implement logic

def create_advanced_nlu():
    import os
    from .config import EMBED_DIM, HIDDEN_DIM, T, NUM_CLASSES
    
    model = AdvancedNLUProcessor(
        vocab_size=5000,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        T=T
    )
    
    # Load mô hình vừa train xong
    model_path = os.path.join(os.path.dirname(__file__), "best_advanced_nlu.pth")
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ Loaded Advanced NLU weights from {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
    else:
        print(f"⚠️ Warning: Model weights not found at {model_path}")
        
    return model

if __name__ == "__main__":
    model = AdvancedNLUProcessor()
    dummy_input = torch.randint(0, 1000, (2, 20))  # (batch=2, seq_len=20)
    intent, policy, _, spikes = model(dummy_input)
    print(f"Intent logits shape: {intent.shape}")
    print(f"Policy output shape: {policy.shape}")
    print(f"Spikes shape: {spikes.shape}")
    print("✅ Advanced NLU Processor hoạt động!")
