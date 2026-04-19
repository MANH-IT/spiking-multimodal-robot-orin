import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from underthesea import word_tokenize
from typing import Dict

class SpikedNLPFree(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        self.intents = ["thong_tin_truong", "tuyen_sinh", "dao_tao", "nghien_cuu", "khac"]
        self._init_vocab()
    
    def _init_vocab(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        words = [
            'trường', 'đại', 'học', 'giao', 'thông', 'vận', 'tải', 'utc',
            'tuyển', 'sinh', 'đào', 'tạo', 'ngành', 'điểm', 'chuẩn', 'xét',
            'hồ', 'sơ', 'chỉ', 'tiêu', 'học', 'phí', 'thời', 'gian',
            'kỹ', 'thuật', 'ô', 'tô', 'công', 'nghệ', 'thông', 'tin',
            'logistics', 'cầu', 'đường', 'nghiên', 'cứu', 'khoa', 'học',
            'xin', 'chào', 'cảm', 'ơn', 'robot', 'tạm', 'biệt', 'ở', 'đâu',
            'địa', 'chỉ', 'cơ', 'sở', 'thành', 'lập', 'lịch', 'sử',
            'bạn', 'khoẻ', 'không', 'giúp', 'cứu', 'trợ', 'lý'
        ]
        for i, w in enumerate(words, 2):
            self.vocab[w] = i
    
    def tokenize(self, text, max_len=20):
        tokens = word_tokenize(text.lower())
        ids = [self.vocab.get(t, 1) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long)
    
    def forward(self, x):
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)
    
    def predict_intent(self, text):
        # Rule-based fallback (ưu tiên hơn model)
        text_lower = text.lower()
        
        # Khac - Chào hỏi
        if any(w in text_lower for w in ['chào', 'xin chào', 'hello', 'hi', 'cảm ơn', 'tạm biệt']):
            return 'khac'
        
        # Thong tin truong
        if any(w in text_lower for w in ['ở đâu', 'địa chỉ', 'cơ sở', 'thành lập', 'lịch sử']):
            return 'thong_tin_truong'
        
        # Dao tao - Ngành học
        if any(w in text_lower for w in ['ngành', 'học gì', 'đào tạo', 'chương trình', 'môn học']):
            return 'dao_tao'
        
        # Tuyen sinh
        if any(w in text_lower for w in ['tuyển sinh', 'điểm chuẩn', 'xét tuyển', 'hồ sơ', 'chỉ tiêu']):
            return 'tuyen_sinh'
        
        # Nghien cuu
        if any(w in text_lower for w in ['nghiên cứu', 'đề tài', 'khoa học']):
            return 'nghien_cuu'
        
        # Fallback to model
        tokens = self.tokenize(text)
        with torch.no_grad():
            out = self.forward(tokens)
            idx = out.argmax().item()
            return self.intents[idx]
    
    def predict_with_confidence(self, text):
        tokens = self.tokenize(text)
        with torch.no_grad():
            out = self.forward(tokens)
            probs = torch.softmax(out, dim=1)
            return {self.intents[i]: float(probs[0][i]) for i in range(5)}


def init_nlp(model_path=None):
    model = SpikedNLPFree()
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'snn_model.pth')
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {model_path}")
        except Exception as e:
            print(f"⚠️ Using untrained model: {e}")
    return model


if __name__ == "__main__":
    model = init_nlp()
    test_queries = [
        "Trường UTC ở đâu?",
        "Ngành Kỹ thuật ô tô học gì?",
        "Xin chào robot"
    ]
    for q in test_queries:
        print(f"'{q}' → {model.predict_intent(q)}")