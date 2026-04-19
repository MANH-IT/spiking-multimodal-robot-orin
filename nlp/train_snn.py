"""
Train Spiking Neural Network cho phân loại ý định tiếng Việt - Robot EEEC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
from underthesea import word_tokenize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spiked_nlp_free import SpikedNLPFree


class IntentDataset(Dataset):
    def __init__(self, data, vocab, max_length=20):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.intent_to_idx = {
            "thong_tin_truong": 0,
            "tuyen_sinh": 1,
            "dao_tao": 2,
            "nghien_cuu": 3,
            "khac": 4
        }
    
    def text_to_indices(self, text):
        tokens = word_tokenize(text.lower())
        indices = []
        for token in tokens:
            idx = self.vocab.get(token, 1)  # 1 là UNK
            indices.append(idx)
        
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices += [0] * (self.max_length - len(indices))
        
        return indices
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        indices = self.text_to_indices(item['text'])
        intent_idx = self.intent_to_idx[item['intent']]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(intent_idx, dtype=torch.long)


def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for item in data:
        tokens = word_tokenize(item['text'].lower())
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def train_model(model, train_loader, epochs=100, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print(f"🚀 Bắt đầu training trên {device.upper()}")
    print(f"📊 Số lượng mẫu: {len(train_loader.dataset)}")
    print(f"📦 Batch size: {train_loader.batch_size}")
    print("-" * 50)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    
    print("-" * 50)
    print(f"✅ Training hoàn tất! Accuracy cuối: {acc:.2f}%")
    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training_data.json')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        sys.exit(1)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"📚 Loaded {len(training_data)} training samples")
    
    # Build vocab
    vocab = build_vocab(training_data)
    print(f"📖 Vocabulary size: {len(vocab)}")
    
    # Setup dataset
    dataset = IntentDataset(training_data, vocab)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model - FIXED: đúng tham số của SpikedNLPFree
    model = SpikedNLPFree(vocab_size=len(vocab), embed_dim=64, num_classes=5)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    
    # Train
    model = train_model(model, train_loader, epochs=100, device=device)
    
    # Save model
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snn_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved model to {save_path}")
    
    # Test
    print("\n🧪 Testing after training:")
    test_queries = [
        "Trường UTC ở đâu?",
        "Ngành Kỹ thuật ô tô học gì?",
        "Xin chào robot"
    ]
    
    model.eval()
    intents = ["thong_tin_truong", "tuyen_sinh", "dao_tao", "nghien_cuu", "khac"]
    for query in test_queries:
        indices = dataset.text_to_indices(query)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            intent_idx = output.argmax().item()
            confidence = probs[0][intent_idx].item()
            print(f"  '{query}' → {intents[intent_idx]} (confidence: {confidence:.2%})")