import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from underthesea import word_tokenize

class SimpleIntentClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=64, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)

class SimpleDataset(Dataset):
    def __init__(self, data, vocab, max_len=20):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.intent_map = {
            "thong_tin_truong": 0, "tuyen_sinh": 1, 
            "dao_tao": 2, "nghien_cuu": 3, "khac": 4
        }
    
    def text_to_ids(self, text):
        tokens = word_tokenize(text.lower())
        ids = [self.vocab.get(t, 1) for t in tokens[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ids = self.text_to_ids(item['text'])
        label = self.intent_map[item['intent']]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_data.json')
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Build vocab
vocab = {"<PAD>": 0, "<UNK>": 1}
idx = 2
for item in data:
    for token in word_tokenize(item['text'].lower()):
        if token not in vocab:
            vocab[token] = idx
            idx += 1

print(f"Vocab size: {len(vocab)}")

# Create dataset
dataset = SimpleDataset(data, vocab)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
model = SimpleIntentClassifier(vocab_size=len(vocab), embed_dim=64, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nTraining...")
for epoch in range(50):
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    if (epoch + 1) % 10 == 0:
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:3d} - Loss: {total_loss/len(loader):.4f}, Acc: {acc:.2f}%")

# Save model
save_path = os.path.join(os.path.dirname(__file__), 'simple_model.pth')
torch.save(model.state_dict(), save_path)
print(f"\nSaved model to {save_path}")

# Test
print("\nTesting:")
test_queries = [
    "Điểm chuẩn năm nay bao nhiêu",
    "Trường có ngành gì",
    "Xin chào robot"
]

intent_map_rev = {0: "thong_tin_truong", 1: "tuyen_sinh", 2: "dao_tao", 3: "nghien_cuu", 4: "khac"}

model.eval()
for q in test_queries:
    ids = dataset.text_to_ids(q)
    with torch.no_grad():
        out = model(torch.tensor([ids]))
        pred = out.argmax().item()
        probs = torch.softmax(out, dim=1)
        conf = probs[0][pred].item()
        print(f"  '{q}' → {intent_map_rev[pred]} (conf: {conf:.2%})")