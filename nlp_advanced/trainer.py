"""
Trainer cho Advanced NLU Processor
Huấn luyện đồng thời:
1. Intent classification (Cross-entropy loss)
2. Dependency parsing (Biaffine loss)
3. Policy network (RL - PPO, tùy chọn)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import modules
import sys
sys.path.append('D:/robot_eeec')
from nlp.spiked_nlp_free import SpikedNLPFree
from nlp_advanced.integration import AdvancedNLUProcessor
from nlp_advanced.config import *

# ==================== DATASET ====================
class AdvancedNLUDataset(Dataset):
    def __init__(self, intent_path, dep_path, tokenizer, max_len=MAX_SENT_LEN):
        # Load intent data
        with open(intent_path, 'r', encoding='utf-8') as f:
            self.intent_data = json.load(f)
        
        # Load dependency data (nếu có)
        self.dep_data = {}
        if os.path.exists(dep_path):
            with open(dep_path, 'r', encoding='utf-8') as f:
                dep_list = json.load(f)
                for item in dep_list:
                    self.dep_data[item['text']] = item
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.intent_to_idx = {
            "thong_tin_truong": 0,
            "tuyen_sinh": 1,
            "dao_tao": 2,
            "nghien_cuu": 3,
            "khac": 4
        }
    
    def __len__(self):
        return len(self.intent_data)
    
    def __getitem__(self, idx):
        item = self.intent_data[idx]
        text = item['text']
        intent = item['intent']
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text, max_len=self.max_len)
        label = self.intent_to_idx[intent]
        
        # Dependency data (nếu có)
        dep_info = self.dep_data.get(text, None)
        if dep_info:
            heads = torch.tensor(dep_info['heads'], dtype=torch.long)
            dep_labels = dep_info.get('dep_labels', [''] * len(heads))
        else:
            heads = torch.tensor([-1] * self.max_len, dtype=torch.long)
            dep_labels = [''] * self.max_len
        
        return {
            'input_ids': tokens.squeeze(0),
            'label': label,
            'text': text,
            'heads': heads,
            'dep_labels': dep_labels
        }


def collate_fn(batch):
    """Ghép batch với padding"""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    labels = []
    texts = []
    heads_list = []
    
    for item in batch:
        ids = item['input_ids']
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item['label'])
        texts.append(item['text'])
        
        # Xử lý heads
        heads = item['heads']
        if len(heads) < max_len:
            heads = torch.cat([heads, torch.full((max_len - len(heads),), -1, dtype=torch.long)])
        heads_list.append(heads)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.tensor(labels, dtype=torch.long),
        'texts': texts,
        'heads': torch.stack(heads_list)
    }


# ==================== LOSS FUNCTIONS ====================
class DependencyLoss(nn.Module):
    """Loss cho dependency parsing (Biaffine)"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, pred_heads, target_heads, mask=None):
        # pred_heads: (B, seq_len, seq_len) scores
        # target_heads: (B, seq_len)
        B, L, _ = pred_heads.shape
        loss = torch.tensor(0.0, device=pred_heads.device)
        for b in range(B):
            for i in range(L):
                if target_heads[b, i] != -1:
                    loss = loss + self.ce(pred_heads[b, i].unsqueeze(0), target_heads[b, i].unsqueeze(0))
        return loss / (B * L)


class CombinedLoss(nn.Module):
    """Kết hợp intent loss + dependency loss"""
    def __init__(self, lambda_intent=1.0, lambda_dep=0.2):
        super().__init__()
        self.intent_loss = nn.CrossEntropyLoss()
        self.dep_loss = DependencyLoss()
        self.lambda_intent = lambda_intent
        self.lambda_dep = lambda_dep
    
    def forward(self, intent_logits, dep_scores, target_intent, target_heads):
        loss_intent = self.intent_loss(intent_logits, target_intent)
        loss_dep = self.dep_loss(dep_scores, target_heads)
        total = self.lambda_intent * loss_intent + self.lambda_dep * loss_dep
        return total, loss_intent, loss_dep


# ==================== TRAINER ====================
class AdvancedNLUTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3) # Tăng L2 phạt trọng lượng chặt hơn vì train rất lâu
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6) # Đổi về decay mượt trôi dần tới điểm hội tụ lớn nhất
        self.best_val_acc = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.patience = 30 # Cắt sớm nếu 30 vòng liên tiếp không cải thiện
        self.epochs_no_improve = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device).long()
            labels = batch['labels'].to(self.device)
            target_heads = batch['heads'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward qua Advanced NLU
            intent_logits, policy_out, dep_scores, spikes = self.model(input_ids, return_parser=True)
            
            # Tính loss
            loss, loss_intent, loss_dep = self.criterion(intent_logits, dep_scores, labels, target_heads)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = intent_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'intent_loss': f'{loss_intent.item():.4f}',
                'dep_loss': f'{loss_dep.item():.4f}'
            })
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)
        return train_acc, avg_loss
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device).long()
                labels = batch['labels'].to(self.device)
                target_heads = batch['heads'].to(self.device)
                
                intent_logits, _, dep_scores, _ = self.model(input_ids, return_parser=True)
                loss, loss_intent, loss_dep = self.criterion(intent_logits, dep_scores, labels, target_heads)
                
                total_loss += loss.item()
                _, predicted = intent_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        return val_acc, avg_loss
    
    def train(self, epochs=50):
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_acc, train_loss = self.train_epoch(epoch)
            val_acc, val_loss = self.validate()
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'nlp_advanced/best_advanced_nlu.pth')
                print(f"  ✅ Saved best model with val_acc={val_acc:.2f}%")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"🛑 Kích hoạt Early Stopping tại epoch {epoch+1} vì Val Acc ngừng tăng!")
                    break
            
            print("-" * 60)
        
        print(f"\n🎉 Training complete! Best validation accuracy: {self.best_val_acc:.2f}%")
        self.evaluate_model()
        self.plot_training_results()
        return self.model
    
    def evaluate_model(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device).long()
                labels = batch['labels'].to(self.device)
                intent_logits, _, _, _ = self.model(input_ids, return_parser=False)
                _, predicted = intent_logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        target_names = ["thong_tin_truong", "tuyen_sinh", "dao_tao", "nghien_cuu", "khac"]
        print("\n📊 BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT):")
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        
        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Intent Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('nlp_advanced/confusion_matrix.png', dpi=300)
        print("✅ Đã lưu ảnh Confusion Matrix tại nlp_advanced/confusion_matrix.png")

    def plot_training_results(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', marker='o')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', marker='x')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', marker='o')
        plt.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', marker='x')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('nlp_advanced/training_curves.png', dpi=300)
        print("✅ Đã lưu ảnh Training Curves tại nlp_advanced/training_curves.png")


# ==================== MAIN ====================
def main():
    # Kiểm tra device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Khởi tạo tokenizer
    print("🔧 Loading tokenizer...")
    tokenizer = SpikedNLPFree()
    
    # Load dataset
    print("📚 Loading datasets...")
    train_dataset = AdvancedNLUDataset(
        intent_path=TRAINING_DATA,
        dep_path=DEP_TRAINING_DATA,
        tokenizer=tokenizer
    )
    val_dataset = AdvancedNLUDataset(
        intent_path=os.path.join(os.path.dirname(TRAINING_DATA), 'training_data.json'),  # same for demo
        dep_path=DEP_TRAINING_DATA,
        tokenizer=tokenizer
    )
    
    # Fix size if there's issue with tensors
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Khởi tạo model
    print("🧠 Initializing Advanced NLU model...")
    model = AdvancedNLUProcessor(
        vocab_size=5000,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        T=T
    )
    
    # Đếm tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Huấn luyện
    trainer = AdvancedNLUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4, # Trải đều chậm lại vì chúng ta cho nó đi đoạn đường xa tới tận 200 vòng
        device=device
    )
    
    # Train nhanh rút gọn xuống 10 epoch để tiết kiệm thời gian
    trainer.train(epochs=50)
    
    print("\n✅ Training script completed!")


if __name__ == "__main__":
    main()
