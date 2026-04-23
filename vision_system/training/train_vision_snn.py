import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vision_system.models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
from vision_system.dataset.synthetic_dataset import SyntheticDynamicDataset
from vision_system.encoding.spike_encoder import DeltaEncoder

def train_snn_vision():
    # 1. Cấu hình
    # CHÚ Ý: Chuyển sang "cpu" để đảm bảo ổn định tối đa trong quá trình demo
    device = torch.device("cpu")
    T = 20
    batch_size = 2 # Giảm batch size để ổn định hơn
    epochs = 5
    lr = 1e-3
    
    print(f"🚀 Training SNN Vision on {device}")
    
    # 2. Dataset & Loader
    dataset = SyntheticDynamicDataset(num_samples=200, T=T)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model & Encoder
    model = ThreeDSpikingCNN(num_classes=2, T=T).to(device)
    encoder = DeltaEncoder(T=T, theta=0.1).to(device)
    
    # Ép kiểu dữ liệu đồng nhất
    model.float()
    encoder.float()
    
    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (sequences, labels) in enumerate(train_loader):
            # sequences: (B, T, 4, 224, 224) -> (T, B, 4, 224, 224)
            sequences = sequences.permute(1, 0, 2, 3, 4).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Encoder: (T, B, 4, 224, 224) -> Spikes (T, B, 4, 224, 224)
            # DeltaEncoder bắt chuyển động
            spikes = encoder(sequences)
            
            # Forward
            pred_boxes, pred_conf, pred_cls = model(spikes)
            
            # Tính Loss (Sơ bộ: chỉ tính cho box và class tại frame cuối)
            # labels: [x1, y1, x2, y2, class_id]
            target_boxes = labels[:, :4]
            target_cls = labels[:, 4].long()
            
            # Lấy prediction có confidence cao nhất (giả định SNN trả về 1 box cho đơn giản hóa)
            # model trả về (B, Hf*Wf, 4), ta lấy trung bình các box có conf > 0.5 hoặc chỉ lấy box đầu nếu chưa train
            # Trong SNN detector này, ta cần map loss cực kỳ cẩn thận.
            # Ở đây ta sử dụng MSE trực tiếp cho output của box_head (đã flattened)
            
            loss_box = mse_loss(pred_boxes.mean(dim=1), target_boxes)
            loss_cls = ce_loss(pred_cls.mean(dim=1), target_cls)
            
            loss = loss_box + 0.5 * loss_cls
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f} (Box: {loss_box.item():.4f}, Cls: {loss_cls.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")
        
    # 5. Lưu model
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), 'models/best_vision_snn.pth')
    print("✅ Model saved to models/best_vision_snn.pth")

if __name__ == "__main__":
    train_snn_vision()
