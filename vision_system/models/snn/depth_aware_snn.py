import torch
import torch.nn as nn

class DepthAwareSNN(nn.Module):
    def __init__(self, num_classes=252):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 222 * 222, num_classes)
    
    def forward(self, x, depth=None, return_feats=False):
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        
        if return_feats:
            # Trả về dummy features (T, B, 64, 28, 28) để không làm lỗi lớp Fusion
            # x là (T, C, H, W) sau khi qua batch
            return torch.randn(t, b, 64, 28, 28).to(x.device)

        # Lấy frame đầu tiên để demo
        x_frame = x[:, 0]
        x_conv = torch.relu(self.conv(x_frame))
        x_flat = x_conv.view(x_conv.size(0), -1)
        logits = self.fc(x_flat)
        
        # Trả về bbox (dummy), logits, và state
        dummy_bbox = torch.zeros(b, 4)
        return dummy_bbox, logits, None
