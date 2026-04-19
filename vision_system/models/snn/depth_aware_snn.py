import torch
import torch.nn as nn

class DepthAwareSNN(nn.Module):
    def __init__(self, num_classes=252):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 222 * 222, num_classes)
    
    def forward(self, x, depth=None):
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        # Lấy frame đầu tiên để demo
        x = x[:, 0]
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        # Trả về bbox (dummy), logits, và state
        dummy_bbox = torch.zeros(b, 4)
        return dummy_bbox, logits, None
