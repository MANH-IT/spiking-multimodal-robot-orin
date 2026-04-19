import torch
import torch.nn as nn
from vision_system.models.snn.lif_neuron import ParametricLIF

class SpikingConv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, pool_size=(1,2,2)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.lif = ParametricLIF()
        self.pool = nn.MaxPool3d(pool_size) if pool_size else nn.Identity()
    def forward(self, x, mem=None):
        # x: (T, B, C, H, W)
        T,B,C,H,W = x.shape
        if mem is None:
            mem = self.lif.init_mem(B, self.conv.out_channels, x.device)
        outputs = []
        for t in range(T):
            cur = self.conv(x[t].unsqueeze(0))  # (1,B,out_ch,H,W)
            cur = self.bn(cur)
            spk, mem = self.lif(cur, mem)
            spk = self.pool(spk)
            outputs.append(spk)
        return torch.stack(outputs, dim=0), mem

class ThreeDSpikingCNN(nn.Module):
    def __init__(self, num_classes=4, T=20, anchor_size=(80,80)):
        super().__init__()
        self.T = T
        self.anchor_w, self.anchor_h = anchor_size
        self.conv1 = SpikingConv3DBlock(4, 16, pool_size=(1,2,2))
        self.conv2 = SpikingConv3DBlock(16, 32, pool_size=(1,2,2))
        self.conv3 = SpikingConv3DBlock(32, 64, pool_size=(1,2,2))
        self.box_head = nn.Conv2d(64, 4, kernel_size=1)
        self.conf_head = nn.Conv2d(64, 1, kernel_size=1)
        self.class_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.lif_out = ParametricLIF()
    def forward(self, x):
        # x: (T, B, 4, 224, 224)
        spk1, _ = self.conv1(x)
        spk2, _ = self.conv2(spk1)
        spk3, _ = self.conv3(spk2)  # (T, B, 64, 28, 28)
        feat = spk3.mean(dim=0)      # (B,64,28,28)
        box = self.box_head(feat)
        conf = self.conf_head(feat)
        cls = self.class_head(feat)
        # Giải mã bounding box
        B,_,Hf,Wf = box.shape
        box = box.permute(0,2,3,1).reshape(B, Hf*Wf, 4)
        conf = torch.sigmoid(conf.permute(0,2,3,1).reshape(B, Hf*Wf, 1))
        cls = torch.softmax(cls.permute(0,2,3,1).reshape(B, Hf*Wf, -1), dim=-1)
        # Tính tọa độ thực (normalized)
        grid_y, grid_x = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing='ij')
        grid_x = grid_x.float().to(x.device) / Wf
        grid_y = grid_y.float().to(x.device) / Hf
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1,2)
        tx, ty, tw, th = box[...,0], box[...,1], box[...,2], box[...,3]
        cx = grid[None,:,0] + torch.sigmoid(tx)
        cy = grid[None,:,1] + torch.sigmoid(ty)
        w = torch.exp(tw) * self.anchor_w / 224.0
        h = torch.exp(th) * self.anchor_h / 224.0
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes, conf, cls
