import torch
import torch.nn as nn

class PoissonEncoder(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T = T
    def forward(self, x):
        # x: (B, C, H, W) giá trị [0,1]
        spikes = torch.zeros(self.T, *x.shape, device=x.device)
        for t in range(self.T):
            spikes[t] = torch.bernoulli(x)
        return spikes  # (T, B, C, H, W)

class TTFSEncoder(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T = T
    def forward(self, x):
        B,C,H,W = x.shape
        spikes = torch.zeros(self.T, B, C, H, W, device=x.device)
        # tf = floor(T*(1-x)) + 1,  nếu x=0 thì tf=T+1 (không phát)
        tf = torch.floor(self.T * (1 - x)).long() + 1
        mask = (tf <= self.T)
        for t in range(self.T):
            spikes[t][mask & (tf == t+1)] = 1.0
        return spikes

class DeltaEncoder(nn.Module):
    def __init__(self, T=20, theta=0.1):
        super().__init__()
        self.T = T
        self.theta = theta
    def forward(self, frames_seq):
        # frames_seq: (T, B, C, H, W)
        spikes = torch.zeros_like(frames_seq)
        for t in range(1, self.T):
            diff = torch.abs(frames_seq[t] - frames_seq[t-1])
            spikes[t] = (diff > self.theta).float()
        return spikes

class AdaptiveSpikeEncoder(nn.Module):
    def __init__(self, T=20, grad_thresh=0.2, theta=0.1):
        super().__init__()
        self.T = T
        self.grad_thresh = grad_thresh
        self.poisson = PoissonEncoder(T)
        self.ttfs = TTFSEncoder(T)
        self.delta = DeltaEncoder(T, theta)
    def forward(self, rgbd_seq):
        # rgbd_seq: (T, B, 4, H, W) đã chuẩn hóa [0,1]
        T,B,C,H,W = rgbd_seq.shape
        # Tính gradient thời gian trên kênh RGB trung bình
        intensity = rgbd_seq[:,:,:3,:,:].mean(dim=2)  # (T,B,H,W)
        grad = torch.abs(intensity[1:] - intensity[:-1])
        grad = torch.cat([torch.zeros_like(grad[:1]), grad], dim=0)  # (T,B,H,W)
        use_ttfs = (grad > self.grad_thresh).float()
        use_poisson = 1.0 - use_ttfs
        
        # Đơn giản: lặp frame
        poisson_spikes = torch.zeros(T,B,C,H,W, device=rgbd_seq.device)
        ttfs_spikes = torch.zeros(T,B,C,H,W, device=rgbd_seq.device)
        for t in range(T):
            frame = rgbd_seq[t]  # (B,C,H,W)
            poisson_spikes[t] = self.poisson(frame)[0]  # chỉ lấy t=0 của PoissonEncoder (do nó trả về T,B,...)
            ttfs_spikes[t] = self.ttfs(frame)[0]
        combined = use_ttfs.unsqueeze(2) * ttfs_spikes + use_poisson.unsqueeze(2) * poisson_spikes
        delta_spikes = self.delta(rgbd_seq)
        return combined + delta_spikes
