import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer

class SpikingMotionDetector(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # Simple Spiking SNN for motion/feature detection
        self.conv = nn.Sequential(
            layer.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(16),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid())
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            layer.Linear(16 * 64 * 64, 2), # Example size
            neuron.IFNode(surrogate_function=surrogate.Sigmoid())
        )

    def forward(self, x):
        # x shape: [T, C, H, W] for time-steps
        x = self.conv(x)
        return x

def create_model():
    return SpikingMotionDetector()
