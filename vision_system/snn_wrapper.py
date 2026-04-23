import torch
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vision_system.models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
    from vision_system.encoding.spike_encoder import AdaptiveSpikeEncoder
except ImportError as e:
    # Fallback for relative imports if called from root
    from .models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
    from .encoding.spike_encoder import AdaptiveSpikeEncoder

class SNNVisionWrapper:
    def __init__(self, model_path='models/best_vision_snn.pth', T=8, input_size=128, num_classes=2):
        self.T = T
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ThreeDSpikingCNN(num_classes=num_classes).to(self.device)
        # Handle absolute path if needed
        if not os.path.isabs(model_path):
             model_path = os.path.join(os.getcwd(), model_path)
             
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.encoder = AdaptiveSpikeEncoder(T=T).to(self.device)
        
    def process_frame_sequence(self, rgbd_frames):
        """
        rgbd_frames: list of T frames, each frame is (H,W,4) RGB+D normalized [0,1]
        Returns: boxes, confidences, classes
        """
        assert len(rgbd_frames) == self.T, f"Need {self.T} frames, got {len(rgbd_frames)}"
        
        # Convert frames to tensors and stack
        # Expected shape for encoder: (T, B, C, H, W)
        tensors = []
        for f in rgbd_frames:
            t = torch.from_numpy(f).permute(2, 0, 1).float() # (C, H, W)
            tensors.append(t)
        
        tensor_frames = torch.stack(tensors) # (T, C, H, W)
        tensor_frames = tensor_frames.unsqueeze(1).to(self.device) # (T, 1, 4, H, W)
        
        # Resize to input_size
        # interpolate expects (B, C, H, W)
        T_val, B_val, C_val, H_val, W_val = tensor_frames.shape
        flat_frames = tensor_frames.view(-1, C_val, H_val, W_val) # (T*B, C, H, W)
        resized_frames = torch.nn.functional.interpolate(
            flat_frames,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False
        )
        tensor_frames = resized_frames.view(T_val, B_val, C_val, self.input_size, self.input_size)
        
        # Encode to spikes
        spikes = self.encoder(tensor_frames)
        
        # Inference
        with torch.no_grad():
            boxes, conf, cls = self.model(spikes)
        
        return boxes.cpu().numpy(), conf.cpu().numpy(), cls.cpu().numpy()
    
    def get_fps_estimate(self):
        # Base estimate from benchmark T=8, size=128
        return 1000 / 15.12 # ms
