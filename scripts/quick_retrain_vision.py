# scripts/quick_retrain_vision.py
"""
Retrain SNN Vision với synthetic data - 50 epochs
Chạy ngay để có weights mới tương thích
"""

import torch
import sys
import os
from pathlib import Path
from types import SimpleNamespace

# Add root to sys.path
root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

def retrain_vision():
    print("🚀 Starting Vision SNN Retraining...")
    print("="*50)
    
    from vision_system.training.train_vision_snn import train_snn_vision
    
    config = {
        'epochs': 50,
        'batch_size': 4, # Giảm để tránh quá nhiệt
        'lr': 0.001,
        'T': 8,
        'input_size': 128,
        'synthetic_samples': 500, # Chạy nhanh hơn mỗi epoch
        'real_data': None,
        'pretrained': None, 
        'save_dir': 'models/vision'
    }
    
    args = SimpleNamespace(**config)
    
    print(f"\n📊 Config:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    cuda_available = torch.cuda.is_available()
    print(f"\n🖥️ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Train
    history = train_snn_vision(args)
    
    print("\n✅ Retraining complete!")
    print(f"   Best model saved to: {config['save_dir']}/best_vision_snn.pth")
    print(f"   (Also copied to models/best_vision_snn.pth)")
    
    return history

if __name__ == "__main__":
    retrain_vision()
