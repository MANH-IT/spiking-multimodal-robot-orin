# scripts/finetune_vision_snn.py
import os
import sys
from pathlib import Path
import torch

# Add root to sys.path
root = Path(__file__).parent.parent.absolute()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from vision_system.training.train_vision_snn import train_snn_vision

def run_finetuning():
    print("🚀 Starting SNN Fine-tuning with YOLO Teacher Data...")
    
    # Cấu hình tối ưu cho Fine-tuning
    args = type('Args', (), {
        'epochs': 30,
        'batch_size': 4,      # Giảm batch size để tránh quá nhiệt GPU
        'lr': 0.0001,         # Learning rate nhỏ để giữ lại kiến thức cũ
        'T': 8,
        'input_size': 128,
        'synthetic_samples': 200, # Vẫn dùng một ít synthetic để tránh overfit
        'real_data': 'data/real_vision.json',
        'pretrained': 'models/best_vision_snn.pth', # Load weights hiện tại
        'save_dir': 'models/vision_finetuned'
    })()

    if not os.path.exists(args.real_data):
        print(f"❌ Error: {args.real_data} not found. Please run yolo_to_snn_collector.py first.")
        return

    # Chạy training
    history = train_snn_vision(args)
    
    # Sau khi xong, copy ra models chính
    import shutil
    best_path = Path(args.save_dir) / "best_vision_snn.pth"
    if best_path.exists():
        target = Path("models/best_vision_snn.pth")
        shutil.copy(best_path, target)
        print(f"✅ Fine-tuning complete! Best model updated at {target}")
    else:
        print("⚠️ Warning: Could not find best model after training.")

if __name__ == "__main__":
    run_finetuning()
