"""
Vision Fine-tuning Script
==========================
Fine-tune model SNN đã train bằng dữ liệu thực tế (Real Data).
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add root to sys.path
root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

from vision_system.training.train_vision_snn import train_snn_vision

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SNN Vision with Real Data")
    parser.add_argument("--real-data", type=str, default="data/real_vision_augmented.json",
                        help="Đường dẫn tới dữ liệu thực tế (đã augment)")
    parser.add_argument("--pretrained", type=str, default="models/best_vision_snn.pth",
                        help="Weights đã train trên synthetic data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate nhỏ để fine-tune")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="models/finetuned_vision")
    
    args = parser.parse_args()

    # Reuse logic từ train_vision_snn.py
    # Chúng ta set --synthetic-samples 100 để giữ một ít synthetic data (tránh catastrophic forgetting)
    
    # Giả lập namespace args cho hàm train_snn_vision
    class Args:
        pass
    
    train_args = Args()
    train_args.epochs = args.epochs
    train_args.batch_size = args.batch_size
    train_args.lr = args.lr
    train_args.T = 8
    train_args.input_size = 128
    train_args.synthetic_samples = 200 # Giữ lại 200 mẫu giả lập để bảo toàn kiến thức nền
    train_args.real_data = args.real_data
    train_args.pretrained = args.pretrained
    train_args.save_dir = args.save_dir

    print("🚀 Bắt đầu Fine-tuning hệ thống Vision...")
    print(f"📍 Base model: {args.pretrained}")
    print(f"📍 Real data: {args.real_data}")
    
    history = train_snn_vision(train_args)
    
    print("\n✅ Fine-tuning hoàn tất!")
    print(f"📦 Model tốt nhất lưu tại: {args.save_dir}/best_vision_snn.pth")

if __name__ == "__main__":
    main()
