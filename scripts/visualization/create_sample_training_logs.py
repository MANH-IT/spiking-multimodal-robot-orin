#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo dữ liệu training logs mẫu để test visualization.

Tạo các file JSON mô phỏng TensorBoard logs cho testing.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create sample training logs for visualization testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/logs/vision_training_logs/sample"),
        help="Output directory for sample logs",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    return parser.parse_args()


def generate_training_curves(num_epochs: int) -> dict:
    """Generate sample training curves."""
    epochs = list(range(1, num_epochs + 1))
    
    # Simulate training loss (decreasing with noise)
    train_loss = []
    val_loss = []
    train_cls_loss = []
    val_cls_loss = []
    train_bbox_loss = []
    val_bbox_loss = []
    val_accuracy = []
    
    base_train_loss = 2.5
    base_val_loss = 2.6
    
    for epoch in epochs:
        # Training loss decreases with some noise
        noise = np.random.normal(0, 0.05)
        decay = np.exp(-epoch / 30)
        train_loss.append((epoch, base_train_loss * decay + noise))
        
        # Validation loss similar but slightly higher
        val_noise = np.random.normal(0, 0.08)
        val_loss.append((epoch, base_val_loss * decay + val_noise))
        
        # Classification loss (part of total)
        train_cls_loss.append((epoch, (base_train_loss * 0.6) * decay + noise * 0.6))
        val_cls_loss.append((epoch, (base_val_loss * 0.6) * decay + val_noise * 0.6))
        
        # BBox loss (part of total)
        train_bbox_loss.append((epoch, (base_train_loss * 0.4) * decay + noise * 0.4))
        val_bbox_loss.append((epoch, (base_val_loss * 0.4) * decay + val_noise * 0.4))
        
        # Accuracy increases
        base_acc = 0.3
        max_acc = 0.85
        acc = base_acc + (max_acc - base_acc) * (1 - np.exp(-epoch / 25)) + np.random.normal(0, 0.02)
        acc = np.clip(acc, 0, 1)
        val_accuracy.append((epoch, acc))
    
    return {
        "Train/Epoch_Loss": train_loss,
        "Val/Loss": val_loss,
        "Train/Epoch_CLS_Loss": train_cls_loss,
        "Val/CLS_Loss": val_cls_loss,
        "Train/Epoch_BBox_Loss": train_bbox_loss,
        "Val/BBox_Loss": val_bbox_loss,
        "Val/Accuracy": val_accuracy,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample training logs for {args.num_epochs} epochs...")
    
    # Generate training curves
    curves = generate_training_curves(args.num_epochs)
    
    # Save as JSON (simulating TensorBoard format)
    output_file = args.output_dir / "training_curves.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    
    print(f"✅ Created sample training logs: {output_file}")
    print(f"   Metrics: {list(curves.keys())}")
    print(f"   Epochs: 1 to {args.num_epochs}")


if __name__ == "__main__":
    main()
