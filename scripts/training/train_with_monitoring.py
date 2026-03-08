#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Training Script với Monitoring và Visualization

Tính năng:
- Real-time monitoring
- Automatic checkpoint management
- Training visualization
- Early stopping với best model tracking
- Export training results

Usage:
    python scripts/training/train_with_monitoring.py \
        --config configs/training/default.yaml \
        --output-dir experiments/training/run_001
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.training.depth_aware_train import (
    DepthAwareLoss,
    train_epoch,
    validate_epoch,
    save_checkpoint,
    load_checkpoint,
)
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN
from vision_system.data.rgbd_sequence_with_gt_dataset import (
    RGBDSequenceWithGTDataset,
    collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced Training với Monitoring"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file (JSON/YAML)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--monitor-metrics",
        nargs="+",
        default=["val_loss", "val_accuracy"],
        help="Metrics to monitor",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best model based on monitored metrics",
    )
    return parser.parse_args()


class TrainingMonitor:
    """Monitor training progress và export results."""
    
    def __init__(self, output_dir: Path, monitor_metrics: list):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metrics = monitor_metrics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "epochs": [],
        }
        self.best_metrics = {}
        self.best_epoch = 0
    
    def update(self, epoch: int, metrics: dict):
        """Update training history."""
        self.history["epochs"].append(epoch)
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # Track best metrics
        for metric in self.monitor_metrics:
            if metric in metrics:
                if metric not in self.best_metrics:
                    self.best_metrics[metric] = metrics[metric]
                    self.best_epoch = epoch
                else:
                    # For loss, lower is better; for accuracy, higher is better
                    if "loss" in metric.lower():
                        if metrics[metric] < self.best_metrics[metric]:
                            self.best_metrics[metric] = metrics[metric]
                            self.best_epoch = epoch
                    else:
                        if metrics[metric] > self.best_metrics[metric]:
                            self.best_metrics[metric] = metrics[metric]
                            self.best_epoch = epoch
    
    def save_history(self):
        """Save training history to JSON."""
        history_file = self.output_dir / "training_history.json"
        with history_file.open("w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Saved training history to {history_file}")
    
    def save_summary(self):
        """Save training summary."""
        summary = {
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "total_epochs": len(self.history["epochs"]),
            "final_metrics": {
                k: v[-1] if v else None
                for k, v in self.history.items()
                if k != "epochs"
            },
        }
        
        summary_file = self.output_dir / "training_summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved training summary to {summary_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Best epoch: {summary['best_epoch']}")
        print("Best metrics:")
        for metric, value in summary['best_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("="*70)


def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"experiments/training/run_{timestamp}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Enhanced Training với Monitoring")
    print("="*70)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # TensorBoard
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"📝 TensorBoard logs: {log_dir}")
    
    # Monitor
    monitor = TrainingMonitor(args.output_dir, args.monitor_metrics)
    
    # Dataset paths (fixed by design for this project)
    rgb_dir = Path("data/01_interim/vision/aligned/rgb")
    depth_dir = Path("data/01_interim/vision/aligned/depth")
    ann_file = Path("data/02_processed/vision/coco_format/hilo_annotations_3d.json")

    print("\n📂 Checking dataset paths...")
    missing = []
    if not rgb_dir.exists():
        missing.append(f"- RGB dir missing: {rgb_dir}")
    if not depth_dir.exists():
        missing.append(f"- Depth dir missing: {depth_dir}")
    if not ann_file.exists():
        missing.append(f"- Annotation JSON missing: {ann_file}")

    if missing:
        print("❌ Dataset paths are not valid. Training aborted.")
        for m in missing:
            print("   " + m)
        print("\n💡 HINT: Kiểm tra lại cấu trúc dữ liệu hoặc chạy lại pipeline HILO để sinh:")
        print("   data/01_interim/vision/aligned/{rgb,depth} và")
        print("   data/02_processed/vision/coco_format/hilo_annotations_3d.json")
        raise SystemExit(1)

    print("\n📂 Loading dataset...")
    dataset = RGBDSequenceWithGTDataset(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        ann_file=ann_file,
        img_size=224,
    )

    if len(dataset) == 0:
        print("❌ Loaded dataset nhưng không có sample nào (len(dataset) == 0). Training aborted.")
        print("   Hãy kiểm tra lại file annotation và thư mục ảnh RGB/Depth.")
        raise SystemExit(1)

    print(f"✅ Dataset size: {len(dataset)} frames")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Train: {train_size} | Val: {val_size}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Model
    print("\n🤖 Initializing model...")
    model = DepthAwareSNN(num_classes=237).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = DepthAwareLoss()
    
    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and args.resume.exists():
        print(f"📂 Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
    
    # Training loop
    print("\n🎯 BẮT ĐẦU TRAINING...")
    print("="*70)
    
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n📌 Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss, train_cls, train_bbox = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            args.log_interval if hasattr(args, 'log_interval') else 10,
            False, None, 1.0, None
        )
        
        # Validate
        val_loss, val_accuracy, val_bbox_mse = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update monitor
        monitor.update(epoch, {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            checkpoint_dir, is_best=is_best
        )
        
        # Print summary
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    # Save history and summary
    monitor.save_history()
    monitor.save_summary()
    
    print("\n✅ TRAINING HOÀN TẤT!")
    print(f"📁 Results: {args.output_dir}")
    print(f"📝 TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
