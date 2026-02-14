#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: depth_aware_train_ann.py
Mục đích: Training DepthAwareSNN với dữ liệu HILO đã có annotation.

Mô tả:
- Đọc dữ liệu từ file COCO JSON (hilo_annotations_3d.json)
- Sử dụng RGBDSequenceWithGTDataset để load ảnh RGB-D và ground truth
- Loss kết hợp: 
    * Classification loss (CrossEntropy) cho object class
    * Regression loss (Smooth L1) cho bounding box 3D (cx, cy, cz, w, h, d)
- Hỗ trợ TensorBoard để theo dõi quá trình training

Yêu cầu:
- Đã chạy hilo_make_annotations.py thành công
- File JSON tồn tại trong data/02_processed/vision/coco_format/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time
from typing import List, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các module cần thiết
from vision_system.data.rgbd_sequence_with_gt_dataset import RGBDSequenceWithGTDataset
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DepthAwareSNN with HILO annotations"
    )
    
    # Data paths
    parser.add_argument(
        "--ann-path",
        type=Path,
        default=Path("data") / "02_processed" / "vision" / "coco_format" / "hilo_annotations_3d.json",
        help="Đường dẫn tới file annotation JSON (COCO format)."
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path("data") / "01_interim" / "vision" / "aligned" / "rgb",
        help="Thư mục chứa ảnh RGB aligned."
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("data") / "01_interim" / "vision" / "aligned" / "depth",
        help="Thư mục chứa ảnh depth aligned."
    )
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=50, help="Số epochs")
    parser.add_argument("--num-classes", type=int, default=237, help="Số classes (từ HILO)")
    parser.add_argument("--img-size", type=int, default=224, help="Kích thước ảnh đầu vào")
    
    # Model config
    parser.add_argument("--use-multiscale", action="store_true", help="Sử dụng multi-scale feature extraction")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension của model")
    
    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("vision_system") / "weights" / "finetuned",
                        help="Thư mục lưu checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để tiếp tục training")
    
    # Logging
    parser.add_argument("--log-dir", type=Path, default=Path("experiments") / "logs" / "vision_training_logs",
                        help="Thư mục lưu tensorboard logs")
    parser.add_argument("--log-interval", type=int, default=10, help="Số steps giữa các lần log")
    
    return parser.parse_args()


def collate_fn(batch):
    """
    Custom collate function để xử lý batch có targets với độ dài khác nhau.
    
    Args:
        batch: List of (rgb, depth, target)
        
    Returns:
        rgb: (B, T, C, H, W)
        depth: (B, T, 1, H, W)
        targets: List of dicts
    """
    rgbs = []
    depths = []
    targets = []
    
    for rgb, depth, target in batch:
        rgbs.append(rgb)  # rgb đã có shape (1, 3, H, W)
        depths.append(depth)  # depth đã có shape (1, 1, H, W)
        targets.append(target)
    
    return torch.stack(rgbs), torch.stack(depths), targets


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Xây dựng dataloaders cho train và validation.
    
    Returns:
        train_loader, val_loader, dataset_info
    """
    # Tạo full dataset
    full_dataset = RGBDSequenceWithGTDataset(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        ann_file=args.ann_path,
        img_size=args.img_size,
        sequence_length=1
    )
    
    # Lấy thông tin categories
    categories = full_dataset.categories
    num_classes = len(categories)
    print(f"📊 Dataset có {num_classes} classes")
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Train: {train_size} samples, Val: {val_size} samples")
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows cần set 0
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    dataset_info = {
        'num_classes': num_classes,
        'categories': categories,
        'train_size': train_size,
        'val_size': val_size
    }
    
    return train_loader, val_loader, dataset_info


class DepthAwareLoss(nn.Module):
    """
    Loss function cho DepthAwareSNN:
    - Classification loss (CrossEntropy) cho object class
    - Regression loss (Smooth L1) cho bounding box 3D
    """
    def __init__(self, cls_weight: float = 1.0, bbox_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, bbox_pred: torch.Tensor, cls_pred: torch.Tensor, 
                targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            bbox_pred: (B, N, 6) - predicted 3D bboxes
            cls_pred: (B, N, num_classes) - predicted class logits
            targets: List of dicts with 'boxes_3d' and 'labels'
            
        Returns:
            Dict with losses
        """
        batch_size = len(targets)
        device = bbox_pred.device
        
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        total_objects = 0
        
        for i in range(batch_size):
            target = targets[i]
            num_gt = target['boxes_3d'].shape[0]
            
            if num_gt == 0:
                # Không có object trong ảnh
                continue
            
            # Lấy predictions cho object đầu tiên (giả sử mỗi ảnh chỉ có 1 object chính)
            # TODO: Xử lý multiple objects per image
            bbox_gt = target['boxes_3d'][0:1].to(device)  # (1, 6)
            label_gt = target['labels'][0:1].to(device)   # (1,)
            
            # Lấy predictions tương ứng
            bbox_i = bbox_pred[i:i+1, 0:1, :]  # (1, 1, 6)
            cls_i = cls_pred[i:i+1, 0:1, :]    # (1, 1, num_classes)
            
            # Tính losses
            cls_loss_i = self.cls_loss(cls_i.squeeze(1), label_gt.squeeze(0))
            bbox_loss_i = self.bbox_loss(bbox_i.squeeze(1), bbox_gt).mean()
            
            total_cls_loss += cls_loss_i
            total_bbox_loss += bbox_loss_i
            total_objects += 1
        
        if total_objects > 0:
            avg_cls_loss = total_cls_loss / total_objects
            avg_bbox_loss = total_bbox_loss / total_objects
            total_loss = self.cls_weight * avg_cls_loss + self.bbox_weight * avg_bbox_loss
        else:
            avg_cls_loss = torch.tensor(0.0, device=device)
            avg_bbox_loss = torch.tensor(0.0, device=device)
            total_loss = torch.tensor(0.0, device=device)
        
        return {
            'total_loss': total_loss,
            'cls_loss': avg_cls_loss,
            'bbox_loss': avg_bbox_loss,
            'num_objects': total_objects
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DepthAwareLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_interval: int = 10
) -> float:
    """
    Train một epoch.
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    num_batches = 0
    
    for batch_idx, (rgb, depth, targets) in enumerate(loader):
        rgb = rgb.to(device)      # (B, 1, 3, H, W)
        depth = depth.to(device)  # (B, 1, 1, H, W)
        
        # Forward pass
        bbox_pred, cls_pred = model(rgb, depth)
        
        # Tính loss
        losses = criterion(bbox_pred, cls_pred, targets)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total_loss'].item()
        total_cls_loss += losses['cls_loss'].item()
        total_bbox_loss += losses['bbox_loss'].item()
        num_batches += 1
        
        # Logging
        if batch_idx % log_interval == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Train/Total_Loss', losses['total_loss'].item(), global_step)
            writer.add_scalar('Train/CLS_Loss', losses['cls_loss'].item(), global_step)
            writer.add_scalar('Train/BBox_Loss', losses['bbox_loss'].item(), global_step)
            writer.add_scalar('Train/Num_Objects', losses['num_objects'], global_step)
            
            print(f"Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                  f"Loss: {losses['total_loss'].item():.4f} | "
                  f"CLS: {losses['cls_loss'].item():.4f} | "
                  f"BBox: {losses['bbox_loss'].item():.4f} | "
                  f"Objects: {losses['num_objects']}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_cls = total_cls_loss / num_batches if num_batches > 0 else 0
    avg_bbox = total_bbox_loss / num_batches if num_batches > 0 else 0
    
    writer.add_scalar('Train/Epoch_Total_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Epoch_CLS_Loss', avg_cls, epoch)
    writer.add_scalar('Train/Epoch_BBox_Loss', avg_bbox, epoch)
    
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: DepthAwareLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """
    Validation.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_objects = 0
    num_batches = 0
    
    for rgb, depth, targets in loader:
        rgb = rgb.to(device)
        depth = depth.to(device)
        
        # Forward pass
        bbox_pred, cls_pred = model(rgb, depth)
        
        # Tính loss
        losses = criterion(bbox_pred, cls_pred, targets)
        
        total_loss += losses['total_loss'].item()
        num_batches += 1
        
        # Tính accuracy
        for i, target in enumerate(targets):
            if target['labels'].shape[0] > 0:
                label_gt = target['labels'][0].item()
                pred_label = cls_pred[i, 0].argmax().item()
                if pred_label == label_gt:
                    total_correct += 1
                total_objects += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = total_correct / total_objects if total_objects > 0 else 0
    
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', accuracy, epoch)
    
    print(f"🔍 Validation | Epoch {epoch:3d} | "
          f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} ({total_correct}/{total_objects})")
    
    return avg_loss


def main() -> None:
    """Hàm chính."""
    args = parse_args()
    
    # Tạo thư mục checkpoint
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # TensorBoard
    log_dir = args.log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"📝 TensorBoard logs: {log_dir}")
    
    # Dataloaders
    print("\n📂 Đang load dữ liệu...")
    train_loader, val_loader, dataset_info = build_dataloaders(args)
    
    # Model
    print("\n🤖 Đang khởi tạo model...")
    model = DepthAwareSNN(
        num_classes=dataset_info['num_classes'],
        use_multiscale=args.use_multiscale
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = DepthAwareLoss(cls_weight=1.0, bbox_weight=1.0)
    
    # Resume từ checkpoint
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"📂 Tiếp tục từ checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"   Bắt đầu từ epoch {start_epoch}")
    
    # Training loop
    print("\n🎯 BẮT ĐẦU TRAINING...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n📌 Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, args.log_interval
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = args.checkpoint_dir / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, checkpoint_path)
            print(f"💾 Đã lưu best model: {checkpoint_path}")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = args.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"💾 Đã lưu checkpoint: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING HOÀN TẤT!")
    print("=" * 70)
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📁 Checkpoints: {args.checkpoint_dir}")
    print(f"📝 TensorBoard logs: {log_dir}")
    print("\nĐể xem tensorboard, chạy:")
    print(f"tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()