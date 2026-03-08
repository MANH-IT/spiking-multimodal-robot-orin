# depth_aware_train.py
# Training DepthAwareSNN với HILO RGB-D dataset - NCKH 2026

import json
import sys
import time
from pathlib import Path
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.data.rgbd_sequence_with_gt_dataset import (
    HILORawDataset, RGBDSequenceWithGTDataset,
    collate_fn, get_hilo_dataset, DEFAULT_HILO_ROOT
)
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass


# ---------------------- CONFIG ----------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DepthAwareSNN với HILO RGB-D dataset - NCKH 2026"
    )
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num-epochs",  type=int,   default=50)
    parser.add_argument("--img-size",    type=int,   default=224)
    parser.add_argument("--seq-len",     type=int,   default=4,
                        help="Số frame liên tiếp / sample")
    parser.add_argument("--max-scenes",  type=int,   default=None,
                        help="Giới hạn số HILO scenes (None=tất cả)")
    parser.add_argument("--arc-ids",     type=int,   nargs='+',
                        default=[0,1,2,3],
                        help="Arc IDs để dùng (0-7)")
    parser.add_argument("--use-snn",     action="store_true", default=True,
                        help="Dùng snntorch LIF neurons")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("vision_system/weights/finetuned"))
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--grad-clip",   type=float, default=1.0)
    parser.add_argument("--use-amp",     action="store_true")
    parser.add_argument("--log-interval",type=int,   default=10)
    # Augmentation flags (fixed: was missing but referenced in code)
    parser.add_argument("--use-augmentation",  action="store_true", default=False)
    parser.add_argument("--aug-color-jitter",  action="store_true", default=True)
    parser.add_argument("--aug-geometric",     action="store_true", default=True)
    parser.add_argument("--aug-depth-noise",   action="store_true", default=True)
    # History export for NCKH plots
    parser.add_argument("--history-out", type=str,
                        default="experiments/vision_training/history.json",
                        help="Path để save training history JSON")
    return parser.parse_args()


# ---------------------- LOSS ----------------------

class DepthAwareLoss(nn.Module):
    def __init__(self, cls_weight: float = 1.0, bbox_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.SmoothL1Loss()

    def forward(self, bbox_pred, cls_pred, targets):
        device = bbox_pred.device
        total_cls = 0.0
        total_box = 0.0
        valid = 0

        for i in range(len(targets)):
            if targets[i]["boxes_3d"].shape[0] == 0:
                continue

            gt_box = targets[i]["boxes_3d"][0].to(device)
            gt_label = targets[i]["labels"][0].to(device)

            total_cls += self.cls_loss(
                cls_pred[i].unsqueeze(0), gt_label.unsqueeze(0)
            )
            total_box += self.box_loss(bbox_pred[i], gt_box)
            valid += 1

        if valid == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {"total": zero, "cls": zero, "bbox": zero, "valid": 0}

        total = self.cls_weight * (total_cls / valid) + self.bbox_weight * (total_box / valid)
        return {
            "total": total,
            "cls": total_cls / valid,
            "bbox": total_box / valid,
            "valid": valid
        }


# ---------------------- METRICS ----------------------

def compute_metrics(bbox_pred, cls_pred, targets, device):
    """Tính toán các metrics: accuracy, bbox mse"""
    total_correct = 0
    total_objects = 0
    total_bbox_error = 0.0
    
    for i in range(len(targets)):
        if targets[i]["boxes_3d"].shape[0] == 0:
            continue
        
        gt_label = targets[i]["labels"][0].item()
        pred_label = cls_pred[i].argmax().item()
        
        if pred_label == gt_label:
            total_correct += 1
        total_objects += 1
        
        # Bbox error (MSE)
        gt_box = targets[i]["boxes_3d"][0].to(device)
        bbox_error = torch.nn.functional.mse_loss(bbox_pred[i], gt_box).item()
        total_bbox_error += bbox_error
    
    accuracy = total_correct / total_objects if total_objects > 0 else 0.0
    avg_bbox_error = total_bbox_error / total_objects if total_objects > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "bbox_mse": avg_bbox_error,
        "num_objects": total_objects
    }


# ---------------------- TRAIN LOOP ----------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer,
                log_interval, use_amp, scaler, grad_clip):
    """Train một epoch — trả về (avg_loss, avg_cls, avg_bbox, avg_spike)"""
    model.train()
    model.reset_spike_stats()
    total_loss = total_cls_loss = total_bbox_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=90)

    for batch_idx, (rgb, depth, targets) in enumerate(pbar):
        rgb   = rgb.to(device)
        depth = depth.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                bbox_pred, cls_pred = model(rgb, depth)
                losses = criterion(bbox_pred, cls_pred, targets)
            scaler.scale(losses["total"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            bbox_pred, cls_pred = model(rgb, depth)
            losses = criterion(bbox_pred, cls_pred, targets)
            losses["total"].backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss     += losses["total"].item()
        total_cls_loss += losses["cls"].item()
        total_bbox_loss+= losses["bbox"].item()

        pbar.set_postfix(
            loss=f'{losses["total"].item():.4f}',
            cls=f'{losses["cls"].item():.4f}',
            spike=f'{model.get_avg_spike_rate():.3f}',
        )

        if batch_idx % log_interval == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar("Train/Loss",      losses["total"].item(), step)
            writer.add_scalar("Train/CLS_Loss",  losses["cls"].item(),   step)
            writer.add_scalar("Train/BBox_Loss", losses["bbox"].item(),  step)
            writer.add_scalar("Train/Spike_Rate",model.get_avg_spike_rate(), step)

    n = max(len(loader), 1)
    return total_loss/n, total_cls_loss/n, total_bbox_loss/n, model.get_avg_spike_rate()


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch, writer):
    """Validate một epoch"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    all_metrics = {"accuracy": 0.0, "bbox_mse": 0.0, "num_objects": 0}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    
    for rgb, depth, targets in pbar:
        rgb = rgb.to(device)
        depth = depth.to(device)
        
        bbox_pred, cls_pred = model(rgb, depth)
        losses = criterion(bbox_pred, cls_pred, targets)
        
        total_loss += losses["total"].item()
        total_cls_loss += losses["cls"].item()
        total_bbox_loss += losses["bbox"].item()
        
        # Compute metrics
        metrics = compute_metrics(bbox_pred, cls_pred, targets, device)
        # Accumulate weighted by number of objects
        all_metrics["accuracy"] += metrics["accuracy"] * metrics["num_objects"]
        all_metrics["bbox_mse"] += metrics["bbox_mse"] * metrics["num_objects"]
        all_metrics["num_objects"] += metrics["num_objects"]
        
        pbar.set_postfix({
            'loss': f'{losses["total"].item():.4f}',
            'acc': f'{metrics["accuracy"]:.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_cls = total_cls_loss / len(loader)
    avg_bbox = total_bbox_loss / len(loader)
    
    if all_metrics["num_objects"] > 0:
        avg_accuracy = all_metrics["accuracy"] / all_metrics["num_objects"]
        avg_bbox_mse = all_metrics["bbox_mse"] / all_metrics["num_objects"]
    else:
        avg_accuracy = 0.0
        avg_bbox_mse = 0.0
    
    # Log to tensorboard
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/CLS_Loss", avg_cls, epoch)
    writer.add_scalar("Val/BBox_Loss", avg_bbox, epoch)
    writer.add_scalar("Val/Accuracy", avg_accuracy, epoch)
    writer.add_scalar("Val/BBox_MSE", avg_bbox_mse, epoch)
    
    return avg_loss, avg_accuracy, avg_bbox_mse


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_dir, 
                   is_best=False, filename=None):
    """Lưu checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    
    if filename is None:
        filename = "checkpoint_latest.pth"
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best model: {best_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint để tiếp tục training"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    print(f"📂 Loaded checkpoint từ epoch {checkpoint['epoch']}")
    return start_epoch, best_val_loss


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔥 depth_aware_train.py (NCKH 2026 — snntorch LIF)")
    print("🚀 Device:", device)
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # TensorBoard
    log_dir = PROJECT_ROOT / 'experiments' / 'logs' / 'vision' / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir)) if HAS_TB else SummaryWriter()
    print("📝 TensorBoard log:", log_dir)

    # ---------------- DATASET ----------------
    print("\n📂 Loading HILO Dataset...")
    print(f"   HILO root: {DEFAULT_HILO_ROOT}")
    print(f"   Seq len  : {args.seq_len}  |  Arc IDs: {args.arc_ids}")

    dataset = HILORawDataset(
        hilo_root=DEFAULT_HILO_ROOT,
        img_size=args.img_size,
        sequence_length=args.seq_len,
        arc_ids=args.arc_ids,
        max_scenes=args.max_scenes,
        use_undistorted=True,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset rỗng! Kiểm tra HILO_Scenes tại:\n  {DEFAULT_HILO_ROOT}"
        )

    num_classes = dataset.num_classes
    print(f"📊 Total sequences: {len(dataset):,}")
    print(f"📊 Classes (objects): {num_classes}")

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"📊 Train: {train_size:,} | Val: {val_size:,}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=0,
                               collate_fn=collate_fn, pin_memory=pin)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                               shuffle=False, num_workers=0,
                               collate_fn=collate_fn, pin_memory=pin)

    # ---------------- MODEL ----------------
    model = DepthAwareSNN(
        num_classes=num_classes,
        backbone_channels=64,
        temporal_hidden_dim=256,
        use_multiscale=False,    # Tắt multi-scale lúc đầu để train ổn định
        use_snn_backbone=args.use_snn,
    ).to(device)
    print(f"\n{model}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = DepthAwareLoss()
    
    # Mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("⚡ Mixed precision training: ENABLED")
    
    # Data augmentation (optional — nếu augmentation module có sẵn)
    augmentation = None
    if args.use_augmentation:
        try:
            from vision_system.data.augmentation import get_augmentation_pipeline
            augmentation = get_augmentation_pipeline(
                training=True,
                apply_color_jitter=args.aug_color_jitter,
                apply_geometric=args.aug_geometric,
                apply_depth_noise=args.aug_depth_noise,
            )
            print("🔄 Data augmentation: ENABLED")
        except Exception as e:
            print(f"⚠️ Augmentation not available: {e} — skipping")

    # Resume training
    start_epoch = 1
    best_val = float("inf")
    if args.resume and Path(args.resume).exists():
        start_epoch, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # Early stopping
    patience_counter = 0
    early_stop_patience = args.early_stop_patience

    print("\n🎯 BẮT ĐẦU TRAINING...\n")
    print("=" * 70)

    # History for NCKH plots
    history = []
    history_path = Path(args.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- EPOCH LOOP ----------------
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n📌 Epoch {epoch}/{args.num_epochs}")

        # ----- TRAIN -----
        train_loss, train_cls, train_bbox, train_spike = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            args.log_interval, args.use_amp, scaler, args.grad_clip
        )

        # ----- VALIDATE -----
        val_loss, val_accuracy, val_bbox_mse = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/Learning_Rate", current_lr, epoch)
        writer.add_scalar("Train/Epoch_Loss", train_loss, epoch)
        writer.add_scalar("Train/Epoch_CLS_Loss", train_cls, epoch)
        writer.add_scalar("Train/Epoch_BBox_Loss", train_bbox, epoch)

        # Print summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch:03d} | "
              f"Train Loss={train_loss:.4f} (CLS={train_cls:.4f}, BBox={train_bbox:.4f}) | "
              f"Val Loss={val_loss:.4f} Acc={val_accuracy:.4f} | "
              f"Spike={train_spike:.3f} | LR={current_lr:.2e}")
        print(f"{'='*70}")
        writer.add_scalar("Train/Spike_Rate_Epoch", train_spike, epoch)

        # ── Save history (for NCKH plot_training_curves.py) ──
        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "train_cls":   round(train_cls, 6),
            "train_bbox":  round(train_bbox, 6),
            "train_acc":   0.0,        # cls acc computed separately in validate
            "val_loss":    round(val_loss, 6),
            "val_acc":     round(val_accuracy * 100, 4),  # percent
            "val_bbox_mse": round(val_bbox_mse, 6),
            "spike_rate":  round(train_spike, 6),
            "lr":          current_lr,
        })
        with open(history_path, 'w') as hf:
            json.dump(history, hf, indent=2)

        # Save checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            args.checkpoint_dir, is_best=is_best
        )
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                args.checkpoint_dir, filename=f"checkpoint_epoch_{epoch}.pth"
            )

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️  Early stopping sau {early_stop_patience} epochs không cải thiện")
            print(f"   Best validation loss: {best_val:.4f}")
            break

    print("\n" + "=" * 70)
    print("✅ TRAINING HOÀN TẤT!")
    print("=" * 70)
    print(f"📊 Best validation loss: {best_val:.4f}")
    print(f"📁 Checkpoints: {args.checkpoint_dir}")
    print(f"📝 TensorBoard logs: {log_dir}")
    print(f"📄 History saved: {history_path}")
    print(f"\nĐể xem tensorboard, chạy:")
    print(f"  tensorboard --logdir {log_dir.parent}")
    print(f"\nĐể vẽ training curves:")
    print(f"  python scripts/visualization/plot_training_curves.py --tag vision --history {history_path}")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    train()
