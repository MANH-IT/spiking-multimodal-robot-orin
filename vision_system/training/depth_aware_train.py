# depth_aware_train.py
# Training DepthAwareSNN với HILO RGB-D dataset

import sys
import time
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.data.rgbd_sequence_with_gt_dataset import (
    RGBDSequenceWithGTDataset, collate_fn
)
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN


# ---------------------- CONFIG ----------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=237)
    return parser.parse_args()


# ---------------------- LOSS ----------------------

class DepthAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
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

        total = total_cls / valid + total_box / valid
        return {
            "total": total,
            "cls": total_cls / valid,
            "bbox": total_box / valid,
            "valid": valid
        }


# ---------------------- TRAIN LOOP ----------------------

def train():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔥 depth_aware_train.py đang chạy...")
    print("🚀 Device:", device)

    # TensorBoard
    log_dir = Path("experiments/logs/vision_training_logs") / time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    print("📝 TensorBoard log:", log_dir)

    # ---------------- DATASET ----------------
    print("\n📂 Loading dataset...")

    dataset = RGBDSequenceWithGTDataset(
        rgb_dir=Path("data/01_interim/vision/aligned/rgb"),
        depth_dir=Path("data/01_interim/vision/aligned/depth"),
        ann_file=Path("data/02_processed/vision/coco_format/hilo_annotations_3d.json"),
        img_size=args.img_size
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset rỗng — kiểm tra đường dẫn hoặc annotation")

    print(f"📊 Tổng samples: {len(dataset)}")

    # split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"📊 Train: {train_size} | Val: {val_size}")

    # loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # ---------------- MODEL ----------------
    model = DepthAwareSNN(num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = DepthAwareLoss()

    best_val = float("inf")

    print("\n🎯 BẮT ĐẦU TRAINING...\n")

    # ---------------- EPOCH LOOP ----------------
    for epoch in range(1, args.num_epochs + 1):

        # ----- TRAIN -----
        model.train()
        train_loss = 0.0

        for rgb, depth, targets in train_loader:
            rgb = rgb.to(device)
            depth = depth.to(device)

            bbox_pred, cls_pred = model(rgb, depth)
            losses = criterion(bbox_pred, cls_pred, targets)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            train_loss += losses["total"].item()

        avg_train = train_loss / len(train_loader)

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for rgb, depth, targets in val_loader:
                rgb = rgb.to(device)
                depth = depth.to(device)

                bbox_pred, cls_pred = model(rgb, depth)
                losses = criterion(bbox_pred, cls_pred, targets)
                val_loss += losses["total"].item()

        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Loss/val", avg_val, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train {avg_train:.4f} | Val {avg_val:.4f} | LR {current_lr:.6f}"
        )

        # save best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_depth_aware_snn.pth")
            print("💾 Saved best model")

    print("\n✅ TRAINING HOÀN TẤT!")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    train()
