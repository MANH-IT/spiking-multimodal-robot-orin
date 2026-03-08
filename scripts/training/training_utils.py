#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Utilities

Các utilities hỗ trợ training:
- Checkpoint management
- Training config management
- Resume training
- Model inspection
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_training_config(config: Dict[str, Any], output_dir: Path):
    """Save training configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "training_config.json"
    
    config["timestamp"] = datetime.now().isoformat()
    
    with config_file.open("w") as f:
        json.dump(config, f, indent=2)
    
    return config_file


def load_training_config(config_file: Path) -> Dict[str, Any]:
    """Load training configuration."""
    with config_file.open() as f:
        return json.load(f)


def inspect_checkpoint(checkpoint_path: Path):
    """Inspect checkpoint file."""
    print(f"📂 Inspecting checkpoint: {checkpoint_path}")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: {checkpoint[key].shape} ({checkpoint[key].dtype})")
            elif isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} keys")
            else:
                print(f"  {key}: {type(checkpoint[key]).__name__}")
        
        # Check for common keys
        if "epoch" in checkpoint:
            print(f"\n📊 Epoch: {checkpoint['epoch']}")
        if "val_loss" in checkpoint:
            print(f"📊 Validation Loss: {checkpoint['val_loss']:.4f}")
        if "optimizer_state_dict" in checkpoint:
            print("✅ Optimizer state found")
        if "scheduler_state_dict" in checkpoint:
            print("✅ Scheduler state found")
    else:
        print("⚠️  Checkpoint is not a dict (might be state_dict only)")
        if isinstance(checkpoint, torch.nn.Module):
            print(f"   Model parameters: {sum(p.numel() for p in checkpoint.parameters())}")
    
    print("="*70)


def find_best_checkpoint(checkpoint_dir: Path, metric: str = "val_loss") -> Optional[Path]:
    """Find best checkpoint based on metric."""
    checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        return None
    
    best_checkpoint = None
    best_value = float("inf") if "loss" in metric.lower() else float("-inf")
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            if isinstance(checkpoint, dict) and metric in checkpoint:
                value = checkpoint[metric]
                
                if "loss" in metric.lower():
                    if value < best_value:
                        best_value = value
                        best_checkpoint = checkpoint_file
                else:
                    if value > best_value:
                        best_value = value
                        best_checkpoint = checkpoint_file
        except Exception as e:
            print(f"⚠️  Error reading {checkpoint_file}: {e}")
            continue
    
    return best_checkpoint


def cleanup_checkpoints(
    checkpoint_dir: Path,
    keep_best: bool = True,
    keep_last_n: int = 5,
    keep_every_n_epochs: int = 10,
):
    """Cleanup old checkpoints, keep only important ones."""
    checkpoint_files = sorted(
        checkpoint_dir.glob("*.pth") + checkpoint_dir.glob("*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    
    if not checkpoint_files:
        print("No checkpoints to clean")
        return
    
    to_keep = set()
    
    # Keep best
    if keep_best:
        best = find_best_checkpoint(checkpoint_dir)
        if best:
            to_keep.add(best)
            print(f"✅ Keeping best checkpoint: {best.name}")
    
    # Keep last N
    for checkpoint_file in checkpoint_files[:keep_last_n]:
        to_keep.add(checkpoint_file)
    
    # Keep every N epochs
    epochs_kept = set()
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            if isinstance(checkpoint, dict) and "epoch" in checkpoint:
                epoch = checkpoint["epoch"]
                if epoch % keep_every_n_epochs == 0:
                    to_keep.add(checkpoint_file)
                    epochs_kept.add(epoch)
        except:
            pass
    
    # Delete others
    deleted = 0
    for checkpoint_file in checkpoint_files:
        if checkpoint_file not in to_keep:
            checkpoint_file.unlink()
            deleted += 1
    
    print(f"🗑️  Deleted {deleted} checkpoints")
    print(f"📦 Kept {len(to_keep)} checkpoints")
    if epochs_kept:
        print(f"📅 Epochs kept: {sorted(epochs_kept)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Utilities")
    parser.add_argument("command", choices=["inspect", "find-best", "cleanup"])
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path (for inspect)")
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory")
    parser.add_argument("--metric", type=str, default="val_loss", help="Metric for find-best")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        if args.checkpoint:
            inspect_checkpoint(args.checkpoint)
        else:
            print("❌ Need --checkpoint for inspect")
    
    elif args.command == "find-best":
        if args.checkpoint_dir:
            best = find_best_checkpoint(args.checkpoint_dir, args.metric)
            if best:
                print(f"✅ Best checkpoint: {best}")
            else:
                print("❌ No checkpoint found")
        else:
            print("❌ Need --checkpoint-dir for find-best")
    
    elif args.command == "cleanup":
        if args.checkpoint_dir:
            cleanup_checkpoints(args.checkpoint_dir)
        else:
            print("❌ Need --checkpoint-dir for cleanup")
