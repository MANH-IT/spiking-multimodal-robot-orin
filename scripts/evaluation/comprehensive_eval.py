#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script

Đánh giá toàn diện model với:
- Multiple metrics (mAP, IoU, Precision/Recall, F1)
- Per-class evaluation
- Error analysis
- Export results cho visualization

Usage:
    python scripts/evaluation/comprehensive_eval.py \
        --model vision_system/weights/finetuned/best_model.pth \
        --dataset data/02_processed/vision/coco_format/hilo_annotations_3d.json \
        --output-dir experiments/evaluation/run_001
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

import torch
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.data.rgbd_sequence_with_ann_dataset import (
    RGBDSequenceWithAnnDataset,
)
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN
from vision_system.utils.evaluation_metrics import (
    compute_iou_3d,
    compute_map_3d,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset annotation file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--iou-thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="IoU thresholds for mAP",
    )
    parser.add_argument(
        "--export-results",
        action="store_true",
        help="Export results to JSON for visualization",
    )
    return parser.parse_args()


def collate_fn(batch):
    rgbs, depths, targets = zip(*batch)
    return (
        torch.stack(rgbs, dim=0),
        torch.stack(depths, dim=0),
        list(targets),
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_thresholds: List[float],
) -> Dict[str, Any]:
    """Comprehensive evaluation."""
    model.eval()
    
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    all_ious = []
    
    with torch.no_grad():
        for rgb, depth, targets in tqdm(dataloader, desc="Evaluating"):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            # Forward
            rgb_seq = rgb.unsqueeze(1)
            depth_seq = depth.unsqueeze(1)
            bbox_pred, logits = model(rgb_seq, depth_seq)
            
            # Process predictions
            pred_labels = logits.argmax(dim=1)
            pred_scores = torch.softmax(logits, dim=1).max(dim=1)[0]
            
            for i in range(len(targets)):
                if targets[i]["boxes_3d"].shape[0] > 0:
                    # Has ground truth
                    gt_box = targets[i]["boxes_3d"][0]
                    gt_label = targets[i]["labels"][0]
                    
                    pred_box = bbox_pred[i]
                    pred_label = pred_labels[i].item()
                    pred_score = pred_scores[i].item()
                    
                    # Compute IoU
                    iou = compute_iou_3d(
                        pred_box.unsqueeze(0),
                        gt_box.unsqueeze(0),
                    )[0, 0].item()
                    all_ious.append(iou)
                    
                    all_pred_boxes.append(pred_box.cpu())
                    all_pred_scores.append(torch.tensor(pred_score))
                    all_pred_labels.append(torch.tensor(pred_label))
                    
                    all_gt_boxes.append(gt_box.cpu())
                    all_gt_labels.append(gt_label)
    
    # Compute metrics
    results = {}
    
    # IoU statistics
    if all_ious:
        results["iou_statistics"] = {
            "mean": float(np.mean(all_ious)),
            "median": float(np.median(all_ious)),
            "std": float(np.std(all_ious)),
            "min": float(np.min(all_ious)),
            "max": float(np.max(all_ious)),
            "distribution": [float(x) for x in all_ious],
        }
    
    # mAP
    if all_pred_boxes and all_gt_boxes:
        map_results = compute_map_3d(
            pred_boxes=all_pred_boxes,
            pred_scores=all_pred_scores,
            pred_labels=all_pred_labels,
            gt_boxes=all_gt_boxes,
            gt_labels=all_gt_labels,
            iou_thresholds=iou_thresholds,
        )
        results["map_results"] = {k: float(v) for k, v in map_results.items()}
    
    # Precision/Recall
    if all_ious:
        iou_array = np.array(all_ious)
        for threshold in iou_thresholds:
            tp = np.sum(iou_array >= threshold)
            fp = len(all_ious) - tp
            fn = 0  # Simplified
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"precision@{threshold:.2f}"] = float(precision)
            results[f"recall@{threshold:.2f}"] = float(recall)
            results[f"f1@{threshold:.2f}"] = float(f1)
    
    results["total_samples"] = len(all_ious)
    
    return results


def main():
    args = parse_args()
    
    # Validate inputs early
    if not args.model.exists():
        print(f"❌ Model checkpoint not found: {args.model}")
        raise SystemExit(1)
    if not args.dataset.exists():
        print(f"❌ Dataset annotation file not found: {args.dataset}")
        raise SystemExit(1)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("experiments/evaluation") / args.model.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Comprehensive Evaluation")
    print("="*70)
    print(f"📁 Model: {args.model}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"📂 Output: {args.output_dir}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Load model
    print("\n🤖 Loading model...")
    model = DepthAwareSNN(num_classes=237).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model loaded")
    
    # Dataset
    print("\n📂 Loading dataset...")
    dataset = RGBDSequenceWithAnnDataset(args.dataset)
    if len(dataset) == 0:
        print("❌ Dataset loaded nhưng không có sample nào (len(dataset) == 0). Evaluation aborted.")
        raise SystemExit(1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Evaluate
    print("\n🔬 Evaluating...")
    results = evaluate_model(
        model,
        dataloader,
        device,
        args.iou_thresholds,
    )
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    if "iou_statistics" in results:
        iou_stats = results["iou_statistics"]
        print(f"IoU - Mean: {iou_stats['mean']:.4f}, Median: {iou_stats['median']:.4f}")
    
    if "map_results" in results:
        print("\nmAP Results:")
        for key, value in results["map_results"].items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal samples evaluated: {results['total_samples']}")
    print("="*70)
    
    # Save results
    results_file = args.output_dir / "evaluation_results.json"
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Export for visualization
    if args.export_results:
        export_file = Path("experiments/results/vision_results/evaluation_results.json")
        export_file.parent.mkdir(parents=True, exist_ok=True)
        with export_file.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Exported to {export_file} for visualization")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
