#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script export kết quả evaluation thành JSON format để visualization script có thể đọc.

Tích hợp với eval_depth_aware_3d.py để export:
- mAP metrics
- Confusion matrix
- IoU distribution
- Performance metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Optional imports for real evaluation (not needed for sample data)
try:
    import torch
    from vision_system.utils.evaluation_metrics import compute_map_3d, compute_iou_3d
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export evaluation results to JSON for visualization"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/vision_results/evaluation_results.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=None,
        help="Path to existing evaluation results (optional)",
    )
    return parser.parse_args()


def export_map_results(
    pred_boxes: List,
    pred_scores: List,
    pred_labels: List,
    gt_boxes: List,
    gt_labels: List,
) -> Dict[str, float]:
    """Compute and export mAP results."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for computing mAP from real data")
    map_results = compute_map_3d(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_thresholds=[0.25, 0.5, 0.75],
    )
    return map_results


def export_confusion_matrix(
    pred_labels: List,
    gt_labels: List,
    num_classes: int,
) -> Dict[str, Any]:
    """Compute and export confusion matrix."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for computing confusion matrix from real data")
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for pred, gt in zip(pred_labels, gt_labels):
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy()
        else:
            pred_np = np.array(pred)
        if isinstance(gt, torch.Tensor):
            gt_np = gt.cpu().numpy()
        else:
            gt_np = np.array(gt)
        
        for p, g in zip(pred_np, gt_np):
            cm[int(g), int(p)] += 1
    
    return {
        "matrix": cm.tolist(),
        "class_names": [f"Class {i}" for i in range(num_classes)],
    }


def export_iou_distribution(
    pred_boxes: List,
    gt_boxes: List,
) -> List[float]:
    """Compute IoU distribution."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for computing IoU distribution from real data")
    iou_values = []
    
    for pred_box_list, gt_box_list in zip(pred_boxes, gt_boxes):
        if len(pred_box_list) == 0 or len(gt_box_list) == 0:
            continue
        
        # Compute pairwise IoU
        ious = compute_iou_3d(pred_box_list, gt_box_list)  # (N, M)
        
        # Get best IoU for each prediction
        best_ious = ious.max(dim=1)[0].cpu().numpy()
        iou_values.extend(best_ious.tolist())
    
    return iou_values


def create_sample_results() -> Dict[str, Any]:
    """Tạo dữ liệu mẫu để test visualization."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create realistic confusion matrix (diagonal dominant)
    cm = np.random.randint(0, 30, size=(10, 10))
    np.fill_diagonal(cm, np.random.randint(50, 100, size=10))
    
    # Precision-Recall curve data
    recalls = np.linspace(0, 1, 50).tolist()
    precisions = (0.9 - 0.3 * np.array(recalls) + np.random.normal(0, 0.05, 50)).tolist()
    precisions = [max(0, min(1, p)) for p in precisions]
    
    # Per-class mAP
    class_names = ["Bottle", "Cup", "Chair", "Table", "Laptop", "Phone", "Book", "Box", "Toy", "Other"]
    per_class_map = {name: float(np.random.uniform(0.5, 0.9)) for name in class_names}
    
    # Ablation study
    ablation_study = {
        "Full Model": 0.653,
        "w/o Depth Branch": 0.582,
        "w/o Attention Fusion": 0.612,
        "w/o MultiScale STDFE": 0.628,
        "w/o Temporal LSTM": 0.601,
        "RGB only": 0.521,
    }
    
    # Error analysis
    error_analysis = {
        "True Positive": 1250,
        "False Positive": 180,
        "False Negative": 220,
        "True Negative": 8900,
    }
    
    # IoU vs Distance (depth)
    depth_ranges = ["0-1m", "1-2m", "2-3m", "3-4m", "4-5m"]
    iou_by_depth = [0.72, 0.68, 0.61, 0.54, 0.48]
    
    # Jetson metrics (Power, Memory, Temperature)
    jetson_metrics = {
        "SNN": {"Power_W": 8.2, "Memory_GB": 2.1, "Temp_C": 52, "FPS": 28.5},
        "ANN": {"Power_W": 12.5, "Memory_GB": 3.8, "Temp_C": 65, "FPS": 22.3},
    }
    
    # ROC curve data
    fpr = np.linspace(0, 1, 50).tolist()
    tpr = (np.array(fpr) ** 0.5 + np.random.normal(0, 0.02, 50)).tolist()
    tpr = [max(0, min(1, t)) for t in tpr]
    
    return {
        "map_results": {
            "mAP@0.25": 0.752,
            "mAP@0.50": 0.685,
            "mAP@0.75": 0.521,
            "mAP": 0.653,
        },
        "iou_distribution": np.random.beta(5, 2, size=1000).tolist(),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "class_names": [f"Class {i}" for i in range(10)],
        },
        "performance_comparison": {
            "SNN": {
                "mAP": 0.653,
                "Accuracy": 0.825,
                "FPS": 28.5,
                "Latency_ms": 35.1,
            },
            "ANN": {
                "mAP": 0.642,
                "Accuracy": 0.818,
                "FPS": 22.3,
                "Latency_ms": 44.8,
            },
        },
        "latency_benchmark": {
            "SNN": [32.1, 33.5, 34.2, 35.8, 36.1, 34.9, 35.2, 33.8, 35.5, 34.3],
            "ANN": [42.3, 43.1, 44.5, 45.2, 44.8, 43.9, 45.1, 44.6, 43.7, 44.2],
        },
        "pr_curve": {
            "recalls": recalls,
            "precisions": precisions,
            "ap": 0.752,
        },
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "auc": 0.912,
        },
        "per_class_map": per_class_map,
        "ablation_study": ablation_study,
        "error_analysis": error_analysis,
        "iou_vs_distance": {
            "depth_ranges": depth_ranges,
            "iou_values": iou_by_depth,
        },
        "jetson_metrics": jetson_metrics,
    }


def main():
    args = parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # If eval_results provided, load and process
    if args.eval_results and args.eval_results.exists():
        # TODO: Load from actual evaluation results
        print(f"Loading evaluation results from {args.eval_results}")
        results = {}
    else:
        # Create sample results for testing
        print("Creating sample evaluation results for testing...")
        results = create_sample_results()
    
    # Save to JSON
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported evaluation results to: {args.output}")
    print(f"   - mAP results: {len(results.get('map_results', {}))} metrics")
    print(f"   - IoU distribution: {len(results.get('iou_distribution', []))} samples")
    print(f"   - Confusion matrix: {len(results.get('confusion_matrix', {}).get('matrix', []))} classes")


if __name__ == "__main__":
    main()
