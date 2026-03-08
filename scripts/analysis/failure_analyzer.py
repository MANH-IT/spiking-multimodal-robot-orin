#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Failure Case Analyzer - Phân tích các trường hợp lỗi

Tự động phát hiện và phân loại:
- False Positives
- False Negatives
- Error categories (occlusion, scale, distance, etc.)
- Generate report với visualizations

Usage:
    python scripts/analysis/failure_analyzer.py \
        --predictions results/predictions.json \
        --ground-truth data/annotations.json \
        --output-dir experiments/analysis/failures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Failure Case Analyzer"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predictions JSON file",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Ground truth annotations JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis/failures"),
        help="Output directory",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images",
    )
    return parser.parse_args()


def compute_iou_2d(bbox1, bbox2):
    """Compute 2D IoU."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def categorize_error(pred_bbox, gt_bbox, image_size):
    """Categorize error type."""
    img_w, img_h = image_size
    
    # Size analysis
    pred_area = pred_bbox[2] * pred_bbox[3]
    gt_area = gt_bbox[2] * gt_bbox[3]
    size_ratio = pred_area / gt_area if gt_area > 0 else 0
    
    # Position analysis
    pred_center = (pred_bbox[0] + pred_bbox[2]/2, pred_bbox[1] + pred_bbox[3]/2)
    gt_center = (gt_bbox[0] + gt_bbox[2]/2, gt_bbox[1] + gt_bbox[3]/2)
    center_dist = np.sqrt((pred_center[0] - gt_center[0])**2 + 
                         (pred_center[1] - gt_center[1])**2)
    
    # Scale error
    if size_ratio < 0.5:
        return "scale_small"
    elif size_ratio > 2.0:
        return "scale_large"
    
    # Position error
    if center_dist > max(pred_bbox[2], pred_bbox[3]):
        return "position_error"
    
    # Partial overlap
    iou = compute_iou_2d(pred_bbox, gt_bbox)
    if iou < 0.3:
        return "low_overlap"
    
    return "other"


def analyze_failures(predictions, ground_truth, iou_threshold=0.5):
    """Analyze failure cases."""
    results = {
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
        "error_categories": defaultdict(list),
        "statistics": {},
    }
    
    # Match predictions với ground truth
    matched_gt = set()
    
    for pred in predictions:
        best_iou = 0
        best_gt_idx = None
        
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            
            iou = compute_iou_2d(pred["bbox2d"], gt["bbox2d"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        if best_iou >= iou_threshold and best_gt_idx is not None:
            # True Positive
            results["true_positives"].append({
                "prediction": pred,
                "ground_truth": ground_truth[best_gt_idx],
                "iou": best_iou,
            })
            matched_gt.add(best_gt_idx)
        else:
            # False Positive
            error_type = "no_match"
            if best_gt_idx is not None:
                error_type = categorize_error(
                    pred["bbox2d"],
                    ground_truth[best_gt_idx]["bbox2d"],
                    (640, 480)  # Default image size
                )
            
            results["false_positives"].append({
                "prediction": pred,
                "iou": best_iou,
                "error_type": error_type,
            })
            results["error_categories"][error_type].append(pred)
    
    # False Negatives (unmatched ground truth)
    for idx, gt in enumerate(ground_truth):
        if idx not in matched_gt:
            results["false_negatives"].append({
                "ground_truth": gt,
            })
    
    # Statistics
    results["statistics"] = {
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth),
        "true_positives": len(results["true_positives"]),
        "false_positives": len(results["false_positives"]),
        "false_negatives": len(results["false_negatives"]),
        "precision": len(results["true_positives"]) / len(predictions) if predictions else 0,
        "recall": len(results["true_positives"]) / len(ground_truth) if ground_truth else 0,
        "error_distribution": {k: len(v) for k, v in results["error_categories"].items()},
    }
    
    return results


def generate_report(results, output_dir):
    """Generate analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    report_path = output_dir / "failure_analysis.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate text report
    report_text = f"""
# Failure Analysis Report

## Statistics

- Total Predictions: {results['statistics']['total_predictions']}
- Total Ground Truth: {results['statistics']['total_ground_truth']}
- True Positives: {results['statistics']['true_positives']}
- False Positives: {results['statistics']['false_positives']}
- False Negatives: {results['statistics']['false_negatives']}

## Metrics

- Precision: {results['statistics']['precision']:.3f}
- Recall: {results['statistics']['recall']:.3f}
- F1-Score: {2 * results['statistics']['precision'] * results['statistics']['recall'] / (results['statistics']['precision'] + results['statistics']['recall']) if (results['statistics']['precision'] + results['statistics']['recall']) > 0 else 0:.3f}

## Error Distribution

"""
    
    for error_type, count in results['statistics']['error_distribution'].items():
        report_text += f"- {error_type}: {count}\n"
    
    report_path = output_dir / "report.md"
    with report_path.open("w") as f:
        f.write(report_text)
    
    print(f"✅ Report saved to {output_dir}")


def main():
    args = parse_args()
    
    print("🔍 Failure Case Analyzer")
    print("="*70)
    
    # Load data
    print("📂 Loading predictions...")
    with args.predictions.open() as f:
        predictions = json.load(f)
    print(f"✅ Loaded {len(predictions)} predictions")
    
    print("📂 Loading ground truth...")
    with args.ground_truth.open() as f:
        ground_truth = json.load(f)
    print(f"✅ Loaded {len(ground_truth)} ground truth")
    
    # Analyze
    print("🔬 Analyzing failures...")
    results = analyze_failures(predictions, ground_truth, args.iou_threshold)
    
    # Generate report
    print("📊 Generating report...")
    generate_report(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    stats = results["statistics"]
    print(f"Precision: {stats['precision']:.3f}")
    print(f"Recall: {stats['recall']:.3f}")
    print(f"False Positives: {stats['false_positives']}")
    print(f"False Negatives: {stats['false_negatives']}")
    print(f"\nError Categories:")
    for error_type, count in stats['error_distribution'].items():
        print(f"  - {error_type}: {count}")


if __name__ == "__main__":
    main()
