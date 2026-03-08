#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Tool

So sánh nhiều models:
- Side-by-side metrics comparison
- Visual comparison
- Export comparison report

Usage:
    python scripts/evaluation/model_comparison.py \
        --models model1.pth model2.pth model3.pth \
        --dataset data/annotations.json \
        --output-dir experiments/comparison
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.data.rgbd_sequence_with_ann_dataset import RGBDSequenceWithAnnDataset
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN
from scripts.evaluation.comprehensive_eval import evaluate_model, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model Comparison Tool"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        required=True,
        help="Model checkpoint paths",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        type=str,
        default=None,
        help="Model names (optional)",
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
        default=Path("experiments/comparison"),
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison visualizations",
    )
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device):
    """Load model from checkpoint."""
    model = DepthAwareSNN(num_classes=237).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def compare_models(
    models: List[Path],
    model_names: List[str],
    dataset: Path,
    output_dir: Path,
    batch_size: int,
    visualize: bool,
):
    """Compare multiple models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset_obj = RGBDSequenceWithAnnDataset(dataset)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Evaluate each model
    all_results = {}
    iou_thresholds = [0.25, 0.5, 0.75]
    
    for idx, model_path in enumerate(models):
        model_name = model_names[idx] if model_names else model_path.stem
        
        print(f"\n🔍 Evaluating {model_name}...")
        print(f"   Model: {model_path}")
        
        # Load model
        model = load_model(model_path, device)
        
        # Evaluate
        results = evaluate_model(
            model,
            dataloader,
            device,
            iou_thresholds,
        )
        
        all_results[model_name] = results
        
        # Print summary
        if "map_results" in results:
            print(f"   mAP: {results['map_results'].get('mAP', 0):.4f}")
        if "iou_statistics" in results:
            print(f"   Mean IoU: {results['iou_statistics']['mean']:.4f}")
    
    # Comparison summary
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    
    comparison_data = {}
    
    # Extract key metrics
    metrics = ["mAP", "mean_iou", "precision@0.50", "recall@0.50", "f1@0.50"]
    
    for model_name, results in all_results.items():
        comparison_data[model_name] = {}
        
        if "map_results" in results:
            comparison_data[model_name]["mAP"] = results["map_results"].get("mAP", 0)
        
        if "iou_statistics" in results:
            comparison_data[model_name]["mean_iou"] = results["iou_statistics"]["mean"]
        
        for threshold in iou_thresholds:
            key = f"precision@{threshold:.2f}"
            if key in results:
                comparison_data[model_name][key] = results[key]
    
    # Print comparison table
    print("\nMetrics Comparison:")
    print("-" * 70)
    print(f"{'Model':<20} {'mAP':<10} {'Mean IoU':<10} {'Precision@0.5':<15} {'Recall@0.5':<15}")
    print("-" * 70)
    
    for model_name, metrics_dict in comparison_data.items():
        mAP = metrics_dict.get("mAP", 0)
        mean_iou = metrics_dict.get("mean_iou", 0)
        precision = metrics_dict.get("precision@0.50", 0)
        recall = metrics_dict.get("recall@0.50", 0)
        
        print(f"{model_name:<20} {mAP:<10.4f} {mean_iou:<10.4f} {precision:<15.4f} {recall:<15.4f}")
    
    print("="*70)
    
    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / "model_comparison.json"
    with comparison_file.open("w") as f:
        json.dump({
            "models": all_results,
            "comparison": comparison_data,
        }, f, indent=2)
    print(f"\n✅ Comparison saved to {comparison_file}")
    
    # Generate visualization
    if visualize:
        try:
            from scripts.visualization.generate_evaluation_figures import (
                plot_performance_comparison,
            )
            
            plot_performance_comparison(
                comparison_data,
                output_dir / "comparison.png",
                language="vi",
                format="png",
                dpi=300,
            )
            print(f"✅ Visualization saved to {output_dir / 'comparison.png'}")
        except Exception as e:
            print(f"⚠️  Could not generate visualization: {e}")


def main():
    args = parse_args()
    
    # Validate model names
    if args.model_names and len(args.model_names) != len(args.models):
        print("⚠️  Number of model names doesn't match number of models")
        print("   Using model file names instead")
        args.model_names = None
    
    if args.model_names is None:
        args.model_names = [m.stem for m in args.models]
    
    print("🔍 Model Comparison Tool")
    print("="*70)
    print(f"📊 Models to compare: {len(args.models)}")
    for name, path in zip(args.model_names, args.models):
        print(f"   - {name}: {path}")
    print(f"📂 Dataset: {args.dataset}")
    print(f"📁 Output: {args.output_dir}")
    
    compare_models(
        args.models,
        args.model_names,
        args.dataset,
        args.output_dir,
        args.batch_size,
        args.visualize,
    )


if __name__ == "__main__":
    main()
