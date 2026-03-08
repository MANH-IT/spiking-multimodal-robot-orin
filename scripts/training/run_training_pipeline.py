#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Training Pipeline

Chạy toàn bộ pipeline:
1. Training với monitoring
2. Evaluation
3. Export results
4. Generate visualizations

Usage:
    python scripts/training/run_training_pipeline.py \
        --train-config configs/training/default.yaml \
        --num-epochs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete Training Pipeline"
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Training config file",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only evaluate",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Model path (if skip training)",
    )
    return parser.parse_args()


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print("-"*70)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"experiments/pipeline/run_{timestamp}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 COMPLETE TRAINING PIPELINE")
    print("="*70)
    print(f"📁 Output directory: {args.output_dir}")
    print("="*70)
    
    model_path = None
    
    # Step 1: Training
    if not args.skip_training:
        training_dir = args.output_dir / "training"
        training_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "scripts/training/train_with_monitoring.py",
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--num-epochs", str(args.num_epochs),
            "--output-dir", str(training_dir),
            "--save-best",
        ]
        
        if run_command(cmd, "Training"):
            # Find best model
            checkpoint_dir = training_dir / "checkpoints"
            best_model = checkpoint_dir / "best_model.pth"
            if best_model.exists():
                model_path = best_model
                print(f"✅ Best model: {model_path}")
            else:
                latest_model = checkpoint_dir / "checkpoint_latest.pth"
                if latest_model.exists():
                    model_path = latest_model
                    print(f"✅ Using latest model: {model_path}")
        else:
            print("⚠️  Training failed, cannot continue")
            return
    else:
        if args.model_path:
            model_path = args.model_path
        else:
            print("❌ Need --model-path when skipping training")
            return
    
    if model_path is None or not model_path.exists():
        print("❌ No model found for evaluation")
        return
    
    # Step 2: Evaluation
    if not args.skip_evaluation:
        eval_dir = args.output_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "scripts/evaluation/comprehensive_eval.py",
            "--model", str(model_path),
            "--dataset", "data/02_processed/vision/coco_format/hilo_annotations_3d.json",
            "--output-dir", str(eval_dir),
            "--export-results",
        ]
        
        run_command(cmd, "Evaluation")
    
    # Step 3: Visualization
    if not args.skip_visualization:
        cmd = [
            sys.executable,
            "scripts/visualization/generate_evaluation_figures.py",
            "--log-dir", str(args.output_dir / "training" / "logs"),
            "--results-dir", str(args.output_dir / "evaluation"),
            "--output-dir", str(args.output_dir / "figures"),
        ]
        
        run_command(cmd, "Generate Visualizations")
    
    print("\n" + "="*70)
    print("✅ PIPELINE HOÀN TẤT!")
    print("="*70)
    print(f"📁 Results: {args.output_dir}")
    print("\nCấu trúc output:")
    print(f"  {args.output_dir}/")
    if not args.skip_training:
        print(f"    training/        - Training logs và checkpoints")
    if not args.skip_evaluation:
        print(f"    evaluation/      - Evaluation results")
    if not args.skip_visualization:
        print(f"    figures/         - Visualization figures")
    print("="*70)


if __name__ == "__main__":
    main()
