#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script master để tạo tất cả các hình đánh giá cho báo cáo NCKH.

Script này sẽ:
1. Tạo dữ liệu mẫu (nếu chưa có)
2. Export evaluation results (nếu chưa có)
3. Generate tất cả các hình visualization

Usage:
    python scripts/visualization/generate_all_figures.py
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, description: str):
    """Chạy một script Python."""
    print(f"\n{'='*70}")
    print(f"📊 {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("⚠️  Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_path}:")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    """Main function để chạy tất cả các bước."""
    scripts_dir = Path(__file__).parent
    
    print("🚀 Bắt đầu tạo các hình đánh giá cho báo cáo NCKH...")
    
    # Step 1: Tạo sample training logs (nếu chưa có)
    sample_logs_dir = Path("experiments/logs/vision_training_logs/sample")
    if not (sample_logs_dir / "training_curves.json").exists():
        print("\n📝 Tạo dữ liệu training logs mẫu...")
        run_script(
            scripts_dir / "create_sample_training_logs.py",
            "Tạo sample training logs"
        )
    else:
        print("✅ Sample training logs đã tồn tại")
    
    # Step 2: Export evaluation results (nếu chưa có)
    results_file = Path("experiments/results/vision_results/evaluation_results.json")
    if not results_file.exists():
        print("\n📝 Tạo dữ liệu evaluation results mẫu...")
        run_script(
            scripts_dir / "export_evaluation_results.py",
            "Export evaluation results"
        )
    else:
        print("✅ Evaluation results đã tồn tại")
    
    # Step 3: Generate all figures
    print("\n📊 Tạo tất cả các hình visualization...")
    success = run_script(
        scripts_dir / "generate_evaluation_figures.py",
        "Generate evaluation figures"
    )
    
    if success:
        print("\n" + "="*70)
        print("✅ HOÀN TẤT! Tất cả các hình đã được tạo.")
        print("="*70)
        print("\n📁 Các hình được lưu tại: experiments/figures/evaluation/")
        print("\nCác hình đã tạo:")
        print("  - training_curves.png/pdf")
        print("  - map_metrics.png/pdf")
        print("  - iou_distribution.png/pdf")
        print("  - confusion_matrix.png/pdf")
        print("  - performance_comparison.png/pdf")
        print("  - latency_fps.png/pdf")
    else:
        print("\n❌ Có lỗi xảy ra khi tạo hình. Vui lòng kiểm tra lại.")


if __name__ == "__main__":
    main()
