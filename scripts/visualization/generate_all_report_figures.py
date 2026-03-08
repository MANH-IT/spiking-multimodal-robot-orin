#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tổng hợp để tạo TẤT CẢ các hình đánh giá cho báo cáo NCKH
Bao gồm cả đề tài 1 (Vision System) và đề tài 2 (NLP System)

Usage:
    python scripts/visualization/generate_all_report_figures.py
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
    except FileNotFoundError:
        print(f"⚠️  Script không tồn tại: {script_path}")
        return False


def main():
    """Main function để tạo tất cả các hình cho báo cáo."""
    scripts_dir = Path(__file__).parent
    
    print("🚀 BẮT ĐẦU TẠO TẤT CẢ CÁC HÌNH ĐÁNH GIÁ CHO BÁO CÁO NCKH")
    print("="*70)
    print("\n📋 Bao gồm:")
    print("  1. Đề tài 1: Vision System - 3D Object Detection")
    print("  2. Đề tài 2: NLP System (nếu có)")
    print("  3. Các hình đánh giá chung")
    print("="*70)
    
    success_count = 0
    total_count = 0
    
    # ========== PHẦN 1: CÁC HÌNH ĐÁNH GIÁ CHUNG ==========
    print("\n" + "="*70)
    print("📊 PHẦN 1: CÁC HÌNH ĐÁNH GIÁ CHUNG")
    print("="*70)
    
    # 1.1. Tạo sample data (nếu chưa có)
    total_count += 1
    if run_script(
        scripts_dir / "create_sample_training_logs.py",
        "Tạo sample training logs"
    ):
        success_count += 1
    
    total_count += 1
    if run_script(
        scripts_dir / "export_evaluation_results.py",
        "Export evaluation results"
    ):
        success_count += 1
    
    # 1.2. Generate evaluation figures (chung)
    total_count += 1
    if run_script(
        scripts_dir / "generate_evaluation_figures.py",
        "Generate evaluation figures (chung)"
    ):
        success_count += 1
    
    # ========== PHẦN 2: ĐỀ TÀI 1 - VISION SYSTEM ==========
    print("\n" + "="*70)
    print("📊 PHẦN 2: ĐỀ TÀI 1 - VISION SYSTEM (3D Object Detection)")
    print("="*70)
    
    total_count += 1
    if run_script(
        scripts_dir / "generate_vision_3d_figures.py",
        "Generate Vision System 3D figures"
    ):
        success_count += 1
    
    # ========== TỔNG KẾT ==========
    print("\n" + "="*70)
    print("📊 TỔNG KẾT")
    print("="*70)
    print(f"✅ Thành công: {success_count}/{total_count}")
    print(f"❌ Thất bại: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 HOÀN TẤT! Tất cả các hình đã được tạo thành công!")
        print("\n📁 Các hình được lưu tại:")
        print("  - experiments/figures/evaluation/ (hình đánh giá chung)")
        print("  - experiments/figures/vision_3d/ (hình Vision System)")
        print("\n📝 Các hình đã tạo:")
        print("\n  📊 Hình đánh giá chung:")
        print("    - training_curves.png")
        print("    - map_metrics.png")
        print("    - iou_distribution.png")
        print("    - confusion_matrix.png")
        print("    - performance_comparison.png")
        print("    - latency_fps.png")
        print("    - precision_recall_curve.png")
        print("    - roc_curve.png")
        print("    - per_class_map.png")
        print("    - ablation_study.png")
        print("    - error_analysis.png")
        print("    - iou_vs_distance.png")
        print("    - jetson_metrics.png")
        print("\n  🎯 Hình Vision System 3D:")
        print("    - dataset_statistics.png")
        print("    - architecture_diagram.png")
        print("    - detection_comparison.png")
        print("    - depth_visualization_*.png (nếu có ảnh)")
    else:
        print("\n⚠️  Một số hình chưa được tạo. Vui lòng kiểm tra lại.")


if __name__ == "__main__":
    main()
