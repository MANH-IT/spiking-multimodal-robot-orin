#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo các hình đánh giá cho báo cáo NCKH

Tạo các hình:
1. Training curves (loss, accuracy over epochs)
2. Confusion matrix
3. Precision-Recall curves
4. mAP metrics visualization
5. IoU distribution
6. Performance comparison (SNN vs ANN)
7. Latency/FPS benchmarks
8. Model architecture visualization

Usage:
    python scripts/visualization/generate_evaluation_figures.py \
        --log-dir experiments/logs/vision_training_logs \
        --results-dir experiments/results/vision_results \
        --output-dir experiments/figures/evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Set style for scientific papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16

# Try to import tensorboard for reading logs
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard không có sẵn. Sẽ dùng dữ liệu từ JSON files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo các hình đánh giá cho báo cáo NCKH"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("experiments/logs/vision_training_logs"),
        help="Thư mục chứa TensorBoard logs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/vision_results"),
        help="Thư mục chứa kết quả evaluation (JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures/evaluation"),
        help="Thư mục output cho các hình",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Định dạng file output",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI cho hình ảnh (300 cho báo cáo chất lượng cao)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        choices=["vi", "en"],
        help="Ngôn ngữ cho labels (vi: Tiếng Việt, en: English)",
    )
    return parser.parse_args()


# ==================== UTILITY FUNCTIONS ====================

def load_tensorboard_logs(log_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load training logs từ TensorBoard event files hoặc JSON files.
    
    Returns:
        Dict với keys là metric names, values là list of (step, value) tuples
    """
    logs = {}
    
    # First try to load from JSON (for sample data)
    json_files = list(log_dir.rglob("*.json"))
    if json_files:
        for json_file in json_files:
            if "training_curves" in json_file.name or "logs" in json_file.name:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert to expected format if needed
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], list) and len(value[0]) == 2:
                                logs[key] = [(v[0], v[1]) for v in value]
                            elif isinstance(value[0], dict):
                                logs[key] = [(v.get("step", v.get("epoch", 0)), v.get("value", v.get("loss", 0))) for v in value]
                    print(f"✅ Loaded logs from JSON: {json_file}")
                    return logs
    
    # Try TensorBoard event files
    if not TENSORBOARD_AVAILABLE:
        return {}
    
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"⚠️  Không tìm thấy TensorBoard logs trong {log_dir}")
        return {}
    
    # Load từ event file đầu tiên (hoặc có thể merge nhiều files)
    event_file = event_files[0]
    ea = EventAccumulator(str(event_file.parent))
    ea.Reload()
    
    # Lấy tất cả scalar tags
    scalar_tags = ea.Tags()['scalars']
    
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        logs[tag] = [(e.step, e.value) for e in scalar_events]
    
    return logs


def load_json_results(results_dir: Path) -> Dict[str, Any]:
    """Load kết quả evaluation từ JSON files."""
    results = {}
    
    json_files = list(results_dir.glob("*.json"))
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Merge data directly into results (not nested under filename)
            if isinstance(data, dict):
                results.update(data)
            else:
                results[json_file.stem] = data
    
    return results


# ==================== FIGURE GENERATORS ====================

def plot_training_curves(
    logs: Dict[str, List[Tuple[float, float]]],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ training curves: loss và accuracy qua các epochs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Training Curves" if language == "en" else "Đường cong Huấn luyện",
        fontsize=16,
        fontweight="bold",
    )
    
    # 1. Total Loss
    ax1 = axes[0, 0]
    if "Train/Epoch_Loss" in logs:
        steps, values = zip(*logs["Train/Epoch_Loss"])
        ax1.plot(steps, values, label="Train", linewidth=2, color="blue")
    if "Val/Loss" in logs:
        steps, values = zip(*logs["Val/Loss"])
        ax1.plot(steps, values, label="Validation", linewidth=2, color="red", linestyle="--")
    ax1.set_xlabel("Epoch" if language == "en" else "Epoch")
    ax1.set_ylabel("Loss" if language == "en" else "Loss")
    ax1.set_title("Total Loss" if language == "en" else "Tổng Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Classification Loss
    ax2 = axes[0, 1]
    if "Train/Epoch_CLS_Loss" in logs:
        steps, values = zip(*logs["Train/Epoch_CLS_Loss"])
        ax2.plot(steps, values, label="Train", linewidth=2, color="blue")
    if "Val/CLS_Loss" in logs:
        steps, values = zip(*logs["Val/CLS_Loss"])
        ax2.plot(steps, values, label="Validation", linewidth=2, color="red", linestyle="--")
    ax2.set_xlabel("Epoch" if language == "en" else "Epoch")
    ax2.set_ylabel("Loss" if language == "en" else "Loss")
    ax2.set_title("Classification Loss" if language == "en" else "Loss Phân loại")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. BBox Loss
    ax3 = axes[1, 0]
    if "Train/Epoch_BBox_Loss" in logs:
        steps, values = zip(*logs["Train/Epoch_BBox_Loss"])
        ax3.plot(steps, values, label="Train", linewidth=2, color="blue")
    if "Val/BBox_Loss" in logs:
        steps, values = zip(*logs["Val/BBox_Loss"])
        ax3.plot(steps, values, label="Validation", linewidth=2, color="red", linestyle="--")
    ax3.set_xlabel("Epoch" if language == "en" else "Epoch")
    ax3.set_ylabel("Loss" if language == "en" else "Loss")
    ax3.set_title("Bounding Box Loss" if language == "en" else "Loss Bounding Box")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy
    ax4 = axes[1, 1]
    if "Val/Accuracy" in logs:
        steps, values = zip(*logs["Val/Accuracy"])
        ax4.plot(steps, values, label="Validation", linewidth=2, color="green", marker="o", markersize=4)
    ax4.set_xlabel("Epoch" if language == "en" else "Epoch")
    ax4.set_ylabel("Accuracy" if language == "en" else "Độ chính xác")
    ax4.set_title("Validation Accuracy" if language == "en" else "Độ chính xác Validation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ confusion matrix.
    
    Args:
        confusion_matrix: (N, N) numpy array
        class_names: List of class names
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype("float") / (
        confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6
    )
    
    # Plot
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix" if language == "en" else "Ma trận Nhầm lẫn",
        ylabel="True Label" if language == "en" else "Nhãn Thực tế",
        xlabel="Predicted Label" if language == "en" else "Nhãn Dự đoán",
    )
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]}\n({cm_normalized[i, j]:.2%})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=8,
            )
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_precision_recall_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    ap: float,
    output_path: Path,
    class_name: Optional[str] = None,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ Precision-Recall curve.
    
    Args:
        precisions: Array of precision values
        recalls: Array of recall values
        ap: Average Precision score
        class_name: Optional class name for title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recalls, precisions, linewidth=2, color="blue", label=f"AP = {ap:.3f}")
    ax.fill_between(recalls, precisions, alpha=0.2, color="blue")
    
    ax.set_xlabel("Recall" if language == "en" else "Recall")
    ax.set_ylabel("Precision" if language == "en" else "Precision")
    title = f"Precision-Recall Curve"
    if class_name:
        title += f" - {class_name}"
    ax.set_title(title if language == "en" else f"Đường cong Precision-Recall - {class_name or ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_map_metrics(
    map_results: Dict[str, float],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ mAP metrics ở các IoU thresholds khác nhau.
    
    Args:
        map_results: Dict với keys như "mAP@0.25", "mAP@0.50", "mAP@0.75", "mAP"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract IoU thresholds and mAP values
    thresholds = []
    map_values = []
    
    for key, value in map_results.items():
        if key.startswith("mAP@") and key != "mAP":
            threshold = float(key.split("@")[1])
            thresholds.append(threshold)
            map_values.append(value)
    
    # Sort by threshold
    sorted_pairs = sorted(zip(thresholds, map_values))
    thresholds, map_values = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # Plot bars
    bars = ax.bar(
        [f"IoU@{t:.2f}" for t in thresholds],
        map_values,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    
    # Add value labels on bars
    for bar, val in zip(bars, map_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    # Add overall mAP if available
    if "mAP" in map_results:
        ax.axhline(
            y=map_results["mAP"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"mAP (mean) = {map_results['mAP']:.3f}",
        )
        ax.legend()
    
    ax.set_ylabel("mAP" if language == "en" else "mAP", fontsize=12)
    ax.set_xlabel("IoU Threshold" if language == "en" else "Ngưỡng IoU", fontsize=12)
    ax.set_title(
        "mAP at Different IoU Thresholds" if language == "en" else "mAP tại các Ngưỡng IoU khác nhau",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_iou_distribution(
    iou_values: np.ndarray,
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ phân phối IoU values.
    
    Args:
        iou_values: Array of IoU values (0-1)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(iou_values, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(
        np.mean(iou_values),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {np.mean(iou_values):.3f}",
    )
    ax1.axvline(
        np.median(iou_values),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(iou_values):.3f}",
    )
    ax1.set_xlabel("IoU" if language == "en" else "IoU")
    ax1.set_ylabel("Frequency" if language == "en" else "Tần suất")
    ax1.set_title("IoU Distribution" if language == "en" else "Phân phối IoU")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(
        [iou_values],
        vert=True,
        patch_artist=True,
        tick_labels=["IoU"],
        showmeans=True,
    )
    bp["boxes"][0].set_facecolor("lightblue")
    ax2.set_ylabel("IoU" if language == "en" else "IoU")
    ax2.set_title("IoU Box Plot" if language == "en" else "Biểu đồ Box IoU")
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_performance_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ so sánh hiệu năng giữa các models (ví dụ: SNN vs ANN).
    
    Args:
        comparison_data: Dict với structure:
            {
                "Model1": {"mAP": 0.75, "Accuracy": 0.85, "FPS": 30, ...},
                "Model2": {"mAP": 0.72, "Accuracy": 0.82, "FPS": 25, ...},
            }
    """
    models = list(comparison_data.keys())
    metrics = list(comparison_data[models[0]].keys())
    
    n_metrics = len(metrics)
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [comparison_data[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison" if language == "en" else f"So sánh {metric}", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_latency_fps(
    latency_data: Dict[str, List[float]],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """
    Vẽ latency và FPS benchmarks.
    
    Args:
        latency_data: Dict với structure:
            {
                "Model1": [latency1, latency2, ...],
                "Model2": [latency2, latency2, ...],
            }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(latency_data.keys())
    latencies = [latency_data[model] for model in models]
    fps_values = [[1000 / lat for lat in lats] for lats in latencies]  # Convert to FPS
    
    # Latency box plot
    ax1 = axes[0]
    bp1 = ax1.boxplot(latencies, tick_labels=models, patch_artist=True, showmeans=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("lightcoral")
    ax1.set_ylabel("Latency (ms)" if language == "en" else "Độ trễ (ms)")
    ax1.set_title("Latency Comparison" if language == "en" else "So sánh Độ trễ")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # FPS box plot
    ax2 = axes[1]
    bp2 = ax2.boxplot(fps_values, tick_labels=models, patch_artist=True, showmeans=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("lightgreen")
    ax2.set_ylabel("FPS" if language == "en" else "FPS")
    ax2.set_title("FPS Comparison" if language == "en" else "So sánh FPS")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_precision_recall_curve_from_data(
    pr_data: Dict[str, Any],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ Precision-Recall curve từ dữ liệu."""
    recalls = np.array(pr_data["recalls"])
    precisions = np.array(pr_data["precisions"])
    ap = pr_data.get("ap", 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, linewidth=2, color="blue", label=f"AP = {ap:.3f}")
    ax.fill_between(recalls, precisions, alpha=0.2, color="blue")
    ax.set_xlabel("Recall" if language == "en" else "Recall")
    ax.set_ylabel("Precision" if language == "en" else "Precision")
    ax.set_title("Precision-Recall Curve" if language == "en" else "Đường cong Precision-Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_roc_curve(
    roc_data: Dict[str, Any],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ ROC curve."""
    fpr = np.array(roc_data["fpr"])
    tpr = np.array(roc_data["tpr"])
    auc = roc_data.get("auc", 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, color="darkorange", label=f"AUC = {auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.2, color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate" if language == "en" else "Tỷ lệ Dương tính Giả")
    ax.set_ylabel("True Positive Rate" if language == "en" else "Tỷ lệ Dương tính Đúng")
    ax.set_title("ROC Curve" if language == "en" else "Đường cong ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_per_class_map(
    per_class_map: Dict[str, float],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ mAP theo từng class."""
    classes = list(per_class_map.keys())
    values = list(per_class_map.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
    bars = ax.barh(classes, values, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", fontsize=9, fontweight="bold")
    
    ax.set_xlabel("mAP" if language == "en" else "mAP")
    ax.set_ylabel("Class" if language == "en" else "Lớp")
    ax.set_title("Per-class mAP" if language == "en" else "mAP theo từng lớp")
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_ablation_study(
    ablation_data: Dict[str, float],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ kết quả ablation study."""
    configs = list(ablation_data.keys())
    values = list(ablation_data.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["green" if v == max(values) else "steelblue" for v in values]
    bars = ax.bar(
        range(len(configs)),
        values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylabel("mAP" if language == "en" else "mAP")
    ax.set_title("Ablation Study" if language == "en" else "Nghiên cứu Ablation")
    ax.grid(True, alpha=0.3, axis="y")
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_error_analysis(
    error_data: Dict[str, int],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ phân tích lỗi (TP, FP, FN, TN)."""
    labels = list(error_data.keys())
    sizes = list(error_data.values())
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]
    explode = (0.1, 0, 0, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    ax1 = axes[0]
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors[:len(labels)],
            autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.set_title("Error Distribution" if language == "en" else "Phân bố Lỗi")
    
    # Bar chart
    ax2 = axes[1]
    bars = ax2.bar(labels, sizes, color=colors[:len(labels)], alpha=0.8)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
                str(val), ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count" if language == "en" else "Số lượng")
    ax2.set_title("Classification Results" if language == "en" else "Kết quả Phân loại")
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_iou_vs_distance(
    iou_data: Dict[str, Any],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ IoU theo khoảng cách (độ sâu)."""
    labels = iou_data["depth_ranges"]
    values = iou_data["iou_values"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="teal", alpha=0.7, edgecolor="black")
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    
    ax.set_xlabel("Depth Range" if language == "en" else "Khoảng cách (depth)")
    ax.set_ylabel("Mean IoU" if language == "en" else "IoU trung bình")
    ax.set_title("IoU vs Distance" if language == "en" else "IoU theo Khoảng cách")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


def plot_jetson_metrics(
    jetson_data: Dict[str, Dict[str, float]],
    output_path: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Vẽ so sánh metrics trên Jetson (Power, Memory, Temperature)."""
    models = list(jetson_data.keys())
    metrics = ["Power_W", "Memory_GB", "Temp_C", "FPS"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Jetson AGX Orin Performance" if language == "en" else "Hiệu năng Jetson AGX Orin",
        fontsize=16,
        fontweight="bold",
    )
    
    metric_names = {
        "Power_W": "Power (W)" if language == "en" else "Công suất (W)",
        "Memory_GB": "Memory (GB)" if language == "en" else "Bộ nhớ (GB)",
        "Temp_C": "Temperature (°C)" if language == "en" else "Nhiệt độ (°C)",
        "FPS": "FPS" if language == "en" else "FPS",
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [jetson_data[m].get(metric, 0) for m in models]
        bars = ax.bar(models, values, color=["#2ecc71", "#e74c3c"][:len(models)], alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel(metric_names.get(metric, metric))
        ax.set_title(metric_names.get(metric, metric))
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo: {output_path}")


# ==================== MAIN FUNCTION ====================

def generate_all_figures(
    log_dir: Path,
    results_dir: Path,
    output_dir: Path,
    language: str = "vi",
    format: str = "png",
    dpi: int = 300,
):
    """Generate tất cả các hình đánh giá."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Đang tạo các hình đánh giá...")
    print(f"📂 Output directory: {output_dir}")
    
    # Load data
    print("\n1️⃣  Đang load dữ liệu...")
    logs = load_tensorboard_logs(log_dir)
    results = load_json_results(results_dir)
    
    # Generate figures
    print("\n2️⃣  Đang tạo các hình...")
    
    # 1. Training curves
    if logs:
        plot_training_curves(
            logs,
            output_dir / f"training_curves.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    else:
        print("⚠️  Không có training logs, bỏ qua training curves")
    
    # 2. mAP metrics (nếu có trong results)
    if "map_results" in results or any("map" in k.lower() for k in results.keys()):
        map_data = results.get("map_results", {})
        if not map_data:
            # Try to find any map-related data
            for key, value in results.items():
                if "map" in key.lower() and isinstance(value, dict):
                    map_data = value
                    break
        
        if map_data:
            plot_map_metrics(
                map_data,
                output_dir / f"map_metrics.{format}",
                language=language,
                format=format,
                dpi=dpi,
            )
    
    # 3. IoU distribution (nếu có)
    if "iou_distribution" in results:
        iou_values = np.array(results["iou_distribution"])
        plot_iou_distribution(
            iou_values,
            output_dir / f"iou_distribution.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 4. Performance comparison (nếu có)
    if "performance_comparison" in results:
        comparison_data = results["performance_comparison"]
        plot_performance_comparison(
            comparison_data,
            output_dir / f"performance_comparison.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 5. Latency/FPS (nếu có)
    if "latency_benchmark" in results:
        latency_data = results["latency_benchmark"]
        plot_latency_fps(
            latency_data,
            output_dir / f"latency_fps.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 6. Confusion matrix (nếu có)
    if "confusion_matrix" in results:
        cm_data = results["confusion_matrix"]
        cm = np.array(cm_data["matrix"])
        class_names = cm_data.get("class_names", [f"Class {i}" for i in range(len(cm))])
        plot_confusion_matrix(
            cm,
            class_names,
            output_dir / f"confusion_matrix.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 7. Precision-Recall curve
    if "pr_curve" in results:
        plot_precision_recall_curve_from_data(
            results["pr_curve"],
            output_dir / f"precision_recall_curve.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 8. ROC curve
    if "roc_curve" in results:
        plot_roc_curve(
            results["roc_curve"],
            output_dir / f"roc_curve.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 9. Per-class mAP
    if "per_class_map" in results:
        plot_per_class_map(
            results["per_class_map"],
            output_dir / f"per_class_map.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 10. Ablation study
    if "ablation_study" in results:
        plot_ablation_study(
            results["ablation_study"],
            output_dir / f"ablation_study.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 11. Error analysis
    if "error_analysis" in results:
        plot_error_analysis(
            results["error_analysis"],
            output_dir / f"error_analysis.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 12. IoU vs Distance
    if "iou_vs_distance" in results:
        plot_iou_vs_distance(
            results["iou_vs_distance"],
            output_dir / f"iou_vs_distance.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    # 13. Jetson metrics
    if "jetson_metrics" in results:
        plot_jetson_metrics(
            results["jetson_metrics"],
            output_dir / f"jetson_metrics.{format}",
            language=language,
            format=format,
            dpi=dpi,
        )
    
    print(f"\n✅ Hoàn thành! Tất cả hình đã được lưu tại: {output_dir}")


def main():
    args = parse_args()
    
    generate_all_figures(
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        language=args.language,
        format=args.format,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
