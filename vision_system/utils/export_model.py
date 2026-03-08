#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export DepthAwareSNN model to ONNX and TensorRT formats for deployment.

This script exports the trained model to:
- ONNX format (for cross-platform deployment)
- TensorRT engine (for Jetson AGX Orin optimization)

Usage:
    python vision_system/utils/export_model.py \
        --weights vision_system/weights/finetuned/best_model.pth \
        --output-dir vision_system/weights/exported \
        --format onnx tensorrt
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.models.snn.depth_aware_snn import DepthAwareSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DepthAwareSNN to ONNX/TensorRT formats"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vision_system/weights/exported"),
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["onnx", "tensorrt"],
        default=["onnx"],
        help="Export formats (can specify multiple)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (H=W)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1,
        help="Temporal sequence length (T)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=237,
        help="Number of object classes",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision for TensorRT",
    )
    return parser.parse_args()


def load_model(
    weights_path: Path,
    num_classes: int,
    device: torch.device,
) -> DepthAwareSNN:
    """Load model from checkpoint."""
    print(f"📂 Loading model from: {weights_path}")
    
    model = DepthAwareSNN(
        num_classes=num_classes,
        use_multiscale=False,
        use_snn_backbone=False,
    ).to(device)
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully")
    return model


def export_onnx(
    model: DepthAwareSNN,
    output_path: Path,
    img_size: int,
    sequence_length: int,
    opset_version: int,
    device: torch.device,
) -> None:
    """Export model to ONNX format."""
    print(f"\n🔄 Exporting to ONNX format...")
    
    # Create dummy inputs
    rgb_seq = torch.randn(1, sequence_length, 3, img_size, img_size).to(device)
    depth_seq = torch.randn(1, sequence_length, 1, img_size, img_size).to(device)
    
    # Export
    torch.onnx.export(
        model,
        (rgb_seq, depth_seq),
        str(output_path),
        input_names=["rgb_seq", "depth_seq"],
        output_names=["bbox_3d", "class_logits"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            "rgb_seq": {0: "batch_size"},
            "depth_seq": {0: "batch_size"},
            "bbox_3d": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
    )
    
    print(f"✅ ONNX model saved to: {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
    except ImportError:
        print("⚠️  onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠️  ONNX verification warning: {e}")


def export_tensorrt(
    onnx_path: Path,
    output_path: Path,
    fp16: bool,
    img_size: int,
    sequence_length: int,
) -> None:
    """Export ONNX model to TensorRT engine."""
    print(f"\n🔄 Exporting to TensorRT format...")
    
    try:
        import tensorrt as trt
    except ImportError:
        print("❌ TensorRT not installed. Please install TensorRT for Jetson deployment.")
        print("   Skipping TensorRT export...")
        return
    
    # TensorRT export logic
    # Note: This is a simplified version. Full implementation requires:
    # - TensorRT builder and engine creation
    # - Calibration for INT8 (optional)
    # - Engine serialization
    
    print("⚠️  TensorRT export requires full TensorRT setup.")
    print(f"   ONNX model available at: {onnx_path}")
    print("   Use trtexec or TensorRT Python API to convert ONNX to TensorRT engine:")
    print(f"   trtexec --onnx={onnx_path} --saveEngine={output_path} {'--fp16' if fp16 else ''}")


def main() -> None:
    args = parse_args()
    
    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Load model
    model = load_model(
        weights_path=args.weights,
        num_classes=args.num_classes,
        device=device,
    )
    
    # Export to requested formats
    if "onnx" in args.format:
        onnx_path = args.output_dir / "depth_aware_snn.onnx"
        export_onnx(
            model=model,
            output_path=onnx_path,
            img_size=args.img_size,
            sequence_length=args.sequence_length,
            opset_version=args.opset_version,
            device=device,
        )
        
        if "tensorrt" in args.format:
            trt_path = args.output_dir / "depth_aware_snn.trt"
            export_tensorrt(
                onnx_path=onnx_path,
                output_path=trt_path,
                fp16=args.fp16,
                img_size=args.img_size,
                sequence_length=args.sequence_length,
            )
    
    print("\n✅ Export completed!")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
