#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tải dataset text-segmentation từ Hugging Face.

Dataset: VSMRC/text-segmentation
Yêu cầu: Cần login Hugging Face trước khi chạy
    - Chạy: huggingface-cli login
    - Hoặc set token: export HF_TOKEN=your_token

Usage:
    python scripts/data_collection/download_text_segmentation.py
    python scripts/data_collection/download_text_segmentation.py --output-dir data/03_external/text_segmentation
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    from huggingface_hub import login, whoami
except ImportError as e:
    print(f"❌ Lỗi: Thiếu thư viện cần thiết. Cài đặt với:")
    print(f"   pip install datasets huggingface-hub")
    sys.exit(1)

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_huggingface_login() -> bool:
    """
    Kiểm tra xem đã login Hugging Face chưa.

    Returns:
        True nếu đã login, False nếu chưa
    """
    try:
        user_info = whoami()
        if user_info:
            print(f"✅ Đã login Hugging Face với user: {user_info.get('name', 'Unknown')}")
            return True
    except Exception:
        pass
    
    print("⚠️  Chưa login Hugging Face!")
    print("   Chạy lệnh sau để login:")
    print("   huggingface-cli login")
    print("   hoặc")
    print("   export HF_TOKEN=your_token_here")
    return False


def download_text_segmentation_dataset(
    output_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    split: Optional[str] = None,
) -> None:
    """
    Tải dataset text-segmentation từ Hugging Face.

    Args:
        output_dir: Thư mục lưu dataset (nếu None, dùng default)
        cache_dir: Thư mục cache cho datasets library
        split: Split cần tải (train, test, validation) hoặc None để tải tất cả
    """
    dataset_name = "VSMRC/text-segmentation"
    
    # Kiểm tra login
    if not check_huggingface_login():
        print("\n❌ Cần login Hugging Face để truy cập dataset này!")
        print("   Dataset này yêu cầu authentication.")
        sys.exit(1)
    
    # Setup output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "03_external" / "text_segmentation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup cache directory
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data" / ".cache" / "huggingface"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Đang tải dataset: {dataset_name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Cache directory: {cache_dir}")
    
    try:
        # Tải dataset
        if split:
            print(f"📊 Tải split: {split}")
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(cache_dir),
                split=split,
            )
        else:
            print("📊 Tải tất cả splits...")
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(cache_dir),
            )
        
        print(f"\n✅ Đã tải dataset thành công!")
        
        # In thông tin dataset
        if isinstance(dataset, dict):
            print(f"\n📋 Dataset có {len(dataset)} splits:")
            for split_name, split_data in dataset.items():
                print(f"   - {split_name}: {len(split_data)} samples")
                if len(split_data) > 0:
                    print(f"     Features: {list(split_data[0].keys())}")
        else:
            print(f"\n📋 Dataset có {len(dataset)} samples")
            if len(dataset) > 0:
                print(f"   Features: {list(dataset[0].keys())}")
        
        # Lưu dataset vào disk (optional - nếu cần)
        print(f"\n💾 Dataset đã được cache tại: {cache_dir}")
        print(f"   Để sử dụng, load lại với:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{dataset_name}', cache_dir='{cache_dir}')")
        
        # Lưu metadata
        metadata_file = output_dir / "dataset_info.txt"
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Cache directory: {cache_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            if isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    f.write(f"\nSplit: {split_name}\n")
                    f.write(f"  Samples: {len(split_data)}\n")
                    if len(split_data) > 0:
                        f.write(f"  Features: {list(split_data[0].keys())}\n")
            else:
                f.write(f"\nSamples: {len(dataset)}\n")
                if len(dataset) > 0:
                    f.write(f"Features: {list(dataset[0].keys())}\n")
        
        print(f"📝 Metadata đã lưu tại: {metadata_file}")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi tải dataset: {e}")
        print(f"\n💡 Gợi ý:")
        print(f"   1. Kiểm tra lại đã login Hugging Face chưa")
        print(f"   2. Kiểm tra quyền truy cập dataset")
        print(f"   3. Kiểm tra kết nối internet")
        sys.exit(1)


def main() -> None:
    """Hàm main."""
    parser = argparse.ArgumentParser(
        description="Tải dataset text-segmentation từ Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tải tất cả splits
  python scripts/data_collection/download_text_segmentation.py
  
  # Tải chỉ train split
  python scripts/data_collection/download_text_segmentation.py --split train
  
  # Chỉ định output directory
  python scripts/data_collection/download_text_segmentation.py --output-dir data/my_dataset
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục lưu metadata (default: data/03_external/text_segmentation)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Thư mục cache cho datasets library (default: data/.cache/huggingface)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "test", "validation"],
        help="Split cần tải (train/test/validation) hoặc None để tải tất cả"
    )
    
    args = parser.parse_args()
    
    download_text_segmentation_dataset(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        split=args.split,
    )


if __name__ == "__main__":
    main()
