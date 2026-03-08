#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script cài đặt dependencies với workaround cho lỗi pip.

Sử dụng python -m pip thay vì pip trực tiếp để tránh lỗi
"Unable to create process" trong conda environment.

Usage:
    python scripts/install_dependencies.py
    python scripts/install_dependencies.py --file requirements.txt
    python scripts/install_dependencies.py --packages datasets huggingface-hub
"""

import sys
import subprocess
import argparse
from pathlib import Path


def install_package(package: str) -> bool:
    """
    Cài đặt một package sử dụng python -m pip.
    
    Args:
        package: Tên package cần cài
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    print(f"📦 Đang cài đặt: {package}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ Đã cài đặt: {package}")
            return True
        else:
            print(f"   ❌ Lỗi khi cài {package}:")
            print(f"   {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Timeout khi cài {package}")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def install_from_file(requirements_file: Path) -> bool:
    """
    Cài đặt từ requirements file.
    
    Args:
        requirements_file: Đường dẫn đến requirements file
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    if not requirements_file.exists():
        print(f"❌ File không tồn tại: {requirements_file}")
        return False
    
    print(f"📄 Đang đọc requirements từ: {requirements_file}")
    
    # Đọc packages từ file
    packages = []
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Bỏ qua comments và empty lines
            if line and not line.startswith("#"):
                # Lấy package name (bỏ version constraints nếu cần)
                package = line.split("#")[0].strip()
                if package:
                    packages.append(package)
    
    print(f"📦 Tìm thấy {len(packages)} packages")
    
    # Cài đặt từng package
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Kết quả: {success_count}/{len(packages)} packages đã cài đặt thành công")
    return success_count == len(packages)


def main() -> None:
    """Hàm main."""
    parser = argparse.ArgumentParser(
        description="Cài đặt dependencies với workaround cho lỗi pip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Cài từ requirements.txt
  python scripts/install_dependencies.py
  
  # Cài từ file cụ thể
  python scripts/install_dependencies.py --file requirements-dev.txt
  
  # Cài packages cụ thể
  python scripts/install_dependencies.py --packages datasets huggingface-hub
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default="requirements.txt",
        help="Requirements file (default: requirements.txt)"
    )
    
    parser.add_argument(
        "--packages",
        nargs="+",
        help="List of packages to install"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Cài đặt Dependencies")
    print("="*60)
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print()
    
    success = False
    
    if args.packages:
        # Cài packages cụ thể
        print(f"📦 Cài đặt {len(args.packages)} packages...")
        success_count = 0
        for package in args.packages:
            if install_package(package):
                success_count += 1
        success = success_count == len(args.packages)
    else:
        # Cài từ requirements file
        requirements_file = Path(args.file)
        if not requirements_file.is_absolute():
            # Tìm trong project root
            project_root = Path(__file__).resolve().parents[1]
            requirements_file = project_root / args.file
        
        success = install_from_file(requirements_file)
    
    print("\n" + "="*60)
    if success:
        print("✅ Hoàn tất! Tất cả packages đã được cài đặt.")
    else:
        print("⚠️  Một số packages có thể chưa được cài đặt.")
        print("   Kiểm tra lại output ở trên để xem chi tiết.")
    print("="*60)


if __name__ == "__main__":
    main()
