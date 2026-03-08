#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script fix lỗi pip trong conda environment.

Lỗi thường gặp:
- "Unable to create process using 'python.exe pip-script.py'"
- Thường xảy ra khi đường dẫn có khoảng trắng hoặc pip bị hỏng

Usage:
    python scripts/fix_pip_conda.py
"""

import sys
import subprocess
import os
from pathlib import Path


def fix_pip_conda() -> None:
    """Fix lỗi pip trong conda environment."""
    print("🔧 Đang fix lỗi pip trong conda environment...")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Kiểm tra xem đang trong conda env không
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        print(f"✅ Đang trong conda environment: {conda_env}")
    else:
        print("⚠️  Không phát hiện conda environment")
    
    # Method 1: Reinstall pip
    print("\n📦 Method 1: Reinstall pip...")
    try:
        # Download get-pip.py
        import urllib.request
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_path = Path.cwd() / "get-pip.py"
        
        print(f"   Đang tải get-pip.py từ {get_pip_url}...")
        urllib.request.urlretrieve(get_pip_url, get_pip_path)
        
        print("   Đang reinstall pip...")
        result = subprocess.run(
            [sys.executable, str(get_pip_path), "--force-reinstall"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Reinstall pip thành công!")
        else:
            print(f"   ⚠️  Reinstall pip có warning: {result.stderr}")
        
        # Xóa file tạm
        if get_pip_path.exists():
            get_pip_path.unlink()
            
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
    
    # Method 2: Fix pip script
    print("\n🔧 Method 2: Fix pip script...")
    try:
        # Tìm pip script
        scripts_dir = Path(sys.executable).parent.parent / "Scripts"
        pip_script = scripts_dir / "pip-script.py"
        
        if pip_script.exists():
            print(f"   Tìm thấy pip-script.py tại: {pip_script}")
            # Đọc nội dung
            content = pip_script.read_text(encoding="utf-8")
            
            # Kiểm tra xem có vấn đề với đường dẫn không
            if '"' in str(sys.executable) or ' ' in str(sys.executable):
                print("   ⚠️  Đường dẫn Python có khoảng trắng hoặc dấu ngoặc kép")
                print("   💡 Khuyến nghị: Tạo conda env mới ở đường dẫn không có khoảng trắng")
        else:
            print(f"   ⚠️  Không tìm thấy pip-script.py tại: {pip_script}")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
    
    # Method 3: Dùng python -m pip thay vì pip
    print("\n💡 Method 3: Sử dụng 'python -m pip' thay vì 'pip'")
    print("   Thay vì: pip install package")
    print("   Dùng:    python -m pip install package")
    
    # Test pip
    print("\n🧪 Test pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"   ✅ Pip hoạt động tốt!")
            print(f"   {result.stdout.strip()}")
        else:
            print(f"   ❌ Pip vẫn có vấn đề: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Lỗi khi test pip: {e}")
    
    print("\n" + "="*60)
    print("📋 Hướng dẫn tiếp theo:")
    print("="*60)
    print("1. Thử cài đặt lại với:")
    print("   python -m pip install datasets huggingface-hub")
    print()
    print("2. Nếu vẫn lỗi, tạo conda env mới:")
    print("   conda create -n new_env python=3.10")
    print("   conda activate new_env")
    print("   python -m pip install -r requirements.txt")
    print()
    print("3. Hoặc dùng venv thay vì conda:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate  # Windows")
    print("   python -m pip install -r requirements.txt")


if __name__ == "__main__":
    fix_pip_conda()
