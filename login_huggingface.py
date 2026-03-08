#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hỗ trợ login Hugging Face.

Chạy: python login_huggingface.py
"""

import sys
import subprocess
import os
from pathlib import Path

try:
    from huggingface_hub import login, whoami
except ImportError:
    print("❌ Chưa cài huggingface-hub!")
    print("   Cài đặt: python -m pip install huggingface-hub")
    sys.exit(1)


def check_login() -> bool:
    """Kiểm tra đã login chưa."""
    try:
        user_info = whoami()
        if user_info:
            print(f"✅ Đã login với user: {user_info.get('name', 'Unknown')}")
            print(f"   Email: {user_info.get('email', 'N/A')}")
            return True
    except Exception:
        pass
    
    print("❌ Chưa login Hugging Face")
    return False


def login_with_token() -> bool:
    """Login bằng token."""
    print("\n🔑 Login bằng Token")
    print("="*60)
    
    # Kiểm tra token trong environment
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"✅ Tìm thấy HF_TOKEN trong environment")
        try:
            login(token=token)
            print("✅ Login thành công!")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi login: {e}")
            return False
    
    # Nhập token từ user
    print("\n💡 Cách lấy token:")
    print("   1. Vào https://huggingface.co/settings/tokens")
    print("   2. Tạo token mới (Read permission là đủ)")
    print("   3. Copy token")
    print()
    
    token = input("Nhập Hugging Face token (hoặc Enter để bỏ qua): ").strip()
    
    if not token:
        print("❌ Không có token, bỏ qua")
        return False
    
    try:
        login(token=token)
        print("✅ Login thành công!")
        
        # Lưu token vào environment (tùy chọn)
        save = input("\nLưu token vào environment? (y/n): ").strip().lower()
        if save == 'y':
            print("\n💡 Để lưu token vĩnh viễn:")
            print("   Windows (CMD):")
            print(f"   setx HF_TOKEN {token}")
            print("   Windows (PowerShell):")
            print(f"   [System.Environment]::SetEnvironmentVariable('HF_TOKEN', '{token}', 'User')")
            print("   Linux/Mac:")
            print(f"   export HF_TOKEN={token}")
            print("   (Thêm vào ~/.bashrc hoặc ~/.zshrc để lưu vĩnh viễn)")
        
        return True
    except Exception as e:
        print(f"❌ Lỗi khi login: {e}")
        return False


def login_with_cli() -> bool:
    """Login bằng huggingface-cli."""
    print("\n🔧 Login bằng huggingface-cli")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["huggingface-cli", "login"],
            input="",  # Interactive
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Login thành công!")
            return True
        else:
            print("❌ Login thất bại")
            return False
    except FileNotFoundError:
        print("❌ huggingface-cli không tìm thấy")
        print("   Cài đặt: python -m pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def main():
    """Hàm main."""
    print("="*60)
    print("🤗 Hugging Face Login Helper")
    print("="*60)
    print()
    
    # Kiểm tra đã login chưa
    if check_login():
        print("\n✅ Bạn đã login rồi! Không cần làm gì thêm.")
        return 0
    
    print("\n📋 Chọn phương thức login:")
    print("   1. Login bằng token (khuyến nghị)")
    print("   2. Login bằng huggingface-cli")
    print("   3. Hướng dẫn manual")
    print("   0. Thoát")
    print()
    
    choice = input("Chọn (1/2/3/0): ").strip()
    
    if choice == "1":
        success = login_with_token()
    elif choice == "2":
        success = login_with_cli()
    elif choice == "3":
        print("\n📖 Hướng dẫn Manual:")
        print("="*60)
        print("1. Vào https://huggingface.co/settings/tokens")
        print("2. Tạo token mới với Read permission")
        print("3. Copy token")
        print("4. Chạy một trong các lệnh sau:")
        print()
        print("   Windows (CMD):")
        print("   set HF_TOKEN=your_token_here")
        print()
        print("   Windows (PowerShell):")
        print("   $env:HF_TOKEN='your_token_here'")
        print()
        print("   Linux/Mac:")
        print("   export HF_TOKEN=your_token_here")
        print()
        print("5. Chạy lại script này hoặc:")
        print("   python scripts/data_collection/download_text_segmentation.py")
        return 0
    elif choice == "0":
        return 0
    else:
        print("❌ Lựa chọn không hợp lệ")
        return 1
    
    if success:
        # Test lại
        print("\n🧪 Test login...")
        if check_login():
            print("\n✅ Hoàn tất! Bạn có thể tải dataset bây giờ:")
            print("   python scripts/data_collection/download_text_segmentation.py")
            return 0
        else:
            print("\n⚠️  Login có vẻ không thành công, thử lại")
            return 1
    else:
        print("\n❌ Login thất bại. Thử lại hoặc xem hướng dẫn manual (chọn 3)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
