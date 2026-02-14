#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: hilo_inspect.py
Mục đích: Kiểm tra cấu trúc thư mục HILO, đếm số scene và số ảnh.
Giải thích:
- Script này dùng để xem nhanh thông tin dataset HILO sau khi tải về.
- Nó duyệt qua tất cả các scene, đọc file camera_poses.json để ước lượng số lượng ảnh.
- In ra một scene mẫu để kiểm tra đường dẫn.
"""

import sys
from pathlib import Path

# Thêm đường dẫn gốc của project vào sys.path để có thể import các module từ src
# __file__ là đường dẫn tuyệt đối của file hiện tại
# parents[2] trỏ lên thư mục gốc (multi_modal_robot_ai)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các module cần thiết từ src.common
from src.common.hilo_dataset import (
    HILODatasetLoader,
    get_default_hilo_scenes_root,
)


def main() -> None:
    """
    Hàm chính:
    - Lấy đường dẫn mặc định chứa các scene HILO (data/00_raw/HILO/HILO_Scenes).
    - Tạo loader để duyệt các scene.
    - Đếm số scene và tổng số ảnh (dựa trên camera_poses.json).
    - In ra thông tin của scene đầu tiên để kiểm tra.
    """
    scenes_root = get_default_hilo_scenes_root()
    loader = HILODatasetLoader(scenes_root)

    num_scenes = 0
    total_images = 0
    example_scene = None

    for scene in loader.iter_scenes():
        num_scenes += 1
        if example_scene is None:
            example_scene = scene

        camera_poses = loader.load_camera_poses(scene)
        total_images += len(camera_poses)

    print(f"HILO_Scenes root: {scenes_root}")
    print(f"Số scene: {num_scenes}")
    print(f"Tổng số ảnh (ước lượng từ camera_poses): {total_images}")

    if example_scene is not None:
        print("\nVí dụ một scene:")
        print(f"  Tên: {example_scene.name}")
        print(f"  Đường dẫn gốc: {example_scene.root}")
        print(f"  Thư mục RGB raw: {example_scene.rgb_raw_dir}")
        print(f"  Thư mục RGB undistorted: {example_scene.rgb_undistorted_dir}")


if __name__ == "__main__":
    main()