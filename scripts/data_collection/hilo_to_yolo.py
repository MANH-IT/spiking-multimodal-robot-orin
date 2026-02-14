#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: hilo_to_yolo.py
Mục đích: Chuyển đổi annotation từ file JSON (đã tạo bởi hilo_make_annotations.py)
sang định dạng YOLO (tệp .txt với mỗi dòng: class_id x_center y_center width height).

Tác giả: [Tên bạn]
Ngày: 2024

Mô tả chi tiết:
- Đọc file hilo_annotations_3d.json (COCO format) chứa 32,256 frames với 237 categories
- Tự động mapping category name sang class_id (0-236)
- Tạo cấu trúc thư mục YOLO chuẩn:
    data/02_processed/vision/yolo_format/
    ├── images/
    │   ├── scene_00_s0_c1/
    │   │   ├── arc0_image0.jpg
    │   │   └── ...
    │   └── ...
    └── labels/
        ├── scene_00_s0_c1/
        │   ├── arc0_image0.txt
        │   └── ...
        └── ...

Yêu cầu:
- Đã chạy hilo_make_annotations.py thành công
- File hilo_annotations_3d.json tồn tại trong data/02_processed/vision/coco_format/
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import time
import random

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import thư viện cần thiết
import cv2
import numpy as np
from tqdm import tqdm

from src.common.hilo_dataset import (
    HILODatasetLoader,
    HILOScenePaths,
    get_default_hilo_scenes_root,
)


def ensure_dir(path: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    path.mkdir(parents=True, exist_ok=True)


def list_rgb_images(scene: HILOScenePaths) -> List[Path]:
    """Liệt kê ảnh RGB của scene (ưu tiên undistorted)."""
    rgb_dir = scene.rgb_undistorted_dir
    if not rgb_dir.exists():
        rgb_dir = scene.rgb_raw_dir
    # Lấy tất cả file .png, .jpg, .jpeg
    return sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def load_coco_annotations(ann_path: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Đọc file COCO JSON và tách thành các dictionary riêng.
    
    Args:
        ann_path: Đường dẫn đến file JSON
        
    Returns:
        Tuple[Dict, Dict, Dict]: (images, annotations, categories)
    """
    with ann_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)
    
    # Tạo dictionary cho images
    images = {img['id']: img for img in coco_data.get('images', [])}
    
    # Tạo dictionary cho categories
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
    
    # Tạo index cho annotations theo image_id
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    return images, annotations_by_image, categories


def create_yolo_label(
    bbox_coco: List[float], 
    img_width: int, 
    img_height: int, 
    class_id: int
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Chuyển đổi từ COCO format [x, y, w, h] sang YOLO format [class_id, x_center, y_center, width, height].
    
    COCO: (x, y) là góc trên bên trái, (w, h) là chiều rộng và cao (pixel)
    YOLO: (x_center, y_center, width, height) là tỷ lệ (0-1) so với kích thước ảnh
    
    Args:
        bbox_coco: [x, y, w, h] trong hệ tọa độ pixel
        img_width: Chiều rộng ảnh
        img_height: Chiều cao ảnh
        class_id: ID của class (0-based)
        
    Returns:
        Tuple[class_id, x_center, y_center, width, height] hoặc None nếu không hợp lệ
    """
    x, y, w, h = bbox_coco
    
    # Kiểm tra hợp lệ
    if w <= 0 or h <= 0:
        return None
    
    # Tính tọa độ trung tâm
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Tính tỷ lệ width và height
    width_norm = w / img_width
    height_norm = h / img_height
    
    # Kiểm tra giá trị hợp lệ (0-1)
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width_norm <= 1 and 0 <= height_norm <= 1):
        return None
    
    return (class_id, x_center, y_center, width_norm, height_norm)


def process_scene_to_yolo(
    scene: HILOScenePaths,
    images_info: Dict[int, Dict],
    annotations_by_image: Dict[int, List[Dict]],
    category_to_id: Dict[str, int],
    yolo_images_dir: Path,
    yolo_labels_dir: Path,
    use_symlink: bool = True
) -> int:
    """
    Xử lý một scene và tạo file YOLO tương ứng.
    
    Args:
        scene: Scene hiện tại
        images_info: Dict mapping image_id -> image info từ COCO
        annotations_by_image: Dict mapping image_id -> list annotations
        category_to_id: Dict mapping category name -> class_id
        yolo_images_dir: Thư mục gốc chứa ảnh YOLO
        yolo_labels_dir: Thư mục gốc chứa label YOLO
        use_symlink: Dùng hardlink (True) hay copy (False) để tiết kiệm dung lượng
        
    Returns:
        Số lượng ảnh đã xử lý
    """
    # Tạo thư mục cho scene
    scene_img_dir = yolo_images_dir / scene.name
    scene_label_dir = yolo_labels_dir / scene.name
    ensure_dir(scene_img_dir)
    ensure_dir(scene_label_dir)
    
    # Liệt kê ảnh RGB
    rgb_images = list_rgb_images(scene)
    
    # Tạo mapping từ tên file (arcX_imageY) sang image_id trong COCO
    # File COCO lưu đường dẫn đầy đủ, cần trích xuất tên file
    filename_to_img_id = {}
    for img_id, img_info in images_info.items():
        file_name = Path(img_info['file_name']).name
        filename_to_img_id[file_name] = img_id
    
    count = 0
    for img_path in tqdm(rgb_images, desc=f"  {scene.name}", leave=False):
        # Tìm image_id tương ứng
        img_filename = img_path.name
        img_id = filename_to_img_id.get(img_filename)
        
        if img_id is None:
            # Không có annotation cho ảnh này
            continue
        
        # Copy/hardlink ảnh
        target_img_path = scene_img_dir / img_filename
        if not target_img_path.exists():
            try:
                if use_symlink:
                    # Thử hardlink trước (tiết kiệm dung lượng)
                    os.link(img_path, target_img_path)
                else:
                    from shutil import copy2
                    copy2(img_path, target_img_path)
            except OSError:
                # Fallback to copy
                from shutil import copy2
                copy2(img_path, target_img_path)
        
        # Lấy annotations cho ảnh này
        anns = annotations_by_image.get(img_id, [])
        
        # Đọc ảnh để lấy kích thước (có thể lấy từ img_info nhưng đọc lại để chắc chắn)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    WARNING: Không thể đọc ảnh {img_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Tạo file label
        label_path = scene_label_dir / (img_path.stem + ".txt")
        
        yolo_lines = []
        for ann in anns:
            bbox = ann.get('bbox')
            category_id = ann.get('category_id')
            
            if not bbox or category_id is None:
                continue
            
            # Chuyển COCO bbox sang YOLO format
            yolo_bbox = create_yolo_label(bbox, img_width, img_height, category_id - 1)  # category_id trong COCO bắt đầu từ 1
            
            if yolo_bbox is not None:
                class_id, xc, yc, wn, hn = yolo_bbox
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        
        # Ghi file label nếu có object
        if yolo_lines:
            with label_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            count += 1
        else:
            # Nếu không có object nào, tạo file rỗng (tùy chọn)
            # with label_path.open("w", encoding="utf-8") as f:
            #     f.write("")
            pass
    
    return count


def create_data_yaml(category_to_id: Dict[str, int], output_path: Path) -> None:
    """
    Tạo file data.yaml cho YOLO training.
    
    Args:
        category_to_id: Dict mapping category name -> class_id
        output_path: Đường dẫn file output
    """
    # Sắp xếp categories theo ID
    sorted_cats = sorted(category_to_id.items(), key=lambda x: x[1])
    
    yaml_content = f"""# YOLO dataset configuration file
# Generated by hilo_to_yolo.py
# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

# Number of classes: {len(sorted_cats)}

# Train/val/test paths (relative to this file or absolute)
path: ../yolo_format  # dataset root dir
train: images  # train images
val: images    # val images
test: images   # test images

# Number of classes
nc: {len(sorted_cats)}

# Class names
names: ["""
    
    # Thêm tên classes
    for i, (name, _) in enumerate(sorted_cats):
        if i > 0:
            yaml_content += ", "
        yaml_content += f"'{name}'"
    
    yaml_content += "]"
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"  ✅ Đã tạo {output_path}")


def split_train_val_test(
    images_root: Path,
    labels_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> None:
    """
    Tạo file train.txt, val.txt, test.txt chứa danh sách đường dẫn ảnh.
    
    Args:
        images_root: Thư mục gốc chứa ảnh
        labels_root: Thư mục gốc chứa label
        train_ratio: Tỷ lệ train
        val_ratio: Tỷ lệ validation
        test_ratio: Tỷ lệ test
    """
    # Kiểm tra thư mục images tồn tại
    if not images_root.exists():
        print(f"  ❌ Thư mục images không tồn tại: {images_root}")
        return
    
    # Lấy tất cả các file ảnh
    image_files = []
    
    print(f"  Đang quét thư mục {images_root}...")
    
    for scene_dir in images_root.iterdir():
        if scene_dir.is_dir():
            # Tìm file .jpg
            for img_file in scene_dir.glob("*.jpg"):
                rel_path = f"{scene_dir.name}/{img_file.name}"
                image_files.append(rel_path)
            
            # Tìm file .png
            for img_file in scene_dir.glob("*.png"):
                rel_path = f"{scene_dir.name}/{img_file.name}"
                image_files.append(rel_path)
    
    n_total = len(image_files)
    print(f"  Tổng số ảnh tìm thấy: {n_total}")
    
    if n_total == 0:
        print("  ⚠️ KHÔNG tìm thấy file ảnh nào!")
        return
    
    # Shuffle ngẫu nhiên
    random.seed(42)
    random.shuffle(image_files)
    
    # Tính số lượng
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Ghi file
    train_path = images_root.parent / "train.txt"
    val_path = images_root.parent / "val.txt"
    test_path = images_root.parent / "test.txt"
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_files))
    
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_files))
    
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_files))
    
    print(f"\n  ✅ ĐÃ TẠO SPLIT DATASET:")
    print(f"    📊 Train: {len(train_files)} ảnh ({len(train_files)/n_total*100:.1f}%) - {train_path}")
    print(f"    📊 Val: {len(val_files)} ảnh ({len(val_files)/n_total*100:.1f}%) - {val_path}")
    print(f"    📊 Test: {len(test_files)} ảnh ({len(test_files)/n_total*100:.1f}%) - {test_path}")


def main() -> None:
    """
    Hàm chính.
    """
    print("=" * 70)
    print("CHUYỂN ĐỔI HILO SANG YOLO FORMAT")
    print("=" * 70)
    
    start_time = time.time()
    
    # Đường dẫn đến file annotation
    ann_path = (
        Path("data")
        / "02_processed"
        / "vision"
        / "coco_format"
        / "hilo_annotations_3d.json"
    )
    
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy annotation JSON: {ann_path}\n"
            f"Hãy chạy scripts/data_collection/hilo_make_annotations.py trước."
        )
    
    print(f"\n📂 1. Đọc file annotation: {ann_path}")
    images, annotations_by_image, categories = load_coco_annotations(ann_path)
    
    print(f"   - Số images: {len(images)}")
    print(f"   - Số annotations: {sum(len(v) for v in annotations_by_image.values())}")
    print(f"   - Số categories: {len(categories)}")
    
    # Tạo mapping category name -> class_id (0-based)
    category_to_id = {}
    for cat_id, cat_info in categories.items():
        category_to_id[cat_info['name']] = cat_id - 1  # COCO id bắt đầu từ 1
    
    print(f"\n📂 2. Tạo thư mục YOLO...")
    yolo_root = Path("data") / "02_processed" / "vision" / "yolo_format"
    images_root = yolo_root / "images"
    labels_root = yolo_root / "labels"
    
    ensure_dir(images_root)
    ensure_dir(labels_root)
    
    # Load danh sách scenes
    scenes_root = get_default_hilo_scenes_root()
    loader = HILODatasetLoader(scenes_root)
    
    print(f"\n📂 3. Xử lý từng scene...")
    total_images = 0
    num_scenes = 0
    
    # Dùng tqdm để hiển thị tiến độ
    scenes_list = list(loader.iter_scenes())
    for scene in tqdm(scenes_list, desc="Tổng số scenes"):
        num_scenes += 1
        n = process_scene_to_yolo(
            scene=scene,
            images_info=images,
            annotations_by_image=annotations_by_image,
            category_to_id=category_to_id,
            yolo_images_dir=images_root,
            yolo_labels_dir=labels_root,
            use_symlink=True  # Dùng hardlink để tiết kiệm dung lượng
        )
        total_images += n
    
    print(f"\n📂 4. Tạo file data.yaml...")
    create_data_yaml(category_to_id, yolo_root / "data.yaml")
    
    print(f"\n📂 5. Split train/val/test...")
    split_train_val_test(images_root, labels_root)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT!")
    print("=" * 70)
    print(f"📊 Số scene đã xử lý: {num_scenes}/{len(scenes_list)}")
    print(f"📊 Số ảnh đã chuyển: {total_images}/{len(images)} ({total_images/len(images)*100:.1f}%)")
    print(f"📊 Số categories: {len(category_to_id)}")
    print(f"📁 Thư mục images: {images_root}")
    print(f"📁 Thư mục labels: {labels_root}")
    print(f"📁 File data.yaml: {yolo_root / 'data.yaml'}")
    print(f"⏱️  Thời gian xử lý: {elapsed_time/60:.2f} phút ({elapsed_time:.2f} giây)")
    print("=" * 70)
    
    # Thống kê nhanh
    print("\n📊 PHÂN BỐ CLASS (TOP 20):")
    class_counter = Counter()
    for ann_list in annotations_by_image.values():
        for ann in ann_list:
            cat_id = ann.get('category_id')
            if cat_id:
                class_counter[cat_id] += 1
    
    for cat_id, count in class_counter.most_common(20):
        cat_name = categories.get(cat_id, {}).get('name', f'Unknown_{cat_id}')
        print(f"   {cat_name}: {count} annotations")


if __name__ == "__main__":
    # Import thêm thư viện ở đây để tránh lỗi nếu chưa cài
    try:
        from tqdm import tqdm
    except ImportError:
        print("Đang cài tqdm...")
        os.system("pip install tqdm")
        from tqdm import tqdm
    
    main()