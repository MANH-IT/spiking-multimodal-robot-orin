#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: hilo_make_annotations.py
Mục đích: Tạo annotation 2D/3D cho toàn bộ dataset HILO dựa trên pose camera,
pose vật thể và kích thước vật thể (từ objects_metadata.csv) cùng với thông số
calibration camera.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Thêm project root vào sys.path trước khi import src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.common.hilo_dataset import (
    HILODatasetLoader,
    HILOScenePaths,
    get_default_hilo_scenes_root,
)
from src.common.annotations_3d import (
    FrameAnnotation,
    ObjectAnnotation,
    BBox2D,
    BBox3D,
    frame_annotation_to_dict,
)
from src.common.hilo_geometry import (
    load_camera_rig_params,
    load_object_dimensions,
    compute_object_bboxes_for_frame,
    parse_arc_from_key,
    CameraIntrinsics,
    ObjectDimensions,
)


def make_frame_annotations_for_scene(
    scene: HILOScenePaths,
    rig_params_by_arc: Dict[int, Any],
    obj_dims_map: Dict[str, ObjectDimensions],
) -> List[FrameAnnotation]:
    """
    Tạo danh sách FrameAnnotation cho một scene cụ thể.
    """
    loader = HILODatasetLoader(scene.root.parent)
    
    # Đọc camera poses - đã được xử lý thành numpy array trong hilo_dataset.py
    camera_poses = loader.load_camera_poses(scene)
    print(f"  Đọc camera_poses: {len(camera_poses)} frames")
    
    if not camera_poses:
        print(f"  WARNING: Không có camera poses cho scene {scene.name}")
        return []
    
    # Đọc object poses
    object_poses = loader.load_object_poses(scene)
    print(f"  Đọc object_poses: {len(object_poses)} objects")
    
    # Đọc danh sách object id có trong scene
    object_ids = loader.load_object_id_list(scene)
    print(f"  Danh sách object IDs: {object_ids}")

    frames: List[FrameAnnotation] = []

    for img_key, pose_cam_mat in camera_poses.items():
        try:
            # Xác định đường dẫn ảnh RGB (ưu tiên undistorted)
            rgb_path = scene.rgb_undistorted_dir / f"{img_key}.png"
            if not rgb_path.exists():
                rgb_path = scene.rgb_raw_dir / f"{img_key}.jpg"
            
            if not rgb_path.exists():
                print(f"  WARNING: Không tìm thấy ảnh RGB cho {img_key}")
                continue

            # Xác định đường dẫn depth (nếu có)
            depth_path = scene.depth_undistorted_dir / f"{img_key}.png"
            if not depth_path.exists():
                depth_path = scene.depth_raw_dir / f"{img_key}.png"
            if not depth_path.exists():
                depth_path = None

            # Đọc ảnh RGB để lấy kích thước
            img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  WARNING: Không thể đọc ảnh {rgb_path}")
                continue
                
            height, width = img.shape[:2]

            # Tính bbox3D và bbox2D cho tất cả các object
            bboxes_by_model = compute_object_bboxes_for_frame(
                img_key=img_key,
                pose_cam_wrt_arc0=pose_cam_mat,
                object_poses=object_poses,
                obj_dims_map=obj_dims_map,
                rig_params_by_arc=rig_params_by_arc,
                image_size=(width, height),
            )

            objects: List[ObjectAnnotation] = []

            # Duyệt qua tất cả object_id có trong scene
            for model_id in object_ids:
                # Kiểm tra model_id có trong object_poses không
                found_pose = False
                for obj_pose in object_poses:
                    if obj_pose.get("Model") == model_id:
                        found_pose = True
                        break
                
                if not found_pose:
                    continue

                bbox3d, bbox2d = bboxes_by_model.get(model_id, (None, None))
                
                objects.append(
                    ObjectAnnotation(
                        object_id=model_id,
                        category=model_id,
                        bbox2d=bbox2d,
                        bbox3d=bbox3d,
                    )
                )

            frames.append(
                FrameAnnotation(
                    image_path=rgb_path,
                    depth_path=depth_path,
                    objects=objects,
                )
            )
            
        except Exception as e:
            print(f"  ERROR: Lỗi xử lý frame {img_key}: {e}")
            continue

    return frames


def create_category_mapping(all_frames: List[FrameAnnotation]) -> Dict[str, int]:
    """
    Tạo mapping từ category name sang integer ID.
    """
    categories = set()
    for frame in all_frames:
        for obj in frame.objects:
            categories.add(obj.category)
    
    category_to_id = {cat: idx + 1 for idx, cat in enumerate(sorted(categories))}
    return category_to_id


def save_as_coco_format(
    all_frames: List[FrameAnnotation], 
    output_path: Path,
    category_to_id: Dict[str, int]
) -> None:
    """
    Lưu annotations dưới định dạng COCO JSON.
    """
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": id, "name": name} for name, id in category_to_id.items()
        ]
    }
    
    ann_id = 0
    
    for img_id, frame in enumerate(all_frames):
        img_info = {
            "id": img_id,
            "file_name": str(frame.image_path),
            "width": 640,
            "height": 480,
        }
        coco_output["images"].append(img_info)
        
        for obj in frame.objects:
            if obj.bbox2d is not None:
                bbox2d = [obj.bbox2d.x, obj.bbox2d.y, obj.bbox2d.w, obj.bbox2d.h]
                area = bbox2d[2] * bbox2d[3]
                cat_id = category_to_id.get(obj.category, 0)
                
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": bbox2d,
                    "area": area,
                    "iscrowd": 0,
                }
                
                if obj.bbox3d is not None:
                    ann["bbox3d"] = [
                        obj.bbox3d.cx, obj.bbox3d.cy, obj.bbox3d.cz,
                        obj.bbox3d.sx, obj.bbox3d.sy, obj.bbox3d.sz
                    ]
                
                coco_output["annotations"].append(ann)
                ann_id += 1
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)


def main() -> None:
    """
    Hàm chính.
    """
    print("=" * 60)
    print("BẮT ĐẦU TẠO ANNOTATIONS CHO HILO DATASET")
    print("=" * 60)
    
    scenes_root = get_default_hilo_scenes_root()
    print(f"Scenes root: {scenes_root}")
    
    if not scenes_root.exists():
        print(f"ERROR: Không tìm thấy thư mục {scenes_root}")
        return
    
    loader = HILODatasetLoader(scenes_root)

    # Load thông số camera calibration
    print("\n1. ĐANG LOAD CAMERA PARAMETERS...")
    try:
        rig_params_by_arc = load_camera_rig_params(scenes_root)
        print(f"   Đã load parameters cho {len(rig_params_by_arc)} arcs")
    except Exception as e:
        print(f"   ERROR: Không thể load camera parameters: {e}")
        return

    # Đường dẫn đến file metadata
    print("\n2. ĐANG LOAD OBJECT DIMENSIONS...")
    
    possible_paths = [
        scenes_root.parent / "HILO_Dataset" / "HILO_Objects" / "objects_metadata.csv",
        scenes_root.parent.parent / "HILO_Objects" / "objects_metadata.csv",
        scenes_root.parent / "HILO_Objects" / "objects_metadata.csv",
        Path("D:/multi_modal_robot_ai/data/00_raw/vision/HILO_Dataset/HILO_Objects/objects_metadata.csv"),
    ]
    
    obj_meta_csv = None
    for path in possible_paths:
        if path.exists():
            obj_meta_csv = path
            print(f"   Tìm thấy file tại: {path}")
            break
    
    if obj_meta_csv is None:
        print("   ERROR: Không tìm thấy objects_metadata.csv")
        return
    
    try:
        obj_dims_map = load_object_dimensions(obj_meta_csv)
        print(f"   Đã load {len(obj_dims_map)} object dimensions")
    except Exception as e:
        print(f"   ERROR: Không thể load object dimensions: {e}")
        return

    # Tạo annotations
    print("\n3. ĐANG TẠO ANNOTATIONS...")
    out_dir = Path("data") / "02_processed" / "vision" / "coco_format"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hilo_annotations_3d.json"

    all_frames: List[FrameAnnotation] = []
    scene_count = 0
    
    for scene in loader.iter_scenes():
        scene_count += 1
        print(f"\n--- Scene {scene_count}: {scene.name} ---")
        
        try:
            frames = make_frame_annotations_for_scene(
                scene,
                rig_params_by_arc=rig_params_by_arc,
                obj_dims_map=obj_dims_map,
            )
            all_frames.extend(frames)
            print(f"   ✅ {len(frames)} frames")
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            continue

    if not all_frames:
        print("\n❌ KHÔNG CÓ FRAME NÀO ĐƯỢC TẠO!")
        return

    # Tạo category mapping
    print(f"\n4. ĐANG TẠO CATEGORY MAPPING...")
    category_to_id = create_category_mapping(all_frames)
    print(f"   {len(category_to_id)} categories")

    # Lưu file
    print(f"\n5. ĐANG LƯU FILE JSON...")
    save_as_coco_format(all_frames, out_path, category_to_id)
    
    # Lưu file đơn giản
    with out_path.with_suffix(".simple.json").open("w", encoding="utf-8") as f:
        json.dump(
            [frame_annotation_to_dict(fr) for fr in all_frames],
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT!")
    print("=" * 60)
    print(f"📊 Tổng số scene đã xử lý: {scene_count}")
    print(f"📊 Tổng số frame: {len(all_frames)}")
    print(f"📊 Số categories: {len(category_to_id)}")
    print(f"📁 Output (COCO format): {out_path}")
    print(f"📁 Output (simple format): {out_path.with_suffix('.simple.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()