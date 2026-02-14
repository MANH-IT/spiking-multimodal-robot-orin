"""
Sinh dữ liệu trung gian cho HILO:
- RGB–Depth alignment (lưu image pairs)
- Pointcloud (từ depth + calib – TODO)
- Chỗ hook để sau này thêm feature maps (backbone/SNN encoder)

Mục tiêu: tạo cấu trúc rõ ràng trong data/01_interim/vision để
pipeline SNN và baseline CNN dùng chung.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np

import sys

# Bảo đảm project root nằm trong sys.path trước khi import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hilo_dataset import (
    HILODatasetLoader,
    HILOScenePaths,
    get_default_hilo_scenes_root,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_depth_images(scene: HILOScenePaths) -> List[Path]:
    depth_dir = scene.depth_undistorted_dir
    if not depth_dir.exists():
        depth_dir = scene.depth_raw_dir
    return sorted(p for p in depth_dir.iterdir() if p.suffix.lower() in {".png"})


def align_rgb_depth_pair(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạm thời: giả sử RGB undistorted & depth undistorted đã căn chỉnh cơ bản.
    Trả về (rgb_bgr, depth_16u hoặc 32f).

    TODO:
    - Dùng calibration matrix để warp depth -> RGB một cách chính xác.
    """
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise RuntimeError(f"Cannot read rgb/depth pair: {rgb_path}, {depth_path}")

    # Resize depth về size RGB nếu khác kích thước (approx)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    return rgb, depth


def save_aligned_pair(
    rgb: np.ndarray,
    depth: np.ndarray,
    out_rgb_path: Path,
    out_depth_path: Path,
) -> None:
    ensure_dir(out_rgb_path.parent)
    ensure_dir(out_depth_path.parent)
    cv2.imwrite(str(out_rgb_path), rgb)
    cv2.imwrite(str(out_depth_path), depth)


def process_scene_to_interim(
    scene: HILOScenePaths,
    aligned_root: Path,
) -> int:
    """
    Tạo RGB–Depth aligned cho một scene.
    Sau này có thể mở rộng để sinh pointcloud/features.
    """
    from scripts.data_collection.hilo_to_yolo import list_rgb_images

    rgb_images = list_rgb_images(scene)
    depth_images = list_depth_images(scene)

    if not rgb_images or not depth_images:
        return 0

    # Giả sử số lượng/frame order tương ứng theo index
    count = 0
    for rgb_path, depth_path in zip(rgb_images, depth_images):
        rgb, depth = align_rgb_depth_pair(rgb_path, depth_path)

        rel_name = f"{scene.name}_{rgb_path.stem}"
        out_rgb = aligned_root / "rgb" / (rel_name + ".png")
        out_depth = aligned_root / "depth" / (rel_name + ".png")

        save_aligned_pair(rgb, depth, out_rgb, out_depth)
        count += 1

    return count


def main() -> None:
    """
    Entry:
    - Duyệt toàn bộ scenes HILO
    - Sinh cặp RGB–Depth aligned lưu trong:
        data/01_interim/vision/aligned/rgb
        data/01_interim/vision/aligned/depth
    - Đây là bước 0 để sau này:
        - generate pointclouds từ depth
        - trích xuất features chung cho SNN/CNN
    """
    scenes_root = get_default_hilo_scenes_root()
    loader = HILODatasetLoader(scenes_root)

    aligned_root = Path("data") / "01_interim" / "vision" / "aligned"
    ensure_dir(aligned_root)

    total_pairs = 0
    num_scenes = 0

    for scene in loader.iter_scenes():
        num_scenes += 1
        n = process_scene_to_interim(scene, aligned_root)
        total_pairs += n
        print(f"[{scene.name}] -> {n} aligned RGB–Depth pairs.")

    print("\n=== HILO -> Interim (aligned RGB–Depth) hoàn tất ===")
    print(f"Số scene: {num_scenes}")
    print(f"Tổng số cặp RGB–Depth: {total_pairs}")
    print(f"Output aligned root: {aligned_root}")


if __name__ == "__main__":
    main()

