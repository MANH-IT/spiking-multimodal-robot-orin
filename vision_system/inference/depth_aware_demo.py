"""
Demo inference pipeline cho DepthAwareSNN:
- Đọc một chuỗi frame RGB–Depth đã aligned từ data/01_interim/vision/aligned
- Đưa qua DepthAwareSNN
- In kích thước output bbox_3d và logits để kiểm tra luồng chạy
"""

from pathlib import Path
from typing import List

import cv2
import torch
import sys

# Bảo đảm project root nằm trong sys.path khi chạy trực tiếp file .py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.models.snn.depth_aware_snn import DepthAwareSNN


def load_aligned_sequence(
    aligned_root: Path,
    num_frames: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Đọc một chuỗi frame từ:
      aligned/rgb/*.png
      aligned/depth/*.png
    và trả về tensor:
      rgb_seq: (1, T, 3, H, W)
      depth_seq: (1, T, 1, H, W)
    """
    rgb_dir = aligned_root / "rgb"
    depth_dir = aligned_root / "depth"

    rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    depth_files = sorted(p for p in depth_dir.iterdir() if p.suffix.lower() in {".png"})

    if not rgb_files or not depth_files:
        raise RuntimeError(f"Không tìm thấy dữ liệu aligned trong: {aligned_root}")

    # Chọn T frame đầu tiên (hoặc ít hơn nếu không đủ)
    T = min(num_frames, len(rgb_files), len(depth_files))
    rgb_files = rgb_files[:T]
    depth_files = depth_files[:T]

    rgb_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            raise RuntimeError(f"Lỗi đọc frame: {rgb_path}, {depth_path}")

        # BGR -> RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Chuẩn hoá đơn giản về [0,1]
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # (3, H, W)

        # Depth để nguyên scale (có thể chuẩn hoá sau)
        if depth.ndim == 2:
            depth = depth[None, ...]  # (1, H, W)
        depth_t = torch.from_numpy(depth).float()  # (1, H, W)

        rgb_list.append(rgb_t)
        depth_list.append(depth_t)

    # (T, C, H, W) -> (1, T, C, H, W)
    rgb_seq = torch.stack(rgb_list, dim=0).unsqueeze(0)
    depth_seq = torch.stack(depth_list, dim=0).unsqueeze(0)
    return rgb_seq, depth_seq


def main() -> None:
    aligned_root = Path("data") / "01_interim" / "vision" / "aligned"

    rgb_seq, depth_seq = load_aligned_sequence(aligned_root, num_frames=8)
    print(f"rgb_seq shape: {tuple(rgb_seq.shape)}")
    print(f"depth_seq shape: {tuple(depth_seq.shape)}")

    model = DepthAwareSNN(num_classes=10)
    model.eval()

    with torch.no_grad():
        bbox_3d, logits = model(rgb_seq, depth_seq)

    print(f"bbox_3d shape: {tuple(bbox_3d.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

