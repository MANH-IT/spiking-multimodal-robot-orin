"""
Đánh giá DepthAwareSNN trên bbox 3D (IoU-based) dùng annotation HILO.

- Dataset: RGBDSequenceWithAnnDataset (frame đơn T=1).
- Metric:
    * Với mỗi frame, model dự đoán 1 bbox_3d.
    * So sánh với tất cả bbox3d GT trong frame, lấy IoU_3D lớn nhất.
    * Nếu IoU_3D >= threshold (mặc định 0.5) và frame có ít nhất 1 GT bbox,
      tính là 1 true positive (TP).
    * Nếu frame không có GT bbox nhưng vẫn dự đoán, tính là false positive (FP).
    * Nếu frame có GT bbox nhưng IoU_3D < threshold, tính là false negative (FN).
- In ra: precision, recall, F1, IoU_3D trung bình trên các frame có GT.

Lưu ý:
- Đây là baseline đơn giản cho bài toán "mỗi frame tối đa 1 dự đoán chính".
- Sau này có thể mở rộng sang nhiều bbox/detect head phức tạp hơn.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from vision_system.data.rgbd_sequence_with_ann_dataset import (
    RGBDSequenceWithAnnDataset,
)
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DepthAwareSNN on 3D bbox IoU using HILO annotations"
    )
    parser.add_argument(
        "--ann-path",
        type=Path,
        default=Path("data")
        / "02_processed"
        / "vision"
        / "coco_format"
        / "hilo_annotations_3d.json",
        help="Đường dẫn tới file annotation JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Đường dẫn tới checkpoint model (.pt/.pth). Nếu bỏ trống sẽ dùng random init.",
    )
    parser.add_argument(
        "--use-snn-backbone",
        action="store_true",
        help="Bật backbone SNN (nếu đã cài snntorch).",
    )
    return parser.parse_args()


def collate_fn(batch):
    rgbs, depths, targets = zip(*batch)
    return (
        torch.stack(rgbs, dim=0),
        torch.stack(depths, dim=0),
        list(targets),
    )


def build_dataloader(ann_path: Path, batch_size: int) -> DataLoader:
    dataset = RGBDSequenceWithAnnDataset(ann_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return loader


def bbox3d_to_minmax(
    bbox: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param bbox: (..., 6) [cx, cy, cz, sx, sy, sz]
    :return: (min_xyz, max_xyz), mỗi cái shape (..., 3)
    """
    cx, cy, cz, sx, sy, sz = torch.unbind(bbox, dim=-1)
    half = torch.stack([sx, sy, sz], dim=-1) / 2.0
    center = torch.stack([cx, cy, cz], dim=-1)
    mins = center - half
    maxs = center + half
    return mins, maxs


def iou_3d_single(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    IoU_3D giữa 1 bbox pred và N bbox GT.

    :param pred: (6,)
    :param gt: (N, 6)
    :return: (N,) IoU cho từng GT.
    """
    if gt.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=pred.device)

    pred = pred.view(1, 6)
    pred_mins, pred_maxs = bbox3d_to_minmax(pred)  # (1, 3)
    gt_mins, gt_maxs = bbox3d_to_minmax(gt)  # (N, 3)

    inter_mins = torch.maximum(pred_mins, gt_mins)
    inter_maxs = torch.minimum(pred_maxs, gt_maxs)
    inter_sizes = torch.clamp(inter_maxs - inter_mins, min=0.0)
    inter_vol = inter_sizes[:, 0] * inter_sizes[:, 1] * inter_sizes[:, 2]

    pred_sizes = torch.clamp(pred_maxs - pred_mins, min=0.0)
    pred_vol = (
        pred_sizes[:, 0] * pred_sizes[:, 1] * pred_sizes[:, 2]
    )  # shape (1,)

    gt_sizes = torch.clamp(gt_maxs - gt_mins, min=0.0)
    gt_vol = gt_sizes[:, 0] * gt_sizes[:, 1] * gt_sizes[:, 2]  # (N,)

    union_vol = pred_vol + gt_vol - inter_vol
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)
    return iou


def evaluate(
    model: DepthAwareSNN,
    loader: DataLoader,
    device: torch.device,
    iou_thresh: float,
    max_batches: int,
) -> None:
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    iou_sum = 0.0
    iou_count = 0

    with torch.no_grad():
        for step, (rgb, depth, targets) in enumerate(loader):
            rgb = rgb.to(device)  # (B, 3, H, W)
            depth = depth.to(device)  # (B, 1, H, W)

            rgb_seq = rgb.unsqueeze(1)  # (B, 1, 3, H, W)
            depth_seq = depth.unsqueeze(1)  # (B, 1, 1, H, W)

            bbox_pred, logits = model(rgb_seq, depth_seq)  # (B, 6)

            for i in range(rgb.size(0)):
                pred_box = bbox_pred[i]  # (6,)
                target = targets[i]
                gt_boxes = target.get("bbox3d")

                if gt_boxes is None or gt_boxes.numel() == 0:
                    # Không có GT mà vẫn dự đoán -> FP
                    fp += 1
                    continue

                ious = iou_3d_single(pred_box, gt_boxes.to(device))
                best_iou = float(ious.max().item())
                iou_sum += best_iou
                iou_count += 1

                if best_iou >= iou_thresh:
                    tp += 1
                else:
                    fn += 1

            if step + 1 >= max_batches:
                break

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    print("\n=== 3D BBox Evaluation (frame-wise, 1 prediction / frame) ===")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"IoU threshold: {iou_thresh}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Mean IoU (over frames with GT): {mean_iou:.4f}")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(args.ann_path, args.batch_size)

    model = DepthAwareSNN(
        num_classes=10,
        use_multiscale=False,
        use_snn_backbone=args.use_snn_backbone,
    ).to(device)

    if args.weights is not None and args.weights.exists():
        state = torch.load(args.weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Loaded weights from {args.weights}")
    else:
        print("Không có weights được cung cấp, dùng model random init (chỉ để debug).")

    evaluate(
        model=model,
        loader=loader,
        device=device,
        iou_thresh=args.iou_thresh,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()

