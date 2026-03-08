"""
Comprehensive evaluation metrics for 3D object detection.

Includes:
- 3D IoU calculation
- mAP (mean Average Precision) for 3D detection
- Precision/Recall curves
- Distance-based metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def bbox3d_to_corners(bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 3D bbox (cx, cy, cz, sx, sy, sz) to 8 corners.
    
    Args:
        bbox: (..., 6) tensor with [cx, cy, cz, sx, sy, sz]
        
    Returns:
        corners: (..., 8, 3) tensor with 8 corner coordinates
        mins, maxs: (..., 3) min and max coordinates
    """
    cx, cy, cz, sx, sy, sz = torch.unbind(bbox, dim=-1)
    half = torch.stack([sx, sy, sz], dim=-1) / 2.0
    center = torch.stack([cx, cy, cz], dim=-1)
    
    # Generate 8 corners
    offsets = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=half.dtype, device=half.device)  # (8, 3)
    
    # Expand to match bbox shape
    while offsets.dim() < half.dim() + 1:
        offsets = offsets.unsqueeze(0)
    
    corners = center.unsqueeze(-2) + offsets * half.unsqueeze(-2)  # (..., 8, 3)
    
    mins = center - half
    maxs = center + half
    
    return corners, (mins, maxs)


def compute_iou_3d(
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 3D IoU between predicted and ground truth bounding boxes.
    
    Args:
        pred_bbox: (N, 6) predicted boxes [cx, cy, cz, sx, sy, sz]
        gt_bbox: (M, 6) ground truth boxes [cx, cy, cz, sx, sy, sz]
        
    Returns:
        iou: (N, M) IoU matrix
    """
    pred_mins, pred_maxs = bbox3d_to_corners(pred_bbox)[1]
    gt_mins, gt_maxs = bbox3d_to_corners(gt_bbox)[1]
    
    # Expand dimensions for broadcasting
    pred_mins = pred_mins.unsqueeze(1)  # (N, 1, 3)
    pred_maxs = pred_maxs.unsqueeze(1)  # (N, 1, 3)
    gt_mins = gt_mins.unsqueeze(0)      # (1, M, 3)
    gt_maxs = gt_maxs.unsqueeze(0)      # (1, M, 3)
    
    # Intersection
    inter_mins = torch.maximum(pred_mins, gt_mins)  # (N, M, 3)
    inter_maxs = torch.minimum(pred_maxs, gt_maxs)  # (N, M, 3)
    inter_sizes = torch.clamp(inter_maxs - inter_mins, min=0.0)  # (N, M, 3)
    inter_vol = inter_sizes[..., 0] * inter_sizes[..., 1] * inter_sizes[..., 2]  # (N, M)
    
    # Union
    pred_sizes = torch.clamp(pred_maxs - pred_mins, min=0.0)  # (N, 1, 3)
    pred_vol = pred_sizes[..., 0] * pred_sizes[..., 1] * pred_sizes[..., 2]  # (N, 1)
    
    gt_sizes = torch.clamp(gt_maxs - gt_mins, min=0.0)  # (1, M, 3)
    gt_vol = gt_sizes[..., 0] * gt_sizes[..., 1] * gt_sizes[..., 2]  # (1, M)
    
    union_vol = pred_vol + gt_vol - inter_vol
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)
    
    return iou


def compute_center_distance(
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Euclidean distance between bbox centers.
    
    Args:
        pred_bbox: (N, 6) predicted boxes
        gt_bbox: (M, 6) ground truth boxes
        
    Returns:
        distances: (N, M) distance matrix
    """
    pred_center = pred_bbox[..., :3]  # (N, 3)
    gt_center = gt_bbox[..., :3]      # (M, 3)
    
    # Compute pairwise distances
    pred_center = pred_center.unsqueeze(1)  # (N, 1, 3)
    gt_center = gt_center.unsqueeze(0)       # (1, M, 3)
    
    diff = pred_center - gt_center  # (N, M, 3)
    distances = torch.norm(diff, dim=-1)  # (N, M)
    
    return distances


def compute_map_3d(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    iou_thresholds: List[float] = [0.25, 0.5, 0.75],
    max_detections: int = 100,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) for 3D object detection.
    
    Args:
        pred_boxes: List of (N_i, 6) predicted boxes per image
        pred_scores: List of (N_i,) confidence scores
        pred_labels: List of (N_i,) predicted class labels
        gt_boxes: List of (M_i, 6) ground truth boxes per image
        gt_labels: List of (M_i,) ground truth class labels
        iou_thresholds: List of IoU thresholds for mAP calculation
        max_detections: Maximum number of detections per image
        
    Returns:
        Dictionary with mAP metrics
    """
    all_classes = set()
    for labels in gt_labels:
        all_classes.update(labels.cpu().tolist())
    all_classes = sorted(all_classes)
    
    results = {}
    
    for iou_thresh in iou_thresholds:
        ap_per_class = []
        
        for class_id in all_classes:
            # Collect predictions and ground truth for this class
            pred_boxes_class = []
            pred_scores_class = []
            gt_boxes_class = []
            is_detected = []  # Track which GT boxes are detected
            
            for img_idx in range(len(pred_boxes)):
                # Filter predictions for this class
                pred_mask = pred_labels[img_idx] == class_id
                if pred_mask.any():
                    pred_boxes_class.append(pred_boxes[img_idx][pred_mask])
                    pred_scores_class.append(pred_scores[img_idx][pred_mask])
                else:
                    pred_boxes_class.append(torch.empty(0, 6, device=pred_boxes[img_idx].device))
                    pred_scores_class.append(torch.empty(0, device=pred_scores[img_idx].device))
                
                # Filter ground truth for this class
                gt_mask = gt_labels[img_idx] == class_id
                if gt_mask.any():
                    gt_boxes_class.append(gt_boxes[img_idx][gt_mask])
                    is_detected.append([False] * gt_mask.sum().item())
                else:
                    gt_boxes_class.append(torch.empty(0, 6, device=gt_boxes[img_idx].device))
                    is_detected.append([])
            
            # Sort all predictions by score
            all_pred_boxes = []
            all_pred_scores = []
            all_pred_img_idx = []
            
            for img_idx, (boxes, scores) in enumerate(zip(pred_boxes_class, pred_scores_class)):
                if len(boxes) > 0:
                    # Sort by score
                    sorted_indices = torch.argsort(scores, descending=True)
                    boxes = boxes[sorted_indices][:max_detections]
                    scores = scores[sorted_indices][:max_detections]
                    
                    all_pred_boxes.append(boxes)
                    all_pred_scores.append(scores)
                    all_pred_img_idx.extend([img_idx] * len(boxes))
            
            if len(all_pred_boxes) == 0:
                ap_per_class.append(0.0)
                continue
            
            # Concatenate all predictions
            all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
            all_pred_scores = torch.cat(all_pred_scores, dim=0)
            
            # Compute IoU with all ground truth boxes
            tp = []
            fp = []
            
            for pred_idx, (pred_box, pred_score, img_idx) in enumerate(
                zip(all_pred_boxes, all_pred_scores, all_pred_img_idx)
            ):
                gt_boxes_img = gt_boxes_class[img_idx]
                
                if len(gt_boxes_img) == 0:
                    fp.append(1)
                    tp.append(0)
                    continue
                
                # Compute IoU with all GT boxes in this image
                ious = compute_iou_3d(pred_box.unsqueeze(0), gt_boxes_img)  # (1, M)
                best_iou, best_gt_idx = ious.max(dim=1)
                best_iou = best_iou.item()
                best_gt_idx = best_gt_idx.item()
                
                if best_iou >= iou_thresh and not is_detected[img_idx][best_gt_idx]:
                    tp.append(1)
                    fp.append(0)
                    is_detected[img_idx][best_gt_idx] = True
                else:
                    tp.append(0)
                    fp.append(1)
            
            # Compute precision and recall
            tp = np.array(tp)
            fp = np.array(fp)
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / max(sum(len(gt) for gt in gt_boxes_class), 1)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # Compute AP using 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11.0
            
            ap_per_class.append(ap)
        
        # Compute mAP
        map_score = np.mean(ap_per_class) if ap_per_class else 0.0
        results[f"mAP@{iou_thresh:.2f}"] = map_score
    
    # Average mAP across all thresholds
    results["mAP"] = np.mean([results[f"mAP@{t:.2f}"] for t in iou_thresholds])
    
    return results


__all__ = [
    "compute_iou_3d",
    "compute_center_distance",
    "compute_map_3d",
    "bbox3d_to_corners",
]
