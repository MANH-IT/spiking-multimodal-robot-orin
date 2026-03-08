"""
Dataclasses cho 3D annotations và bounding boxes.

Module này định nghĩa các cấu trúc dữ liệu cho:
- BBox2D: Bounding box 2D (x, y, width, height)
- BBox3D: Bounding box 3D (center x,y,z và size x,y,z)
- ObjectAnnotation: Annotation cho một object
- FrameAnnotation: Annotation cho một frame
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class BBox2D:
    x: float
    y: float
    w: float
    h: float


@dataclass
class BBox3D:
    cx: float
    cy: float
    cz: float
    sx: float
    sy: float
    sz: float


@dataclass
class ObjectAnnotation:
    """Annotation cho một object trong frame."""
    object_id: str
    category: str
    bbox2d: Optional[BBox2D]
    bbox3d: Optional[BBox3D]


@dataclass
class FrameAnnotation:
    """Annotation cho một frame (ảnh RGB-D và danh sách objects)."""
    image_path: Path
    depth_path: Optional[Path]
    objects: List[ObjectAnnotation]


def frame_annotation_to_dict(ann: FrameAnnotation) -> Dict[str, Any]:
    """
    Chuyển đổi FrameAnnotation sang dictionary format (cho JSON serialization).

    Args:
        ann: FrameAnnotation cần chuyển đổi

    Returns:
        Dictionary chứa thông tin frame annotation
    """
    return {
        "image_path": str(ann.image_path),
        "depth_path": str(ann.depth_path) if ann.depth_path is not None else None,
        "objects": [
            {
                "object_id": obj.object_id,
                "category": obj.category,
                "bbox2d": None
                if obj.bbox2d is None
                else {
                    "x": obj.bbox2d.x,
                    "y": obj.bbox2d.y,
                    "w": obj.bbox2d.w,
                    "h": obj.bbox2d.h,
                },
                "bbox3d": None
                if obj.bbox3d is None
                else {
                    "cx": obj.bbox3d.cx,
                    "cy": obj.bbox3d.cy,
                    "cz": obj.bbox3d.cz,
                    "sx": obj.bbox3d.sx,
                    "sy": obj.bbox3d.sy,
                    "sz": obj.bbox3d.sz,
                },
            }
            for obj in ann.objects
        ],
    }


__all__ = [
    "BBox2D",
    "BBox3D",
    "ObjectAnnotation",
    "FrameAnnotation",
    "frame_annotation_to_dict",
]

