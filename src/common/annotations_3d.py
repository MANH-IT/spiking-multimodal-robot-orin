from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


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
    object_id: str
    category: str
    bbox2d: BBox2D | None
    bbox3d: BBox3D | None


@dataclass
class FrameAnnotation:
    image_path: Path
    depth_path: Path | None
    objects: List[ObjectAnnotation]


def frame_annotation_to_dict(ann: FrameAnnotation) -> Dict[str, Any]:
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

