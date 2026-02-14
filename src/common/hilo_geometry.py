"""
Hỗ trợ hình học 3D cho HILO:
- Đọc camera intrinsics/extrinsics (cam_params_arc0123/4567).
- Đọc metadata kích thước object từ objects_metadata.csv.
- Tính bbox3D (trong toạ độ camera) và bbox2D (trên ảnh) từ:
    * pose object_wrt_arc0_image7
    * camera_poses[img_key]

Ghi chú:
- Hiện tại giả định ma trận trong camera_poses và object_pose_wrt_arc0_image7
  đều ở cùng hệ toạ độ camera depth/color; để đơn giản, ta:
    * Tính bbox3D trong hệ toạ độ camera tương ứng với img_key.
    * Project trực tiếp bằng K_color (color_K) của arc đó.
- Nếu sau này cần tách depth vs color camera chính xác hơn, có thể
  bổ sung bước depth->color thông qua color_R, color_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import csv
import numpy as np

from .annotations_3d import BBox3D, BBox2D


@dataclass
class CameraIntrinsics:
    """
    Lớp biểu diễn ma trận nội tại (intrinsic matrix) của camera.
    
    Attributes:
        fx: Tiêu cự theo trục x (pixel)
        fy: Tiêu cự theo trục y (pixel)
        cx: Tọa độ x của điểm chính (principal point)
        cy: Tọa độ y của điểm chính
    """
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_K(cls, K: List[List[float]]) -> "CameraIntrinsics":
        """
        Tạo đối tượng CameraIntrinsics từ ma trận K 3x3.
        
        Args:
            K: Ma trận intrinsic 3x3 dạng [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            
        Returns:
            CameraIntrinsics: Đối tượng chứa các thông số nội tại
        """
        return cls(
            fx=float(K[0][0]),
            fy=float(K[1][1]),
            cx=float(K[0][2]),
            cy=float(K[1][2]),
        )
    
    def to_K_matrix(self) -> np.ndarray:
        """
        Chuyển đổi thành ma trận K 3x3.
        
        Returns:
            np.ndarray: Ma trận intrinsic 3x3
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=float)


@dataclass
class CameraRigParams:
    """
    Tham số calib cho một nhóm arc (0–3 hoặc 4–7).
    
    Attributes:
        depth_K: Intrinsics của camera depth
        color_K: Intrinsics của camera color (RGB)
        color_R: Ma trận xoay (3x3) từ depth sang color
        color_t: Vector tịnh tiến (3,) từ depth sang color
    """
    depth_K: CameraIntrinsics
    color_K: CameraIntrinsics
    color_R: np.ndarray  # (3, 3)
    color_t: np.ndarray  # (3,)


def load_camera_rig_params(hilo_scenes_root: Path) -> Dict[int, CameraRigParams]:
    """
    Đọc hai file:
      cam_params_arc0123.json  -> áp dụng cho arc 0,1,2,3
      cam_params_arc4567.json  -> áp dụng cho arc 4,5,6,7

    Args:
        hilo_scenes_root: Đường dẫn đến thư mục HILO_Scenes
        
    Returns:
        Dict[int, CameraRigParams]: Dictionary với key là arc index (0-7)
        
    Raises:
        FileNotFoundError: Nếu không tìm thấy file camera parameters
    """
    rig_params: Dict[int, CameraRigParams] = {}

    def _load_one(path: Path) -> CameraRigParams:
        """
        Đọc một file camera parameters.
        """
        if not path.exists():
            raise FileNotFoundError(f"Camera parameters file not found: {path}")
            
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        depth_K = CameraIntrinsics.from_K(data["depth_K"])
        color_K = CameraIntrinsics.from_K(data["color_K"])
        color_R = np.array(data["color_R"], dtype=float).reshape(3, 3)
        color_t = np.array(data["color_t"], dtype=float).reshape(3)
        
        return CameraRigParams(
            depth_K=depth_K,
            color_K=color_K,
            color_R=color_R,
            color_t=color_t,
        )

    # Load cho arc 0-3
    path_0123 = hilo_scenes_root / "cam_params_arc0123.json"
    rig_0123 = _load_one(path_0123)
    for arc in (0, 1, 2, 3):
        rig_params[arc] = rig_0123

    # Load cho arc 4-7
    path_4567 = hilo_scenes_root / "cam_params_arc4567.json"
    rig_4567 = _load_one(path_4567)
    for arc in (4, 5, 6, 7):
        rig_params[arc] = rig_4567

    print(f"Đã load camera parameters cho {len(rig_params)} arcs")
    return rig_params


@dataclass
class ObjectDimensions:
    """
    Kích thước thực tế của object (đơn vị: mm).
    
    Attributes:
        length_x: Chiều dài theo trục X (mm)
        width_y: Chiều rộng theo trục Y (mm)
        height_z: Chiều cao theo trục Z (mm)
    """
    length_x: float  # mm
    width_y: float   # mm
    height_z: float  # mm
    
    def to_meters(self) -> Tuple[float, float, float]:
        """
        Chuyển đổi đơn vị từ mm sang m.
        
        Returns:
            Tuple[float, float, float]: (length_x, width_y, height_z) tính bằng mét
        """
        return (self.length_x / 1000.0, self.width_y / 1000.0, self.height_z / 1000.0)


def load_object_dimensions(
    metadata_csv: Path,
) -> Dict[str, ObjectDimensions]:
    """
    Đọc objects_metadata.csv, map ID# -> (Length X, Width Y, Height Z) (mm).

    Args:
        metadata_csv: Đường dẫn đến file objects_metadata.csv
        
    Returns:
        Dict[str, ObjectDimensions]: Dictionary với key là object ID
        
    Raises:
        FileNotFoundError: Nếu không tìm thấy file metadata
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"objects_metadata.csv not found: {metadata_csv}")

    dims: Dict[str, ObjectDimensions] = {}
    
    with metadata_csv.open("r", encoding="utf-8") as f:
        # Đọc file CSV, tự động phát hiện delimiter
        sample = f.read(1024)
        f.seek(0)
        
        # Xác định delimiter (có thể là dấu phẩy hoặc tab)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # In ra tên các cột để debug
        print(f"Các cột trong file CSV: {reader.fieldnames}")
        
        for row_num, row in enumerate(reader, start=2):
            try:
                # Xử lý linh hoạt tên cột (có thể có hoặc không có dấu ngoặc kép)
                obj_id = None
                lx = None
                wy = None
                hz = None
                
                # Tìm object ID
                for key in ['ID#', 'ID', 'Object ID', 'object_id']:
                    if key in row and row[key].strip():
                        obj_id = row[key].strip()
                        break
                
                if not obj_id:
                    print(f"Warning: Dòng {row_num}: Không tìm thấy object ID, bỏ qua")
                    continue
                
                # Tìm kích thước Length (X)
                for key in ['"Length (X, mm)"', 'Length (X, mm)', 'Length_X', 'length_x']:
                    if key in row and row[key]:
                        try:
                            lx = float(row[key])
                            break
                        except ValueError:
                            continue
                
                # Tìm kích thước Width (Y)
                for key in ['"Width (Y, mm)"', 'Width (Y, mm)', 'Width_Y', 'width_y']:
                    if key in row and row[key]:
                        try:
                            wy = float(row[key])
                            break
                        except ValueError:
                            continue
                
                # Tìm kích thước Height (Z)
                for key in ['"Height (Z, mm)"', 'Height (Z, mm)', 'Height_Z', 'height_z']:
                    if key in row and row[key]:
                        try:
                            hz = float(row[key])
                            break
                        except ValueError:
                            continue
                
                # Kiểm tra đã đọc đủ dữ liệu chưa
                if lx is None or wy is None or hz is None:
                    print(f"Warning: Dòng {row_num} - Object {obj_id}: Thiếu kích thước, bỏ qua")
                    continue
                
                dims[obj_id] = ObjectDimensions(length_x=lx, width_y=wy, height_z=hz)
                print(f"  - {obj_id}: {lx} x {wy} x {hz} mm")
                
            except Exception as e:
                print(f"Warning: Dòng {row_num}: Lỗi xử lý: {e}")
                continue
    
    print(f"Đã load {len(dims)} object dimensions từ {metadata_csv}")
    return dims


def make_aabb_corners_local(dims: ObjectDimensions) -> np.ndarray:
    """
    Tạo 8 đỉnh của axis-aligned bounding box trong hệ tọa độ local
    (trung tâm tại gốc, các trục song song với trục object).

    Args:
        dims: Kích thước object (mm)

    Returns:
        np.ndarray: Mảng (8, 4) với tọa độ homogeneous (x, y, z, 1)
    """
    lx, wy, hz = dims.length_x, dims.width_y, dims.height_z
    hx, hy, hz2 = lx / 2.0, wy / 2.0, hz / 2.0

    corners = []
    for sx in (-hx, hx):
        for sy in (-hy, hy):
            for sz in (-hz2, hz2):
                corners.append([sx, sy, sz, 1.0])
                
    return np.array(corners, dtype=float)


def transform_points(T: np.ndarray, pts_h: np.ndarray) -> np.ndarray:
    """
    Áp dụng phép biến đổi 4x4 lên các điểm homogeneous.

    Args:
        T: Ma trận biến đổi (4, 4)
        pts_h: Các điểm homogeneous (N, 4)

    Returns:
        np.ndarray: Tọa độ 3D (N, 3) sau khi chia cho w
    """
    out = (T @ pts_h.T).T
    w = out[:, 3:4]
    w[w == 0.0] = 1.0  # Tránh chia cho 0
    out[:, :3] /= w
    return out[:, :3]


def bbox3d_from_corners(cam_xyz: np.ndarray) -> BBox3D:
    """
    Tính bounding box 3D axis-aligned trong hệ tọa độ camera từ các đỉnh.

    Args:
        cam_xyz: Mảng (N, 3) tọa độ các điểm trong hệ camera

    Returns:
        BBox3D: Bounding box 3D với (cx, cy, cz, sx, sy, sz)
    """
    mins = cam_xyz.min(axis=0)
    maxs = cam_xyz.max(axis=0)
    cx, cy, cz = (mins + maxs) / 2.0
    sx, sy, sz = (maxs - mins)
    
    return BBox3D(
        cx=float(cx),
        cy=float(cy),
        cz=float(cz),
        sx=float(sx),
        sy=float(sy),
        sz=float(sz),
    )


def project_points(
    intr: CameraIntrinsics,
    cam_xyz: np.ndarray,
) -> np.ndarray:
    """
    Chiếu các điểm 3D (trong hệ tọa độ camera) lên ảnh 2D.

    Args:
        intr: Thông số nội tại camera
        cam_xyz: Mảng (N, 3) tọa độ các điểm (X, Y, Z) trong hệ camera
                (đơn vị có thể là mm hoặc m, miễn là tỷ lệ X/Z, Y/Z đúng)

    Returns:
        np.ndarray: Mảng (N, 2) tọa độ pixel (u, v)
    """
    X = cam_xyz[:, 0]
    Y = cam_xyz[:, 1]
    Z = cam_xyz[:, 2]
    
    # Tránh chia cho 0
    Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    
    u = intr.fx * (X / Z_safe) + intr.cx
    v = intr.fy * (Y / Z_safe) + intr.cy
    
    return np.stack([u, v], axis=-1)


def bbox2d_from_projected(
    uv: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[BBox2D]:
    """
    Tạo bounding box 2D axis-aligned từ các điểm đã chiếu, clip trong ảnh.

    Args:
        uv: Mảng (N, 2) tọa độ pixel
        image_width: Chiều rộng ảnh
        image_height: Chiều cao ảnh

    Returns:
        Optional[BBox2D]: Bounding box 2D (x, y, w, h) hoặc None nếu không hợp lệ
    """
    if uv.size == 0:
        return None

    u = uv[:, 0]
    v = uv[:, 1]

    # Clip tọa độ trong phạm vi ảnh
    u_min = float(np.clip(u.min(), 0, image_width - 1))
    v_min = float(np.clip(v.min(), 0, image_height - 1))
    u_max = float(np.clip(u.max(), 0, image_width - 1))
    v_max = float(np.clip(v.max(), 0, image_height - 1))

    w = u_max - u_min
    h = v_max - v_min
    
    # Kiểm tra bounding box hợp lệ (có diện tích > 1 pixel)
    if w <= 1.0 or h <= 1.0:
        return None

    return BBox2D(x=u_min, y=v_min, w=w, h=h)


def parse_arc_from_key(img_key: str) -> int:
    """
    Trích xuất arc index từ image key.

    Args:
        img_key: Image key dạng 'arc0_image7', 'arc1_image0', ...

    Returns:
        int: Arc index (0-7)

    Raises:
        ValueError: Nếu key không đúng định dạng
    """
    if not img_key.startswith("arc"):
        raise ValueError(f"Unrecognized img_key format: {img_key}")
        
    head = img_key.split("_")[0]  # 'arc0'
    return int(head.replace("arc", ""))


def compute_object_bboxes_for_frame(
    img_key: str,
    pose_cam_wrt_arc0: np.ndarray,
    object_poses: List[Dict],
    obj_dims_map: Dict[str, ObjectDimensions],
    rig_params_by_arc: Dict[int, CameraRigParams],
    image_size: Tuple[int, int],
) -> Dict[str, Tuple[Optional[BBox3D], Optional[BBox2D]]]:
    """
    Tính bounding box 3D/2D cho tất cả object trong một frame.

    Quy trình:
        1. Xác định arc index từ img_key để lấy camera parameters
        2. Với mỗi object:
            - Lấy kích thước thực tế từ obj_dims_map
            - Tạo 8 điểm góc của bounding box trong hệ tọa độ object
            - Biến đổi sang hệ tọa độ camera: T_cam_obj = T_cam_arc0 @ T_arc0_obj
            - Tính bbox3D từ 8 điểm góc
            - Chiếu 8 điểm xuống ảnh để có bbox2D

    Args:
        img_key: Key của frame, ví dụ 'arc0_image7'
        pose_cam_wrt_arc0: Ma trận (4,4) từ camera_poses[img_key]
        object_poses: List từ object_pose_wrt_arc0_image7.json,
                     mỗi phần tử có 'Model' và 'trans' (4x4)
        obj_dims_map: Map Model -> ObjectDimensions (mm)
        rig_params_by_arc: Map arc_idx -> CameraRigParams
        image_size: (width, height) của ảnh RGB

    Returns:
        Dict[str, Tuple[Optional[BBox3D], Optional[BBox2D]]]:
            Dictionary với key là Model ID, value là (bbox3d, bbox2d)
            Có thể là None nếu thiếu dữ liệu hoặc tính toán thất bại
    """
    # Xác định arc index để lấy camera parameters
    arc_idx = parse_arc_from_key(img_key)
    
    if arc_idx not in rig_params_by_arc:
        print(f"Warning: Không tìm thấy camera parameters cho arc {arc_idx}")
        return {}
        
    rig = rig_params_by_arc[arc_idx]
    color_intr = rig.color_K

    width, height = image_size

    result: Dict[str, Tuple[Optional[BBox3D], Optional[BBox2D]]] = {}

    for obj in object_poses:
        # Lấy model ID
        model_id = obj.get("Model")
        if not model_id:
            continue
            
        # Lấy ma trận biến đổi
        trans = obj.get("trans")
        if not trans:
            result[model_id] = (None, None)
            continue
            
        trans = np.array(trans, dtype=float)
        if trans.shape != (4, 4):
            result[model_id] = (None, None)
            continue

        # Lấy kích thước object
        dims = obj_dims_map.get(model_id)
        if dims is None:
            print(f"Warning: Không tìm thấy kích thước cho object {model_id}")
            result[model_id] = (None, None)
            continue

        # T_arc0_obj: object -> arc0_image7 camera
        T_arc0_obj = trans
        
        # T_cam_arc0: arc0_image7 camera -> camera hiện tại
        T_cam_arc0 = pose_cam_wrt_arc0

        # object -> camera hiện tại
        T_cam_obj = T_cam_arc0 @ T_arc0_obj

        # Tạo 8 điểm góc của bounding box trong hệ object
        corners_local = make_aabb_corners_local(dims)  # (8,4)
        
        # Chuyển sang hệ camera
        cam_xyz = transform_points(T_cam_obj, corners_local)  # (8,3)

        # Tính bbox3D
        bbox3d = bbox3d_from_corners(cam_xyz)
        
        # Chiếu xuống ảnh để có bbox2D
        uv = project_points(color_intr, cam_xyz)
        bbox2d = bbox2d_from_projected(uv, width, height)

        result[model_id] = (bbox3d, bbox2d)

    return result


# Hàm tiện ích để kiểm tra kết quả
def visualize_bbox_on_image(
    image_path: Path,
    bbox2d: BBox2D,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Vẽ bounding box 2D lên ảnh (dùng để kiểm tra).

    Args:
        image_path: Đường dẫn đến file ảnh
        bbox2d: Bounding box 2D cần vẽ
        color: Màu sắc (B, G, R)
        thickness: Độ dày nét vẽ

    Returns:
        np.ndarray: Ảnh đã được vẽ bounding box
    """
    import cv2
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    x, y, w, h = int(bbox2d.x), int(bbox2d.y), int(bbox2d.w), int(bbox2d.h)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    return img


__all__ = [
    "CameraIntrinsics",
    "CameraRigParams",
    "ObjectDimensions",
    "load_camera_rig_params",
    "load_object_dimensions",
    "compute_object_bboxes_for_frame",
    "parse_arc_from_key",
    "project_points",
    "bbox2d_from_projected",
    "bbox3d_from_corners",
    "visualize_bbox_on_image",
]