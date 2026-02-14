"""
Module đọc dữ liệu HILO dataset.
Xử lý các file JSON và cấu trúc thư mục của HILO.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Union


@dataclass
class HILOScenePaths:
    """
    Đại diện cho một scene trong HILO_Scenes.
    Chỉ quản lý đường dẫn, không load dữ liệu nặng.
    """

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def camera_poses_path(self) -> Path:
        return self.root / "camera_poses.json"

    @property
    def object_pose_path(self) -> Path:
        return self.root / "object_pose_wrt_arc0_image7.json"

    @property
    def object_id_list_path(self) -> Path:
        return self.root / "object_id_list.json"

    @property
    def rgb_raw_dir(self) -> Path:
        return self.root / "rgb_raw"

    @property
    def rgb_undistorted_dir(self) -> Path:
        return self.root / "rgb_undistorted"

    @property
    def depth_raw_dir(self) -> Path:
        return self.root / "depth_raw"

    @property
    def depth_undistorted_dir(self) -> Path:
        return self.root / "depth_undistorted"

    @property
    def mask_undistorted_dir(self) -> Path:
        return self.root / "mask_undistorted"


class HILODatasetLoader:
    """
    Loader mức thấp cho HILO_Dataset/HILO_Scenes.
    """

    def __init__(self, dataset_root: Path) -> None:
        """
        :param dataset_root: Thư mục HILO_Scenes
        """
        self.dataset_root = dataset_root
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"HILO_Scenes root not found: {self.dataset_root}")

    def iter_scenes(self) -> Iterator[HILOScenePaths]:
        for scene_dir in sorted(self.dataset_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            yield HILOScenePaths(root=scene_dir)

    @staticmethod
    def _read_json(path: Path) -> Any:
        """Đọc file JSON với xử lý lỗi."""
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  ERROR: Không thể đọc {path}: {e}")
            return None

    def load_camera_poses(self, scene: HILOScenePaths) -> Dict[str, np.ndarray]:
        """
        Đọc camera_poses.json và trả về dictionary các ma trận 4x4.
        
        Định dạng HILO:
        {
            "arc0_image0": [
                [r11, r12, r13, tx],
                [r21, r22, r23, ty],
                [r31, r32, r33, tz],
                [0,   0,   0,   1]
            ],
            ...
        }
        """
        data = self._read_json(scene.camera_poses_path)
        if data is None:
            return {}
        
        result = {}
        
        # HILO dùng dictionary với key là tên frame
        if isinstance(data, dict):
            for frame_name, pose_matrix in data.items():
                try:
                    # Kiểm tra pose_matrix có đúng định dạng list of lists không
                    if isinstance(pose_matrix, list) and len(pose_matrix) == 4:
                        # Kiểm tra mỗi hàng có 4 phần tử
                        valid = True
                        matrix_data = []
                        for i, row in enumerate(pose_matrix):
                            if isinstance(row, list) and len(row) == 4:
                                matrix_data.extend(row)
                            else:
                                valid = False
                                print(f"    WARNING: {frame_name} hàng {i} không hợp lệ: {row}")
                                break
                        
                        if valid and len(matrix_data) == 16:
                            # Chuyển thành numpy array 4x4
                            matrix = np.array(matrix_data, dtype=float).reshape(4, 4)
                            result[frame_name] = matrix
                        else:
                            print(f"    WARNING: {frame_name} không phải ma trận 4x4 hợp lệ")
                    else:
                        print(f"    WARNING: {frame_name} có định dạng không xác định: {type(pose_matrix)}")
                        
                except Exception as e:
                    print(f"    ERROR: Lỗi xử lý {frame_name}: {e}")
                    continue
        
        elif isinstance(data, list):
            # Một số phiên bản HILO có thể dùng list
            print(f"  DEBUG: camera_poses là list với {len(data)} phần tử")
            for i, item in enumerate(data):
                if isinstance(item, dict) and 'frame' in item and 'pose' in item:
                    frame_name = item['frame']
                    pose_matrix = item['pose']
                    # Xử lý tương tự như trên
                    try:
                        if isinstance(pose_matrix, list) and len(pose_matrix) == 4:
                            matrix_data = []
                            for row in pose_matrix:
                                if isinstance(row, list) and len(row) == 4:
                                    matrix_data.extend(row)
                            
                            if len(matrix_data) == 16:
                                matrix = np.array(matrix_data, dtype=float).reshape(4, 4)
                                result[frame_name] = matrix
                    except Exception as e:
                        print(f"    ERROR: Lỗi xử lý frame {i}: {e}")
        
        print(f"  Đã đọc {len(result)} camera poses hợp lệ")
        return result

    def load_object_poses(self, scene: HILOScenePaths) -> List[Dict[str, Any]]:
        """
        Đọc file object_pose_wrt_arc0_image7.json.
        
        Định dạng HILO:
        [
            {
                "Model": "M_ChipsAhoy",
                "trans": [4x4 matrix - list of lists]
            },
            ...
        ]
        """
        data = self._read_json(scene.object_pose_path)
        if data is None:
            return []
        
        result = []
        
        # Xử lý nếu data là list (format chuẩn của HILO)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    model_id = item.get('Model', '')
                    trans = item.get('trans', [])
                    
                    # Chuyển đổi trans thành ma trận
                    try:
                        if isinstance(trans, list) and len(trans) == 4:
                            matrix_data = []
                            for row in trans:
                                if isinstance(row, list) and len(row) == 4:
                                    matrix_data.extend(row)
                            
                            if len(matrix_data) == 16:
                                matrix = np.array(matrix_data, dtype=float).reshape(4, 4)
                                result.append({
                                    "Model": model_id,
                                    "trans": matrix.tolist()  # Lưu dưới dạng list để dễ xử lý
                                })
                            else:
                                print(f"    WARNING: Object {model_id} có trans không hợp lệ")
                        else:
                            print(f"    WARNING: Object {model_id} có trans không phải ma trận 4x4")
                    except Exception as e:
                        print(f"    ERROR: Lỗi xử lý object {model_id}: {e}")
        
        # Xử lý nếu data là dict (một số phiên bản khác)
        elif isinstance(data, dict):
            for model_id, trans in data.items():
                try:
                    if isinstance(trans, list) and len(trans) == 4:
                        matrix_data = []
                        for row in trans:
                            if isinstance(row, list) and len(row) == 4:
                                matrix_data.extend(row)
                        
                        if len(matrix_data) == 16:
                            matrix = np.array(matrix_data, dtype=float).reshape(4, 4)
                            result.append({
                                "Model": model_id,
                                "trans": matrix.tolist()
                            })
                except Exception as e:
                    print(f"    ERROR: Lỗi xử lý object {model_id}: {e}")
        
        print(f"  Đã đọc {len(result)} object poses")
        return result

    def load_object_id_list(self, scene: HILOScenePaths) -> List[str]:
        """
        Đọc file object_id_list.json - danh sách object IDs trong scene.
        """
        data = self._read_json(scene.object_id_list_path)
        if data is None:
            return []
        
        if isinstance(data, list):
            return data
        else:
            print(f"  WARNING: object_id_list không phải list: {type(data)}")
            return []


def get_default_hilo_scenes_root() -> Path:
    """
    Trả về đường dẫn mặc định tới HILO_Scenes.
    """
    return (
        Path.cwd()
        / "data"
        / "00_raw"
        / "vision"
        / "HILO_Dataset"
        / "HILO_Scenes"
    )


__all__ = [
    "HILOScenePaths",
    "HILODatasetLoader",
    "get_default_hilo_scenes_root",
]