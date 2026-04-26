# Task: [D] Safety Override Mechanism
# File: d:\robot_eeec\scripts\safety_controller.py

import numpy as np
import logging

logger = logging.getLogger("SafetyController")

class SafetyController:
    """
    Cơ chế an toàn: Tự động dừng robot khi phát hiện vật cản quá gần.
    Hoạt động độc lập với lệnh từ Fusion Transformer.
    """
    def __init__(self, min_distance: float = 0.5):
        self.min_distance = min_distance
        self.emergency_stop = False
        
    def check_safety(self, depth_map: np.ndarray, current_action: dict) -> dict:
        """
        Kiểm tra bản đồ độ sâu để quyết định có dừng khẩn cấp không.
        
        Args:
            depth_map: (H, W) array, giá trị 0-1 (càng gần 1 càng gần, hoặc tùy theo model)
                       Trong MiDaS-small, giá trị cao = gần.
            current_action: action dict hiện tại {'type': ..., 'target': ...}
            
        Returns:
            action: Action đã được kiểm tra (có thể bị ghi đè thành 'idle')
        """
        if depth_map is None:
            return current_action
            
        # Tính toán giá trị đại diện cho khoảng cách gần nhất
        # Giả sử depth_map được chuẩn hóa 0-1 (1 là gần nhất)
        # Nếu dùng MiDaS, ta có thể lấy vùng trung tâm 
        h, w = depth_map.shape
        center_region = depth_map[h//4:3*h//4, w//4:3*w//4]
        max_proximity = np.max(center_region)
        
        # Ngưỡng an toàn (giả định 0.8 là rất gần)
        SAFETY_THRESHOLD = 0.85 
        
        if max_proximity > SAFETY_THRESHOLD:
            if not self.emergency_stop:
                logger.warning(f"🚨 EMERGENCY STOP! Object detected at high proximity ({max_proximity:.2f})")
                self.emergency_stop = True
            
            # Ghi đè hành động thành dừng lại
            return {
                'type': 'idle',
                'target': None,
                'reason': 'safety_override',
                'proximity': float(max_proximity),
                'response': "Dừng lại! Có vật cản quá gần."
            }
        
        self.emergency_stop = False
        return current_action

if __name__ == "__main__":
    # Test simple
    sc = SafetyController()
    safe_map = np.zeros((128, 128))
    danger_map = np.ones((128, 128))
    
    action = {'type': 'follow', 'target': [0.5, 0.5]}
    
    print(f"Safe test: {sc.check_safety(safe_map, action)['type']}")
    print(f"Danger test: {sc.check_safety(danger_map, action)['type']}")
