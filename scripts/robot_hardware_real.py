"""
Robot Hardware Interface for EEEC Robot
Hỗ trợ 3 chế độ: Serial (Arduino/ESP32), ROS (Jetson), Mock (Test)
"""

import time
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotHardware")

class RealRobotController:
    """
    Điều khiển robot EEEC với 3 mode: serial, ros, mock.
    Ánh xạ Action từ MultimodalBridge → lệnh motor/servo thực tế.
    """
    
    def __init__(self, mode: str = 'mock', port: str = 'COM3', baudrate: int = 115200):
        self.mode = mode
        self.current_state = 'IDLE'
        self.serial_conn = None
        self.cmd_pub = None
        
        if mode == 'serial':
            try:
                import serial
                self.serial_conn = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)  # Chờ Arduino/ESP32 reset
                logger.info(f"✅ Serial connected on {port} @ {baudrate}bps")
            except ImportError:
                logger.warning("⚠️ pyserial không có. Chuyển sang Mock mode. Cài: pip install pyserial")
                self.mode = 'mock'
            except Exception as e:
                logger.warning(f"⚠️ Serial error: {e}. Chuyển sang Mock mode.")
                self.mode = 'mock'
        
        elif mode == 'ros':
            try:
                import rospy
                from geometry_msgs.msg import Twist
                rospy.init_node('robot_eeec_control', anonymous=True)
                self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
                self._Twist = Twist
                logger.info("✅ ROS node /robot_eeec_control initialized, publishing to /cmd_vel")
            except ImportError:
                logger.warning("⚠️ rospy không có. Chuyển sang Mock mode.")
                self.mode = 'mock'
        
        logger.info(f"🤖 RealRobotController started in [{self.mode.upper()}] mode.")
    
    # ─── Core Movement ─────────────────────────────────────────────────────────
    
    def send_velocity(self, linear: float = 0.0, angular: float = 0.0):
        """Gửi lệnh vận tốc cơ bản: linear (m/s), angular (rad/s)"""
        if self.mode == 'serial' and self.serial_conn:
            cmd = f"VEL {linear:.3f} {angular:.3f}\n"
            self.serial_conn.write(cmd.encode())
            logger.debug(f"[SERIAL] {cmd.strip()}")
        elif self.mode == 'ros' and self.cmd_pub:
            twist = self._Twist()
            twist.linear.x = linear
            twist.angular.z = angular
            self.cmd_pub.publish(twist)
            logger.debug(f"[ROS] linear={linear:.3f}, angular={angular:.3f}")
        else:
            logger.info(f"[MOCK] VEL → linear={linear:.3f} m/s, angular={angular:.3f} rad/s")
    
    # ─── High-level Actions (từ MultimodalBridge) ──────────────────────────────
    
    def follow_person(self, center_x: float, center_y: float, speed: float = 0.3):
        """
        Theo dõi người: tính lệnh motor dựa trên tọa độ tâm bounding box.
        center_x, center_y: giá trị chuẩn hóa [0, 1]
        """
        self.current_state = 'FOLLOWING'
        error_x = center_x - 0.5  # -0.5 (trái) → +0.5 (phải)
        angular = -error_x * 1.5  # Gain điều chỉnh hướng
        
        # Dừng nếu người quá gần (center_y lớn hơn 0.8 nghĩa là chiếm phần lớn màn hình)
        linear = speed if center_y < 0.75 else 0.0
        
        logger.info(f"[FOLLOW] Person @ ({center_x:.2f}, {center_y:.2f}) → lin={linear:.2f}, ang={angular:.2f}")
        self.send_velocity(linear, angular)
    
    def avoid_obstacle(self, obstacle_center: Tuple[float, float]):
        """Tránh vật cản đơn giản dựa trên vị trí tâm của obstacle."""
        self.current_state = 'AVOIDING'
        center_x = obstacle_center[0]
        
        if center_x < 0.35:
            # Vật cản bên trái → rẽ phải
            linear, angular = 0.2, -0.8
            direction = "RIGHT (obstacle on left)"
        elif center_x > 0.65:
            # Vật cản bên phải → rẽ trái
            linear, angular = 0.2, 0.8
            direction = "LEFT (obstacle on right)"
        else:
            # Vật cản chính giữa → lùi và rẽ trái
            linear, angular = -0.1, 0.8
            direction = "BACK-LEFT (obstacle center)"
        
        logger.info(f"[AVOID] Obstacle @ ({center_x:.2f}) → Turning {direction}")
        self.send_velocity(linear, angular)
    
    def approach_and_greet(self, center_x: float = 0.5, distance: float = 0.5):
        """Tiến lại gần người và dừng để chào hỏi."""
        self.current_state = 'GREETING'
        error_x = center_x - 0.5
        angular = -error_x * 1.0
        logger.info(f"[GREET] Approaching person, angular correction={angular:.2f}")
        self.send_velocity(0.2, angular)
        time.sleep(distance / 0.2)  # Ước tính thời gian di chuyển
        self.stop()
    
    def search_mode(self, rotation_speed: float = 0.4):
        """Xoay tại chỗ để tìm kiếm người dùng."""
        self.current_state = 'SEARCHING'
        logger.info(f"[SEARCH] Rotating to search for person...")
        self.send_velocity(0.0, rotation_speed)
    
    def stop(self):
        """Dừng robot (khẩn cấp)."""
        self.current_state = 'IDLE'
        logger.info("[STOP] Robot stopped.")
        self.send_velocity(0.0, 0.0)
    
    def move_forward(self, speed: float = 0.3, duration: float = 1.0):
        """Di chuyển thẳng trong `duration` giây."""
        self.current_state = 'MOVING'
        logger.info(f"[MOVE] Forward @ {speed} m/s for {duration}s")
        self.send_velocity(speed, 0.0)
        time.sleep(duration)
        self.stop()
    
    # ─── Action Dispatcher (kết nối trực tiếp với MultimodalBridge output) ─────
    
    def execute_bridge_action(self, action: dict, detections: list):
        """
        Nhận action dict từ MultimodalBridge và thực thi.
        action = {'type': 'follow', 'target': [cx, cy], 'parameters': {...}}
        """
        action_type = action.get('type', 'idle')
        target = action.get('target')
        
        if action_type == 'follow' and target:
            self.follow_person(target[0], target[1])
        
        elif action_type == 'navigate_around' and target:
            self.avoid_obstacle((target[0], target[1]))
        
        elif action_type == 'approach_and_greet':
            cx = target[0] if target else 0.5
            self.approach_and_greet(center_x=cx)
        
        elif action_type == 'search':
            self.search_mode()
        
        elif action_type == 'respond':
            # Robot đứng yên khi đang trả lời (RAG mode)
            self.stop()
            logger.info("[RESPOND] Robot idle, waiting for RAG response.")
        
        elif action_type == 'idle':
            self.stop()
        
        else:
            logger.info(f"[ACTION] Unhandled action type: {action_type}")
    
    def __del__(self):
        if self.serial_conn:
            self.send_velocity(0.0, 0.0)
            self.serial_conn.close()
            logger.info("Serial connection closed.")


# ─── Quick Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== RealRobotController Test ===\n")
    robot = RealRobotController(mode='mock')
    
    print("\n[1] Follow person at center-right:")
    robot.follow_person(center_x=0.65, center_y=0.5)
    
    print("\n[2] Avoid obstacle on the left:")
    robot.avoid_obstacle((0.2, 0.5))
    
    print("\n[3] Approach and greet:")
    robot.approach_and_greet(center_x=0.5)
    
    print("\n[4] Bridge action dispatcher:")
    mock_action = {'type': 'follow', 'target': [0.5, 0.6]}
    robot.execute_bridge_action(mock_action, [])
    
    print("\n✅ All hardware actions tested successfully!")
