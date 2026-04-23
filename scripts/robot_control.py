import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotControl")

class RobotController:
    """
    Điều khiển di chuyển cho Robot EEEC.
    Trong môi trường thực tế, lớp này sẽ giao tiếp với ROS hoặc GPIO/Serial.
    """
    def __init__(self, mode="mock"):
        self.mode = mode
        logger.info(f"🤖 Robot Controller initialized in {mode} mode.")
        self.current_state = "IDLE"

    def execute_action(self, intent):
        """Thực thi hành động dựa trên Intent từ AI"""
        logger.info(f"🎯 Executing action for intent: {intent}")
        
        if intent == "Chào hỏi":
            self.wave_hand()
        elif intent == "Tìm phòng":
            self.search_mode()
        elif intent == "Dẫn đường":
            self.move_forward()
        elif intent == "Dừng lại":
            self.stop()
        else:
            logger.info("⏸️ No specific movement for this intent.")

    def move_forward(self, duration=2):
        logger.info("🚀 Moving Forward...")
        self.current_state = "MOVING_FORWARD"
        if self.mode == "real":
            # Gửi lệnh qua Serial/ROS ở đây
            pass
        time.sleep(duration)
        self.stop()

    def turn_left(self, angle=90):
        logger.info(f"↪️ Turning Left {angle} degrees...")
        self.current_state = "TURNING_LEFT"
        # Logic rẽ
        time.sleep(1)
        self.stop()

    def turn_right(self, angle=90):
        logger.info(f"↩️ Turning Right {angle} degrees...")
        self.current_state = "TURNING_RIGHT"
        # Logic rẽ
        time.sleep(1)
        self.stop()

    def stop(self):
        logger.info("🛑 Stopping.")
        self.current_state = "IDLE"

    def wave_hand(self):
        logger.info("👋 Waving Hand / Greeting...")
        # Lệnh điều khiển servo tay
        time.sleep(2)

    def search_mode(self):
        logger.info("🔍 Entering Search Mode (Rotating)...")
        self.turn_left(360)

if __name__ == "__main__":
    # Test controller
    controller = RobotController()
    controller.execute_action("Dẫn đường")
    controller.execute_action("Chào hỏi")
