import cv2
import torch
import collections
import numpy as np
import os

# Import components
import sys
sys.path.append(os.getcwd())

from vision_system.snn_wrapper import SNNVisionWrapper
from multimodal_fusion.learned_bridge import LearnedMultimodalBridge
from nlp_advanced.real_nlp_processor import RealNLPProcessor
from scripts.robot_hardware_real import RealRobotController
from scripts.safety_controller import SafetyController # NEW

# Configuration
CONFIG = {
    'snn': {
        'enabled': True,
        'T': 8,
        'input_size': 128,
        'model_path': 'models/best_vision_snn.pth',
        'num_classes': 2
    },
    'nlp': {
        'enabled': True,
        'intent_model': 'nlp_advanced/best_advanced_nlu.pth',
        'use_rag': True
    },
    'fusion': {
        'model_path': 'models/fusion/best_fusion.pth',  # ← Learned Fusion Transformer
        'confidence_thr': 0.6,   # Dưới ngưỡng này → yêu cầu xác nhận
        'use_phobert': False,    # True nếu có GPU đủ RAM cho PhoBERT
    },
    'robot': {
        'mode': 'mock',  # ← Đổi sang 'serial' hoặc 'ros' khi có hardware thật
        'port': 'COM3',  # Serial port (Windows: COM3, Linux: /dev/ttyUSB0)
        'baudrate': 115200,
        'max_speed': 0.5,
        'safety_distance': 0.5
    }
}

def main():
    print("=" * 50)
    print("🤖 Robot EEEC - Multimodal AI System")
    print("=" * 50)
    
    # Initialize components
    print("\n🔧 Initializing...")
    
    # 1. SNN Vision
    vision = None
    if CONFIG['snn']['enabled']:
        try:
            vision = SNNVisionWrapper(
                model_path=CONFIG['snn']['model_path'],
                T=CONFIG['snn']['T'],
                input_size=CONFIG['snn']['input_size'],
                num_classes=CONFIG['snn']['num_classes']
            )
            print(f"✅ SNN Vision: {vision.get_fps_estimate():.1f} FPS")
        except Exception as e:
            print(f"❌ SNN Vision init failed: {e}")
    
    # 2. NLP Processor
    nlp = None
    if CONFIG['nlp']['enabled']:
        try:
            nlp = RealNLPProcessor(intent_model_path=CONFIG['nlp']['intent_model'])
            print("✅ NLP Processor ready")
        except Exception as e:
            print(f"⚠️ NLP init failed: {e}, using rule-based fallback")
            nlp = RealNLPProcessor(intent_model_path="")
    
    # 3. Multimodal Bridge (Learned Fusion Transformer)
    bridge = LearnedMultimodalBridge(
        vision_model=vision,
        nlp_processor=nlp,
        model_path=CONFIG['fusion']['model_path'],
        config={
            'confidence_thr': CONFIG['fusion']['confidence_thr'],
            'use_phobert':    CONFIG['fusion']['use_phobert'],
        },
        fallback_to_rules=True,  # Fallback sang rule-based nếu model fail
    )
    print("✅ Multimodal Bridge (Fusion Transformer) ready")
    
    # 4. Safety Controller
    safety = SafetyController(min_distance=CONFIG['robot']['safety_distance'])
    
    # 5. Robot Hardware Controller
    robot = RealRobotController(
        mode=CONFIG['robot']['mode'],
        port=CONFIG['robot']['port'],
        baudrate=CONFIG['robot']['baudrate']
    )
    
    # 4. Camera
    cap = cv2.VideoCapture(0)
    mock_mode = False
    if not cap.isOpened():
        print("❌ Cannot open camera, running in MOCK mode.")
        mock_mode = True
    else:
        print("✅ Camera ready")
    
    # Frame buffer
    frame_buffer = collections.deque(maxlen=CONFIG['snn']['T'])
    
    print("\n🎮 System ready!")
    print("Commands:")
    print("  - Type 'q' to quit")
    print("  - Type 's' to toggle SNN")
    print("  - Type any text command in terminal and press Enter\n")
    
    snn_active = True
    last_command = ""
    
    # Để nhận input không chặn (non-blocking) trên Windows có thể khó,
    # nên ta sẽ dùng logic đơn giản: mỗi khi nhấn phím 'c' thì cho nhập command.
    
    while True:
        if mock_mode:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret: break
        
        # Prepare RGB-D frame
        frame_resized = cv2.resize(frame, (CONFIG['snn']['input_size'], CONFIG['snn']['input_size']))
        rgb_normalized = frame_resized.astype(np.float32) / 255.0
        depth_channel = np.zeros((CONFIG['snn']['input_size'], CONFIG['snn']['input_size'], 1), dtype=np.float32)
        rgbd_frame = np.concatenate([rgb_normalized, depth_channel], axis=2)
        
        frame_buffer.append(rgbd_frame)
        
        # Process with SNN
        display_frame = frame.copy()
        
        if snn_active and len(frame_buffer) == CONFIG['snn']['T']:
            # Check for command
            user_command = None
            if last_command:
                user_command = last_command
                last_command = ""
            
            # Multimodal processing
            result = bridge.process_frame_with_query(list(frame_buffer), user_command)
            
            # Draw detections
            h, w = frame.shape[:2]
            for det in result['detections']:
                bbox = det['bbox']
                x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
                color = (0, 255, 0) if det['class'] == 'person' else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{det['class']}: {det['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display robot action
            if result.get('action'):
                action = result['action']
                
                # === SAFETY OVERRIDE ===
                # Giả định lấy depth map từ vision (với fallback laplacian)
                # SNNVisionWrapper xử lý depth ngầm định
                # Ta có thể mock hoặc lấy nếu SNN trả về
                # Ở đây ta lấy depth_map giả định từ processing
                depth_map = np.zeros((128, 128)) # Fallback
                if vision and hasattr(vision, 'depth_estimator') and vision.depth_estimator:
                    # Lấy frame cuối cùng
                    depth_map = vision.depth_estimator.estimate(frame)
                
                final_action = safety.check_safety(depth_map, action)
                
                # Execute robot hardware action
                if user_command:
                    robot.execute_bridge_action(final_action, result['detections'])
                
                # Display info
                cv2.putText(display_frame, f"Action: {final_action['type']}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if final_action.get('reason') == 'safety_override':
                    cv2.putText(display_frame, "⚠️ SAFETY OVERRIDE!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif action.get('response'):
                    cv2.putText(display_frame, action['response'][:40], (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show robot state
                cv2.putText(display_frame, f"Robot: {robot.current_state}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Sliding window
            frame_buffer.popleft()
        
        # Display status
        cv2.putText(display_frame, f"Mode: {'SNN' if snn_active else 'OFF'}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Buffer: {len(frame_buffer)}/{CONFIG['snn']['T']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Robot EEEC - Multimodal AI", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snn_active = not snn_active
            print(f"SNN mode: {'ON' if snn_active else 'OFF'}")
        elif key == ord('c'):
            print("\nEnter command: ", end='', flush=True)
            # Lưu ý: cv2.waitKey không chặn được input terminal tốt, 
            # nhưng đây là cách đơn giản nhất cho bản demo.
            cmd = input()
            last_command = cmd
    
    if not mock_mode:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
