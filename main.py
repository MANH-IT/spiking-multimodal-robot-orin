import cv2
import torch
import collections
from camera.camera import Camera
from detection.yolo_detector import YOLODetector
from tracking.deepsort_tracker import ObjectTracker
from multimodal_fusion.bridges.vision_nlp_bridge import VisionNLPBridge
from scripts.robot_control import RobotController

def main():
    # 1. Initialize Components
    cam = Camera(0)
    detector = YOLODetector('models/yolov8n.pt')
    tracker = ObjectTracker()
    controller = RobotController(mode="mock")
    
    # 2. Initialize Spiking Multimodal Bridge
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ưu tiên load model SNN Vision vừa huấn luyện
    vision_model_path = 'models/best_vision_snn.pth'
    bridge = VisionNLPBridge(vision_model_path=vision_model_path, device=device)
    
    # Frame buffer for 3D SNN (T=20)
    frame_buffer = collections.deque(maxlen=20)
    current_intent = "Giám sát"
    current_response = "Đang chờ yêu cầu..."

    print(f"🚀 Starting Spiking Robot AI System on {device}...")
    print("💡 Press 's' to trigger speech/text input mock.")
    print("💡 Press 'q' to quit.")

    while True:
        # 3. Capture Frame
        ret, frame = cam.get_frame()
        if not ret:
            print("❌ Không lấy được frame")
            break

        # Thêm vào buffer để xử lý 3D SNN
        resized_frame = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
        frame_buffer.append(frame_tensor)

        # 4. Object Detection & Tracking (Legacy / 2D)
        results = detector.detect(frame)
        tracker_input = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                w, h = x2 - x1, y2 - y1
                tracker_input.append([[int(x1), int(y1), int(w), int(h)], float(box.conf[0]), int(box.cls[0])])

        tracks = tracker.update(frame, tracker_input)

        # 5. Multimodal Processing (Key Trigger 's')
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(frame_buffer) == 20:
                print("🧠 Processing Spiking Multimodal Fusion...")
                # Giả lập text input - bạn có thể đổi query ở đây
                query = "Bạn có thể dẫn tôi đến thư viện không?" 
                
                # Stack frames (T, C, H, W)
                rgb_seq = torch.stack(list(frame_buffer), dim=0)
                
                # Xử lý qua Bridge (Sơ đồ SNN Multimodal)
                result = bridge.process(rgb_seq, query)
                current_intent = result.action
                current_response = result.speech_response
                
                print(f"🎯 INTENT: {current_intent}")
                print(f"💬 AI: {current_response}")
                
                # 6. Điều khiển hardware robot thực tế dựa trên Intent
                controller.execute_action(current_intent)
            else:
                print(f"⏳ Đang thu thập frame ({len(frame_buffer)}/20)...")

        # 6. Draw Results
        for track in tracks:
            if not track.is_confirmed(): continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (600, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Intent: {current_intent}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"AI: {current_response[:50]}...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("UTC Robot - Spiking Multimodal AI", frame)

        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()