# scripts/yolo_to_snn_collector.py
import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import base64
import sys

# Add root to sys.path
root = Path(__file__).parent.parent.absolute()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

class YOLOTeacherCollector:
    def __init__(self, json_path='data/real_vision.json'):
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO model
        print("⏳ Loading YOLOv8 Teacher Model...")
        self.model = YOLO('yolov8n.pt')
        
        # Classes: 0=person. Gộp các class COCO khác thành obstacle
        self.target_classes = {0: 'person'} 
        # Danh sách các class COCO thường gặp làm vật cản
        self.obstacle_ids = [24, 26, 56, 62, 63, 67] 
        
        self.samples = []
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.samples = json.load(f)
                print(f"📂 Loaded {len(self.samples)} existing samples.")
            except: pass

    def collect_auto(self, duration_sec=300, fps_limit=2):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        start_time = time.time()
        last_capture_time = 0
        new_count = 0
        
        print(f"🎥 YOLO Teacher Auto-Labeling for {duration_sec}s...")
        print("   Mode: Distillation (YOLO -> SNN)")
        print("   Press 'q' to stop early")
        
        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret: break
            
            # Chỉ xử lý theo fps_limit để tránh dữ liệu quá trùng lặp
            if time.time() - last_capture_time >= (1.0 / fps_limit):
                # Inference
                results = self.model(frame, verbose=False)[0]
                
                found_in_frame = False
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf >= 0.45: # Chỉ lấy nhãn có độ tin cậy cao
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = 'person' if cls_id == 0 else 'obstacle'
                        
                        # Encode image to B64
                        _, buffer = cv2.imencode('.jpg', frame)
                        img_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Save in SNN format [x, y, w, h]
                        self.samples.append({
                            'image_b64': img_b64,
                            'bbox': [x1, y1, x2-x1, y2-y1],
                            'class': label,
                            'source': 'yolo_teacher',
                            'timestamp': time.time()
                        })
                        new_count += 1
                        found_in_frame = True
                        last_capture_time = time.time()
                        break # Mỗi frame chỉ lấy 1 object chính để đơn giản hóa training
                
                # Visualize
                annotated = results.plot()
                cv2.putText(annotated, f"New Samples: {new_count}", (10, 30), 1, 1.5, (0,255,0), 2)
                cv2.imshow('YOLO Teacher -> SNN Data Collection', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()
        
        # Save results
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, indent=2)
        print(f"✅ Total samples: {len(self.samples)} (New: {new_count})")
        print(f"📂 Saved to {self.json_path}")

if __name__ == "__main__":
    collector = YOLOTeacherCollector()
    # Chạy thu thập 5 phút, giới hạn 2 ảnh/giây để đa dạng hóa data
    collector.collect_auto(duration_sec=300, fps_limit=2)
