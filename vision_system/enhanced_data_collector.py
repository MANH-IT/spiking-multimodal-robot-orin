"""
Enhanced Data Collector
========================
Tích hợp GUI hướng dẫn, manual bbox drawing và COCO format export.
"""

import cv2
import json
import numpy as np
import time
import base64
from pathlib import Path
from datetime import datetime
import os

class EnhancedDataCollector:
    def __init__(self, save_path='data/real_vision.json'):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.save_path.parent / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples = []
        self.current_bbox = None # [x1, y1, x2, y2]
        self.drawing = False
        self.class_name = 'person' # Default mode
        
        # Load existing
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    self.samples = json.load(f)
                print(f"📂 Loaded {len(self.samples)} existing samples")
            except: pass

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_bbox = [x, y, x, y]
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_bbox[2] = x
            self.current_bbox[3] = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_bbox[2] = x
            self.current_bbox[3] = y

    def run_collector(self, camera_id=0, duration=600):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
            
        win_name = 'EEEC Data Collector - P=Person, O=Obstacle, S=Save, Q=Quit'
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.mouse_callback)
        
        start_time = time.time()
        print(f"🎥 Collecting data for {duration}s...")
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret: break
                
            display_frame = frame.copy()
            H, W = frame.shape[:2]
            
            # Draw current bbox
            if self.current_bbox:
                color = (0, 255, 0) if self.class_name == 'person' else (0, 0, 255)
                cv2.rectangle(display_frame, 
                            (self.current_bbox[0], self.current_bbox[1]),
                            (self.current_bbox[2], self.current_bbox[3]),
                            color, 2)
                cv2.putText(display_frame, f"Mode: {self.class_name.upper()}", 
                           (self.current_bbox[0], self.current_bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # UI Overlay
            cv2.rectangle(display_frame, (0, 0), (W, 40), (0,0,0), -1)
            status = f"Samples: {len(self.samples)} | Mode: {self.class_name.upper()} | Time: {int(duration - (time.time()-start_time))}s"
            cv2.putText(display_frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            cv2.imshow(win_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.class_name = 'person'
            elif key == ord('o'):
                self.class_name = 'obstacle'
            elif key == ord('s'):
                if self.current_bbox and abs(self.current_bbox[2]-self.current_bbox[0]) > 5:
                    # Convert [x1,y1,x2,y2] -> [x,y,w,h] for internal storage consistency
                    x1, y1, x2, y2 = self.current_bbox
                    bbox = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
                    
                    # Store
                    _, buf = cv2.imencode('.jpg', frame)
                    img_b64 = base64.b64encode(buf).decode()
                    
                    self.samples.append({
                        'image_b64': img_b64,
                        'bbox': bbox,
                        'class': self.class_name,
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"✅ Saved {self.class_name} | {bbox}")
                    self.current_bbox = None
                else:
                    print("⚠️ Draw a valid box first!")
            elif key == ord('q') or key == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.save_all()

    def save_all(self):
        if not self.samples: return
        
        # Save JSON
        with open(self.save_path, 'w') as f:
            json.dump(self.samples, f, indent=2)
            
        # Save COCO
        coco = {
            'images': [], 'annotations': [],
            'categories': [{'id': 0, 'name': 'person'}, {'id': 1, 'name': 'obstacle'}]
        }
        
        for idx, s in enumerate(self.samples):
            img_name = f"real_{idx:04d}.jpg"
            img_path = self.img_dir / img_name
            
            # Write image file
            img_data = base64.b64decode(s['image_b64'])
            with open(img_path, 'wb') as f:
                f.write(img_data)
                
            coco['images'].append({
                'id': idx, 'file_name': img_name, 'width': 640, 'height': 480
            })
            coco['annotations'].append({
                'id': idx, 'image_id': idx,
                'category_id': 0 if s['class'] == 'person' else 1,
                'bbox': s['bbox'],
                'area': s['bbox'][2] * s['bbox'][3],
                'iscrowd': 0
            })
            
        with open(self.save_path.parent / "real_vision_coco.json", 'w') as f:
            json.dump(coco, f, indent=2)
            
        print(f"🚀 Saved {len(self.samples)} samples (JSON + COCO)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--duration', type=int, default=600)
    args = parser.parse_args()
    
    collector = EnhancedDataCollector()
    collector.run_collector(camera_id=args.camera, duration=args.duration)
