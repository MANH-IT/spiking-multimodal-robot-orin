"""
test_real_vision.py - Kiểm tra hệ thống Vision SNN thực tế
"""
import cv2
import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from vision_system.snn_wrapper import SNNVisionWrapper

def main():
    print("🚀 Starting Real-time SNN Vision Test...")
    
    # Init wrapper
    vision = SNNVisionWrapper(
        model_path='models/best_vision_snn.pth',
        use_depth_estimation=False # Dùng Laplacian gradient mặc định trong wrapper
    )
    
    # Đo FPS thực tế
    measured_fps = vision.get_fps_estimate()
    print(f"⏱️  Hardware Benchmark: {measured_fps:.1f} FPS")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Camera not found")
        return

    print("💡 Press 'Q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # SNN xử lý sequence (8 frames) - Ở đây ta giả lập sequence bằng frame hiện tại
        # Trong thực tế wrapper sẽ quản lý buffer nếu cần, nhưng bản hiện tại nhận list frames
        frames_seq = [frame] * 8 
        
        boxes, confs, clss = vision.process_frame_sequence(frames_seq)
        
        # BGR frame để vẽ
        display = frame.copy()
        H, W = display.shape[:2]
        
        # Detections
        # confs shape: (1, N, 1), boxes: (1, N, 4)
        for i in range(confs.shape[1]):
            c = confs[0, i, 0]
            if c > 0.35:
                box = boxes[0, i] # x1, y1, x2, y2 normalized
                x1, y1, x2, y2 = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
                
                cls_id = np.argmax(clss[0, i])
                label = "Person" if cls_id == 0 else "Obstacle"
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{label} {c:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # FPS overlay
        cv2.putText(display, f"Hardware FPS: {measured_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("EEEC SNN Vision - Real-time Test", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
