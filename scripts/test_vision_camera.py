# scripts/test_vision_camera.py
"""
Test vision với webcam sau khi retrain
"""

import cv2
import torch
import numpy as np
import sys
import os
import time
import collections
from pathlib import Path

# Add root to sys.path
root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

from vision_system.snn_wrapper import SNNVisionWrapper

def test_vision():
    print("🎥 Testing Vision System with Webcam...")
    
    # Path to the new model
    model_path = 'models/best_vision_snn.pth'
    if not os.path.exists(model_path):
        model_path = 'models/vision/best_vision_snn.pth'

    # Load model
    try:
        vision = SNNVisionWrapper(
            model_path=model_path,
            T=8,
            input_size=128,
            num_classes=2
        )
        print(f"✅ Vision system loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load vision system: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera ready. Press 'q' to quit")
    print("📊 Showing detections in real-time...")
    
    # Frame buffer for T=8
    frame_buffer = collections.deque(maxlen=8)
    
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Add to buffer
        frame_buffer.append(frame)
        
        # Process if buffer is full
        if len(frame_buffer) == 8:
            try:
                # SNNVisionWrapper expects list of frames
                boxes, conf, cls = vision.process_frame_sequence(list(frame_buffer))
                
                # Draw detections
                # boxes: (B, N, 4)
                for i in range(len(boxes[0])):
                    score = float(conf[0][i][0])
                    if score > 0.35:
                        bbox = boxes[0][i]
                        x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
                        cls_idx = int(cls[0][i].argmax())
                        label_name = "person" if cls_idx == 0 else "obstacle"
                        
                        color = (0, 255, 0) if cls_idx == 0 else (0, 0, 255)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"{label_name}: {score:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                cv2.putText(display_frame, f"Error: {e}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Calculate FPS
        frame_count += 1
        if time.time() - start_time > 1:
            fps = frame_count
            frame_count = 0
            start_time = time.time()
        
        # Show info
        cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Buffer: {len(frame_buffer)}/8", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('SNN Vision Test - Press q to quit', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test complete")

if __name__ == "__main__":
    test_vision()
