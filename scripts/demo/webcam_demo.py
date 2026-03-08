#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Webcam Demo cho Vision System - 3D Object Detection

Demo trực quan với:
- Webcam input
- Depth estimation (monocular depth)
- Real-time 3D detection visualization
- FPS counter
- Performance metrics

Usage:
    python scripts/demo/webcam_demo.py
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV không có sẵn. Cài đặt: pip install opencv-python")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch không có sẵn. Sẽ dùng mock model cho demo.")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Webcam Demo cho Vision System"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height",
    )
    parser.add_argument(
        "--fps-display",
        action="store_true",
        help="Hiển thị FPS",
    )
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="Save output video to file",
    )
    return parser.parse_args()


class MockDepthEstimator:
    """Mock depth estimator từ RGB (cho demo)."""
    
    def __init__(self):
        self.name = "Mock Depth Estimator"
    
    def estimate_depth(self, rgb_img):
        """Estimate depth từ RGB image."""
        # Simple mock: depth tỷ lệ nghịch với brightness
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        depth = 255 - gray  # Darker = closer
        depth = depth.astype(np.uint16) * 10  # Scale to mm
        return depth


class MockDetector:
    """Mock 3D detector cho demo."""
    
    def __init__(self):
        self.name = "Mock 3D Detector"
        self.class_names = ["Bottle", "Cup", "Chair", "Table", "Laptop", "Phone", "Book", "Box"]
    
    def detect(self, rgb_img, depth_img):
        """Detect objects trong image."""
        h, w = rgb_img.shape[:2]
        
        # Mock detections - random boxes
        num_detections = np.random.randint(0, 3)
        detections = []
        
        for i in range(num_detections):
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            width = np.random.randint(50, 150)
            height = np.random.randint(50, 150)
            
            # Get depth value
            depth_val = depth_img[y + height//2, x + width//2] if depth_img is not None else 1000
            
            detection = {
                "bbox2d": [x, y, width, height],
                "bbox3d": {
                    "cx": x + width/2 - w/2,
                    "cy": y + height/2 - h/2,
                    "cz": depth_val / 1000.0,  # Convert to meters
                    "sx": width / 100.0,
                    "sy": height / 100.0,
                    "sz": depth_val / 2000.0,
                },
                "class": np.random.choice(self.class_names),
                "confidence": np.random.uniform(0.6, 0.95),
            }
            detections.append(detection)
        
        return detections


class RealModelDetector:
    """Real model detector (nếu có model weights)."""
    
    def __init__(self, model_path=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for real model")
        
        # Try to load model
        try:
            from vision_system.models.snn.depth_aware_snn import DepthAwareSNN
            self.model = DepthAwareSNN(num_classes=10)
            
            if model_path and model_path.exists():
                checkpoint = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"✅ Loaded model from {model_path}")
            else:
                print("⚠️  No model weights, using random initialization")
            
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.name = "DepthAwareSNN"
        except Exception as e:
            print(f"⚠️  Could not load real model: {e}")
            raise
    
    def detect(self, rgb_img, depth_img):
        """
        Detect objects using real model.

        Stub implementation:
        - Chạy forward model để đảm bảo checkpoint/shape hợp lệ.
        - Trả về []

        TODO (sau khi có yêu cầu cụ thể):
        - Chuyển bbox_pred + logits thành danh sách detections giống MockDetector:
          {
            "bbox2d": [x, y, w, h],
            "bbox3d": {"cx","cy","cz","sx","sy","sz"},
            "class": str,
            "confidence": float,
          }
        """
        if depth_img is None:
            return []

        # Preprocess tối thiểu để validate model
        rgb_resized = cv2.resize(rgb_img, (224, 224))
        depth_resized = cv2.resize(depth_img, (224, 224))

        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).float() / 1000.0
        
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Add temporal dimension
        rgb_seq = rgb_tensor.unsqueeze(1)
        depth_seq = depth_tensor.unsqueeze(1)
        
        with torch.no_grad():
            _bbox_pred, _logits = self.model(rgb_seq, depth_seq)
        
        # Chưa implement post-processing -> trả về rỗng để không crash UI/demo
        return []


def draw_detections(img, detections, depth_img=None):
    """Vẽ detections lên image."""
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        
        # Draw 2D bbox
        x, y, w, h = det["bbox2d"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{det['class']} {det['confidence']:.2f}"
        if "bbox3d" in det:
            depth = det["bbox3d"]["cz"]
            label += f" | {depth:.2f}m"
        
        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img,
            (x, max(0, y - text_height - 5)),
            (x + text_width, y),
            color,
            -1,
        )
        
        # Text
        cv2.putText(
            img,
            label,
            (x, max(text_height, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    return img


def overlay_depth(rgb_img, depth_img, alpha=0.5):
    """Overlay depth map lên RGB."""
    if depth_img is None:
        return rgb_img
    
    # Normalize depth
    depth_norm = (depth_img / depth_img.max() * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Blend
    result = cv2.addWeighted(rgb_img, 1 - alpha, depth_colored, alpha, 0)
    return result


def main():
    args = parse_args()
    
    print("🎥 Real-time Webcam Demo - Vision System 3D Detection")
    print("="*70)
    
    # Initialize camera
    print(f"📷 Đang mở camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở camera {args.camera}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print("✅ Camera đã mở")
    
    # Initialize depth estimator
    print("🔍 Khởi tạo depth estimator...")
    depth_estimator = MockDepthEstimator()
    print(f"✅ {depth_estimator.name}")
    
    # Initialize detector
    print("🤖 Khởi tạo detector...")
    try:
        if args.model and args.model.exists():
            detector = RealModelDetector(args.model)
        else:
            detector = MockDetector()
    except Exception as e:
        print(f"⚠️  Không thể load real model, dùng mock: {e}")
        detector = MockDetector()
    
    print(f"✅ {detector.name}")
    
    # Video writer (nếu cần)
    video_writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(args.save_output),
            fourcc,
            30.0,
            (args.width, args.height)
        )
        print(f"💾 Sẽ lưu video vào: {args.save_output}")
    
    # FPS tracking
    fps_counter = []
    frame_count = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU DEMO")
    print("="*70)
    print("Nhấn 'q' để thoát")
    print("Nhấn 'd' để toggle depth overlay")
    print("Nhấn 's' để screenshot")
    print("Màn hình 1: AI Detection | Màn hình 2: Giao diện người dùng (view người dùng nhìn thấy)")
    print("="*70 + "\n")
    
    show_depth = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Không đọc được frame")
                break
            
            frame_count += 1
            
            # Estimate depth
            depth = depth_estimator.estimate_depth(frame)
            
            # Detect objects
            det_start = time.time()
            detections = detector.detect(frame, depth)
            det_time = (time.time() - det_start) * 1000  # ms
            
            # Draw detections
            display_frame = frame.copy()
            if show_depth:
                display_frame = overlay_depth(display_frame, depth, alpha=0.4)
            
            display_frame = draw_detections(display_frame, detections, depth)
            
            # Draw FPS
            if args.fps_display or True:  # Always show
                fps = 1.0 / (time.time() - start_time) if frame_count > 0 else 0
                fps_counter.append(fps)
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                avg_fps = np.mean(fps_counter)
                
                fps_text = f"FPS: {avg_fps:.1f} | Detection: {det_time:.1f}ms"
                cv2.putText(
                    display_frame,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            
            # Draw info
            info_text = f"Detections: {len(detections)} | Model: {detector.name}"
            cv2.putText(
                display_frame,
                info_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            
            # Show frame
            cv2.imshow("Vision System 3D Detection Demo", display_frame)
            
            # Save video
            if video_writer:
                video_writer.write(display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            
            start_time = time.time()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("✅ DEMO KẾT THÚC")
        print("="*70)
        if fps_counter:
            print(f"📊 Average FPS: {np.mean(fps_counter):.2f}")
            print(f"📊 Total frames: {frame_count}")


if __name__ == "__main__":
    main()
