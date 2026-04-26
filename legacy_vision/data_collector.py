"""
Vision Data Collector
======================
Thu thập data thực từ webcam với bounding box annotation.

Usage:
    python vision_system/data_collector.py
    python vision_system/data_collector.py --save data/real_vision.json --duration 300

Controls:
    SPACE      → Vẽ bounding box để annotate frame hiện tại
    P          → Mark frame là 'person' và lưu (quick capture)
    O          → Mark frame là 'obstacle' và lưu (quick capture)
    D          → Toggle xem depth channel (Laplacian)
    Q / ESC    → Thoát và lưu
"""

import cv2
import json
import time
import base64
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Depth từ Laplacian (không cần RealSense)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_depth_laplacian(frame_bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) → (H,W) float32 [0,1] — pseudo depth từ edge"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap  = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    # Gaussian blur để smooth
    lap  = cv2.GaussianBlur(lap, (5, 5), 0)
    mx   = lap.max()
    return (lap / mx if mx > 0 else lap).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Collector class
# ─────────────────────────────────────────────────────────────────────────────

class VisionDataCollector:
    """Thu thập và annotate dữ liệu từ webcam."""

    CLASSES = {ord('p'): 'person', ord('o'): 'obstacle'}

    def __init__(self, camera_id: int = 0, save_path: str = "data/real_vision.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data nếu có
        if self.save_path.exists():
            with open(self.save_path, "r") as f:
                self.samples = json.load(f)
            print(f"📂 Loaded {len(self.samples)} existing samples")
        else:
            self.samples = []

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.show_depth = False

    def _encode_frame(self, frame_bgr: np.ndarray) -> str:
        """Encode frame to base64 JPEG."""
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ UI overlay lên frame."""
        vis = frame.copy()
        H, W = vis.shape[:2]

        # Status bar
        cv2.rectangle(vis, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(vis, f"Samples: {len(self.samples)} | SPACE=annotate P=person O=obstacle D=depth Q=quit",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 100), 1)

        # Class counts
        persons   = sum(1 for s in self.samples if s["class"] == "person")
        obstacles = sum(1 for s in self.samples if s["class"] == "obstacle")
        cv2.putText(vis, f"P:{persons} O:{obstacles}",
                    (W - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

        if self.show_depth:
            depth = estimate_depth_laplacian(frame)
            depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
            # Blend
            vis[40:, :] = cv2.addWeighted(vis[40:, :], 0.5, depth_colored[40:, :], 0.5, 0)
            cv2.putText(vis, "DEPTH PREVIEW", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return vis

    def _quick_capture(self, frame_bgr: np.ndarray, cls_name: str):
        """Capture toàn frame với YOLO-style center bbox."""
        H, W = frame_bgr.shape[:2]
        # Bbox = center 60% của frame
        margin_x = int(W * 0.2)
        margin_y = int(H * 0.2)
        bbox = [margin_x, margin_y, W - 2 * margin_x, H - 2 * margin_y]

        depth = estimate_depth_laplacian(frame_bgr)

        self.samples.append({
            "image_b64": self._encode_frame(frame_bgr),
            "bbox":      bbox,
            "class":     cls_name,
            "depth_max": float(depth.max()),
            "timestamp": time.time(),
        })
        print(f"  ✅ Quick capture '{cls_name}' → {len(self.samples)} total")

    def _annotate_frame(self, frame_bgr: np.ndarray):
        """Manual annotation: user vẽ bbox và chọn class."""
        # Vẽ bbox
        print("\n  📌 Vẽ bounding box (kéo chuột), nhấn ENTER để confirm, C để cancel")
        bbox = cv2.selectROI("Annotate — ENTER to confirm, C to cancel",
                             frame_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Annotate — ENTER to confirm, C to cancel")

        if bbox[2] == 0 or bbox[3] == 0:
            print("  ⚠️ Empty bbox, skipping")
            return

        # Chọn class
        cls_win = np.zeros((120, 300, 3), dtype=np.uint8)
        cv2.putText(cls_win, "Class:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cls_win, "P = Person", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        cv2.putText(cls_win, "O = Obstacle", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
        cv2.putText(cls_win, "C = Cancel", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.imshow("Select class", cls_win)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Select class")

        cls_map = {ord('p'): 'person', ord('o'): 'obstacle'}
        if key not in cls_map:
            print("  ⚠️ Cancelled")
            return

        cls_name = cls_map[key]
        depth    = estimate_depth_laplacian(frame_bgr)

        self.samples.append({
            "image_b64": self._encode_frame(frame_bgr),
            "bbox":      list(bbox),
            "class":     cls_name,
            "depth_max": float(depth.max()),
            "timestamp": time.time(),
        })
        print(f"  ✅ Annotated '{cls_name}' bbox={bbox} → {len(self.samples)} total")

    def save(self):
        """Lưu dataset."""
        with open(self.save_path, "w") as f:
            json.dump(self.samples, f)
        persons   = sum(1 for s in self.samples if s["class"] == "person")
        obstacles = sum(1 for s in self.samples if s["class"] == "obstacle")
        print(f"\n💾 Saved {len(self.samples)} samples to {self.save_path}")
        print(f"   Person: {persons} | Obstacle: {obstacles}")

    def run(self, duration_sec: int = 600):
        """Main collection loop."""
        print("=" * 55)
        print("  VISION DATA COLLECTOR")
        print("=" * 55)
        print("  SPACE → Manual annotate (vẽ bbox)")
        print("  P     → Quick capture as 'person'")
        print("  O     → Quick capture as 'obstacle'")
        print("  D     → Toggle depth preview")
        print("  Q/ESC → Quit & save")
        print("=" * 55)

        start = time.time()
        last_save = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Camera frame read failed")
                break

            vis = self._draw_overlay(frame)
            cv2.imshow("Robot EEEC — Vision Data Collector", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):         # Manual annotate
                self._annotate_frame(frame)
            elif key == ord('p'):       # Quick person
                self._quick_capture(frame, 'person')
            elif key == ord('o'):       # Quick obstacle
                self._quick_capture(frame, 'obstacle')
            elif key == ord('d'):       # Toggle depth
                self.show_depth = not self.show_depth
                print(f"  Depth preview: {'ON' if self.show_depth else 'OFF'}")
            elif key in (ord('q'), 27): # Quit
                break

            # Auto-save mỗi 60 giây
            if time.time() - last_save > 60 and self.samples:
                self.save()
                last_save = time.time()

            # Timeout
            if duration_sec and (time.time() - start) > duration_sec:
                print(f"\n⏰ Duration {duration_sec}s reached")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.save()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vision Data Collector")
    p.add_argument("--camera",   type=int, default=0)
    p.add_argument("--save",     type=str, default="data/real_vision.json")
    p.add_argument("--duration", type=int, default=600,
                   help="Timeout (giây), 0=không giới hạn")
    args = p.parse_args()

    collector = VisionDataCollector(camera_id=args.camera, save_path=args.save)
    collector.run(duration_sec=args.duration)
