"""
SNNVisionWrapper — Fixed Version
==================================
Fixes:
1. ✅ FPS đo thực tế bằng timeit, không hardcode
2. ✅ Mono Depth Estimation (MiDaS-small) thay depth=0
3. ✅ Warm-up inference trước khi đo FPS
4. ✅ Compatible với ThreeDSpikingCNN mới (NMS, multi-anchor)
5. ✅ Raw spike features cho Fusion Transformer
6. ✅ get_real_fps() đo thực tế
"""

import torch
import numpy as np
import time
import sys
import os
import cv2  # Moved to top
from pathlib import Path
from typing import List, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent))

try:
    from vision_system.models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
    from vision_system.encoding.spike_encoder import AdaptiveSpikeEncoder
except ImportError:
    from .models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
    from .encoding.spike_encoder import AdaptiveSpikeEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Mono Depth Estimator (Optional — dùng MiDaS-small)
# ─────────────────────────────────────────────────────────────────────────────

class MonoDepthEstimator:
    """
    Ước lượng depth từ RGB ảnh đơn dùng MiDaS-small.
    Nhẹ hơn DPT-Hybrid, phù hợp Jetson Orin (< 5ms/frame @GPU).

    Fallback sang Laplacian gradient nếu không có torch.hub.
    """

    def __init__(self, use_midas: bool = True, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.transform = None
        self.use_midas = use_midas

        if use_midas:
            self._load_midas()

    def _load_midas(self):
        """Load MiDaS-small từ torch.hub."""
        try:
            import torch
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False
            )
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False
            )
            self.transform = midas_transforms.small_transform
            self.model.to(self.device)
            self.model.eval()
            print("✅ MiDaS-small depth estimator loaded")
        except Exception as e:
            print(f"⚠️ MiDaS không load được: {e}")
            print("   Dùng Laplacian gradient depth (fallback)")
            self.model = None

    def estimate(self, rgb_frame: np.ndarray, target_size: int = 128) -> np.ndarray:
        """
        Args:
            rgb_frame:   (H, W, 3) uint8 RGB
            target_size: output size (vuông)
        Returns:
            depth:       (target_size, target_size) float32 [0, 1]
        """
        if self.model is not None:
            return self._midas_depth(rgb_frame, target_size)
        else:
            return self._gradient_depth(rgb_frame, target_size)

    def _midas_depth(self, rgb_frame: np.ndarray, target_size: int) -> np.ndarray:
        """MiDaS inference."""
        inp = self.transform(rgb_frame).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(target_size, target_size),
                mode="bicubic", align_corners=False
            ).squeeze()
        depth = pred.cpu().numpy()
        # Normalize [0, 1] — MiDaS output là disparity (ngược depth)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth.astype(np.float32)

    def _gradient_depth(self, rgb_frame: np.ndarray, target_size: int) -> np.ndarray:
        """
        Fallback: dùng Laplacian gradient để ước lượng saliency/depth.
        Vùng edge nhiều = gần hơn (rough approximation).
        """
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap = cv2.resize(lap, (target_size, target_size))
        lap_max = lap.max()
        if lap_max > 0:
            lap = lap / lap_max
        return lap.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SNNVisionWrapper — Upgraded
# ─────────────────────────────────────────────────────────────────────────────

class SNNVisionWrapper:
    """
    Wrapper cho ThreeDSpikingCNN.
    Drop-in replacement — interface giống hệt phiên bản cũ.

    Thay đổi:
      - Depth không còn hardcode = 0: dùng MonoDepthEstimator (optional)
      - FPS đo thực tế: gọi get_real_fps()
      - Trả về raw spike features khi cần (cho Fusion Transformer)
    """

    def __init__(
        self,
        model_path: str = "models/best_vision_snn.pth",
        T: int = 8,
        input_size: int = 128,
        num_classes: int = 2,
        use_depth_estimation: bool = False,  # Bật khi muốn depth thật
        conf_threshold: float = 0.35,
        nms_iou_threshold: float = 0.45,
    ):
        self.T = T
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._measured_fps: float = None  # Cache sau lần đo đầu tiên

        # Model
        self.model = ThreeDSpikingCNN(
            num_classes=num_classes,
            T=T,
            input_size=input_size,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
        ).to(self.device)

        # Load weights
        abs_path = model_path if os.path.isabs(model_path) \
                   else os.path.join(os.getcwd(), model_path)
        if os.path.exists(abs_path):
            try:
                state = torch.load(abs_path, map_location=self.device)
                # Thử load — nếu weights cũ không match architecture mới, bỏ qua
                result = self.model.load_state_dict(state, strict=False)
                if result.missing_keys:
                    print(f"⚠️ SNN: {len(result.missing_keys)} keys missing (new layers will be random)")
                    print(f"   → Cần retrain: python vision_system/training/train_vision_snn.py")
                else:
                    print(f"✅ SNN weights loaded from {abs_path}")
            except Exception as e:
                print(f"⚠️ SNN weight load failed: {e} — using random init")
        else:
            print(f"⚠️ SNN model not found at {abs_path} — using random init")

        self.model.eval()

        # Encoder
        self.encoder = AdaptiveSpikeEncoder(T=T).to(self.device)

        # Depth estimator
        self.depth_estimator = None
        if use_depth_estimation:
            self.depth_estimator = MonoDepthEstimator(use_midas=True)

    # ── Frame preprocessing ───────────────────────────────────────────────────

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Chuyển frame (H,W,3 hoặc H,W,4) → (input_size, input_size, 4) float32 [0,1].
        """

        if frame.shape[2] == 4:
            rgb = frame[:, :, :3]
            depth = frame[:, :, 3:4]
        else:
            rgb = frame
            if self.depth_estimator is not None:
                depth_2d = self.depth_estimator.estimate(rgb, self.input_size)
                depth = depth_2d[:, :, np.newaxis]
                rgb = cv2.resize(rgb, (self.input_size, self.input_size))
            else:
                # Fallback: Laplacian gradient — tốt hơn toàn số 0
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
                lap_resized = cv2.resize(lap, (self.input_size, self.input_size))
                mx = lap_resized.max()
                depth = (lap_resized / mx if mx > 0 else lap_resized)[:, :, np.newaxis]
                rgb = cv2.resize(rgb, (self.input_size, self.input_size))

        # Resize + normalize
        rgb = cv2.resize(rgb, (self.input_size, self.input_size)).astype(np.float32) / 255.0
        depth = cv2.resize(depth.squeeze(), (self.input_size, self.input_size)).astype(np.float32)
        depth = depth[:, :, np.newaxis]

        return np.concatenate([rgb, depth], axis=2)  # (H, W, 4)

    # ── Main inference ────────────────────────────────────────────────────────

    def process_frame_sequence(
        self,
        rgbd_frames,
        return_raw_spikes: bool = False,
    ):
        """
        Args:
            rgbd_frames:      list of T frames, each (H,W,3) or (H,W,4)
            return_raw_spikes: Nếu True → trả về spike feature tensor
                               (T, B, 64, Hf, Wf) cho Fusion Transformer

        Returns:
            (boxes, conf, cls)  — numpy arrays
            HOẶC
            spike_feats         — torch.Tensor (T,1,64,Hf,Wf) nếu return_raw_spikes=True
        """
        # Preprocess
        processed = [self._preprocess_frame(f) for f in rgbd_frames]

        tensors = [torch.from_numpy(f).permute(2, 0, 1).float() for f in processed]
        tensor_frames = torch.stack(tensors).unsqueeze(1).to(self.device)
        # Shape: (T, 1, 4, H, W)

        # Encode → spikes
        with torch.no_grad():
            spikes = self.encoder(tensor_frames)

            if return_raw_spikes:
                feats = self.model(spikes, return_feats=True)  # (T, B, 64, Hf, Wf)
                return feats

            boxes, conf, cls = self.model(spikes, apply_nms_=True)

        return boxes.cpu().numpy(), conf.cpu().numpy(), cls.cpu().numpy()

    def get_spike_features(self, rgbd_frames: List[np.ndarray]) -> torch.Tensor:
        """
        Lấy đặc trưng xung (raw spikes) cho Fusion Transformer.
        Shape: (T, B=1, 64, Hf, Wf)
        """
        return self.process_frame_sequence(rgbd_frames, return_raw_spikes=True)

    # ── FPS benchmark ─────────────────────────────────────────────────────────

    def get_fps_estimate(self) -> float:
        """Trả về cached FPS (đo 1 lần, cache lại)."""
        if self._measured_fps is None:
            self._measured_fps = self.get_real_fps(n_runs=20, warmup=5)
        return self._measured_fps

    def get_real_fps(self, n_runs: int = 50, warmup: int = 10) -> float:
        """
        Đo FPS thực tế bằng cách chạy inference n_runs lần.
        Bao gồm encoding + forward + NMS.
        """
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames = [dummy_frame] * self.T

        # Warmup (tránh CUDA cold start)
        for _ in range(warmup):
            self.process_frame_sequence(frames)

        # Đo
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.process_frame_sequence(frames)
        elapsed = time.perf_counter() - t0

        fps = n_runs / elapsed
        ms  = elapsed / n_runs * 1000
        print(f"⏱️  SNN Inference: {ms:.1f} ms/frame → {fps:.1f} FPS")
        return fps


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SNNVisionWrapper Test ===\n")

    wrapper = SNNVisionWrapper(
        model_path="models/best_vision_snn.pth",
        use_depth_estimation=False,   # Tắt MiDaS để test nhanh
    )

    # Test với dummy frames (RGB)
    dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)] * 8
    boxes, conf, cls = wrapper.process_frame_sequence(dummy_frames)

    print(f"boxes shape : {boxes.shape}")
    print(f"conf shape  : {conf.shape}")
    print(f"cls shape   : {cls.shape}")
    detections = [(conf[0, i, 0], cls[0, i].argmax()) for i in range(conf.shape[1]) if conf[0, i, 0] > 0.35]
    print(f"Detections (conf>0.35): {len(detections)}")

    print("\n--- FPS Benchmark ---")
    fps = wrapper.get_real_fps(n_runs=20, warmup=5)
    print(f"FPS: {fps:.1f}")
