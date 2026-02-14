"""
Benchmark đơn giản cho DepthAwareSNN:
- Sinh input giả (hoặc có thể dùng dữ liệu aligned thật nếu muốn)
- Đo latency trung bình và FPS trên CPU/GPU hiện tại

Đây là bước chuẩn bị để sau này so sánh với Jetson AGX Orin.
"""

import time
from typing import Tuple

import torch
import sys
from pathlib import Path

# Bảo đảm project root nằm trong sys.path khi chạy trực tiếp file .py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_system.models.snn.depth_aware_snn import DepthAwareSNN


def run_benchmark(
    device: torch.device,
    batch_size: int = 1,
    time_steps: int = 8,
    height: int = 224,
    width: int = 224,
    warmup: int = 5,
    iters: int = 50,
    use_multiscale: bool = False,
) -> Tuple[float, float]:
    model = DepthAwareSNN(
        num_classes=10,
        use_multiscale=use_multiscale,
    ).to(device)
    model.eval()

    rgb = torch.randn(batch_size, time_steps, 3, height, width, device=device)
    depth = torch.randn(batch_size, time_steps, 1, height, width, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(rgb, depth)

    torch.cuda.empty_cache() if device.type == "cuda" else None

    times = []
    with torch.no_grad():
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(rgb, depth)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_latency = sum(times) / len(times)  # seconds
    fps = batch_size * time_steps / avg_latency
    return avg_latency, fps


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for use_ms in (False, True):
        tag = "Baseline" if not use_ms else "With MultiScaleSTDFE"
        avg_latency, fps = run_benchmark(
            device=device,
            batch_size=1,
            time_steps=8,
            height=224,
            width=224,
            use_multiscale=use_ms,
        )
        print(f"\n[{tag}]")
        print(f"  Avg latency per sequence: {avg_latency * 1000:.2f} ms")
        print(f"  Effective FPS (B*T / latency): {fps:.2f}")


if __name__ == "__main__":
    main()

