"""
Inference Benchmark — LearnedMultimodalBridge
"""
import time, sys, torch
sys.path.insert(0, '.')
import numpy as np
from multimodal_fusion.learned_bridge import LearnedMultimodalBridge

bridge = LearnedMultimodalBridge(
    model_path="models/fusion/best_fusion.pth",
    fallback_to_rules=False
)
bridge.model.eval()

test_cases = [
    ("Đi theo người phía trước",  [{"class":"person",   "confidence":0.92,"bbox":[0.3,0.1,0.7,0.9],"center":[0.5,0.5]}]),
    ("Tránh vật cản bên trái",    [{"class":"obstacle", "confidence":0.88,"bbox":[0.0,0.2,0.35,0.8],"center":[0.17,0.5]}]),
    ("Xin chào! Bạn có khỏe không?", [{"class":"person","confidence":0.85,"bbox":[0.4,0.1,0.7,0.9],"center":[0.55,0.5]}]),
    ("Phòng 301 ở đâu?",           []),
    ("Dừng lại đi",               []),
    ("Tìm người dùng",            []),
    ("Học phí trường bao nhiêu?", []),
    ("Follow that person please", [{"class":"person","confidence":0.91,"bbox":[0.3,0.2,0.6,0.8],"center":[0.45,0.5]}]),
]

print("=" * 65)
print("  INFERENCE TEST — SpikingFusionTransformer")
print("  Model: models/fusion/best_fusion.pth")
print("=" * 65)

for query, mock_dets in test_cases:
    token_ids    = bridge._encode_text(query)
    vision_spikes = bridge._mock_vision_spikes(mock_dets).unsqueeze(1)
    with torch.no_grad():
        pred = bridge.model.predict(vision_spikes, token_ids)

    conf_bar = "█" * int(pred["confidence"] * 20) + "░" * (20 - int(pred["confidence"] * 20))
    flag = "⚠️ " if pred["needs_confirm"] else "✅"
    print(f"\n{flag} '{query}'")
    print(f"   Action    : {pred['action_type']}")
    print(f"   Confidence: [{conf_bar}] {pred['confidence']:.1%}")
    top2 = sorted(pred["action_probs"].items(), key=lambda x: -x[1])[:2]
    print(f"   Top-2     : {top2[0][0]}={top2[0][1]:.2%}  {top2[1][0]}={top2[1][1]:.2%}")

# ── Latency benchmark ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  LATENCY BENCHMARK (100 inference calls)")
print("=" * 65)
token_ids    = bridge._encode_text("theo doi nguoi")
vision_spikes = bridge._mock_vision_spikes([{"class":"person","confidence":0.9,"bbox":[0.3,0.2,0.6,0.8],"center":[0.45,0.5]}]).unsqueeze(1)

# Warmup
for _ in range(10):
    with torch.no_grad():
        bridge.model.predict(vision_spikes, token_ids)

latencies = []
for _ in range(100):
    t0 = time.perf_counter()
    with torch.no_grad():
        bridge.model.predict(vision_spikes, token_ids)
    latencies.append((time.perf_counter() - t0) * 1000)

latencies = np.array(latencies)
device = "CUDA" if torch.cuda.is_available() else "CPU"
print(f"  Device   : {device}")
print(f"  Mean     : {latencies.mean():.2f} ms")
print(f"  Median   : {np.median(latencies):.2f} ms")
print(f"  P95      : {np.percentile(latencies, 95):.2f} ms")
print(f"  Min/Max  : {latencies.min():.2f} / {latencies.max():.2f} ms")
print(f"  FPS      : {1000/latencies.mean():.1f} fps")
target = 50.0
status = "✅ Đạt target <50ms" if latencies.mean() < target else "⚠️ Chưa đạt target <50ms"
print(f"  Target   : {status}")
print("=" * 65)
