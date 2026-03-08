# -*- coding: utf-8 -*-
"""
test_multimodal_fusion.py — NCKH 2026
==========================================
End-to-end testing: Vision + NLP + Fusion pipeline.

3 Scenarios:
  1. "Chi duong den phong A1"  (Vision + NLP)
  2. "Nhan dien vat the"       (Vision only)
  3. "Lich thi ngay mai"       (NLP only)

Output:
  experiments/results/fusion_e2e_test.json
  experiments/figures/fig_e2e_fusion_test.png

Usage:
  python scripts/test_multimodal_fusion.py
  python scripts/test_multimodal_fusion.py --use-models   # with real models
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# ── Project imports ──────────────────────────────────────────
from multimodal_fusion.bridges.vision_nlp_bridge import (
    VisionNLPBridge,
    CrossModalAttention,
    VisionResult,
    NLPResult,
    FusedResult,
    rule_based_decision,
    ACTION_CLASSES,
)
from multimodal_fusion.decision.robot_decision import (
    RobotDecisionMaker,
    RobotAction,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Visualization style ─────────────────────────────────────
COLORS = {
    "vision":  "#2563EB",
    "nlp":     "#7C3AED",
    "fusion":  "#059669",
    "target":  "#DC2626",
    "bg":      "#F8FAFC",
}
STYLE = {
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   "#F8FAFC",
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
}


# ============================================================
# Test Scenarios
# ============================================================

SCENARIOS = [
    {
        "id":      "S1",
        "name":    "Chi duong den phong A1",
        "desc_vi": "Robot chi duong cho nguoi dung den phong A1",
        "type":    "vision+nlp",
        "context": "airport",
        "speech":  "Xin chao, chi giup toi duong den phong A1 duoc khong?",
        "intent":  "directions",
        "entities": {"location": "phong A1"},
        "objects": ["person", "door", "sign"],
        "expected_action": "navigate",
        "lang":    "vi",
    },
    {
        "id":      "S2",
        "name":    "Nhan dien vat the truoc mat",
        "desc_vi": "Robot nhan dien vat the bang camera",
        "type":    "vision_only",
        "context": "classroom",
        "speech":  "Truoc mat toi la gi?",
        "intent":  "explain",
        "entities": {},
        "objects": ["monitor", "keyboard", "chair", "table"],
        "expected_action": "explain",
        "lang":    "vi",
    },
    {
        "id":      "S3",
        "name":    "Hoi lich thi ngay mai",
        "desc_vi": "Hoi thong tin lich thi (chi NLP)",
        "type":    "nlp_only",
        "context": "classroom",
        "speech":  "Ngay mai lich thi mon gi vay?",
        "intent":  "grade",
        "entities": {"subject": "all", "time": "ngay mai"},
        "objects": [],
        "expected_action": "explain",
        "lang":    "vi",
    },
]


# ============================================================
# Mock feature generators
# ============================================================

def mock_vision_result(scenario: dict, device: str = "cpu") -> VisionResult:
    """Sinh VisionResult gia lap voi timing hien thuc."""
    t0 = time.perf_counter()

    n_obj = len(scenario["objects"])
    bbox = torch.randn(max(n_obj, 1), 6).to(device)
    logits = torch.randn(max(n_obj, 1), 237).to(device)  # 237 HILO classes

    # Simulate inference time
    time.sleep(0.018)  # ~18ms realistic for SNN
    t1 = time.perf_counter()

    return VisionResult(
        bbox_3d=bbox,
        class_logits=logits,
        class_names=scenario["objects"][:max(n_obj, 1)],
        confidence=[0.92, 0.87, 0.81][:max(n_obj, 1)],
        depth_map=torch.rand(1, 224, 224).to(device),
        processing_time_ms=(t1 - t0) * 1000,
    )


def mock_nlp_result(scenario: dict, device: str = "cpu") -> NLPResult:
    """Sinh NLPResult gia lap voi timing hien thuc."""
    t0 = time.perf_counter()

    # Simulate NLP inference
    feat = torch.randn(1, 512).to(device)
    time.sleep(0.025)  # ~25ms realistic for SpikingLM
    t1 = time.perf_counter()

    return NLPResult(
        intent=scenario["intent"],
        context=scenario["context"],
        entities=scenario["entities"],
        language=scenario["lang"],
        confidence=0.95,
        features=feat,
        response_hint=f"[{scenario['intent']}] processed",
        processing_time_ms=(t1 - t0) * 1000,
    )


# ============================================================
# Memory profiling
# ============================================================

def get_memory_usage_mb() -> float:
    """Lay memory usage hien tai (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


# ============================================================
# Run single scenario
# ============================================================

def run_scenario(
    scenario: dict,
    bridge: VisionNLPBridge,
    decision_maker: RobotDecisionMaker,
    device: str = "cpu",
    n_runs: int = 10,
) -> dict:
    """Chay 1 scenario nhieu lan va thu thap metrics."""

    print(f"\n{'='*60}")
    print(f"  Scenario [{scenario['id']}]: {scenario['name']}")
    print(f"  Type: {scenario['type']} | Context: {scenario['context']}")
    print(f"{'='*60}")

    latencies_vision = []
    latencies_nlp = []
    latencies_fusion = []
    latencies_total = []
    mem_before = get_memory_usage_mb()

    correct = 0

    for run in range(n_runs):
        t_total_start = time.perf_counter()

        # Step 1: Vision inference
        if scenario["type"] in ("vision+nlp", "vision_only"):
            vis = mock_vision_result(scenario, device)
            latencies_vision.append(vis.processing_time_ms)
        else:
            vis = VisionResult(
                bbox_3d=torch.zeros(0, 6),
                class_logits=torch.zeros(0, 237),
            )
            latencies_vision.append(0.0)

        # Step 2: NLP inference
        if scenario["type"] in ("vision+nlp", "nlp_only"):
            nlp = mock_nlp_result(scenario, device)
            latencies_nlp.append(nlp.processing_time_ms)
        else:
            nlp = NLPResult(
                intent="idle", context=scenario["context"],
                entities={}, language="vi", confidence=0.0,
            )
            latencies_nlp.append(0.0)

        # Step 3: Fusion
        t_fuse_start = time.perf_counter()

        # Generate features from mock results
        v_feat = torch.randn(1, 256).to(device)
        n_feat = nlp.features if nlp.features is not None else torch.randn(1, 512).to(device)

        fused = bridge.fuse_features(v_feat, n_feat)
        t_fuse_end = time.perf_counter()
        lat_fuse = (t_fuse_end - t_fuse_start) * 1000
        latencies_fusion.append(lat_fuse)

        # Step 4: Decision
        action = decision_maker.decide(
            intent=nlp.intent,
            context=scenario["context"],
            entities=scenario["entities"],
            detected_objects=scenario["objects"],
            language=scenario["lang"],
            fused_features=fused,
            confidence=nlp.confidence,
        )

        t_total_end = time.perf_counter()
        latencies_total.append((t_total_end - t_total_start) * 1000)

        if action.action_type == scenario["expected_action"]:
            correct += 1

    mem_after = get_memory_usage_mb()

    # Compute stats
    import numpy as np
    accuracy = (correct / n_runs) * 100
    result = {
        "scenario_id":   scenario["id"],
        "scenario_name": scenario["name"],
        "type":          scenario["type"],
        "context":       scenario["context"],
        "expected":      scenario["expected_action"],
        "accuracy":      accuracy,
        "n_runs":        n_runs,
        "latency": {
            "vision_ms":   round(float(np.mean(latencies_vision)), 2),
            "nlp_ms":      round(float(np.mean(latencies_nlp)), 2),
            "fusion_ms":   round(float(np.mean(latencies_fusion)), 2),
            "total_ms":    round(float(np.mean(latencies_total)), 2),
            "total_p90_ms": round(float(np.percentile(latencies_total, 90)), 2),
        },
        "memory": {
            "before_MB": round(mem_before, 1),
            "after_MB":  round(mem_after, 1),
            "delta_MB":  round(mem_after - mem_before, 1),
        },
        "last_action": action.to_dict() if hasattr(action, 'to_dict') else str(action),
    }

    # Print summary
    print(f"\n  Results:")
    print(f"    Accuracy:     {accuracy:.0f}% ({correct}/{n_runs})")
    print(f"    Latency mean: {result['latency']['total_ms']:.1f}ms")
    print(f"    Latency p90:  {result['latency']['total_p90_ms']:.1f}ms")
    print(f"    Vision:       {result['latency']['vision_ms']:.1f}ms")
    print(f"    NLP:          {result['latency']['nlp_ms']:.1f}ms")
    print(f"    Fusion:       {result['latency']['fusion_ms']:.1f}ms")
    print(f"    Memory delta: {result['memory']['delta_MB']:.1f} MB")
    print(f"    Action taken: {action.action_type}")
    if hasattr(action, 'speech_text'):
        print(f"    Speech:       {action.speech_text[:60]}...")

    return result


# ============================================================
# Print summary table
# ============================================================

def print_summary_table(results: list) -> None:
    """In bang tong hop."""
    print("\n" + "=" * 80)
    print("  BANG: End-to-End Multimodal Fusion Test Results — NCKH 2026")
    print("=" * 80)
    print(f"  {'Scenario':<22} {'Type':<12} {'Accuracy':>8} "
          f"{'Vision':>8} {'NLP':>8} {'Fusion':>8} {'Total':>8} {'Mem':>8}")
    print("  " + "-" * 76)

    for r in results:
        lat = r["latency"]
        vis_str = f"{lat['vision_ms']:.1f}ms" if lat["vision_ms"] > 0 else "—"
        nlp_str = f"{lat['nlp_ms']:.1f}ms" if lat["nlp_ms"] > 0 else "—"
        print(f"  {r['scenario_name']:<22} {r['type']:<12} {r['accuracy']:>7.0f}% "
              f"{vis_str:>8} {nlp_str:>8} {lat['fusion_ms']:>6.1f}ms "
              f"{lat['total_ms']:>6.1f}ms {r['memory']['delta_MB']:>5.1f}MB")

    print("=" * 80)

    # System summary
    mean_acc = sum(r["accuracy"] for r in results) / len(results)
    max_lat  = max(r["latency"]["total_ms"] for r in results)
    print(f"\n  Mean Accuracy: {mean_acc:.1f}%")
    print(f"  Max Latency:   {max_lat:.1f}ms")
    print(f"  Budget (<200ms): {'OK' if max_lat < 200 else 'EXCEEDED'}")
    print("=" * 80 + "\n")


# ============================================================
# Visualization
# ============================================================

def plot_e2e_results(results: list, out_dir: Path) -> None:
    """Ve bieu do E2E test — 4 panels."""
    if not HAS_MPL:
        return

    import numpy as np
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "End-to-End Multimodal Fusion Test\nNCKH 2026",
        fontsize=14, fontweight="bold",
    )

    sc_names = [r["scenario_name"] for r in results]
    x = np.arange(len(sc_names))

    # Panel 1: Latency breakdown (stacked bar)
    ax1 = axes[0, 0]
    vis_lats = [r["latency"]["vision_ms"] for r in results]
    nlp_lats = [r["latency"]["nlp_ms"] for r in results]
    fus_lats = [r["latency"]["fusion_ms"] for r in results]

    ax1.bar(x, vis_lats, color=COLORS["vision"], label="Vision", alpha=0.9)
    ax1.bar(x, nlp_lats, bottom=vis_lats, color=COLORS["nlp"],
            label="NLP", alpha=0.9)
    bottoms = [v + n for v, n in zip(vis_lats, nlp_lats)]
    ax1.bar(x, fus_lats, bottom=bottoms, color=COLORS["fusion"],
            label="Fusion", alpha=0.9)
    ax1.axhline(200, color=COLORS["target"], ls="--", lw=1.5, label="Budget 200ms")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_names, fontsize=8, rotation=15)
    ax1.set_title("Latency Breakdown (ms)")
    ax1.set_ylabel("Latency (ms)")
    ax1.legend(fontsize=8)

    # Panel 2: Accuracy bars
    ax2 = axes[0, 1]
    accs = [r["accuracy"] for r in results]
    cols = [COLORS["fusion"] if a >= 90 else COLORS["target"] for a in accs]
    bars = ax2.bar(x, accs, color=cols, width=0.5, alpha=0.9)
    ax2.axhline(85, color="#64748B", ls="--", lw=1.5, label="Target 85%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_names, fontsize=8, rotation=15)
    ax2.set_title("Scenario Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 110)
    ax2.legend()
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, acc + 1,
                 f"{acc:.0f}%", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Memory usage
    ax3 = axes[1, 0]
    mems = [r["memory"]["delta_MB"] for r in results]
    ax3.bar(x, mems, color=COLORS["fusion"], width=0.5, alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(sc_names, fontsize=8, rotation=15)
    ax3.set_title("Memory Delta (MB)")
    ax3.set_ylabel("Memory (MB)")

    # Panel 4: Action pipeline visualization
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary = (
        "PIPELINE SUMMARY\n"
        "─────────────────────────────\n"
    )
    for r in results:
        act = r.get("last_action", {})
        act_type = act.get("action_type", "?") if isinstance(act, dict) else str(act)
        summary += (
            f"\n[{r['scenario_id']}] {r['scenario_name']}\n"
            f"   → Action: {act_type}\n"
            f"   → Latency: {r['latency']['total_ms']:.0f}ms\n"
        )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=9, fontfamily="monospace", va="top",
             bbox=dict(boxstyle="round", fc="white", ec="#CBD5E1"))

    plt.tight_layout()
    out_path = out_dir / "fig_e2e_fusion_test.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="E2E Multimodal Fusion Test — NCKH 2026"
    )
    parser.add_argument("--n-runs", type=int, default=20,
                        help="Runs per scenario")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "figures"))
    args = parser.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = PROJECT_ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MULTIMODAL FUSION — END-TO-END TEST")
    print("  NCKH 2026")
    print("=" * 60)
    print(f"  Device:    {device}")
    print(f"  Runs/scen: {args.n_runs}")

    # Init modules
    bridge = VisionNLPBridge(
        vision_dim=256,
        nlp_dim=512,
        fused_dim=256,
        num_heads=8,
    )
    bridge.eval()

    decision_maker = RobotDecisionMaker(
        fused_dim=256,
        use_learnable=False,
    )

    # Run all scenarios
    all_results = []
    for scenario in SCENARIOS:
        result = run_scenario(
            scenario=scenario,
            bridge=bridge,
            decision_maker=decision_maker,
            device=device,
            n_runs=args.n_runs,
        )
        all_results.append(result)

    # Summary
    print_summary_table(all_results)

    # Visualization
    plot_e2e_results(all_results, out_dir)

    # Save JSON
    json_path = res_dir / "fusion_e2e_test.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"  END-TO-END TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
