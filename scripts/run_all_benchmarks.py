# -*- coding: utf-8 -*-
"""
run_all_benchmarks.py — NCKH 2026
=====================================
Chạy TẤT CẢ benchmarks và sinh toàn bộ figures chỉ với 1 lệnh.

Các bước:
  1. Architecture diagrams (SNN + Fusion)
  2. NLP benchmark + confusion matrix
  3. Vision benchmark
  4. Fusion strategy comparison
  5. Training curves (demo)

Usage:
    python scripts/run_all_benchmarks.py
    python scripts/run_all_benchmarks.py --demo  (nhanh, không cần model)
    python scripts/run_all_benchmarks.py --out-dir my_figures/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
PYTHON  = sys.executable


def run_step(name: str, cmd: list[str]) -> bool:
    """Chạy 1 bước benchmark."""
    print(f"\n{'='*60}")
    print(f"  ▶ {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  ❌ Failed (code={result.returncode}) after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all NCKH 2026 benchmarks"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode (no real model — fast)")
    parser.add_argument("--out-dir", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "figures"))
    args = parser.parse_args()

    out_flag = ["--out-dir", args.out_dir]
    demo_flag = ["--demo"] if args.demo else []

    print("\n🚀 NCKH 2026 — Running All Benchmarks")
    print(f"   Output dir : {args.out_dir}")
    print(f"   Demo mode  : {args.demo}")

    total_start = time.time()
    results = {}

    # ── Step 1: Architecture diagrams ──────────────────────────
    results["Architecture Diagrams"] = run_step(
        "Architecture Diagrams (SNN + Fusion)",
        [PYTHON, str(SCRIPTS/"visualization"/"plot_architecture.py")],
    )

    # ── Step 2: Training curves ─────────────────────────────────
    results["NLP Training Curves"] = run_step(
        "NLP Training Curves",
        [PYTHON, str(SCRIPTS/"visualization"/"plot_training_curves.py"),
         "--tag", "nlp", "--demo", *out_flag],
    )
    results["Vision Training Curves"] = run_step(
        "Vision Training Curves",
        [PYTHON, str(SCRIPTS/"visualization"/"plot_training_curves.py"),
         "--tag", "vision", "--demo", *out_flag],
    )

    # ── Step 3: NLP Benchmark ───────────────────────────────────
    results["NLP Benchmark"] = run_step(
        "NLP System Benchmark (Tables 3 & 4)",
        [PYTHON, str(SCRIPTS/"benchmarks"/"benchmark_nlp.py"),
         *demo_flag, *out_flag],
    )

    # ── Step 4: Confusion Matrix ────────────────────────────────
    results["Confusion Matrix"] = run_step(
        "Confusion Matrix (Intent Classification)",
        [PYTHON, str(SCRIPTS/"visualization"/"plot_confusion_matrix.py"),
         "--demo", *out_flag],
    )

    # ── Step 5: Vision Benchmark ────────────────────────────────
    results["Vision Benchmark"] = run_step(
        "Vision System Benchmark (Tables 1, 2, 7, 8)",
        [PYTHON, str(SCRIPTS/"benchmarks"/"benchmark_vision.py"),
         *demo_flag, *out_flag],
    )

    # ── Step 6: Fusion Benchmark ────────────────────────────────
    results["Fusion Benchmark"] = run_step(
        "Fusion Strategy Benchmark (Tables 5 & 6)",
        [PYTHON, str(SCRIPTS/"benchmarks"/"benchmark_fusion.py"),
         *demo_flag, *out_flag],
    )

    # ── Step 7: Jetson Benchmark ──────────────────────────────
    results["Jetson Benchmark"] = run_step(
        "Jetson AGX Orin Performance Benchmark",
        [PYTHON, str(SCRIPTS/"benchmarks"/"benchmark_jetson.py"),
         *out_flag],
    )

    # ── Step 8: E2E Fusion Test ───────────────────────────────
    results["E2E Fusion Test"] = run_step(
        "End-to-End Multimodal Fusion Test",
        [PYTHON, str(SCRIPTS/"test_multimodal_fusion.py"),
         "--n-runs", "20", *out_flag],
    )

    # ── Summary ─────────────────────────────────────────────────
    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  📊 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for step, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {step}")

    success_count = sum(results.values())
    total_count   = len(results)

    print(f"\n  Completed: {success_count}/{total_count} steps")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"\n  📁 Figures output: {args.out_dir}")
    print(f"  📁 Results JSON  : {PROJECT_ROOT}/experiments/results/")

    # List figures
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        figs = list(out_dir.glob("*.png"))
        print(f"\n  Generated {len(figs)} figures:")
        for fig in sorted(figs):
            size_kb = fig.stat().st_size / 1024
            print(f"    📈 {fig.name:45s}  ({size_kb:.0f} KB)")

    print(f"\n{'='*60}")
    print(f"  ✅ All benchmarks complete for NCKH 2026 report!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
