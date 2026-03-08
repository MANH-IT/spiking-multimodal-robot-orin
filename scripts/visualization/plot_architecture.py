# -*- coding: utf-8 -*-
"""
plot_architecture.py — NCKH 2026
==================================
Sinh architecture diagrams cho SNN và Multimodal Fusion.
Dùng matplotlib patches (không cần graphviz/draw.io).

Output:
    experiments/figures/fig1_snn_architecture.png
    experiments/figures/fig4_fusion_architecture.png

Usage:
    python scripts/visualization/plot_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] pip install matplotlib numpy")


# ── Color palette ──────────────────────────────────────────────
C = {
    "input":    "#1D4ED8",  # deep blue
    "encoder":  "#7C3AED",  # purple
    "fusion":   "#B45309",  # amber
    "temporal": "#065F46",  # dark green
    "head":     "#DC2626",  # red
    "output":   "#1E293B",  # dark gray
    "bridge":   "#0891B2",  # cyan
    "decision": "#BE185D",  # pink
    "accent":   "#059669",  # green  ← fixed: was missing
    "white":    "white",
    "bg":       "#F1F5F9",
}


def _box(ax, x, y, w, h, color, text, fontsize=9, text_color="white",
         style="round,pad=0.05", alpha=1.0, subtext=None):
    """Vẽ box với text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style,
        linewidth=1.5,
        edgecolor="white",
        facecolor=color,
        alpha=alpha,
        zorder=4,
    )
    ax.add_patch(box)
    ty = y if subtext is None else y + h * 0.12
    ax.text(x, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=text_color, zorder=5)
    if subtext:
        ax.text(x, y - h * 0.22, subtext, ha="center", va="center",
                fontsize=fontsize - 2, color=text_color, alpha=0.85, zorder=5)


def _arrow(ax, x0, y0, x1, y1, color="#64748B", lw=1.5, style="->"):
    """Vẽ mũi tên."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=6,
    )


def _label(ax, x, y, text, fontsize=8, color="#475569"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic", zorder=7)


def draw_snn_architecture(out_path: Path) -> None:
    """
    Vẽ DepthAwareSNN architecture diagram.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle(
        "DepthAwareSNN — Architecture Diagram\n"
        "Spiking Neural Network for 3D Object Detection from RGB-D Sequences  |  NCKH 2026",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Input streams ──────────────────────────────────────────
    # RGB stream (top lane: y=7.5)
    _box(ax, 1.2, 7.5, 2.0, 1.0, C["input"],   "RGB Input",    subtext="(B,T,3,H,W)")
    _box(ax, 4.0, 7.5, 2.2, 1.1, C["encoder"], "SpikingConv\nEncoder (RGB)",
         subtext="Conv→BN→LIF ×3", fontsize=8)

    # Depth stream (bottom lane: y=5.5)
    _box(ax, 1.2, 5.5, 2.0, 1.0, C["input"],   "Depth Input",  subtext="(B,T,1,H,W)")
    _box(ax, 4.0, 5.5, 2.2, 1.1, C["encoder"], "SpikingConv\nEncoder (Depth)",
         subtext="Conv→BN→LIF ×3", fontsize=8)

    # Arrows: Input → Encoder
    _arrow(ax, 2.2, 7.5, 2.9, 7.5)
    _arrow(ax, 2.2, 5.5, 2.9, 5.5)

    # ── Fusion ─────────────────────────────────────────────────
    _box(ax, 7.0, 6.5, 2.5, 1.8, C["fusion"],
         "DepthAware\nAttentionFusion",
         subtext="Channel + Spatial Attn", fontsize=8)
    _arrow(ax, 5.1, 7.5, 5.8, 7.0)   # rgb → fusion
    _arrow(ax, 5.1, 5.5, 5.8, 6.0)   # depth → fusion
    _arrow(ax, 5.9, 6.5, 5.75, 6.5)  # connect
    _arrow(ax, 8.25, 6.5, 9.0, 6.5)  # fusion → multiscale

    # LIF notation on encoders
    _label(ax, 4.0, 8.8, "snntorch.Leaky (LIF)", fontsize=7.5)
    _label(ax, 4.0, 4.3, "snntorch.Leaky (LIF)", fontsize=7.5)

    # ── Multi-Scale STDFE ──────────────────────────────────────
    _box(ax, 10.0, 6.5, 2.3, 1.4, C["encoder"],
         "MultiScale\nSTDFE",
         subtext="3 scales → concat", fontsize=8)
    _arrow(ax, 11.15, 6.5, 11.9, 6.5)

    # ── Temporal ──────────────────────────────────────────────
    _box(ax, 13.0, 6.5, 2.2, 1.4, C["temporal"],
         "SpikingTemporal\nRNN",
         subtext="snntorch.RLeaky", fontsize=8)
    _arrow(ax, 14.1, 6.5, 14.5, 6.5)

    # ── Detection Head ────────────────────────────────────────
    _box(ax, 15.2, 8.0, 1.4, 0.85, C["head"], "BBox 3D",    subtext="(B,6)", fontsize=8)
    _box(ax, 15.2, 5.2, 1.4, 0.85, C["head"], "Class\nLogits", subtext="(B,253)", fontsize=8)
    _arrow(ax, 14.55, 6.5, 15.0, 7.9, color=C["head"])
    _arrow(ax, 14.55, 6.5, 15.0, 5.3, color=C["head"])

    # ── Temporal loop notation ────────────────────────────────
    ax.annotate("", xy=(7.0, 9.2), xytext=(14.0, 9.2),
                arrowprops=dict(arrowstyle="<->", color="#94A3B8", lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(10.5, 9.6, "T timesteps (carry hidden state)", ha="center",
            fontsize=8, color="#64748B", style="italic")

    # ── Spike rate monitoring ─────────────────────────────────
    _box(ax, 4.0, 2.5, 3.0, 0.9, "#64748B",
         "Spike Rate Monitor",
         subtext="mean activity → energy metric", fontsize=7.5, alpha=0.7)
    ax.annotate("", xy=(4.0, 2.95), xytext=(4.0, 4.95),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=1,
                                linestyle="dashed"))

    # ── Stats box ─────────────────────────────────────────────
    stats_text = (
        "Model Statistics\n"
        "─────────────────────────\n"
        "Parameters  :  ~2.1M\n"
        "FLOPs/frame :  ~4.8 GFLOPs\n"
        "SNN mode    :  snntorch.Leaky\n"
        "Classes     :  253 (HILO)\n"
        "Input size  :  224×224"
    )
    ax.text(1.0, 2.5, stats_text, ha="left", va="center",
            fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#CBD5E1", alpha=0.85),
            zorder=8)

    # ── Legend ────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C["input"],    label="Input data"),
        mpatches.Patch(color=C["encoder"],  label="SNN encoder"),
        mpatches.Patch(color=C["fusion"],   label="Fusion module"),
        mpatches.Patch(color=C["temporal"], label="Temporal RNN"),
        mpatches.Patch(color=C["head"],     label="Detection head"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="#CBD5E1",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ SNN Architecture saved: {out_path}")


def draw_fusion_architecture(out_path: Path) -> None:
    """
    Vẽ Multimodal Fusion architecture diagram.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle(
        "Multimodal Fusion Architecture — VisionNLPBridge\n"
        "Cross-Modal Attention for Robot Decision Making  |  NCKH 2026",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Input ─────────────────────────────────────────────────
    _box(ax, 1.5, 6.5, 2.2, 1.0, C["input"],  "RGB-D\nFrames",  subtext="(B,T,4,H,W)")
    _box(ax, 1.5, 4.0, 2.2, 1.0, C["input"],  "Speech\nText",   subtext="Vietnamese")
    _box(ax, 1.5, 1.5, 2.2, 1.0, "#475569",   "Sensor\nData",   subtext="IMU / GPS")

    # ── Parallel processing ───────────────────────────────────
    _box(ax, 4.5, 6.5, 2.2, 1.0, C["encoder"], "DepthAwareSNN",
         subtext="3D Detection")
    _box(ax, 4.5, 4.0, 2.2, 1.0, C["encoder"], "SpikingLM",
         subtext="NLU/Intent")
    _box(ax, 4.5, 1.5, 2.2, 1.0, "#475569",    "Sensor\nProcessor",
         subtext="Filtering")

    for ya, yb in [(6.5, 6.5), (4.0, 4.0), (1.5, 1.5)]:
        _arrow(ax, 2.6, ya, 3.4, yb)

    # Layer labels
    ax.text(4.5, 7.9, "Parallel Inference\n(ThreadPoolExecutor)", ha="center",
            fontsize=8, color="#64748B", style="italic",
            bbox=dict(boxstyle="round", fc="#E2E8F0", ec="none", alpha=0.7))

    # ── Feature extraction ────────────────────────────────────
    _box(ax, 7.2, 5.6, 1.8, 0.9, C["bridge"], "Vision\nFeatures",
         subtext="(B, 256)", fontsize=8)
    _box(ax, 7.2, 4.4, 1.8, 0.9, C["bridge"], "NLP\nFeatures",
         subtext="(B, 512)", fontsize=8)

    _arrow(ax, 5.6, 6.5, 6.5, 5.8)
    _arrow(ax, 5.6, 4.0, 6.5, 4.6)

    # ── Cross-Modal Attention ─────────────────────────────────
    _box(ax, 9.8, 5.0, 2.8, 2.2, C["fusion"],
         "CrossModal\nAttention",
         subtext="8 heads, dim=256\nVision→Query, NLP→K/V", fontsize=8)
    _arrow(ax, 8.1, 5.6, 8.4, 5.3)
    _arrow(ax, 8.1, 4.4, 8.4, 4.7)
    _arrow(ax, 11.2, 5.0, 11.8, 5.0)

    # Self‑attn + FFN sub-boxes
    _box(ax, 9.8, 3.5, 2.2, 0.75, C["bridge"],
         "Self-Attention + FFN", fontsize=7.5, alpha=0.5)
    ax.annotate("", xy=(9.8, 3.88), xytext=(9.8, 3.90),
                arrowprops=dict(arrowstyle="->", color="gray"))

    # ── Fused representation ──────────────────────────────────
    _box(ax, 12.8, 5.0, 1.8, 0.9, C["accent"],
         "Fused\nFeature",
         subtext="(B, 256)", fontsize=8)
    _arrow(ax, 13.7, 5.0, 14.2, 6.3, color=C["head"])
    _arrow(ax, 13.7, 5.0, 14.2, 5.0, color=C["head"])
    _arrow(ax, 13.7, 5.0, 14.2, 3.7, color=C["head"])

    # ── Output heads ─────────────────────────────────────────
    _box(ax, 15.1, 6.3, 1.6, 0.85, C["head"],     "Action\nClassifier",
         subtext="8 actions", fontsize=8)
    _box(ax, 15.1, 5.0, 1.6, 0.85, C["decision"], "Confidence\nEstimator",
         subtext="[0,1]", fontsize=8)
    _box(ax, 15.1, 3.7, 1.6, 0.85, "#475569",     "Rule-Based\nFallback",
         subtext="quick path", fontsize=8)

    # ── Fusion strategies annotation ──────────────────────────
    strat_text = (
        "Fusion Strategies:\n"
        "① Early  — feature concat\n"
        "② Late   — decision ensemble\n"
        "③ Hybrid — cross-modal attn ✓"
    )
    ax.text(0.3, 0.5, strat_text, ha="left", va="bottom",
            fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#CBD5E1", alpha=0.9),
            zorder=8)

    legend_items = [
        mpatches.Patch(color=C["input"],    label="Input"),
        mpatches.Patch(color=C["encoder"],  label="Model backbone"),
        mpatches.Patch(color=C["bridge"],   label="Feature bridge"),
        mpatches.Patch(color=C["fusion"],   label="Cross-modal attention"),
        mpatches.Patch(color=C["head"],     label="Output head"),
        mpatches.Patch(color=C["decision"], label="Decision"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#CBD5E1",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Fusion Architecture saved: {out_path}")


def main():
    out_dir = PROJECT_ROOT / "experiments" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🎨 Generating Architecture Diagrams — NCKH 2026")

    draw_snn_architecture(out_dir / "fig1_snn_architecture.png")
    draw_fusion_architecture(out_dir / "fig4_fusion_architecture.png")

    print(f"\n✅ Done! Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
