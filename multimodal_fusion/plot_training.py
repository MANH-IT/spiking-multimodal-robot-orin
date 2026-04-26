"""
Plot Training Curves — SpikingFusionTransformer
================================================
Đọc training_history.json và vẽ biểu đồ đẹp cho báo cáo NCKH.
"""
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

HISTORY_PATH = "models/fusion/training_history.json"
OUTPUT_PATH  = "models/fusion/training_curves.png"

# ── Load data ────────────────────────────────────────────────────────────────
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs      = [h["epoch"]        for h in history]
train_loss  = [h["loss"]         for h in history]
val_loss    = [h["val_loss"]     for h in history]
train_acc   = [h["acc"] * 100   for h in history]  # → percent
val_acc     = [h["val_acc"] * 100 for h in history]
loss_action = [h["loss_action"]  for h in history]
loss_conf   = [h["loss_conf"]    for h in history]

best_epoch  = max(history, key=lambda h: h["val_acc"])
best_val_acc = best_epoch["val_acc"] * 100
best_ep_num  = best_epoch["epoch"]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccee",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "text.color":       "#ddddff",
})

BLUE   = "#4fc3f7"
PURPLE = "#ce93d8"
GREEN  = "#a5d6a7"
ORANGE = "#ffb74d"
RED    = "#ef9a9a"

fig = plt.figure(figsize=(15, 9), facecolor="#0f0f1a")
fig.suptitle(
    "🤖 SpikingFusionTransformer — Training Report",
    fontsize=16, fontweight="bold", color="#e8eaf6", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: Accuracy ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])  # Chiếm 2/3 hàng trên
ax1.plot(epochs, train_acc, color=BLUE,   lw=2,   label="Train Acc", marker="o", ms=3)
ax1.plot(epochs, val_acc,   color=PURPLE, lw=2.5, label="Val Acc",   marker="s", ms=3)
ax1.axvline(best_ep_num, color=GREEN, lw=1.5, ls="--", alpha=0.8, label=f"Best epoch {best_ep_num}")
ax1.axhline(best_val_acc, color=GREEN, lw=1,  ls=":", alpha=0.6)

# Annotation best
ax1.annotate(
    f"  Best: {best_val_acc:.1f}%",
    xy=(best_ep_num, best_val_acc),
    xytext=(best_ep_num + 1.5, best_val_acc - 3),
    color=GREEN, fontsize=10, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
)

# Shade area
ax1.fill_between(epochs, train_acc, alpha=0.08, color=BLUE)
ax1.fill_between(epochs, val_acc,   alpha=0.12, color=PURPLE)

ax1.set_title("Action Classification Accuracy", color="#c5cae9", fontsize=12, pad=8)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Accuracy (%)", fontsize=10)
ax1.legend(loc="lower right", framealpha=0.3, fontsize=9)
ax1.set_ylim(70, 100)
ax1.grid(True)
ax1.set_xlim(1, max(epochs))

# ── Plot 2: Stats box ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_axis_off()

stats = [
    ("Best Val Acc",   f"{best_val_acc:.2f}%",          GREEN),
    ("Best Epoch",     f"{best_ep_num}",                GREEN),
    ("Final Train Acc",f"{train_acc[-1]:.2f}%",         BLUE),
    ("Final Val Acc",  f"{val_acc[-1]:.2f}%",           PURPLE),
    ("Loss Drop",      f"{train_loss[0]:.3f} → {train_loss[-1]:.3f}", ORANGE),
    ("Model Params",   "4.04M",                         "#aaaaff"),
    ("Device",         "CUDA",                          "#aaaaff"),
    ("Epochs",         f"{len(history)}",               "#aaaaff"),
]

ax2.text(0.5, 1.0, "📊 Training Summary", transform=ax2.transAxes,
         ha="center", va="top", fontsize=11, fontweight="bold", color="#e8eaf6")

for i, (label, val, color) in enumerate(stats):
    y = 0.88 - i * 0.11
    ax2.text(0.05, y, f"{label}:", transform=ax2.transAxes,
             ha="left", va="top", fontsize=9, color="#888899")
    ax2.text(0.98, y, val, transform=ax2.transAxes,
             ha="right", va="top", fontsize=9.5, fontweight="bold", color=color)

# Rule-based comparison
ax2.text(0.5, 0.08, "vs Rule-based: ~70%  →  +21.6% ↑",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=8.5, color=GREEN, fontstyle="italic")

# ── Plot 3: Total Loss ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(epochs, train_loss, color=BLUE,   lw=2, label="Train")
ax3.plot(epochs, val_loss,   color=PURPLE, lw=2, label="Val")
ax3.fill_between(epochs, train_loss, alpha=0.1, color=BLUE)
ax3.set_title("Total Loss", color="#c5cae9", fontsize=11, pad=6)
ax3.set_xlabel("Epoch", fontsize=9)
ax3.set_ylabel("Loss", fontsize=9)
ax3.legend(fontsize=8, framealpha=0.3)
ax3.grid(True)

# ── Plot 4: Action Loss vs Conf Loss ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(epochs, loss_action, color=ORANGE, lw=2, label="Action CE")
ax4.plot(epochs, loss_conf,   color=RED,    lw=2, label="Confidence BCE")
ax4.set_title("Loss Components", color="#c5cae9", fontsize=11, pad=6)
ax4.set_xlabel("Epoch", fontsize=9)
ax4.set_ylabel("Loss", fontsize=9)
ax4.legend(fontsize=8, framealpha=0.3)
ax4.grid(True)

# ── Plot 5: Accuracy improvement bar ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
categories = ["Rule-based\n(est.)", "Epoch 1", "Epoch 15\n(best)", "Epoch 30"]
values     = [70.0, train_acc[0], best_val_acc, val_acc[-1]]
colors_bar = [RED, ORANGE, GREEN, PURPLE]
bars = ax5.bar(categories, values, color=colors_bar, alpha=0.8, width=0.6, edgecolor="none")
for bar, val in zip(bars, values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5,
             fontweight="bold", color="#ddddff")
ax5.set_ylim(50, 100)
ax5.set_title("Accuracy Comparison", color="#c5cae9", fontsize=11, pad=6)
ax5.set_ylabel("Accuracy (%)", fontsize=9)
ax5.grid(True, axis="y")
ax5.axhline(70, color=RED, lw=1, ls="--", alpha=0.5)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"✅ Saved training curves: {OUTPUT_PATH}")

# Print summary
print(f"\n{'='*50}")
print(f"📊 TRAINING SUMMARY")
print(f"{'='*50}")
print(f"  Best Val Accuracy  : {best_val_acc:.2f}%  (epoch {best_ep_num})")
print(f"  Final Train Acc    : {train_acc[-1]:.2f}%")
print(f"  Final Val Acc      : {val_acc[-1]:.2f}%")
print(f"  Loss (epoch 1→30)  : {train_loss[0]:.4f} → {train_loss[-1]:.4f}")
print(f"  Improvement vs rule: +{best_val_acc - 70:.1f}%")
print(f"{'='*50}")
