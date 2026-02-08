#!/usr/bin/env python3
"""Generate eval_results.pdf for the paper.
Style matches characterization.py (large fonts, same color palette, grid).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Color palette matching characterization.py
C_BASELINE = "#FF9800"   # orange (same as LLM Thinking in exec_overview)
C_BPF      = "#2196F3"   # blue   (primary data color in characterization)

labels = ["No Isolation", "BPF"]

# --- (a) OOM Survival Rate ---
values_a = [66, 100]
bars1 = ax1.bar(labels, values_a, width=0.5, color=[C_BASELINE, C_BPF],
                alpha=0.85, edgecolor="white")
ax1.set_ylabel("Survival Rate (%)", fontsize=22)
ax1.set_ylim(0, 115)
ax1.set_title("(a) OOM Survival (1100 MB)", fontsize=24)
for bar, val in zip(bars1, values_a):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
             f"{val}%", ha="center", va="center", fontsize=24,
             fontweight="bold", color="white")
ax1.tick_params(axis="both", labelsize=20)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(axis="y", alpha=0.3)

# --- (b) HIGH P95 Allocation Latency ---
values_b = [70.97, 50.14]
bars2 = ax2.bar(labels, values_b, width=0.5, color=[C_BASELINE, C_BPF],
                alpha=0.85, edgecolor="white")
ax2.set_ylabel("P95 Latency (ms)", fontsize=22)
ax2.set_ylim(0, 90)
ax2.set_title("(b) HIGH P95 Latency (1300 MB)", fontsize=24)
for bar, val in zip(bars2, values_b):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
             f"{val:.1f} ms", ha="center", va="center", fontsize=24,
             fontweight="bold", color="white")
# Improvement annotation
ax2.annotate("\u221229%", xy=(1, 52), xytext=(1.35, 72),
             fontsize=22, fontweight="bold", color=C_BPF,
             arrowprops=dict(arrowstyle="->", color=C_BPF, lw=2))
ax2.tick_params(axis="both", labelsize=20)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/yunwei37/workspace/agentcgroup/paper-repo/docs/img/eval_results.pdf",
            bbox_inches="tight")
print("Saved eval_results.pdf")
