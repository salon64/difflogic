"""One-off: gradient-norm-through-time carousel figure for P1 (draft App. A.3).

Reads grad_profile arrays from existing result JSONs and saves
mlgn/seqlgn/results/curves_gradnorm.png. Control zeros (exact vanishing) are drawn at a
floor line and annotated. Run from repo root:  python <this file>
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join("mlgn", "seqlgn", "results")
FLOOR = 1e-8  # display floor for exact zeros on the log axis

PANELS = [
    ("copy-50 (synthetic recall)",
     "copy_gated_gated_L50_20260612-110149.json",
     "copy_rddlgn_rddlgn_L50_20260612-101043.json"),
    ("delayed-recall MNIST, D=50",
     "smnist-pixel_gated_rec_d50_gated_20260619-132912.json",
     "smnist-pixel_rddlgn_rec_d50_rddlgn_20260619-123018.json"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (title, f_gated, f_ctrl) in zip(axes, PANELS):
    for fname, color, label in [(f_gated, "tab:blue", "gated (ours)"),
                                (f_ctrl, "tab:red", "rddlgn (control)")]:
        gp = json.load(open(os.path.join(RESULTS, fname)))["grad_profile"]
        ts = list(range(len(gp)))
        vals = [max(v, FLOOR) for v in gp]
        ax.plot(ts, vals, "-", color=color, label=label, lw=2)
        zeros = [t for t, v in zip(ts, gp) if v == 0.0]
        if zeros:
            ax.plot(zeros, [FLOOR] * len(zeros), "x", color=color, ms=4)
            ax.annotate("exactly 0 (no signal reaches the write step)",
                        xy=(zeros[len(zeros) // 2], FLOOR), xytext=(0.05, 0.16),
                        textcoords="axes fraction", color=color, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
    ax.axvline(0, ls=":", color="k", alpha=0.4)
    ax.text(0.4, 0.97, "write step", rotation=90, va="top", fontsize=8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 3, None)
    ax.set_title(f"gradient norm through time — {title}\n(carousel carries signal to t=0 · control vanishes)")
    ax.set_xlabel("timestep t")
    ax.set_ylabel(r"$\|\partial \mathcal{L} / \partial h_t\|$")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=9)
plt.tight_layout()
out = os.path.join(RESULTS, "curves_gradnorm.png")
plt.savefig(out, dpi=120)
print("saved", out)
