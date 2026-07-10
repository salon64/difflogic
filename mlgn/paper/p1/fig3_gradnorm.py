"""Figure 3 — gradient norm through time: the carousel, mechanically (draft §5.1 / A.3).

On both recall tasks the control's gradient at the write step is EXACTLY zero (below
float precision; red ×'s on the display floor) and dies within ~3–5 steps of the
readout, while the gated cell's carousel carries signal to t=0 at full strength (2–3
orders of magnitude ABOVE its late-step norm — the exploding-side tendency §3.4 manages).

Reads the `grad_profile` arrays recorded by --grad-analysis in the named runs (matched by
tag, robust to timestamps). Run from repo root: python mlgn/paper/p1/fig3_gradnorm.py
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from figstyle import C_GATED, C_CTRL, RESULTS, apply_style, save

FLOOR = 1e-8  # display floor for exact zeros on the log axis

PANELS = [
    ("copy-50 (synthetic recall)",
     "copy_gated_gated_L50_2*.json", "copy_rddlgn_rddlgn_L50_2*.json"),
    ("delayed-recall MNIST, D=50",
     "smnist-pixel_gated_rec_d50_gated_2*.json", "smnist-pixel_rddlgn_rec_d50_rddlgn_2*.json"),
]


def profile(pattern):
    for f in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        gp = json.load(open(f)).get("grad_profile")
        if gp:
            return gp, os.path.basename(f)
    raise FileNotFoundError(f"no grad_profile found for {pattern}")


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, (title, pat_gated, pat_ctrl) in zip(axes, PANELS):
        for pat, color, label in [(pat_gated, C_GATED, "gated (ours)"),
                                  (pat_ctrl, C_CTRL, "control (concat-recurrence)")]:
            gp, src = profile(pat)
            ts = list(range(len(gp)))
            ax.plot(ts, [max(v, FLOOR) for v in gp], "-", color=color, label=label)
            zeros = [t for t, v in zip(ts, gp) if v == 0.0]
            if zeros:
                ax.plot(zeros, [FLOOR] * len(zeros), "x", color=color, ms=4)
                ax.annotate("exactly 0 — no signal reaches the write step",
                            xy=(zeros[len(zeros) // 2], FLOOR), xytext=(0.06, 0.18),
                            textcoords="axes fraction", color=color, fontsize=8,
                            arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
        ax.axvline(0, ls=":", color="#666666", lw=1.2, alpha=0.6)
        ax.annotate("write step", xy=(0, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points",
                    rotation=90, va="top", fontsize=7.5, color="#666666")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 3, None)
        ax.set_title(f"gradient norm through time — {title}")
        ax.set_xlabel("timestep t")
        ax.set_ylabel(r"$\|\partial \mathcal{L} / \partial h_t\|$")
        ax.grid(alpha=0.15, which="both")
        ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig3_gradnorm")


if __name__ == "__main__":
    main()
