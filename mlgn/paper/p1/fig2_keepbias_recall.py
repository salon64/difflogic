"""Figure 2 — keep-bias task-dependence + the delayed-recall headline (draft §5.1–5.2).

Left (B):  one dial, opposite slopes — the keep-bias sweep RISES on recall (dMNIST-50)
           and FALLS on integration (psMNIST-28). Both curves are the `gated` cell; hue
           distinguishes the TASK here. Error bars where a point has 3 seeds.
Right (C): real-data recall vs delay — gated (kb 6, mean ± s.d., 3 seeds) holds ~3×
           chance through 100 blank steps; the control sits exactly on the MNIST
           majority-class baseline (0.1135) at every delay ≥ 25, and stays there even at
           equal gate count (open diamond at D=50, run p1f_dm50_rddlgn_eqgates).

Run from repo root: python mlgn/paper/p1/fig2_keepbias_recall.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from figstyle import (C_GATED, C_CTRL, C_TASK2, apply_style, load_records, select,
                      curve, eband, refline, seed_stats, save)

CLEAN = dict(lr_min=0.0003, iters=20000)
MAJORITY = 0.1135  # MNIST test-set majority class (digit 1)


def main():
    apply_style()
    recs = load_records()
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ── B: keep-bias dial, opposite slopes (both curves are `gated`; hue = task) ─────
    xs, m, sd, n = curve(recs, "keep_bias", task="smnist-pixel", mechanism="gated",
                         seq_len=51, hidden=1000, **CLEAN)
    eband(axb, xs, m, sd, C_GATED, "recall — delayed MNIST, D=50", z=4)
    xs, m, sd, n = curve(recs, "keep_bias", task="psmnist", mechanism="gated",
                         seq_len=28, hidden=1000, **CLEAN)
    eband(axb, xs, m, sd, C_TASK2, "integration — psMNIST-28", z=3)
    refline(axb, 1 / 10, "chance 10%")
    axb.set_title("keep-bias is a task-dependent dial\n(high to HOLD · low to ABSORB; bars where 3 seeds)")
    axb.set_xlabel("keep-bias (initialization strength)")
    axb.set_ylabel("test accuracy (discrete)")
    axb.set_xticks([0, 1, 2, 3, 4, 6])
    axb.set_ylim(0, 0.85)
    axb.legend(loc="upper right")

    # ── C: recall vs delay (headline) ────────────────────────────────────────────────
    delay = lambda r: r["seq_len"] - 1
    xs, m, sd, n = curve(recs, delay, task="smnist-pixel", mechanism="gated",
                         keep_bias=6.0, hidden=1000, **CLEAN)
    eband(axc, xs, m, sd, C_GATED, "gated, keep-bias 6 (3 seeds)", z=4)
    xs, m, sd, _ = curve(recs, delay, task="smnist-pixel", mechanism="rddlgn",
                         hidden=1000, **CLEAN)
    eband(axc, xs, m, sd, C_CTRL, "control (concat-recurrence)", z=3)
    # equal-gates control at D=50: still exactly at the majority baseline, grad[t=0] = 0
    eq = select(recs, task="smnist-pixel", mechanism="rddlgn", hidden=2000, seq_len=51, **CLEAN)
    if eq:
        y, _, _ = seed_stats(eq)
        axc.plot([50], [y], marker="D", ls="", color=C_CTRL, mfc="white", zorder=5,
                 label="control, equal gates (D=50)")
        axc.annotate("still at baseline at 2× gates\n(write-step gradient = 0)",
                     xy=(50, y), xytext=(28, 0.24), fontsize=7.5, color=C_CTRL,
                     arrowprops=dict(arrowstyle="->", color=C_CTRL, alpha=0.7))
    refline(axc, MAJORITY, "majority class 11.35%")
    axc.set_title("delayed-recall MNIST — accuracy vs delay\n(control collapses to the majority baseline · gated holds)")
    axc.set_xlabel("delay D (blank steps after the one-step image)")
    axc.set_ylabel("test accuracy (discrete)")
    axc.set_xticks([0, 25, 50, 75, 100])
    axc.set_ylim(0, 0.85)
    handles, labels = axc.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: ("gated" not in labels[i], "equal" in labels[i]))
    axc.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right")

    fig.tight_layout()
    save(fig, "fig2_keepbias_recall")


if __name__ == "__main__":
    main()
