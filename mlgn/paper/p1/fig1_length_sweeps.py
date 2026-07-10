"""Figure 1 — length sweeps (draft §5.1 / §5.2 cautionary panel).

Left:  copy(L) — deployed (discrete) accuracy vs delay, mean ± s.d. over 3 seeds for the
       gated cell and BOTH controls (equal-width and equal-gates); the dual-state
       ablations (lstm / gru_cell, 1 seed) cold-start past L20. Gated's soft accuracy is
       the faint dashed line — its distance from the solid line at L50 is the computation
       gap of §5.5.
Right: psMNIST-28 length sweep at the recall-tuned keep-bias (kb 4) — the cautionary
       panel: the control beats a WRONGLY-BIASED gated cell at every length (§5.2).

Run from repo root: python mlgn/paper/p1/fig1_length_sweeps.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from figstyle import (C_GATED, C_CTRL, C_LSTM, C_GRU, apply_style, load_records,
                      curve, eband, refline, save)

CLEAN = dict(lr_min=0.0003, iters=20000)  # the clean-sweep config (App. A.2)


def main():
    apply_style()
    recs = load_records()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ── left: copy(L), discrete ──────────────────────────────────────────────────────
    xs, m, sd, n = curve(recs, "seq_len", task="copy", mechanism="gated",
                         hidden=1024, keep_bias=3.0, **CLEAN)
    eband(axl, xs, m, sd, C_GATED, "gated (ours)", z=5)
    xs, m, sd, _ = curve(recs, "seq_len", field="test_soft", task="copy",
                         mechanism="gated", hidden=1024, keep_bias=3.0, **CLEAN)
    eband(axl, xs, m, sd, C_GATED, "_nolegend_", ls="--", marker="",
          alpha=0.4, z=4, bars=False)
    # direct-label the soft line mid-segment (clear of the L20/L35 error bars)
    axl.annotate("gated — soft (relaxed)", xy=(26, 0.938), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=8,
                 color=C_GATED, alpha=0.7)
    xs, m, sd, _ = curve(recs, "seq_len", task="copy", mechanism="rddlgn", hidden=1024, **CLEAN)
    eband(axl, xs, m, sd, C_CTRL, "control (concat-recurrence)", z=4)
    xs, m, sd, _ = curve(recs, "seq_len", task="copy", mechanism="rddlgn", hidden=2048, **CLEAN)
    eband(axl, xs, m, sd, C_CTRL, "control, equal gates (2×width)", ls="--", mfc="white", z=3)
    # the two dual-state ablations sit on near-identical values — dodge them apart
    xs, m, sd, _ = curve(recs, "seq_len", task="copy", mechanism="lstm",
                         hidden=1024, keep_bias=4.0, **CLEAN)
    eband(axl, xs, m, sd, C_LSTM, "lstm (ablation)", marker="s", z=2, dodge=-0.45)
    xs, m, sd, _ = curve(recs, "seq_len", task="copy", mechanism="gru_cell",
                         hidden=1024, keep_bias=3.0, **CLEAN)
    eband(axl, xs, m, sd, C_GRU, "gru_cell (ablation)", marker="D", z=2, dodge=0.45)
    refline(axl, 1 / 8, "chance 12.5%")
    axl.set_title("copy — long-range recall\n(deployed circuit accuracy; soft accuracy of every control run ≈ chance)")
    axl.set_xlabel("delay L (blank steps after the cue-flagged symbol)")
    axl.set_ylabel("test accuracy (discrete)")
    axl.set_xticks([20, 35, 50])
    axl.set_ylim(0, 1.04)
    axl.legend(loc="upper right")

    # ── right: psMNIST-28 length sweep at the recall-tuned kb (cautionary) ───────────
    xs, m, sd, _ = curve(recs, "seq_len", task="psmnist", mechanism="gated",
                         hidden=1000, keep_bias=4.0, **CLEAN)
    eband(axr, xs, m, sd, C_GATED, "gated, keep-bias 4 (recall-tuned = wrong)", z=4)
    xs, m, sd, _ = curve(recs, "seq_len", task="psmnist", mechanism="rddlgn", hidden=1000, **CLEAN)
    eband(axr, xs, m, sd, C_CTRL, "control (concat-recurrence)", z=3)
    refline(axr, 1 / 10, "chance 10%")
    axr.set_title("psMNIST-28 (chunked) — integration, WRONG keep-bias\n(the cautionary panel: hold-biased gating loses at every length)")
    axr.set_xlabel("sequence length (steps of 784/L pixels)")
    axr.set_ylabel("test accuracy (discrete)")
    axr.set_ylim(0, 1.04)
    axr.legend(loc="upper right")

    fig.tight_layout()
    save(fig, "fig1_length_sweeps")


if __name__ == "__main__":
    main()
