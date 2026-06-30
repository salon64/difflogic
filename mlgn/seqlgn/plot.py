"""
plot.py — accuracy-vs-sequence-length curves for the copy and psMNIST sweeps.
Saves results/curves.png and prints the grouped (seed-averaged) table.

    python -m mlgn.seqlgn.plot

Filters to the clean sweep config (lr_min=3e-4, iters=20000, hidden per task), groups by
(mechanism, seq_len), prefers the highest keep_bias per group (so the fixed lstm wins over
its cold-start run), and averages over seeds. Solid = discrete test acc; dashed faint =
soft (relaxed) acc; dotted = chance.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")
COL = {"rddlgn": "tab:red", "gated": "tab:blue", "lstm": "tab:green", "gru_cell": "tab:purple"}


def load():
    recs = []
    for f in glob.glob(os.path.join(RESULTS, "*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    return recs


def curve(recs, task, hidden):
    rs = [r for r in recs if r.get("task") == task and r.get("hidden") == hidden
          and r.get("lr_min") == 0.0003 and r.get("iters") == 20000]
    groups = defaultdict(list)
    for r in rs:
        groups[(r["mechanism"], r["seq_len"])].append(r)
    out = defaultdict(dict)
    for (mech, seq), g in groups.items():
        kbmax = max((x.get("keep_bias") or 0) for x in g)
        gg = [x for x in g if (x.get("keep_bias") or 0) == kbmax]
        tm = sum(x["test_acc"] for x in gg) / len(gg)
        sm = sum((x.get("test_soft") or float("nan")) for x in gg) / len(gg)
        out[mech][seq] = (tm, sm, len(gg))
    return out


def main():
    recs = load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [("copy", 1024, 1 / 8, "copy (synthetic recall) — chance 12.5%"),
              ("psmnist", 1000, 1 / 10, "psMNIST (real, chunked) — chance 10%")]
    for ax, (task, hidden, chance, title) in zip(axes, panels):
        c = curve(recs, task, hidden)
        for mech, d in sorted(c.items()):
            xs = sorted(d)
            ax.plot(xs, [d[x][0] for x in xs], "-o", color=COL.get(mech, "gray"), label=f"{mech}")
            ax.plot(xs, [d[x][1] for x in xs], "--", color=COL.get(mech, "gray"), alpha=0.35)
        ax.axhline(chance, ls=":", color="k", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("sequence length (timesteps)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, title="solid=test  dashed=soft")
    plt.tight_layout()
    out = os.path.join(RESULTS, "curves.png")
    plt.savefig(out, dpi=120)
    print("saved", out)

    # --- second figure: B (keep-bias sweep) + C (real-data recall vs delay) ---
    fig2, (axb, axc) = plt.subplots(1, 2, figsize=(13, 5))
    # B: psMNIST-28, gated, keep-bias sweep, vs rddlgn baseline
    bg = sorted([(r["keep_bias"], r["test_acc"], r.get("test_soft"))
                 for r in recs if r.get("task") == "psmnist" and r.get("seq_len") == 28
                 and r["mechanism"] == "gated"])
    # fair baseline = equal-gates rddlgn (hidden 2000 = 4000 gates = gated's count)
    br = [r["test_acc"] for r in recs if r.get("task") == "psmnist" and r.get("seq_len") == 28
          and r["mechanism"] == "rddlgn" and r.get("hidden") == 2000]
    if bg:
        axb.plot([x[0] for x in bg], [x[1] for x in bg], "-o", color="tab:blue", label="gated test")
        axb.plot([x[0] for x in bg], [x[2] for x in bg], "--o", color="tab:blue", alpha=0.4, label="gated soft")
    if br:
        axb.axhline(br[0], color="tab:red", label="rddlgn test (equal gates)")
    axb.axhline(0.1, ls=":", color="k", alpha=0.5)
    axb.set_title("B: psMNIST-28 — accuracy vs keep-bias\n(low keep-bias → better integration)")
    axb.set_xlabel("keep-bias (init)")
    axb.set_ylabel("accuracy")
    axb.set_ylim(0, 1)
    axb.grid(alpha=0.2)
    axb.legend(fontsize=8)
    # C: real-data recall — accuracy vs delay (smnist-pixel, 1-step encode + delay)
    cre = [r for r in recs if r.get("task") == "smnist-pixel"]
    for mech, col in [("gated", "tab:blue"), ("rddlgn", "tab:red")]:
        pts = sorted([(r["seq_len"] - 1, r["test_acc"], r.get("test_soft"))
                      for r in cre if r["mechanism"] == mech])
        if pts:
            axc.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, label=f"{mech} test")
            axc.plot([p[0] for p in pts], [p[2] for p in pts], "--", color=col, alpha=0.4)
    axc.axhline(0.1, ls=":", color="k", alpha=0.5, label="chance")
    axc.set_title("C: real-data RECALL — accuracy vs delay\n(hold digit through blank steps)")
    axc.set_xlabel("delay (blank steps after image)")
    axc.set_ylabel("accuracy")
    axc.set_ylim(0, 1)
    axc.grid(alpha=0.2)
    axc.legend(fontsize=8)
    plt.tight_layout()
    out2 = os.path.join(RESULTS, "curves_bc.png")
    plt.savefig(out2, dpi=120)
    print("saved", out2)

    for task, hidden in [("copy", 1024), ("psmnist", 1000)]:
        print(f"\n{task} (hidden {hidden}):")
        c = curve(recs, task, hidden)
        for mech in sorted(c):
            for seq in sorted(c[mech]):
                t, s, n = c[mech][seq]
                print(f"  {mech:8} L{seq:<4} test={t:.3f}  soft={s:.3f}  gap={s - t:+.3f}  (n={n})")


if __name__ == "__main__":
    main()
