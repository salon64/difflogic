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

    for task, hidden in [("copy", 1024), ("psmnist", 1000)]:
        print(f"\n{task} (hidden {hidden}):")
        c = curve(recs, task, hidden)
        for mech in sorted(c):
            for seq in sorted(c[mech]):
                t, s, n = c[mech][seq]
                print(f"  {mech:8} L{seq:<4} test={t:.3f}  soft={s:.3f}  gap={s - t:+.3f}  (n={n})")


if __name__ == "__main__":
    main()
