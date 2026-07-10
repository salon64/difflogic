"""figstyle.py — shared style + data loading for the P1 paper figures.

The fig*_*.py scripts in this directory are the CANONICAL source of the paper's figures
(mlgn/seqlgn/plot.py remains the quick working-figure tool). Run from the repo root:

    python mlgn/paper/p1/fig1_length_sweeps.py
    python mlgn/paper/p1/fig2_keepbias_recall.py
    python mlgn/paper/p1/fig3_gradnorm.py

Outputs land in mlgn/paper/p1/figs/ as both .png (for the markdown draft) and .pdf
(vector, for the LaTeX camera-ready).

Data hygiene (the App.-A.3 rules, enforced here so no figure can silently regress):
  * P1 mechanisms only (rddlgn / gated / lstm / gru_cell) — later-project runs
    (latch/clatch/combo) are excluded by mechanism.
  * No run trained with --deep-sup / --margin-reg / --anneal (P2 training signals that
    share P1's configs and would pollute seed averages).
  * The clean-sweep config only (lr cosine-decayed 3e-3→3e-4, 20k iters) unless a series
    spec says otherwise.
  * Duplicate runs of the same (series, x, seed) are collapsed by mean BEFORE seed
    statistics (the results dir contains a few same-config re-runs); error bars are the
    sample s.d. (ddof=1) across distinct seeds, matching the draft's tables.

Colors: Okabe-Ito (colorblind-safe; validated CVD ΔE ≥ 17.9 for this subset). Color
follows the MECHANISM across all figures; the equal-gates control is the same vermillion
as the control (same entity), distinguished by dashed line + open markers.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(ROOT, "mlgn", "seqlgn", "results")
FIGS = os.path.join(HERE, "figs")

# ── palette: color follows the mechanism, everywhere ─────────────────────────────────
C_GATED = "#0072B2"   # blue        — gated (ours)
C_CTRL = "#D55E00"    # vermillion  — rddlgn control (equal-gates variant: dashed + open)
C_LSTM = "#009E73"    # bluish green
C_GRU = "#CC79A7"     # reddish purple — gru_cell
C_TASK2 = "#E69F00"   # orange — second TASK (fig 2B only, both curves are `gated`)
C_REF = "#666666"     # reference lines (chance / majority baseline), gray, never a series

P1_MECHS = {"rddlgn", "gated", "lstm", "gru_cell"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.88,
        "legend.edgecolor": "none",
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "errorbar.capsize": 3,
    })


def load_records():
    """All result JSONs passing the P1 hygiene filter (see module docstring)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("mechanism") not in P1_MECHS:
            continue
        if r.get("deep_sup") or r.get("margin_reg") or r.get("anneal"):
            continue
        r["_file"] = os.path.basename(f)
        recs.append(r)
    return recs


def select(recs, **crit):
    """Filter records by exact-match criteria; a criterion value of callable is a predicate."""
    out = []
    for r in recs:
        ok = True
        for k, v in crit.items():
            rv = r.get(k)
            ok = v(rv) if callable(v) else (rv == v)
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def seed_stats(runs, field="test_acc"):
    """(mean, sd, n) across DISTINCT seeds; same-seed duplicates collapsed by mean first.
    sd is the sample s.d. (ddof=1), None when n < 2 — matching the draft's tables."""
    by_seed = defaultdict(list)
    for r in runs:
        v = r.get(field)
        if v is not None:
            by_seed[r.get("seed")].append(v)
    vals = [sum(v) / len(v) for v in by_seed.values()]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
    return m, sd, len(vals)


def curve(recs, xkey, field="test_acc", **crit):
    """xs, means, sds, ns for one series, grouped by xkey (e.g. seq_len). xkey may be a
    callable(record) -> x (e.g. delay = seq_len - 1)."""
    groups = defaultdict(list)
    for r in select(recs, **crit):
        x = xkey(r) if callable(xkey) else r.get(xkey)
        groups[x].append(r)
    xs = sorted(groups)
    stats = [seed_stats(groups[x], field) for x in xs]
    return xs, [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats]


def eband(ax, xs, means, sds, color, label, ls="-", marker="o", mfc=None, alpha=1.0, z=3,
          dodge=0.0, bars=True):
    """Line + error bars (bars only where a sd exists). `dodge` shifts x slightly so
    curves that sit on identical values (e.g. lstm/gru_cell both at chance) stay visible."""
    xs = [x + dodge for x in xs]
    yerr = [sd if sd is not None else 0.0 for sd in sds] if bars else None
    ax.errorbar(xs, means, yerr=yerr, color=color, label=label, ls=ls, marker=marker,
                mfc=color if mfc is None else mfc, mec=color, alpha=alpha, zorder=z)


def refline(ax, y, text):
    ax.axhline(y, ls=":", color=C_REF, lw=1.2, alpha=0.8, zorder=1)
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points",
                ha="right", fontsize=7.5, color=C_REF)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    print("saved", os.path.join(FIGS, name) + ".{png,pdf}")
