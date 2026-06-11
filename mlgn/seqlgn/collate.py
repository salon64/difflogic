"""
collate.py — summarize all results/*.json runs into one table (+ optional CSV).
================================================================================

As the experiment sweep grows, this pulls every run record in ``results/`` into a single
sorted table so we can read the accuracy-vs-length curves and build the paper figures
without opening JSONs by hand.

    python -m mlgn.seqlgn.collate                      # all runs, table
    python -m mlgn.seqlgn.collate --task copy          # filter by task
    python -m mlgn.seqlgn.collate --mechanism gated    # filter by mechanism
    python -m mlgn.seqlgn.collate --csv                # emit CSV (for plotting)

Sorted by (task, mechanism, seq_len, hidden, seed). `test`/`soft`/`gap` are the
discrete-locked test acc, the soft-model acc, and the discretization gap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COLS = [
    ("task", "task", "{}"),
    ("mech", "mechanism", "{}"),
    ("seq", "seq_len", "{}"),
    ("hidden", "hidden", "{}"),
    ("seed", "seed", "{}"),
    ("kb", "keep_bias", "{}"),
    ("test", "test_acc", "{:.3f}"),
    ("soft", "test_soft", "{:.3f}"),
    ("gap", "discretization_gap", "{:+.3f}"),
    ("gates", "logic_gates", "{:,}"),
    ("skip", "n_skipped", "{}"),
    ("min", "train_minutes", "{:.0f}"),
]


def load_records():
    recs = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        try:
            with open(path) as f:
                r = json.load(f)
            r["_file"] = os.path.basename(path)
            recs.append(r)
        except Exception as e:
            print(f"[skip] {os.path.basename(path)}: {e}")
    return recs


def fmt(r, key, spec):
    v = r.get(key)
    if v is None:
        return "-"
    try:
        return spec.format(v)
    except Exception:
        return str(v)


def main():
    p = argparse.ArgumentParser(description="Collate seqlgn run results.")
    p.add_argument("--task")
    p.add_argument("--mechanism")
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    recs = load_records()
    if args.task:
        recs = [r for r in recs if r.get("task") == args.task]
    if args.mechanism:
        recs = [r for r in recs if r.get("mechanism") == args.mechanism]
    recs.sort(key=lambda r: (str(r.get("task")), str(r.get("mechanism")),
                             r.get("seq_len") or 0, r.get("hidden") or 0, r.get("seed") or 0))

    if not recs:
        print("no matching results in", RESULTS_DIR)
        return

    if args.csv:
        keys = [k for _, k, _ in COLS]
        print(",".join(h for h, _, _ in COLS))
        for r in recs:
            print(",".join(str(r.get(k, "")) for k in keys))
        return

    headers = [h for h, _, _ in COLS]
    rows = [[fmt(r, k, s) for _, k, s in COLS] for r in recs]
    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print(f"\n{len(recs)} run(s).")


if __name__ == "__main__":
    main()
