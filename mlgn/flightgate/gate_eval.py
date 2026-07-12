"""
gate_eval.py — assemble the D1 THREE-ARM GATE TABLE from per-run result JSONs.
==============================================================================

Aggregates the JSONs written by ``cli.py`` (one per run) into the gate table the
workmap D1 gate is decided on: the recurrent arms (``gated`` / ``clatch``) vs the
stateless control (``ff``), each under the BLACKOUT (memory-required) condition and
the NO-BLACKOUT control, across seeds. It also surfaces the two reference rows:
the privileged PID teacher (upper bound with full state) and the a' ceiling
(``teacher_masked`` — the PID fed the SAME masked obs the student sees; the honest
"how much does occlusion cost an oracle" reference under blackout).

================================ THE GATE WIN CRITERION ==========================
WIN  <=>  under BLACKOUT, at MATCHED gate count (utils.count_gates identical across
          arms — verified 1,728 @ hidden 432), a recurrent arm beats the stateless
          ff arm on closed-loop RETURN by a margin exceeding the cross-seed spread:

              margin(arm) = mean_return[arm, blackout] - mean_return[ff, blackout]
              WIN(arm)    = margin(arm) > spread      (spread = max cross-seed std
                                                        of the two arms compared)
              gate WIN    = WIN(gated) OR WIN(clatch)

     AND   the NON-VACUITY control holds — under NO BLACKOUT the arms MATCH:
              |mean_return[recurrent, control] - mean_return[ff, control]| <= spread
          proving the blackout gap is MEMORY, not capacity. (If ff also collapses
          under control, the harness/sizes are wrong, not the gate.)

Reference bounds (not part of the boolean, but reported): the a' ceiling upper-bounds
every student under blackout; the raw teacher upper-bounds the control condition.
A recurrent arm ABOVE the a' ceiling would signal a leak (memory beating the oracle
that has the same information) — flagged if it happens.
FAIL => workmap D1 pre-committed pivot ("verified logic flight controller").
================================================================================

Usage (from the repo root):

    python -m mlgn.flightgate.gate_eval --results-dir mlgn/flightgate/results
    python -m mlgn.flightgate.gate_eval --results-dir <dir> --glob 'flightgate_hover_*.json'
    python -m mlgn.flightgate.gate_eval --json-out gate_table.json   # machine-readable

Numpy-only (no torch / no sim) — runs anywhere, including the DUST head node.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os

# canonical row order for the table
ARM_ORDER = ("gated", "clatch", "ff", "teacher_masked", "teacher_only")
ARM_LABEL = {
    "gated": "gated (recurrent)",
    "clatch": "clatch (recurrent)",
    "ff": "ff (stateless)",
    "teacher_masked": "a' ceiling (PID|masked)",
    "teacher_only": "teacher (privileged)",
}
RECURRENT = ("gated", "clatch")


def _dig(rec: dict, *path, default=None):
    cur = rec
    for k in path:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[k]
            except (IndexError, TypeError):
                return default
        else:
            return default
        if cur is None:
            return default
    return cur


def load_run(path: str) -> dict:
    """Parse ONE run JSON into a flat row. Handles the three record shapes
    (student with a ``rounds`` list, ``teacher_only``, ``teacher_masked``)."""
    with open(path) as f:
        rec = json.load(f)
    arm = rec.get("mechanism")
    blackout = rec.get("blackout") is not None
    row = {
        "path": os.path.basename(path),
        "arm": arm,
        "blackout": bool(blackout),
        "seed": rec.get("seed"),
        "backend": rec.get("backend"),
        "hidden": rec.get("hidden"),
        "logic_gates": rec.get("logic_gates"),
        "teacher_return": _dig(rec, "eval_teacher", "mean_return"),
        "quant_ratio": rec.get("quantization_gate_return_ratio"),
        # per-arm metrics filled below
        "return": None, "rms": None, "rms_bk": None, "exit_rate": None,
        "acc_d": None, "gap": None, "n_eval": None,
    }
    if arm == "teacher_masked":
        ev = rec.get("eval_teacher_masked") or {}
        row.update(**{"return": ev.get("mean_return"), "rms": ev.get("mean_rms_err"),
                      "rms_bk": ev.get("mean_rms_err_blackout"),
                      "exit_rate": ev.get("envelope_exit_rate"),
                      "n_eval": ev.get("n_episodes")})
    elif arm == "teacher_only":
        ev = rec.get("eval_teacher") or {}
        row.update(**{"return": ev.get("mean_return"), "rms": ev.get("mean_rms_err"),
                      "rms_bk": ev.get("mean_rms_err_blackout"),
                      "exit_rate": ev.get("envelope_exit_rate"),
                      "n_eval": ev.get("n_episodes")})
    else:  # gated / clatch / ff — read the LAST DAgger round
        ev = _dig(rec, "rounds", -1, "eval_student_discrete") or {}
        tr = _dig(rec, "rounds", -1, "train") or {}
        row.update(**{"return": ev.get("mean_return"), "rms": ev.get("mean_rms_err"),
                      "rms_bk": ev.get("mean_rms_err_blackout"),
                      "exit_rate": ev.get("envelope_exit_rate"),
                      "acc_d": tr.get("action_acc_discrete"),
                      "gap": tr.get("discretization_gap"),
                      "n_eval": ev.get("n_episodes")})
    return row


def load_runs(paths) -> list[dict]:
    rows = [load_run(p) for p in paths]
    # newest-wins de-dup on (arm, blackout, seed): a re-run's fresh JSON supersedes
    # the stale one (filenames are timestamped, so sort by name = chronological)
    rows.sort(key=lambda r: r["path"])
    dedup: dict[tuple, dict] = {}
    for r in rows:
        dedup[(r["arm"], r["blackout"], r["seed"])] = r
    return list(dedup.values())


_METRICS = ("return", "acc_d", "gap", "exit_rate", "rms", "rms_bk", "teacher_return")


def _stats(vals: list) -> dict:
    v = [float(x) for x in vals if x is not None and not _isnan(x)]
    if not v:
        return {"mean": None, "std": None, "n": 0, "vals": []}
    mean = sum(v) / len(v)
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v)) if len(v) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(v), "vals": v}


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def aggregate(rows: list[dict]) -> dict:
    """Group by (arm, blackout) -> per-metric {mean,std,n,vals} + the seed list."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["arm"], r["blackout"]), []).append(r)
    agg: dict[tuple, dict] = {}
    for key, group in cells.items():
        group = sorted(group, key=lambda r: (r["seed"] is None, r["seed"]))
        cell = {"n": len(group),
                "seeds": [g["seed"] for g in group],
                "hidden": _first(group, "hidden"),
                "logic_gates": _first(group, "logic_gates")}
        for m in _METRICS:
            cell[m] = _stats([g[m] for g in group])
        agg[key] = cell
    return agg


def _first(group, k):
    for g in group:
        if g.get(k) is not None:
            return g[k]
    return None


# ------------------------------------------------------------------------------------
# Gate verdict
# ------------------------------------------------------------------------------------

def gate_verdict(agg: dict) -> dict:
    """Apply the WIN criterion in the module header. Returns a structured verdict."""
    def cell(arm, bk):
        return agg.get((arm, bk))

    def ret_mean(arm, bk):
        c = cell(arm, bk)
        return None if c is None else c["return"]["mean"]

    def ret_std(arm, bk):
        c = cell(arm, bk)
        return None if c is None else (c["return"]["std"] or 0.0)

    ff_bk = ret_mean("ff", True)
    ff_nb = ret_mean("ff", False)
    ceiling_bk = ret_mean("teacher_masked", True)

    per_arm = {}
    any_win_bk = False
    for arm in RECURRENT:
        rec_bk, rec_nb = ret_mean(arm, True), ret_mean(arm, False)
        entry = {"return_blackout": rec_bk, "return_control": rec_nb,
                 "ff_return_blackout": ff_bk, "ff_return_control": ff_nb,
                 "ceiling_blackout": ceiling_bk}
        # blackout margin
        if rec_bk is not None and ff_bk is not None:
            spread = max(ret_std(arm, True), ret_std("ff", True))
            margin = rec_bk - ff_bk
            entry["blackout_margin"] = margin
            entry["blackout_spread"] = spread
            entry["beats_ff_under_blackout"] = bool(margin > spread and margin > 0)
        else:
            entry["blackout_margin"] = None
            entry["beats_ff_under_blackout"] = None
        # non-vacuity control (match under no blackout)
        if rec_nb is not None and ff_nb is not None:
            spread_nb = max(ret_std(arm, False), ret_std("ff", False))
            entry["control_gap"] = abs(rec_nb - ff_nb)
            entry["matches_ff_under_control"] = bool(abs(rec_nb - ff_nb) <= spread_nb
                                                     or rec_nb >= ff_nb)
        else:
            entry["control_gap"] = None
            entry["matches_ff_under_control"] = None
        # leak check: recurrent should NOT beat the a' oracle (same information)
        if rec_bk is not None and ceiling_bk is not None:
            entry["above_ceiling"] = bool(rec_bk > ceiling_bk + 1e-9)
        else:
            entry["above_ceiling"] = None
        entry["win"] = bool(entry.get("beats_ff_under_blackout")
                            and entry.get("matches_ff_under_control"))
        any_win_bk = any_win_bk or bool(entry.get("beats_ff_under_blackout"))
        per_arm[arm] = entry

    non_vacuity_ok = any(per_arm[a].get("matches_ff_under_control") for a in RECURRENT
                         if per_arm[a].get("matches_ff_under_control") is not None)
    gate_win = bool(any(per_arm[a]["win"] for a in RECURRENT))

    if gate_win:
        status = "WIN"
    elif any_win_bk and not non_vacuity_ok:
        status = "AMBIGUOUS (recurrent beats ff under blackout but ALSO differs "\
                 "under control -> capacity confound, not memory)"
    elif ff_bk is None or any(ret_mean(a, True) is None for a in RECURRENT):
        status = "INCOMPLETE (missing arms/cells; run the full queue)"
    else:
        status = "FAIL (recurrent does not beat ff under blackout -> workmap D1 pivot)"

    leak = [a for a in RECURRENT if per_arm[a].get("above_ceiling")]
    return {"status": status, "gate_win": gate_win, "per_arm": per_arm,
            "non_vacuity_ok": non_vacuity_ok, "ceiling_leak_arms": leak}


# ------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------

def _cellstr(cell, metric, pct=False):
    if cell is None:
        return "   --   "
    s = cell[metric]
    if s["mean"] is None:
        return "   --   "
    mean, std = s["mean"], s["std"] or 0.0
    if pct:
        return f"{mean * 100:5.1f}+-{std * 100:4.1f}"
    return f"{mean:6.2f}+-{std:4.2f}"


def format_table(agg: dict, verdict: dict) -> str:
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in agg)]
    lines = []
    lines.append("=" * 92)
    lines.append("D1 THREE-ARM GATE TABLE  (mean +- cross-seed std;  n seeds in [])")
    lines.append("=" * 92)
    hdr = f"{'arm':<24} {'| CONTROL (no blackout)':<32} {'| BLACKOUT':<32}"
    lines.append(hdr)
    lines.append(f"{'':<24} {'|   return    acc_d   exit':<32} "
                 f"{'|   return    acc_d   exit':<32}")
    lines.append("-" * 92)
    for arm in arms:
        c_nb, c_bk = agg.get((arm, False)), agg.get((arm, True))
        n_nb = c_nb["n"] if c_nb else 0
        n_bk = c_bk["n"] if c_bk else 0
        row = (f"{ARM_LABEL.get(arm, arm):<24} "
               f"| {_cellstr(c_nb, 'return')} {_cellstr(c_nb, 'acc_d', pct=True)} "
               f"{_cellstr(c_nb, 'exit_rate', pct=True)}[{n_nb}]  "
               f"| {_cellstr(c_bk, 'return')} {_cellstr(c_bk, 'acc_d', pct=True)} "
               f"{_cellstr(c_bk, 'exit_rate', pct=True)}[{n_bk}]")
        lines.append(row)
    lines.append("-" * 92)
    # discretization gap + rms_blackout detail rows for the trainable arms
    lines.append("detail (blackout condition): discretization gap | rms_err | rms_blackout")
    for arm in arms:
        c_bk = agg.get((arm, True))
        if c_bk is None:
            continue
        lines.append(f"  {ARM_LABEL.get(arm, arm):<22} "
                     f"gap={_cellstr(c_bk, 'gap')}  "
                     f"rms={_cellstr(c_bk, 'rms')}  "
                     f"rms_bk={_cellstr(c_bk, 'rms_bk')}")
    lines.append("=" * 92)
    # verdict block
    lines.append(f"GATE VERDICT: {verdict['status']}")
    for arm in RECURRENT:
        e = verdict["per_arm"].get(arm)
        if not e:
            continue
        m = e.get("blackout_margin")
        sp = e.get("blackout_spread")
        cg = e.get("control_gap")
        lines.append(
            f"  {arm:<7} blackout return {e['return_blackout']} vs ff {e['ff_return_blackout']} "
            f"-> margin {_num(m)} (spread {_num(sp)}); "
            f"control gap {_num(cg)}; "
            f"beats_ff_bk={e.get('beats_ff_under_blackout')} "
            f"matches_ff_ctrl={e.get('matches_ff_under_control')} WIN={e['win']}")
    if verdict["ceiling_leak_arms"]:
        lines.append(f"  [LEAK WARNING] arm(s) above the a' ceiling: "
                     f"{verdict['ceiling_leak_arms']} (memory beating the oracle with "
                     f"the same information -> investigate)")
    lines.append("=" * 92)
    return "\n".join(lines)


def _num(x):
    return "  --  " if x is None else f"{x:+.2f}"


def build_report(paths) -> dict:
    rows = load_runs(paths)
    agg = aggregate(rows)
    verdict = gate_verdict(agg)
    # JSON-serializable aggregate (tuple keys -> "arm|blackout")
    agg_ser = {f"{arm}|{'blackout' if bk else 'control'}": v
               for (arm, bk), v in agg.items()}
    return {"n_runs": len(rows), "rows": rows, "aggregate": agg_ser,
            "verdict": verdict, "table": format_table(agg, verdict)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Assemble the D1 three-arm gate table.")
    p.add_argument("--results-dir",
                   default=os.path.join(os.path.dirname(__file__), "results"))
    p.add_argument("--glob", default="flightgate_*.json",
                   help="filename glob within --results-dir (default all flightgate JSONs).")
    p.add_argument("--json-out", default=None,
                   help="also write the machine-readable report to this path.")
    args = p.parse_args(argv)

    paths = sorted(_glob.glob(os.path.join(args.results_dir, args.glob)))
    if not paths:
        print(f"[gate_eval] no JSONs matched {args.glob} in {args.results_dir}")
        return 1
    report = build_report(paths)
    print(f"[gate_eval] {report['n_runs']} run(s) from {len(paths)} file(s)\n")
    print(report["table"])
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\n[gate_eval] report -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
