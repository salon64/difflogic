"""
falsify.py — the P3a one-afternoon falsifier, end to end.
=========================================================

checkpoint → (RNG-replay rebuild) → accuracy gate → netlist → bit-exact equivalence
→ protocol trajectory analysis → property BLIFs → ABC (sat / pdr / bmc3) → verdicts.

Run from the repo root, e.g.:

    python -m mlgn.netlist.falsify \
        --ckpt mlgn/seqlgn/results/ckpt_cp50A_curr_c35.pt \
        --json mlgn/seqlgn/results/copy_combo_cp50A_curr_c35_20260704-023800.json

Requires ABC in WSL at ~/abc/abc (override with --abc-path / --no-wsl for a native
binary). Everything before ABC is pure Python — use --skip-abc to run just the
export + equivalence stage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.netlist import blif, props, sim  # noqa: E402
from mlgn.netlist.extract import (check_accuracy, build_task, rebuild_model,  # noqa: E402
                                  spec_from_json)
from mlgn.netlist.ir import extract_netlist  # noqa: E402


def win_to_wsl(p: str) -> str:
    p = os.path.abspath(p)
    assert p[1] == ":", p
    return f"/mnt/{p[0].lower()}{p[2:]}".replace("\\", "/")


def run_abc(script: str, workdir: str, abc_path: str, use_wsl: bool, timeout: int) -> tuple[str, float]:
    if use_wsl:
        inner = f'cd {win_to_wsl(workdir)} && {abc_path} -c "{script}"'
        cmd = ["wsl", "-e", "bash", "-lc", inner]
    else:
        cmd = [abc_path, "-c", script]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout + 180, cwd=workdir)
        log = res.stdout + res.stderr
    except subprocess.TimeoutExpired as e:
        log = ((e.stdout or "") + (e.stderr or "")
               if isinstance(e.stdout, str) else "") + "\n[falsify] hard subprocess timeout"
    return log, time.time() - t0


def parse_verdict(log: str, comb: bool) -> dict:
    v = {"verdict": "UNKNOWN", "detail": ""}
    stats = re.search(r"i/o\s*=\s*(\d+)/\s*(\d+)\s+lat\s*=\s*(\d+)\s+and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)", log)
    if stats:
        v["aig"] = {"pi": int(stats.group(1)), "po": int(stats.group(2)),
                    "latches": int(stats.group(3)), "ands": int(stats.group(4)),
                    "levels": int(stats.group(5))}
    if comb:
        if "UNSATISFIABLE" in log:
            v["verdict"] = "PROVED"
            v["detail"] = "UNSAT: holds for ALL states (reachable or not)"
        elif "SATISFIABLE" in log:
            v["verdict"] = "CEX"
            v["detail"] = "SAT: some state exists where a blank step rewrites bits"
        elif "UNDECIDED" in log:
            v["verdict"] = "UNDECIDED"
        return v
    m = re.search(r"asserted in frame\s+(\d+)", log)
    if "Property proved" in log:
        v["verdict"] = "PROVED"
        v["detail"] = "invariant holds on all reachable states"
    elif m:
        v["verdict"] = "CEX"
        v["detail"] = f"counterexample at frame {m.group(1)}"
    elif re.search(r"No output (?:failed|asserted)", log) or "none of the outputs" in log.lower():
        v["verdict"] = "BOUNDED_CLEAN"
        fr = re.search(r"(\d+)\s+frames", log)
        v["detail"] = f"no cex within explored bound{' (' + fr.group(1) + ' frames)' if fr else ''}"
    elif "imeout" in log or "esource limit" in log:
        v["verdict"] = "UNDECIDED"
        v["detail"] = "engine hit its limit"
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a seqlgn checkpoint to a netlist and model-check hold properties.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", required=True, help="the run's results JSON (spec + recorded test_acc)")
    ap.add_argument("--alphabet", type=int, default=8, help="NOT recorded in the JSON; from run_queue.sh")
    ap.add_argument("--out", default=None, help="output dir (default mlgn/netlist/out/<ckpt-stem>)")
    ap.add_argument("--eval-batches", type=int, default=None, help="accuracy gate on N batches only (default: full test set)")
    ap.add_argument("--equiv-batches", type=int, default=10, help="equivalence-check batches (default 10; 0 = skip)")
    ap.add_argument("--props", default="comb_hold,seq_hold,protocol_hold,protocol_hold_anyx0")
    ap.add_argument("--settle", default="auto",
                    help="arming delay K for the protocol properties: the theorem is "
                         "'settles within K steps, then holds forever'. 'auto' = max "
                         "observed settle step + 1 from the trajectory analysis")
    ap.add_argument("--timeout", type=int, default=600, help="per ABC engine call, seconds")
    ap.add_argument("--abc-path", default="~/abc/abc")
    ap.add_argument("--no-wsl", action="store_true", help="abc-path is a native binary, not inside WSL")
    ap.add_argument("--skip-abc", action="store_true")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    outdir = args.out or os.path.join(os.path.dirname(__file__), "out", stem)
    os.makedirs(outdir, exist_ok=True)
    report: dict = {"ckpt": args.ckpt, "json": args.json}

    # 1. rebuild -------------------------------------------------------------------
    spec = spec_from_json(args.json, alphabet=args.alphabet)
    report["spec"] = spec.__dict__.copy()
    print(f"[1/6] rebuild: {spec.task}/{spec.mechanism} L={spec.seq_len} hidden={spec.hidden} "
          f"seed={spec.seed} (recorded test_acc={spec.test_acc})")
    model = rebuild_model(spec, args.ckpt)
    task = build_task(spec)

    # 2. accuracy gate --------------------------------------------------------------
    t0 = time.time()
    gate = check_accuracy(model, spec, task=task, max_batches=args.eval_batches)
    gate["seconds"] = round(time.time() - t0, 1)
    report["accuracy_gate"] = gate
    print(f"[2/6] accuracy gate: rebuilt={gate['rebuilt_test_acc']:.4f} "
          f"recorded={gate['recorded_test_acc']} full_set={gate['full_test_set']} "
          f"({gate['seconds']}s)")
    if not gate["gate_passed"]:
        print("[FAIL] rebuilt accuracy does not reproduce the recorded test_acc — the RNG "
              "replay did not reconstruct the training-time wiring (torch version drift?). "
              "Fallback: dump LogicLayer indices on the training host and load them here.")
        _write_report(outdir, report)
        return 2

    # 3. netlist + equivalence ------------------------------------------------------
    net = extract_netlist(model)
    report["netlist"] = {"pis": net.n_pi, "latches": net.n_state, "gates": int(net.n_gates)}
    print(f"[3/6] netlist: {net.n_pi} PIs, {net.n_state} latches, {net.n_gates} gates")
    if args.equiv_batches:
        eq = sim.equivalence_check(model, net, task.test_loader, max_batches=args.equiv_batches)
        report["equivalence"] = eq
        print(f"      equivalence vs torch: bit_exact={eq['bit_exact']} "
              f"(samples={eq['samples']}, pred_mismatch={eq['mismatched_predictions']}, "
              f"traj_bits_mismatch={eq['trajectory_bits_mismatched']}/{eq['trajectory_bits_checked']})")
        if not eq["bit_exact"]:
            print("[FAIL] netlist does not match the torch model bit-for-bit.")
            _write_report(outdir, report)
            return 3

    # 4. protocol trajectory analysis ------------------------------------------------
    settle_legal = settle_any = 1
    if spec.task in ("copy", "distcopy", "selcopy"):
        traj = sim.analyze_protocol(net, spec.alphabet)
        report["protocol_analysis"] = traj
        settled = sum(t["settled"] for t in traj)
        correct = sum(t["decode_correct_at_settle"] for t in traj)
        depths = sorted({t["settle_step"] for t in traj if t["settled"]})
        print(f"[4/6] protocol trajectories: {settled}/{len(traj)} symbols reach a fixed point "
              f"(settle steps {depths}), {correct}/{len(traj)} decode correctly there")
        for t in traj:
            if not (t["settled"] and t["decode_correct_at_settle"]):
                print(f"      symbol {t['symbol']}: settled={t['settled']} "
                      f"step={t['settle_step']} decodes={t['decode_trajectory_head']}")
        # exhaustive closed-system analysis: every possible first input (2^n_pi).
        # For cue-then-blank protocols this is a COMPLETE case analysis and doubles
        # as the proof of whatever it establishes; it also gives the honest arming
        # delays for the two protocol properties.
        ex = sim.exhaustive_x0(net, spec.alphabet)
        report["exhaustive_x0"] = ex
        print(f"      exhaustive x0 (all {2 ** net.n_pi}): "
              f"legal settled {ex['legal']['settled']}/{ex['legal']['n']} "
              f"(max {ex['legal']['max_settle']}, decode ok {ex['legal_decode_correct']}), "
              f"garbage settled {ex['garbage']['settled']}/{ex['garbage']['n']} "
              f"(max {ex['garbage']['max_settle']}), cycling {ex['n_cycling']}")
        if ex["legal"]["max_settle"] is not None and ex["legal"]["unsettled"] == 0:
            settle_legal = ex["legal"]["max_settle"] + 1
        elif depths:
            settle_legal = max(depths) + 1
        ms = [m for m in (ex["legal"]["max_settle"], ex["garbage"]["max_settle"]) if m is not None]
        if ms and ex["legal"]["unsettled"] == 0 and ex["garbage"]["unsettled"] == 0:
            settle_any = max(ms) + 1
        else:
            settle_any = settle_legal
    if args.settle != "auto":
        settle_legal = settle_any = int(args.settle)
    report["settle_legal"] = settle_legal
    report["settle_any"] = settle_any
    print(f"      protocol arming delay K: legal={settle_legal} anyx0={settle_any}")

    # 5. emit property BLIFs ----------------------------------------------------------
    wanted = [p.strip() for p in args.props.split(",") if p.strip()]
    all_props = props.build_all(net, spec.alphabet, settle_legal=settle_legal,
                                settle_any=settle_any)
    paths = {}
    for name in wanted:
        pnet, comb = all_props[name]
        path = os.path.join(outdir, f"{name}.blif")
        blif.emit_blif(pnet, path, model=f"{stem}_{name}")
        paths[name] = (path, comb)
        print(f"[5/6] emitted {name}.blif  ({pnet.n_pi} PIs, {pnet.n_state} latches, "
              f"{pnet.n_gates} gates)")

    if args.skip_abc:
        _write_report(outdir, report)
        print("[skip] --skip-abc: stopping before model checking.")
        return 0

    # 6. ABC ---------------------------------------------------------------------------
    results = {}
    prop_settle = {"protocol_hold": settle_legal, "protocol_hold_anyx0": settle_any}
    for name, (path, comb) in paths.items():
        fname = os.path.basename(path)
        engines = ["sat"] if comb else ["pdr", "bmc3", "tempor_pdr"]
        results[name] = {}
        for eng in engines:
            if eng == "sat":
                script = f"read_blif {fname}; strash; print_stats; sat -C 1000000"
            elif eng == "pdr":
                script = f"read_blif {fname}; strash; print_stats; pdr -T {args.timeout}"
            elif eng == "tempor_pdr":
                # the recipe that actually closes settle-then-hold properties: unroll
                # past the transient (simulation-derived K), then inductive signal
                # correspondence collapses the settled cone; pdr finishes the residue.
                k = prop_settle.get(name, 1) + 2
                script = (f"read_blif {fname}; strash; print_stats; tempor -F {k}; "
                          f"scorr; dc2; print_stats; pdr -T {args.timeout}")
            else:  # bmc3 only hunts counterexamples (it cannot prove) — cap its budget
                script = f"read_blif {fname}; strash; print_stats; bmc3 -T {min(args.timeout, 240)}"
            log, secs = run_abc(script, outdir, args.abc_path, not args.no_wsl, args.timeout)
            with open(os.path.join(outdir, f"{name}.{eng}.log"), "w") as f:
                f.write(log)
            v = parse_verdict(log, comb)
            v["seconds"] = round(secs, 1)
            results[name][eng] = v
            print(f"[6/6] {name:22s} {eng:5s} -> {v['verdict']:14s} "
                  f"({v['seconds']}s)  {v.get('detail', '')}")
    report["abc"] = results

    _write_report(outdir, report)
    print(f"\nreport -> {os.path.join(outdir, 'report.json')}")
    return 0


def _write_report(outdir: str, report: dict) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=_default)


if __name__ == "__main__":
    sys.exit(main())
