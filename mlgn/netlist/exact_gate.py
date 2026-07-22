"""
exact_gate.py — feasibility falsifier for exact function-space analysis of trained LGNs.
========================================================================================

Gate experiment for research/26 (exact Boolean-function analysis of trained LGNs).
Question: on a real trained combinational LGN ('ff' arm, CAN-syn IDS), can we compute
EXACTLY — via BDDs (dd.autoref, pure Python) —

  (i)  the whole-input-space class-1 volume  Pr_x[f(x) = attack]        (#SAT analogue)
  (ii) per-variable influences Inf_i(f) = Pr_x[f(x) != f(x ^ e_i)],
       total influence == average sensitivity                            (O'Donnell)
  (iii) exact decision-boundary size  |{(x,i): f(x) != f(x^e_i)}| = 2^n * sum_i Inf_i
  (iv) exact disagreement mass between two independently trained models,
       Pr_x[f_A(x) != f_B(x)]  (both compiled into one shared BDD manager)

PASS = all quantities computed within budget on the h64 model and the machinery
either survives or fails *informatively* on the h512 model. Every stage is wall-clock
budgeted; a blow-up is recorded in the report, not raised.

The decision function is the full pipeline: 120 input bits -> gate layers -> GroupSum
head -> argmax. With k=2 classes and np.argmax tie-breaking (class 0 wins ties),
f(x) = [popcount(group1 bits) > popcount(group0 bits)], built symbolically by a
partial-sum DP over the +-1-weighted hidden bits.

Exactness is cross-checked against Monte-Carlo estimates from the bit-exact netlist
simulator (ff_forward), which is itself equivalence-gated against the torch model.

Usage (repo root):
    python -m mlgn.netlist.exact_gate \
        --run TAG_A path/to/results_A.json path/to/ckpt_A.pt \
        --run TAG_B path/to/results_B.json path/to/ckpt_B.pt \
        --out mlgn/netlist/out/exact_gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # dd.cudd (C, ~100x faster, dynamic sifting) if built — Linux/DUST; else pure Python
    from dd.cudd import BDD  # noqa: E402
    DD_ENGINE = "cudd"
except ImportError:
    from dd.autoref import BDD  # noqa: E402
    DD_ENGINE = "autoref"

from mlgn.netlist.extract import (  # noqa: E402
    build_task, extract_netlist_ff, ff_equivalence_check, ff_forward,
    rebuild_model, spec_from_json,
)

sys.setrecursionlimit(200_000)


class Budget:
    """Wall-clock budget shared across the stages of one model's analysis."""

    def __init__(self, seconds: float):
        self.t0 = time.perf_counter()
        self.seconds = seconds

    def spent(self) -> float:
        return time.perf_counter() - self.t0

    def check(self, stage: str) -> None:
        if self.spent() > self.seconds:
            raise TimeoutError(f"budget exhausted during {stage} "
                               f"({self.spent():.0f}s > {self.seconds:.0f}s)")


def gate_bdd(bdd: BDD, op: int, a, b):
    """difflogic op id -> BDD function (op indexing as in ir.GATE_FN)."""
    if op == 0:
        return bdd.false
    if op == 1:
        return a & b
    if op == 2:
        return a & ~b
    if op == 3:
        return a
    if op == 4:
        return ~a & b
    if op == 5:
        return b
    if op == 6:
        return bdd.apply("xor", a, b)
    if op == 7:
        return a | b
    if op == 8:
        return ~(a | b)
    if op == 9:
        return ~bdd.apply("xor", a, b)
    if op == 10:
        return ~b
    if op == 11:
        return a | ~b
    if op == 12:
        return ~a
    if op == 13:
        return ~a | b
    if op == 14:
        return ~(a & b)
    if op == 15:
        return bdd.true
    raise ValueError(op)


def output_supports(net) -> list[set[int]]:
    """PI-index support set of every output bit (walk the gate DAG)."""
    sup: dict[int, set[int]] = {0: set(), 1: set()}
    for i in range(net.n_pi):
        sup[net.pi(i)] = {i}
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        for g in range(s, e):
            sup[base + g] = sup[int(net.src_a[g])] | sup[int(net.src_b[g])]
    return [sup[o] for o in net.outputs]


def greedy_bit_order(weighted: list[tuple[int, int, set[int]]]) -> list[tuple[int, int, set[int]]]:
    """Order (out_idx, weight, support) triples for support locality: repeatedly pick
    the bit introducing the fewest new variables (tie: most overlap with covered)."""
    remaining = list(weighted)
    covered: set[int] = set()
    ordered = []
    while remaining:
        best = min(remaining,
                   key=lambda t: (len(t[2] - covered), -len(t[2] & covered), len(t[2])))
        remaining.remove(best)
        covered |= best[2]
        ordered.append(best)
    return ordered


def declare_support_order(bdd: BDD, ordered_bits, n: int) -> None:
    """Declare x-vars in first-use order along the greedy bit sequence."""
    seen: set[int] = set()
    for (_, _, sup) in ordered_bits:
        for i in sorted(sup):
            if i not in seen:
                seen.add(i)
                bdd.declare(f"x{i}")
    for i in range(n):
        if i not in seen:
            bdd.declare(f"x{i}")


def _check(bdd: BDD, budget: Budget, stats: dict, node_cap: int, where: str) -> None:
    stats["nodes_peak"] = max(stats.get("nodes_peak", 0), len(bdd))
    if len(bdd) > node_cap:
        raise MemoryError(f"node cap {node_cap} exceeded at {where} ({len(bdd)} nodes)")
    budget.check(where)


def _ripple_add(bdd: BDD, a: list, b: list):
    """Symbolic unsigned add of two LSB-first BDD bit-vectors."""
    k = max(len(a), len(b))
    a = a + [bdd.false] * (k - len(a))
    b = b + [bdd.false] * (k - len(b))
    out, carry = [], bdd.false
    for i in range(k):
        s = bdd.apply("xor", bdd.apply("xor", a[i], b[i]), carry)
        carry = (a[i] & b[i]) | (carry & bdd.apply("xor", a[i], b[i]))
        out.append(s)
    out.append(carry)
    return out


def _popcount_tree(bdd: BDD, bits: list, budget: Budget, stats: dict,
                   node_cap: int, label: str) -> list:
    """Balanced adder tree over support-locality-ordered bits -> LSB-first sum vector.
    Adjacent merges keep intermediate sums on localized variable clusters."""
    level = [[u] for u in bits]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_ripple_add(bdd, level[i], level[i + 1]))
            _check(bdd, budget, stats, node_cap,
                   f"{label} adder tree, {len(level)} nodes at this level")
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def _unsigned_gt(bdd: BDD, a: list, b: list):
    """BDD for [a > b], a/b LSB-first bit-vectors."""
    k = max(len(a), len(b))
    a = a + [bdd.false] * (k - len(a))
    b = b + [bdd.false] * (k - len(b))
    gt, eq = bdd.false, bdd.true
    for i in reversed(range(k)):
        gt = gt | (eq & a[i] & ~b[i])
        eq = eq & ~bdd.apply("xor", a[i], b[i])
    return gt


def build_decision_bdd(bdd: BDD, net, budget: Budget, stats: dict, node_cap: int,
                       head_mode: str = "tree"):
    """Compile the ff netlist's argmax decision into a single BDD over x0..x{n-1}.

    Returns the BDD for f(x) = [class 1 wins]. Two-class heads only (CAN IDS).
    The head popcount comparator is the blow-up risk (measured: 216 -> 89M nodes with
    natural order at h=64; flat DP stalls at h=512 even under CUDD), so: greedy
    support-local bit order, then either a balanced adder tree + comparator
    (head_mode='tree', default) or the flat partial-sum DP with sign-decided state
    pruning (head_mode='dp'), under a wall budget and a hard node cap."""
    n = net.n_pi
    sig = {0: bdd.false, 1: bdd.true}
    for i in range(n):
        sig[net.pi(i)] = bdd.var(f"x{i}")

    t = time.perf_counter()
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        for g in range(s, e):
            sig[base + g] = gate_bdd(bdd, int(net.ops[g]),
                                     sig[int(net.src_a[g])], sig[int(net.src_b[g])])
        budget.check(f"gate layer ending at {e}")
    stats["gates_s"] = round(time.perf_counter() - t, 2)
    stats["nodes_after_gates"] = len(bdd)

    k, gs = net.head
    assert k == 2, f"decision compilation implemented for 2 classes, got {k}"
    sups = output_supports(net)

    t = time.perf_counter()
    if head_mode == "tree":
        # per-group support-local order, then popcount adder trees + comparator
        g1 = greedy_bit_order([(gs + i, +1, sups[gs + i]) for i in range(gs)])
        g0 = greedy_bit_order([(i, -1, sups[i]) for i in range(gs)])
        s1 = _popcount_tree(bdd, [sig[net.outputs[oi]] for (oi, _, _) in g1],
                            budget, stats, node_cap, "group1")
        s0 = _popcount_tree(bdd, [sig[net.outputs[oi]] for (oi, _, _) in g0],
                            budget, stats, node_cap, "group0")
        f = _unsigned_gt(bdd, s1, s0)
    else:
        # flat partial-sum DP over the +-1-weighted hidden bits: dp[d] = "diff == d".
        # States whose final sign is already decided leave the DP immediately:
        #   d - neg_left  > 0  -> always class 1: accumulate into f, drop
        #   d + pos_left <= 0  -> never  class 1: drop
        ordered = greedy_bit_order([(gs + i, +1, sups[gs + i]) for i in range(gs)]
                                   + [(i, -1, sups[i]) for i in range(gs)])
        pos_left = sum(1 for (_, w, _) in ordered if w > 0)
        neg_left = len(ordered) - pos_left
        dp = {0: bdd.true}
        f = bdd.false
        for step, (oi, w, _) in enumerate(ordered):
            u = sig[net.outputs[oi]]
            if w > 0:
                pos_left -= 1
            else:
                neg_left -= 1
            nxt: dict[int, object] = {}
            for d, cond in dp.items():
                for dn, br in ((d + w, cond & u), (d, cond & ~u)):
                    if br == bdd.false:
                        continue
                    if dn - neg_left > 0:
                        f = f | br
                    elif dn + pos_left > 0:
                        nxt[dn] = (nxt[dn] | br) if dn in nxt else br
            dp = nxt
            if DD_ENGINE == "autoref":
                bdd.collect_garbage()  # cudd GCs (and reorders) on its own
            _check(bdd, budget, stats, node_cap, f"head DP bit {step + 1}/{len(ordered)}")
    stats["head_s"] = round(time.perf_counter() - t, 2)
    stats["nodes_after_head"] = len(bdd)
    stats["decision_bdd_nodes"] = f.dag_size
    return f


def exact_quantities(bdd: BDD, f, n: int, budget: Budget) -> dict:
    """Volume, per-variable influences, average sensitivity, boundary size — all exact."""
    t = time.perf_counter()
    vol = bdd.count(f, nvars=n) / (2 ** n)
    vol_s = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    infl = []
    for i in range(n):
        f0 = bdd.let({f"x{i}": False}, f)
        f1 = bdd.let({f"x{i}": True}, f)
        g = bdd.apply("xor", f0, f1)
        infl.append(bdd.count(g, nvars=n) / (2 ** n))
        if i % 20 == 19:
            budget.check(f"influence {i + 1}/{n}")
    infl_s = round(time.perf_counter() - t, 2)

    total = float(sum(infl))
    return {
        "class1_volume_exact": vol,
        "volume_s": vol_s,
        "influences": [float(v) for v in infl],
        "influences_s": infl_s,
        "avg_sensitivity_exact": total,          # == total influence
        "boundary_edges_log2": (float(np.log2(total) + n) if total > 0 else None),
        "n_influential_vars": int(sum(v > 0 for v in infl)),
    }


def mc_estimates(net, n: int, samples: int, seed: int = 0) -> dict:
    """Monte-Carlo volume + sensitivity from the bit-exact simulator (cross-check)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(samples, n), dtype=np.uint8).astype(bool)
    preds, _ = ff_forward(net, x)
    # sensitivity: for each sample flip one uniformly random coordinate
    flip = rng.integers(0, n, size=samples)
    xf = x.copy()
    xf[np.arange(samples), flip] ^= True
    preds_f, _ = ff_forward(net, xf)
    return {
        "samples": samples,
        "class1_volume_mc": float((preds == 1).mean()),
        "avg_sensitivity_mc": float(n * (preds != preds_f).mean()),
    }


def analyze_run(tag: str, json_path: str, ckpt_path: str, budget_s: float,
                equiv_batches: int, mc_samples: int, node_cap: int,
                head_mode: str) -> tuple[dict, BDD | None, object]:
    rep: dict = {"tag": tag, "json": json_path, "ckpt": ckpt_path, "head_mode": head_mode}
    spec = spec_from_json(json_path)
    model = rebuild_model(spec, ckpt_path)
    net = extract_netlist_ff(model)
    rep["n_pi"], rep["n_gates"], rep["head"] = net.n_pi, net.n_gates, net.head

    # bit-exactness gate: netlist vs torch on real test batches
    task = build_task(spec)
    eq = ff_equivalence_check(model, net, task.test_loader, max_batches=equiv_batches)
    rep["equivalence"] = eq
    if not eq["bit_exact"]:
        rep["status"] = "FAIL: netlist not bit-exact vs torch model"
        return rep, None, None

    # own manager per run: vars declared in this net's support-local order
    bdd = BDD()
    if DD_ENGINE == "cudd":
        bdd.configure(reordering=True)  # dynamic sifting on top of the static order
    k, gs = net.head
    sups = output_supports(net)
    weighted = ([(gs + i, +1, sups[gs + i]) for i in range(gs)]
                + [(i, -1, sups[i]) for i in range(gs)])
    declare_support_order(bdd, greedy_bit_order(weighted), net.n_pi)

    budget = Budget(budget_s)
    stats: dict = {}
    try:
        f = build_decision_bdd(bdd, net, budget, stats, node_cap, head_mode)
        rep["bdd_build"] = stats
        rep.update(exact_quantities(bdd, f, net.n_pi, budget))
        rep["mc_check"] = mc_estimates(net, net.n_pi, mc_samples)
        rep["total_s"] = round(budget.spent(), 2)
        rep["status"] = "PASS"
        return rep, bdd, f
    except (TimeoutError, MemoryError) as e:
        rep["bdd_build"] = stats
        rep["status"] = f"BLOWUP: {e}"
        rep["nodes_at_failure"] = len(bdd)
        rep["total_s"] = round(budget.spent(), 2)
        return rep, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", nargs=3, action="append", metavar=("TAG", "JSON", "CKPT"),
                    required=True, help="a trained ff run to analyze (repeatable)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "exact_gate"))
    ap.add_argument("--budget-s", type=float, default=1800.0, help="per-model wall budget")
    ap.add_argument("--equiv-batches", type=int, default=5)
    ap.add_argument("--mc-samples", type=int, default=20_000)
    ap.add_argument("--node-cap", type=int, default=8_000_000,
                    help="abort a model's BDD build past this many live nodes")
    ap.add_argument("--head", choices=("tree", "dp"), default="tree",
                    help="head compilation: balanced adder tree (default) or flat DP")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[exact_gate] BDD engine: {DD_ENGINE}", flush=True)
    n = spec_from_json(args.run[0][1]).input_dim

    reports, passed = [], {}
    for tag, jp, cp in args.run:
        print(f"[exact_gate] analyzing {tag} ...", flush=True)
        rep, mgr, f = analyze_run(tag, jp, cp, args.budget_s,
                                  args.equiv_batches, args.mc_samples, args.node_cap,
                                  args.head)
        reports.append(rep)
        if f is not None:
            passed[tag] = (mgr, f)
        print(f"[exact_gate] {tag}: {rep['status']} ({rep.get('total_s', '?')}s)", flush=True)

    # exact disagreement mass between passing models: copy one decision BDD into the
    # other's manager (dd translates between orders), then count the XOR
    pairwise = []
    tags = list(passed)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            mgr_i, f_i = passed[tags[i]]
            mgr_j, f_j = passed[tags[j]]
            entry: dict = {"pair": [tags[i], tags[j]]}
            try:
                fj_in_i = mgr_j.copy(f_j, mgr_i)
                d = mgr_i.apply("xor", f_i, fj_in_i)
                entry["disagreement_mass_exact"] = mgr_i.count(d, nvars=n) / (2 ** n)
            except (MemoryError, RecursionError) as e:
                entry["status"] = f"BLOWUP during transfer: {e}"
            pairwise.append(entry)

    out = {"engine": DD_ENGINE, "runs": reports, "pairwise_disagreement": pairwise}
    path = os.path.join(args.out, "report.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[exact_gate] report -> {path}")


if __name__ == "__main__":
    main()
