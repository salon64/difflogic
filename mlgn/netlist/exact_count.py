"""
exact_count.py — #SAT / knowledge-compilation route for exact LGN function quantities.
======================================================================================

Escalation of exact_gate.py (research/26): the monolithic-BDD route dies on the h512
CAN model — the popcount-comparator head grows ~28x per adder-tree level under CUDD
(19k -> 546k -> OOM), a genuine ordering-independent blow-up. This module recasts the
same exact quantities as MODEL COUNTING, where cardinality structure is routine:

  volume        Pr_x[f(x) = class1]      = #SAT(circuit CNF ∧ output) / 2^n
  avg. sens.    sum_i Pr_x[f != f o e_i] = #SAT(selector-miter CNF)   / 2^n
  influences    Inf_i                    = #SAT(miter ∧ selector=i)   / 2^n

Circuits are built as plain gate netlists (ir.NetlistBuilder): the base LGN is replayed
verbatim, the GroupSum argmax head becomes an adder tree + comparator IN GATES, and the
sensitivity circuit is two copies of the decision circuit — copy B reads x XOR
onehot(selector s), so ONE count over (x, s) yields the total influence. Tseitin CNF
keeps every auxiliary variable functionally defined, so model counts project exactly
onto input assignments (no overcounting).

Counting engines:
  * pysdd (pip-installable everywhere): bottom-up SDD compilation with dynamic vtree
    minimization; counts are computed by an arbitrary-precision traversal here because
    the C library's int64 model counter overflows beyond 2^64 (n=120 -> counts ~2^118).
  * --dimacs writes plain DIMACS CNF files instead, for external exact counters
    (ganak, sharpSAT-TD, d4) on machines that have them.

Validation ladder: the decision netlist is bit-checked against the torch model's
predictions; h64 counts must reproduce the BDD-exact numbers from exact_gate.py
(volume 0.40566793796054385, avg sensitivity 6.496114226954371); h512 counts are
cross-checked by Monte-Carlo.

Usage (repo root):
    python -m mlgn.netlist.exact_count \
        --run h512 mlgn/seqlgn/results/can-syn_ff_xg_ff_h512_*.json \
                   mlgn/seqlgn/results/ckpt_xg_ff_h512.pt \
        --influences
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.netlist.ir import (  # noqa: E402
    AND, A_AND_NOT_B, CONST0, CONST1, GATE_FN, NOT_A, OR, XOR,
    Netlist, NetlistBuilder, copy_gates_with_remap,
)
from mlgn.netlist.extract import (  # noqa: E402
    build_task, extract_netlist_ff, ff_equivalence_check, ff_forward,
    rebuild_model, spec_from_json,
)

XNOR = 9
sys.setrecursionlimit(1_000_000)


# =====================================================================================
# circuit construction
# =====================================================================================

def _add_vec(b: NetlistBuilder, a: list[int], c: list[int]) -> list[int]:
    """Ripple-add two LSB-first signal vectors; every gate is its own layer slice."""
    k = max(len(a), len(c))
    a = a + [CONST0] * (k - len(a))
    c = c + [CONST0] * (k - len(c))
    out, carry = [], CONST0
    for i in range(k):
        axb = b.add_gate(XOR, a[i], c[i]); b.end_layer()
        s = b.add_gate(XOR, axb, carry); b.end_layer()
        c1 = b.add_gate(AND, a[i], c[i]); b.end_layer()
        c2 = b.add_gate(AND, carry, axb); b.end_layer()
        carry = b.add_gate(OR, c1, c2); b.end_layer()
        out.append(s)
    out.append(carry)
    return out


def _popcount(b: NetlistBuilder, bits: list[int]) -> list[int]:
    level = [[u] for u in bits]
    while len(level) > 1:
        nxt = [_add_vec(b, level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def _support_greedy(net: Netlist, idxs: list[int]) -> list[int]:
    """Order output indices for support locality (min new PIs first) — the ordering
    that collapsed the BDD head blow-up 3000x carries over to CNF/vtree structure."""
    sup: dict[int, set[int]] = {CONST0: set(), CONST1: set()}
    for i in range(net.n_pi):
        sup[net.pi(i)] = {i}
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        for g in range(s, e):
            sup[base + g] = sup[int(net.src_a[g])] | sup[int(net.src_b[g])]
    supp = {i: sup[int(net.outputs[i])] for i in idxs}
    remaining, covered, ordered = list(idxs), set(), []
    while remaining:
        best = min(remaining, key=lambda i: (len(supp[i] - covered),
                                             -len(supp[i] & covered), len(supp[i])))
        remaining.remove(best)
        covered |= supp[best]
        ordered.append(best)
    return ordered


def build_decision_netlist(net: Netlist) -> Netlist:
    """ff netlist (GroupSum head, 2 classes) -> single-output netlist for
    f(x) = [popcount(group1) > popcount(group0)] with the head realized IN GATES."""
    k, gs = net.head
    assert k == 2 and net.n_state == 0, (net.head, net.n_state)
    ord1 = _support_greedy(net, list(range(gs, 2 * gs)))
    ord0 = _support_greedy(net, list(range(gs)))
    b = NetlistBuilder(net.n_pi, 0)
    m = {CONST0: CONST0, CONST1: CONST1}
    for i in range(net.n_pi):
        m[net.pi(i)] = b.pi(i)
    m = copy_gates_with_remap(b, net, m)
    out = [m[int(s)] for s in net.outputs]

    s1 = _popcount(b, [out[i] for i in ord1])
    s0 = _popcount(b, [out[i] for i in ord0])
    kk = max(len(s1), len(s0))
    s1 = s1 + [CONST0] * (kk - len(s1))
    s0 = s0 + [CONST0] * (kk - len(s0))
    gt, eq = CONST0, CONST1
    for i in reversed(range(kk)):
        w = b.add_gate(A_AND_NOT_B, s1[i], s0[i]); b.end_layer()
        t = b.add_gate(AND, eq, w); b.end_layer()
        gt = b.add_gate(OR, gt, t); b.end_layer()
        x = b.add_gate(XNOR, s1[i], s0[i]); b.end_layer()
        eq = b.add_gate(AND, eq, x); b.end_layer()
    return b.build(next_state=[], outputs=[gt], head=None)


def build_sensitivity_netlist(dec: Netlist) -> tuple[Netlist, int]:
    """Two copies of the decision circuit; copy B reads x XOR onehot(s) for a
    ceil(log2 n)-bit selector s appended to the inputs. Output = f_A XOR f_B.
    Model count over (x, s) = 2^n * avg sensitivity (dead selector codes >= n
    flip nothing, so they contribute zero)."""
    n = dec.n_pi
    nb = max(1, (n - 1).bit_length())
    b = NetlistBuilder(n + nb, 0)

    m1 = {CONST0: CONST0, CONST1: CONST1}
    for i in range(n):
        m1[dec.pi(i)] = b.pi(i)
    m1 = copy_gates_with_remap(b, dec, m1)
    out_a = m1[int(dec.outputs[0])]

    s_sig = [b.pi(n + j) for j in range(nb)]
    ns_sig = []
    for s in s_sig:
        ns_sig.append(b.add_gate(NOT_A, s, s)); b.end_layer()
    xprime = []
    for j in range(n):
        lits = [s_sig[bit] if (j >> bit) & 1 else ns_sig[bit] for bit in range(nb)]
        eq = b.and_tree(lits)
        xprime.append(b.add_gate(XOR, b.pi(j), eq)); b.end_layer()

    m2 = {CONST0: CONST0, CONST1: CONST1}
    for i in range(n):
        m2[dec.pi(i)] = xprime[i]
    m2 = copy_gates_with_remap(b, dec, m2)
    out_b = m2[int(dec.outputs[0])]

    miter = b.add_gate(XOR, out_a, out_b); b.end_layer()
    return b.build(next_state=[], outputs=[miter], head=None), nb


def eval_outputs(net: Netlist, x: np.ndarray) -> np.ndarray:
    """Vectorized eval of a combinational netlist's outputs. x: [B, n_pi] bool."""
    B = x.shape[0]
    sig = np.empty((B, net.n_signals), dtype=bool)
    sig[:, 0] = False
    sig[:, 1] = True
    sig[:, 2:2 + net.n_pi] = x
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        a = sig[:, net.src_a[s:e]]
        bb = sig[:, net.src_b[s:e]]
        out = np.empty_like(a)
        ops = net.ops[s:e]
        for code in np.unique(ops):
            msk = ops == code
            out[:, msk] = GATE_FN[int(code)](a[:, msk], bb[:, msk])
        sig[:, base + s:base + e] = out
    return sig[:, net.outputs]


# =====================================================================================
# Tseitin CNF (every aux var functionally defined -> counts project onto inputs)
# =====================================================================================

def netlist_to_cnf(net: Netlist, assert_output: bool = True) -> tuple[int, list[list[int]]]:
    """DIMACS-style clauses; var id = signal id + 1 (c0=1, c1=2, x_i=i+3, gates after)."""
    v = lambda s: int(s) + 1  # noqa: E731
    clauses: list[list[int]] = [[-v(CONST0)], [v(CONST1)]]
    base = 2 + net.n_pi + net.n_state
    one = np.ones(1, dtype=bool)
    zero = np.zeros(1, dtype=bool)
    for g in range(net.n_gates):
        op = int(net.ops[g])
        gv = v(base + g)
        if op == 0:
            clauses.append([-gv])
            continue
        if op == 15:
            clauses.append([gv])
            continue
        av, bv = v(net.src_a[g]), v(net.src_b[g])
        for va in (0, 1):
            for vb in (0, 1):
                r = bool(GATE_FN[op](one if va else zero, one if vb else zero)[0])
                clauses.append([-av if va else av,
                                -bv if vb else bv,
                                gv if r else -gv])
    if assert_output:
        clauses.append([v(net.outputs[0])])
    return net.n_signals, clauses


def write_dimacs(path: str, nvars: int, clauses: list[list[int]]) -> None:
    with open(path, "w") as fh:
        fh.write(f"p cnf {nvars} {len(clauses)}\n")
        for c in clauses:
            fh.write(" ".join(map(str, c)) + " 0\n")


# =====================================================================================
# SDD compilation + arbitrary-precision counting
# =====================================================================================

def firstuse_var_order(net: Netlist) -> list[int]:
    """DIMACS var order by first use: each PI var lands right before the first gate
    that reads it, gates in topo order — the SDD/BDD locality that a plain
    inputs-then-gates order destroys (measured: natural order stalls, this doesn't)."""
    order, seen = [CONST0 + 1, CONST1 + 1], {CONST0, CONST1}
    base = 2 + net.n_pi + net.n_state
    for g in range(net.n_gates):
        for s in (int(net.src_a[g]), int(net.src_b[g])):
            if s not in seen and 2 <= s < 2 + net.n_pi:
                seen.add(s)
                order.append(s + 1)
        seen.add(base + g)
        order.append(base + g + 1)
    for i in range(net.n_pi):
        if 2 + i not in seen:
            order.append(2 + i + 1)
    return order


def compile_sdd(nvars: int, clauses: list[list[int]], label: str,
                minimize_every: int = 0, var_order: list[int] | None = None):
    """Bottom-up SDD compilation. Continuous auto-minimization is OFF — a full vtree
    search per conjoin over ~10^4 vars stalls for tens of minutes (measured); with
    minimize_every=N a manual minimize runs every N clauses instead."""
    from pysdd.sdd import SddManager, Vtree
    if var_order is not None:
        vtree = Vtree(var_count=nvars, var_order=var_order, vtree_type="right")
    else:
        vtree = Vtree(var_count=nvars, vtree_type="right")
    mgr = SddManager.from_vtree(vtree)
    f = mgr.true()
    t0 = time.perf_counter()
    for i, clause in enumerate(clauses):
        node = mgr.false()
        for lit in clause:
            node = node | mgr.literal(lit)
        f = f & node
        if minimize_every and (i + 1) % minimize_every == 0:
            mgr.minimize_limited()
        if (i + 1) % 500 == 0:
            print(f"[exact_count]   {label}: clause {i + 1}/{len(clauses)}, "
                  f"sdd size {f.size()}, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[exact_count]   {label}: compiled, sdd size {f.size()}, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return mgr, f


def sdd_prob(node) -> Fraction:
    """Pr over ALL manager vars (uniform) that `node` is satisfied — exact Fractions,
    iterative postorder (the C library's model counter overflows past 2^64)."""
    memo: dict[int, Fraction] = {}
    stack = [node]
    while stack:
        u = stack[-1]
        if u.id in memo:
            stack.pop()
            continue
        if u.is_true():
            memo[u.id] = Fraction(1)
            stack.pop()
        elif u.is_false():
            memo[u.id] = Fraction(0)
            stack.pop()
        elif u.is_literal():
            memo[u.id] = Fraction(1, 2)
            stack.pop()
        else:
            pending = [w for (p, s) in u.elements() for w in (p, s) if w.id not in memo]
            if pending:
                stack.extend(pending)
            else:
                memo[u.id] = sum((memo[p.id] * memo[s.id] for (p, s) in u.elements()),
                                 Fraction(0))
                stack.pop()
    return memo[node.id]


# =====================================================================================
# driver
# =====================================================================================

def analyze(tag: str, json_path: str, ckpt_path: str, args) -> dict:
    rep: dict = {"tag": tag, "json": json_path, "ckpt": ckpt_path}
    spec = spec_from_json(json_path)
    model = rebuild_model(spec, ckpt_path)
    net = extract_netlist_ff(model)
    n = net.n_pi

    # validation gate 1: base netlist bit-exact vs torch
    task = build_task(spec)
    eq = ff_equivalence_check(model, net, task.test_loader, max_batches=args.equiv_batches)
    rep["equivalence_base"] = eq["bit_exact"]

    # validation gate 2: gates-only head reproduces the argmax decision
    dec = build_decision_netlist(net)
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2, size=(4096, n), dtype=np.uint8).astype(bool)
    preds, _ = ff_forward(net, xs)
    dec_out = eval_outputs(dec, xs)[:, 0]
    rep["equivalence_decision_head"] = bool(((preds == 1) == dec_out).all())
    rep["n_pi"], rep["decision_gates"] = n, dec.n_gates
    if not (eq["bit_exact"] and rep["equivalence_decision_head"]):
        rep["status"] = "FAIL: equivalence gate"
        return rep

    # Monte-Carlo cross-checks from the simulator
    flip = rng.integers(0, n, size=len(xs))
    xf = xs.copy()
    xf[np.arange(len(xs)), flip] ^= True
    predsf, _ = ff_forward(net, xf)
    rep["mc_check"] = {
        "samples": len(xs),
        "class1_volume_mc": float((preds == 1).mean()),
        "avg_sensitivity_mc": float(n * (preds != predsf).mean()),
    }

    nv_vol, cnf_vol = netlist_to_cnf(dec, assert_output=True)
    sens_net, nb = build_sensitivity_netlist(dec)
    nv_sens, cnf_sens = netlist_to_cnf(sens_net, assert_output=True)
    rep["cnf"] = {"volume": {"vars": nv_vol, "clauses": len(cnf_vol)},
                  "sensitivity": {"vars": nv_sens, "clauses": len(cnf_sens)}}

    if args.dimacs:
        os.makedirs(args.out, exist_ok=True)
        pv = os.path.join(args.out, f"{tag}_volume.cnf")
        ps = os.path.join(args.out, f"{tag}_sensitivity.cnf")
        write_dimacs(pv, nv_vol, cnf_vol)
        write_dimacs(ps, nv_sens, cnf_sens)
        rep["dimacs"] = {"volume": pv, "sensitivity": ps,
                         "note": f"exact count / 2^{n} = volume; "
                                 f"sensitivity count / 2^{n} = avg sensitivity "
                                 f"(selector vars {n + 1}..{n + nb} inside)"}
        rep["status"] = "DIMACS_WRITTEN"
        return rep

    t0 = time.perf_counter()
    mgr_v, f_v = compile_sdd(nv_vol, cnf_vol, f"{tag} volume", args.minimize_every,
                             firstuse_var_order(dec))
    p = sdd_prob(f_v)
    count = p * Fraction(2) ** nv_vol            # exact model count
    vol = count / Fraction(2) ** n
    rep["class1_volume_exact"] = float(vol)
    rep["class1_volume_exact_frac"] = f"{vol.numerator}/{vol.denominator}"
    rep["volume_s"] = round(time.perf_counter() - t0, 2)
    del mgr_v, f_v

    t0 = time.perf_counter()
    mgr_s, f_s = compile_sdd(nv_sens, cnf_sens, f"{tag} sensitivity",
                             args.minimize_every, firstuse_var_order(sens_net))
    total = sdd_prob(f_s) * Fraction(2) ** nv_sens / Fraction(2) ** n
    rep["avg_sensitivity_exact"] = float(total)
    rep["sensitivity_s"] = round(time.perf_counter() - t0, 2)

    if args.influences:
        t0 = time.perf_counter()
        infl = []
        # selector input j is PI n+j -> signal 2+n+j -> DIMACS var 3+n+j
        infl_vars = [3 + n + j for j in range(nb)]
        for j in range(n):
            g = f_s
            for bit in range(nb):
                lit = infl_vars[bit] if (j >> bit) & 1 else -infl_vars[bit]
                g = g & mgr_s.literal(lit)
            infl.append(float(sdd_prob(g) * Fraction(2) ** nv_sens / Fraction(2) ** n))
        rep["influences"] = infl
        rep["influences_sum_check"] = float(sum(infl))
        rep["influences_s"] = round(time.perf_counter() - t0, 2)

    rep["status"] = "PASS"
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", nargs=3, action="append", metavar=("TAG", "JSON", "CKPT"),
                    required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "exact_count"))
    ap.add_argument("--equiv-batches", type=int, default=5)
    ap.add_argument("--influences", action="store_true",
                    help="also compute all per-variable influences (selector conditioning)")
    ap.add_argument("--dimacs", action="store_true",
                    help="write DIMACS CNFs for an external exact counter instead of pysdd")
    ap.add_argument("--minimize-every", type=int, default=0,
                    help="run limited vtree minimization every N clauses (0 = never)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    reports = []
    path = os.path.join(args.out, "report.json")
    for tag, jp, cp in args.run:
        print(f"[exact_count] analyzing {tag} ...", flush=True)
        rep = analyze(tag, jp, cp, args)
        reports.append(rep)
        with open(path, "w") as fh:   # incremental: a later hang cannot lose this run
            json.dump(reports, fh, indent=2)
        print(f"[exact_count] {tag}: {rep['status']}", flush=True)
    print(f"[exact_count] report -> {path}")


if __name__ == "__main__":
    main()
