"""
test_head.py — self-checks for the in-netlist GroupSum head (head.py).
=======================================================================

Run from the repo root:

    python -m mlgn.netlist.test_head

Three test banks, all comparing the netlist against numpy ground truth bit-for-bit:

1. ``popcount`` on >= 5000 random bit-vectors per width in {1, 2, 3, 5, 8, 9, 128}
   (plus the all-zeros / all-ones rows).
2. ``vec_gt`` / ``vec_ge`` — EXHAUSTIVE over all input assignments for several
   width pairs, including unequal widths (padding) and, by exhaustiveness, every
   equal-value case.
3. ``decode_ok`` with head=(8, 128) on >= 20000 random 1024-bit states with random
   one-hot shadows, PLUS engineered popcount ties (equal group scores with the
   winning index both below and above the loser) and all-zero-shadow rows.

sim.py stays untouched: this module carries its own output-signal evaluator
(``eval_netlist``), which replicates sim.step's layer loop and additionally reads
``sig[:, net.outputs]``. The integration gate (out/a1_gate) reuses it.
"""

from __future__ import annotations

import sys

import numpy as np

from .head import decode_ok, popcount, vec_ge, vec_gt
from .ir import GATE_FN, Netlist, NetlistBuilder


# -----------------------------------------------------------------------------------
# local evaluator (sim.step's layer loop + output readout; sim.py is read-only)
# -----------------------------------------------------------------------------------
def eval_netlist(net: Netlist, x: np.ndarray, state: np.ndarray | None = None):
    """One clock tick that also reads the declared outputs.

    x: [B, n_pi] bool; state: [B, n_state] bool (required iff net.n_state > 0).
    Returns (outputs [B, n_out] bool, next_state [B, n_state] bool | None)."""
    B = x.shape[0]
    sig = np.empty((B, net.n_signals), dtype=bool)
    sig[:, 0] = False
    sig[:, 1] = True
    sig[:, 2:2 + net.n_pi] = x
    if net.n_state:
        sig[:, 2 + net.n_pi:2 + net.n_pi + net.n_state] = state
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        a = sig[:, net.src_a[s:e]]
        b = sig[:, net.src_b[s:e]]
        out = np.empty_like(a)
        ops = net.ops[s:e]
        for code in np.unique(ops):
            m = ops == code
            out[:, m] = GATE_FN[int(code)](a[:, m], b[:, m])
        sig[:, base + s:base + e] = out
    outs = sig[:, net.outputs] if net.outputs else np.empty((B, 0), dtype=bool)
    nxt = sig[:, net.next_state].copy() if net.next_state else None
    return outs, nxt


def eval_batched(net: Netlist, x: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Combinational-only evaluation in memory-bounded chunks (big signal spaces)."""
    assert net.n_state == 0
    return np.concatenate([eval_netlist(net, x[i:i + chunk])[0]
                           for i in range(0, len(x), chunk)])


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    """[B, w] little-endian bool -> int64."""
    return (bits.astype(np.int64) << np.arange(bits.shape[1], dtype=np.int64)).sum(1)


# -----------------------------------------------------------------------------------
# 1. popcount
# -----------------------------------------------------------------------------------
def test_popcount(n_per_width: int = 5000) -> None:
    rng = np.random.default_rng(0)
    for w in (1, 2, 3, 5, 8, 9, 128):
        b = NetlistBuilder(n_pi=w, n_state=0)
        bits = popcount(b, [b.pi(i) for i in range(w)])
        net = b.build(next_state=[], outputs=bits)
        # varied densities so every count region is exercised; pin the extremes
        x = rng.random((n_per_width, w)) < rng.random((n_per_width, 1))
        x[0, :] = False
        x[1, :] = True
        got = bits_to_int(eval_batched(net, x))
        want = x.sum(1)
        assert (got == want).all(), \
            f"popcount w={w}: {int((got != want).sum())} mismatches"
        print(f"  popcount width {w:4d}: {n_per_width} vectors OK "
              f"(count bits={len(bits)}, gates={net.n_gates})")


# -----------------------------------------------------------------------------------
# 2. comparators
# -----------------------------------------------------------------------------------
def test_comparators() -> None:
    for wa, wb in ((1, 1), (3, 3), (4, 2), (2, 5), (8, 8), (1, 7), (5, 5)):
        b = NetlistBuilder(n_pi=wa + wb, n_state=0)
        av = [b.pi(i) for i in range(wa)]
        bv = [b.pi(wa + i) for i in range(wb)]
        net = b.build(next_state=[], outputs=[vec_gt(b, av, bv), vec_ge(b, av, bv)])
        # exhaustive over all 2^(wa+wb) assignments — includes EVERY equal pair
        n = 1 << (wa + wb)
        x = ((np.arange(n)[:, None] >> np.arange(wa + wb)[None, :]) & 1).astype(bool)
        aval = bits_to_int(x[:, :wa])
        bval = bits_to_int(x[:, wa:])
        out = eval_batched(net, x)
        assert (out[:, 0] == (aval > bval)).all(), f"vec_gt ({wa},{wb})"
        assert (out[:, 1] == (aval >= bval)).all(), f"vec_ge ({wa},{wb})"
        n_eq = int((aval == bval).sum())
        print(f"  vec_gt/vec_ge widths ({wa},{wb}): exhaustive {n} pairs OK "
              f"({n_eq} equal-value cases)")


# -----------------------------------------------------------------------------------
# 3. decode_ok, head = (8, 128)
# -----------------------------------------------------------------------------------
def _state_with_counts(counts, gs: int, rng) -> np.ndarray:
    """A k*gs bit row whose per-group popcounts are exactly ``counts``."""
    row = np.zeros(len(counts) * gs, dtype=bool)
    for c, cnt in enumerate(counts):
        row[c * gs + rng.permutation(gs)[:cnt]] = True
    return row


def test_decode_ok(n_random: int = 20000) -> None:
    k, gs = 8, 128
    b = NetlistBuilder(n_pi=k * gs + k, n_state=0)
    ok = decode_ok(b, (k, gs),
                   state_sigs=[b.pi(i) for i in range(k * gs)],
                   shadow_sigs=[b.pi(k * gs + i) for i in range(k)])
    net = b.build(next_state=[], outputs=[ok])
    print(f"  decode_ok netlist: {net.n_gates} gates, {len(net.layers)} layers")
    rng = np.random.default_rng(1)

    # (a) random states (per-group densities vary => argmax varies), one-hot shadows
    states = (rng.random((n_random, k, gs)) < rng.random((n_random, k, 1))
              ).reshape(n_random, k * gs)
    cls = rng.integers(0, k, size=n_random)

    # (b) engineered exact ties: for every ordered pair (win, lose) build a state
    # whose groups win/lose share the exact maximal count; argmax must pick the
    # LOWER index. Tested with shadow=winner (expect 1 via the min-index side of
    # the tie) and shadow=loser (expect 0) — i.e. winning index below AND above.
    tie_states, tie_cls = [], []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            top = int(rng.integers(40, gs + 1))
            counts = rng.integers(0, top, size=k)   # everyone else strictly below
            counts[i] = counts[j] = top
            row = _state_with_counts(counts, gs, rng)
            for shadow_c in (min(i, j), max(i, j), int(rng.integers(0, k))):
                tie_states.append(row)
                tie_cls.append(shadow_c)
    # 3-way and 8-way ties, winner = lowest tied index
    for tied in ([0, 3, 7], [2, 3, 4], list(range(k))):
        counts = rng.integers(0, 50, size=k)
        for c in tied:
            counts[c] = 50
        row = _state_with_counts(counts, gs, rng)
        for shadow_c in range(k):
            tie_states.append(row)
            tie_cls.append(shadow_c)
    states = np.concatenate([states, np.asarray(tie_states, dtype=bool)])
    cls = np.concatenate([cls, np.asarray(tie_cls)])

    n = len(states)
    shadows = np.zeros((n, k), dtype=bool)
    shadows[np.arange(n), cls] = True
    # (c) a few all-zero shadows: decode_ok must be constant 0
    n_zero = 64
    zero_rows = rng.integers(0, n, size=n_zero)
    shadows[zero_rows] = False

    got = eval_batched(net, np.concatenate([states, shadows], axis=1))[:, 0]
    scores = states.reshape(n, k, gs).sum(-1)
    want = (scores.argmax(-1) == cls) & shadows.any(1)   # first max wins, np.argmax
    assert (got == want).all(), \
        f"decode_ok: {int((got != want).sum())} mismatches of {n}"
    n_tied = int((np.sort(scores, axis=-1)[:, -1] == np.sort(scores, axis=-1)[:, -2]).sum())
    print(f"  decode_ok: {n} cases OK ({len(tie_states)} engineered tie cases, "
          f"{n_tied} rows with a tied maximum, {n_zero} all-zero shadows)")


def main() -> int:
    print("[1/3] popcount vs numpy")
    test_popcount()
    print("[2/3] vec_gt / vec_ge vs numpy (exhaustive)")
    test_comparators()
    print("[3/3] decode_ok vs np.argmax, head=(8,128)")
    test_decode_ok()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
