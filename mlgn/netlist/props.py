"""
props.py — property circuits over an extracted netlist (single 'bad' output each).
==================================================================================

Every builder returns a NEW Netlist whose single output is a *bad* signal: the
property holds iff *bad* can never be 1. Conventions match ABC: ``sat`` on a
combinational miter (UNSAT = holds for ALL input assignments), ``pdr``/``bmc3`` on a
sequential circuit (proved = holds on all REACHABLE states from the latch init).

Properties (for the copy-task circuits; x = [symbol one-hot (alphabet) ; cue]):

1. ``comb_hold``      — ∀h: f(blank, h) = h. State bits become free PIs, x is tied to
   0, bad = OR_i (next_i XOR h_i). The strongest hold: if UNSAT, the learned gate
   layer implements "blank ⇒ keep" for EVERY state, reachable or not, and every
   sequential hold below follows immediately.
2. ``seq_hold``       — G (blank(x) → next q = q) with inputs FREE every step, from
   the all-zero reset. Reachability-aware version of (1). Note the reset state itself
   is subject to the property (a t=0 counterexample means "a blank step disturbs the
   zero state" — honest, if unexciting).
3. ``protocol_hold``  — the copy-protocol theorem. Extra latches: ``first`` (init 1,
   next 0) gates the input (x_eff = x AND first ⇒ blanks are FORCED after t=0) and
   ``legal`` captures at t=0 whether x_0 was a legal protocol input (cue=1 and
   exactly one symbol bit). bad = ¬first ∧ legal ∧ (next q ≠ q).
   PROVED ⇒ for every legal write, the state NEVER changes after the write step —
   combined with the 8 simulated write trajectories (sim.analyze_protocol showing
   decode(h_1) = symbol), this is a complete functional-correctness proof of the
   deployed circuit for ARBITRARY delay length.
4. ``protocol_hold_anyx0`` — (3) without the legality constraint: hold after ANY
   first input, including garbage the net never saw. Stricter; a free robustness
   bonus if it proves.
"""

from __future__ import annotations

from .ir import (A_AND_NOT_B, AND, CONST0, CONST1, NOT_A, NOT_A_AND_B, OR, XOR,
                 Netlist, NetlistBuilder, copy_gates_with_remap)


def _base_sigmap(net: Netlist, b: NetlistBuilder, pi_map, q_map) -> dict[int, int]:
    m = {CONST0: CONST0, CONST1: CONST1}
    for i in range(net.n_pi):
        m[net.pi(i)] = pi_map(i)
    for i in range(net.n_state):
        m[net.q(i)] = q_map(i)
    return m


def _diff_tree(b: NetlistBuilder, net: Netlist, m: dict[int, int], q_of) -> int:
    diffs = [b.add_gate(XOR, m[ns], q_of(i)) for i, ns in enumerate(net.next_state)]
    b.end_layer()
    return b.or_tree(diffs)


def comb_hold(net: Netlist) -> Netlist:
    """∀h miter: PIs are the state bits, x ≡ 0, no latches. Check with ABC ``sat``."""
    b = NetlistBuilder(n_pi=net.n_state, n_state=0)
    m = _base_sigmap(net, b, pi_map=lambda i: CONST0, q_map=lambda i: b.pi(i))
    m = copy_gates_with_remap(b, net, m)
    bad = _diff_tree(b, net, m, q_of=lambda i: b.pi(i))
    return b.build(next_state=[], outputs=[bad])


def seq_hold(net: Netlist) -> Netlist:
    """G(blank → hold) with free inputs, from the reset state. Check with pdr/bmc3."""
    b = NetlistBuilder(net.n_pi, net.n_state, init=net.init)
    m = _base_sigmap(net, b, pi_map=lambda i: b.pi(i), q_map=lambda i: b.q(i))
    m = copy_gates_with_remap(b, net, m)
    difftree = _diff_tree(b, net, m, q_of=lambda i: b.q(i))
    any_x = b.or_tree([b.pi(i) for i in range(net.n_pi)])
    blank = b.add_gate(NOT_A, any_x, CONST0)
    bad = b.add_gate(AND, blank, difftree)
    b.end_layer()
    return b.build(next_state=[m[ns] for ns in net.next_state], outputs=[bad])


def protocol_hold(net: Netlist, alphabet: int | None, settle: int = 1) -> Netlist:
    """Copy-protocol hold: write at t=0 (legality-checked iff ``alphabet`` given),
    forced blanks afterwards; bad = armed ∧ legal ∧ (state changes), where ``armed``
    becomes true ``settle`` steps after the write. pdr/bmc3.

    ``settle`` exists because the trained circuits do NOT latch instantly: the
    trajectory analysis shows a deterministic ~10-15-step convergence to the fixed
    point after the write. The honest theorem is therefore "settles within K steps,
    then holds FOREVER"; ``settle=K`` arms the checker at t=K. A one-hot token
    shifting through ``settle`` extra latches implements the arming delay (w_0 also
    serves as the ``first``/write-step flag gating the inputs)."""
    assert settle >= 1, settle
    b = NetlistBuilder(net.n_pi, net.n_state + settle + 1,
                       init=net.init + [1] + [0] * (settle - 1) + [1])
    warm = [b.q(net.n_state + i) for i in range(settle)]   # w_0 (=first) .. w_{K-1}
    first = warm[0]
    legal_q = b.q(net.n_state + settle)

    x_eff = [b.add_gate(AND, b.pi(i), first) for i in range(net.n_pi)]
    b.end_layer()
    m = _base_sigmap(net, b, pi_map=lambda i: x_eff[i], q_map=lambda i: b.q(i))
    m = copy_gates_with_remap(b, net, m)
    difftree = _diff_tree(b, net, m, q_of=lambda i: b.q(i))

    if alphabet is not None:
        sym = [b.pi(i) for i in range(alphabet)]
        cue = b.pi(alphabet)
        at_least_one = b.or_tree(sym)
        pairs = [b.add_gate(AND, sym[i], sym[j])
                 for i in range(alphabet) for j in range(i + 1, alphabet)]
        b.end_layer()
        two_plus = b.or_tree(pairs)
        one_hot = b.add_gate(A_AND_NOT_B, at_least_one, two_plus)
        legal_now = b.add_gate(AND, cue, one_hot)
        b.end_layer()
    else:
        legal_now = CONST1

    # legal latch: sample legal(x) at t=0, then hold:  legal' = MUX(first, legal_now, legal)
    t_keep = b.add_gate(AND, first, legal_now)
    t_hold = b.add_gate(NOT_A_AND_B, first, legal_q)
    b.end_layer()
    legal_next = b.add_gate(OR, t_keep, t_hold)
    b.end_layer()
    # armed at t >= settle: the shift token has left every warm-up latch
    any_warm = b.or_tree(list(warm))
    not_warm = b.add_gate(NOT_A, any_warm, CONST0)
    b.end_layer()
    armed = b.add_gate(AND, not_warm, legal_q)
    b.end_layer()
    bad = b.add_gate(AND, armed, difftree)
    b.end_layer()

    # warm-up token shifts right and falls off; legal latch holds its sample
    next_state = ([m[ns] for ns in net.next_state]
                  + [CONST0] + list(warm[:-1])
                  + [legal_next])
    return b.build(next_state=next_state, outputs=[bad])


def build_all(net: Netlist, alphabet: int, settle_legal: int = 1,
              settle_any: int | None = None) -> dict[str, tuple[Netlist, bool]]:
    """name -> (property netlist, is_combinational). ``settle_any`` (default =
    settle_legal) arms the anyx0 variant — garbage writes settle later than legal
    ones, so the two theorems have different honest K."""
    if settle_any is None:
        settle_any = settle_legal
    return {
        "comb_hold": (comb_hold(net), True),
        "seq_hold": (seq_hold(net), False),
        "protocol_hold": (protocol_hold(net, alphabet, settle_legal), False),
        "protocol_hold_anyx0": (protocol_hold(net, None, settle_any), False),
    }
