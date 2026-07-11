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
5. ``protocol_decode``   — (3) plus the READOUT: a one-hot shadow register samples
   the written symbol at t=0 and the GroupSum head (head.decode_ok) is built into
   the netlist. bad = armed ∧ legal ∧ ¬(argmax == shadow). PROVED ⇒ full functional
   correctness of write → settle → readout at ARBITRARY delay ≥ K, with no
   simulated-trajectory side conditions left.
6. ``distractor_hold``   — free-input robustness. Legality-checked write at t=0 as
   in (3), but afterwards every frame may carry ANY token from the distractor
   alphabet {blank} ∪ {non-cued one-hot symbols} (a filter aliases everything else
   to blank, so the input set is encoded soundly). bad = armed ∧ legal ∧ (state
   changes). PROVED ⇒ no distractor placement can move the register — a genuinely
   free-input property (exponential input space; simulation cannot close it).
7. ``distractor_decode`` — (6)'s input envelope with (5)'s readout check:
   bad = armed ∧ legal ∧ ¬decode_ok. PROVED ⇒ no distractor stream can corrupt the
   readout, ever.
"""

from __future__ import annotations

from .head import decode_ok
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
    b.end_layer()          # own layer: bad reads blank (simulator discipline)
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
        b.end_layer()      # own layer: legal_now reads one_hot (simulator discipline)
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


def protocol_decode(net: Netlist, alphabet: int, settle: int = 1) -> Netlist:
    """Copy-protocol DECODE: full functional correctness including the GroupSum
    readout as ONE model-checking query. Same frame skeleton as protocol_hold —
    legality-checked write at t=0, forced blanks afterwards, warm-up arming delay —
    plus a one-hot SHADOW register remembering WHICH symbol was written and the
    in-netlist head (head.decode_ok) comparing the argmax readout against it.

    Frame timeline (settle = K):
        t = 0       first = w_0 = 1 ⇒ x_eff = x (the write). legal' samples
                    cue ∧ one-hot(sym); shadow' samples the symbol bits x_0..x_{A-1}.
        t = 1..K-1  warm token sits in w_t; first = 0 ⇒ x_eff = blank; armed = 0.
        t ≥ K       every warm latch is 0 ⇒ armed = legal;
                    bad = armed ∧ ¬decode_ok(q_t, shadow).

    PROVED ⇒ for EVERY legal write, from step K on the readout equals the written
    symbol FOREVER — the copy-task correctness theorem at arbitrary delay ≥ K.
    pdr/bmc3 (use the tempor -F K+2; scorr; dc2; pdr recipe)."""
    assert settle >= 1, settle
    assert net.head is not None and net.head[0] == alphabet, (net.head, alphabet)
    b = NetlistBuilder(net.n_pi, net.n_state + settle + 1 + alphabet,
                       init=net.init + [1] + [0] * (settle - 1) + [1] + [0] * alphabet)
    warm = [b.q(net.n_state + i) for i in range(settle)]   # w_0 (=first) .. w_{K-1}
    first = warm[0]
    legal_q = b.q(net.n_state + settle)
    shadow = [b.q(net.n_state + settle + 1 + i) for i in range(alphabet)]

    x_eff = [b.add_gate(AND, b.pi(i), first) for i in range(net.n_pi)]
    b.end_layer()
    m = _base_sigmap(net, b, pi_map=lambda i: x_eff[i], q_map=lambda i: b.q(i))
    m = copy_gates_with_remap(b, net, m)

    # write legality: cue ∧ exactly-one symbol bit (sampled at t=0 by the legal latch)
    sym = [b.pi(i) for i in range(alphabet)]
    cue = b.pi(alphabet)
    at_least_one = b.or_tree(sym)
    pairs = [b.add_gate(AND, sym[i], sym[j])
             for i in range(alphabet) for j in range(i + 1, alphabet)]
    b.end_layer()
    two_plus = b.or_tree(pairs)
    one_hot = b.add_gate(A_AND_NOT_B, at_least_one, two_plus)
    b.end_layer()          # own layer: legal_now reads one_hot (simulator discipline)
    legal_now = b.add_gate(AND, cue, one_hot)
    b.end_layer()

    # legal latch: sample legal(x) at t=0, then hold:  legal' = MUX(first, legal_now, legal)
    t_keep = b.add_gate(AND, first, legal_now)
    t_hold = b.add_gate(NOT_A_AND_B, first, legal_q)
    b.end_layer()
    legal_next = b.add_gate(OR, t_keep, t_hold)
    b.end_layer()
    # shadow register: sample the symbol bits at t=0, then hold:
    #   shadow_i' = MUX(first, x_i, shadow_i)   (one-hot whenever legal is 1)
    s_keep = [b.add_gate(AND, first, sym[i]) for i in range(alphabet)]
    s_hold = [b.add_gate(NOT_A_AND_B, first, shadow[i]) for i in range(alphabet)]
    b.end_layer()
    shadow_next = [b.add_gate(OR, s_keep[i], s_hold[i]) for i in range(alphabet)]
    b.end_layer()

    # armed at t >= settle: the shift token has left every warm-up latch
    any_warm = b.or_tree(list(warm))
    not_warm = b.add_gate(NOT_A, any_warm, CONST0)
    b.end_layer()
    armed = b.add_gate(AND, not_warm, legal_q)
    b.end_layer()

    # the readout check on the CURRENT state latches (h_t at frame t)
    ok = decode_ok(b, net.head, [b.q(i) for i in range(net.n_state)], shadow)
    not_ok = b.add_gate(NOT_A, ok, CONST0)
    b.end_layer()
    bad = b.add_gate(AND, armed, not_ok)   # armed already includes legal
    b.end_layer()

    next_state = ([m[ns] for ns in net.next_state]
                  + [CONST0] + list(warm[:-1])
                  + [legal_next] + shadow_next)
    return b.build(next_state=next_state, outputs=[bad])


def _distractor(net: Netlist, alphabet: int, settle: int, from_start: bool,
                decode: bool) -> Netlist:
    """Shared skeleton of the distractor properties (see the public wrappers).

    Latch layout: [ original state | warm w_0..w_{K-1} | legal | shadow (decode only) ].

    Frame timeline (settle = K):
        t = 0       first = w_0 = 1 ⇒ x_eff = x: the legality-checked write
                    (legal' samples cue ∧ one-hot(sym); shadow' samples the symbol).
                    The distractor path is off (phase_gate = 0 at t=0 in BOTH modes:
                    ¬first = 0, and armed = 0 because w_0 = 1).
        t ≥ 1       the DISTRACTOR alphabet: cue is forced 0 and the symbol bits
                    pass only while at-most-one of them is set, else the whole
                    input is aliased to blank —
                        x_eff_i = (x_i ∧ first) ∨ (x_i ∧ at_most_one ∧ phase_gate)
                        cue_eff = cue ∧ first
                    This encodes the input set {blank} ∪ {non-cued one-hot tokens}
                    SOUNDLY: every allowed sequence is realizable (drive the PIs
                    with the desired token) and nothing outside the set is
                    reachable (illegal assignments alias to blank, itself allowed).
        phase_gate  from_start=True:  ¬first — distractors from t=1 onward.
                    from_start=False: armed — inputs forced blank for t=1..K-1
                    (the warm latches are reused), distractors only after arming.
        t ≥ K       armed = legal;  bad = armed ∧ (next q ≠ q over the ORIGINAL
                    net.n_state latches)          (hold variant)
                    bad = armed ∧ ¬decode_ok       (decode variant)
    """
    assert settle >= 1, settle
    assert net.n_pi == alphabet + 1, (net.n_pi, alphabet)  # copy-task input layout
    if decode:
        assert net.head is not None and net.head[0] == alphabet, (net.head, alphabet)
    n_shadow = alphabet if decode else 0
    b = NetlistBuilder(net.n_pi, net.n_state + settle + 1 + n_shadow,
                       init=net.init + [1] + [0] * (settle - 1) + [1] + [0] * n_shadow)
    warm = [b.q(net.n_state + i) for i in range(settle)]   # w_0 (=first) .. w_{K-1}
    first = warm[0]
    legal_q = b.q(net.n_state + settle)
    shadow = [b.q(net.n_state + settle + 1 + i) for i in range(n_shadow)]

    # write legality (t=0 sample) and the every-frame distractor filter share the
    # symbol-pair tree:  at_most_one = ¬(any two symbol bits set together)
    sym = [b.pi(i) for i in range(alphabet)]
    cue = b.pi(alphabet)
    at_least_one = b.or_tree(sym)
    pairs = [b.add_gate(AND, sym[i], sym[j])
             for i in range(alphabet) for j in range(i + 1, alphabet)]
    b.end_layer()
    two_plus = b.or_tree(pairs)
    at_most_one = b.add_gate(NOT_A, two_plus, CONST0)
    b.end_layer()
    one_hot = b.add_gate(AND, at_least_one, at_most_one)
    b.end_layer()
    legal_now = b.add_gate(AND, cue, one_hot)
    b.end_layer()

    # armed at t >= settle on legal runs (also the phase gate when from_start=False)
    any_warm = b.or_tree(list(warm))
    not_warm = b.add_gate(NOT_A, any_warm, CONST0)
    b.end_layer()
    armed = b.add_gate(AND, not_warm, legal_q)
    b.end_layer()
    if from_start:
        phase_gate = b.add_gate(NOT_A, first, CONST0)
        b.end_layer()
    else:
        phase_gate = armed

    # effective inputs (see the frame timeline above)
    dist_pass = b.add_gate(AND, at_most_one, phase_gate)
    b.end_layer()
    wr = [b.add_gate(AND, sym[i], first) for i in range(alphabet)]
    di = [b.add_gate(AND, sym[i], dist_pass) for i in range(alphabet)]
    cue_eff = b.add_gate(AND, cue, first)
    b.end_layer()
    x_eff = [b.add_gate(OR, wr[i], di[i]) for i in range(alphabet)] + [cue_eff]
    b.end_layer()

    m = _base_sigmap(net, b, pi_map=lambda i: x_eff[i], q_map=lambda i: b.q(i))
    m = copy_gates_with_remap(b, net, m)

    # legal latch: sample legal(x) at t=0, then hold:  legal' = MUX(first, legal_now, legal)
    t_keep = b.add_gate(AND, first, legal_now)
    t_hold = b.add_gate(NOT_A_AND_B, first, legal_q)
    b.end_layer()
    legal_next = b.add_gate(OR, t_keep, t_hold)
    b.end_layer()

    shadow_next: list[int] = []
    if decode:
        # shadow register: shadow_i' = MUX(first, x_i, shadow_i)
        s_keep = [b.add_gate(AND, first, sym[i]) for i in range(alphabet)]
        s_hold = [b.add_gate(NOT_A_AND_B, first, shadow[i]) for i in range(alphabet)]
        b.end_layer()
        shadow_next = [b.add_gate(OR, s_keep[i], s_hold[i]) for i in range(alphabet)]
        b.end_layer()
        ok = decode_ok(b, net.head, [b.q(i) for i in range(net.n_state)], shadow)
        not_ok = b.add_gate(NOT_A, ok, CONST0)
        b.end_layer()
        bad = b.add_gate(AND, armed, not_ok)   # armed already includes legal
    else:
        difftree = _diff_tree(b, net, m, q_of=lambda i: b.q(i))
        bad = b.add_gate(AND, armed, difftree)
    b.end_layer()

    next_state = ([m[ns] for ns in net.next_state]
                  + [CONST0] + list(warm[:-1])
                  + [legal_next] + shadow_next)
    return b.build(next_state=next_state, outputs=[bad])


def distractor_hold(net: Netlist, alphabet: int, settle: int = 1,
                    from_start: bool = False) -> Netlist:
    """No distractor stream can move the register: legal write at t=0, then ANY
    sequence over {blank} ∪ {non-cued one-hot symbols}; bad = armed ∧ (state
    changes over the ORIGINAL latches). from_start=True allows distractors already
    during the settle window (t ≥ 1); from_start=False forces blanks until arming
    (t = K) and lets distractors loose afterwards. pdr/bmc3."""
    return _distractor(net, alphabet, settle, from_start, decode=False)


def distractor_decode(net: Netlist, alphabet: int, settle: int = 1,
                      from_start: bool = False) -> Netlist:
    """No distractor stream can corrupt the READOUT: same input envelope as
    distractor_hold, but bad = armed ∧ ¬decode_ok (in-netlist GroupSum head vs the
    shadow register of the written symbol). pdr/bmc3."""
    return _distractor(net, alphabet, settle, from_start, decode=True)


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
        "protocol_decode": (protocol_decode(net, alphabet, settle_legal), False),
    }
