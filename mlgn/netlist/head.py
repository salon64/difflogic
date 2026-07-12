"""
head.py — the GroupSum readout as an in-netlist circuit (popcount + argmax compare).
====================================================================================

The trained models' classifier head is GroupSum: class c's score is the POPCOUNT of
the c-th CONTIGUOUS block of ``group_size`` state bits, and the prediction is the
argmax over the k scores (tau only rescales the scores, so it drops out of the
argmax; torch/numpy argmax breaks ties by the FIRST maximum). Until now the head
lived outside the netlist (sim.head_scores), so every ABC property could only speak
about raw state bits — settle-and-hold, but never "the READOUT is correct". This
module builds the head INSIDE the netlist: unsigned popcount, unsigned vector
comparators, and the exact first-max-wins argmax test. That is the missing encoding
for free-input decode properties (props.protocol_decode / props.distractor_decode):
"no matter what the environment injects, the readout equals the written symbol".

Construction notes:
  * Only ops from the 16-gate vocabulary are used: XOR=6, AND=1, OR=7 for the
    adders; A_AND_NOT_B=2, XNOR=9, NOT_A=12 for the comparators.
  * Simulator discipline (sim.step evaluates layer slices in order): every gate's
    inputs must come from strictly earlier layers, so each adder/comparator stage
    closes its layer with ``b.end_layer()``.
  * ``popcount`` is a balanced pairwise-combine tree of ripple-carry adders. All
    adders of one tree level — and, inside ``decode_ok``, across all k groups —
    are emitted in LOCKSTEP, so the layer count is O(width · log n) rather than
    O(n · width) and the numpy simulator stays fast.
  * Bit vectors are little-endian throughout: ``bits[i]`` has weight ``2**i``.
    Vectors of different widths may be mixed freely; the shorter side is padded
    with CONST0.
"""

from __future__ import annotations

from .ir import AND, A_AND_NOT_B, CONST0, CONST1, NOT_A, OR, XOR, NetlistBuilder

XNOR = 9  # a == b, in ir's 16-op indexing (truth-table bit order AB=00,01,10,11)


# -----------------------------------------------------------------------------------
# adders / popcount
# -----------------------------------------------------------------------------------
def _full_adder_stage(b: NetlistBuilder, triples) -> tuple[list[int], list[int]]:
    """One ripple-carry bit position for a whole BANK of adders, in lockstep.

    ``triples`` = [(a, x, cin), ...], one per adder. Returns (sums, carry-outs).
    Exactly three layers regardless of bank size:
        layer 1:  p = a XOR x   and   g = a AND x        (propagate / generate)
        layer 2:  s = p XOR cin and   t = p AND cin      (sum / propagated carry)
        layer 3:  cout = g OR t                          (== majority(a, x, cin))
    """
    p = [b.add_gate(XOR, a, x) for (a, x, _) in triples]
    g = [b.add_gate(AND, a, x) for (a, x, _) in triples]
    b.end_layer()
    s = [b.add_gate(XOR, p[i], c) for i, (_, _, c) in enumerate(triples)]
    t = [b.add_gate(AND, p[i], c) for i, (_, _, c) in enumerate(triples)]
    b.end_layer()
    cout = [b.add_gate(OR, g[i], t[i]) for i in range(len(triples))]
    b.end_layer()
    return s, cout


def _add_vec_bank(b: NetlistBuilder, pairs) -> list[list[int]]:
    """Ripple-carry add a bank of little-endian vector pairs, in lockstep.

    Shorter vectors are padded with CONST0. Every result has width
    ``max_width + 1`` (the final carry is appended), so no sum can overflow."""
    width = max(max(len(av), len(bv)) for av, bv in pairs)
    carries = [CONST0] * len(pairs)
    sums: list[list[int]] = [[] for _ in pairs]
    for i in range(width):
        triples = [(av[i] if i < len(av) else CONST0,
                    bv[i] if i < len(bv) else CONST0,
                    carries[k]) for k, (av, bv) in enumerate(pairs)]
        s, c = _full_adder_stage(b, triples)
        for k in range(len(pairs)):
            sums[k].append(s[k])
            carries[k] = c[k]
    for k in range(len(pairs)):
        sums[k].append(carries[k])
    return sums


def _popcount_bank(b: NetlistBuilder, groups) -> list[list[int]]:
    """Popcount several signal groups at once. Balanced tree: each level pairs up
    the current partial-count vectors within every group and adds ALL pairs (across
    all groups) through one lockstep adder bank. Returns one little-endian count
    vector per group."""
    banks = [[[s] for s in grp] if grp else [[CONST0]] for grp in groups]
    while max(len(vecs) for vecs in banks) > 1:
        pairs = []
        for vecs in banks:
            for i in range(0, len(vecs) - 1, 2):
                pairs.append((vecs[i], vecs[i + 1]))
        sums = iter(_add_vec_bank(b, pairs))
        nxt: list[list[list[int]]] = [[] for _ in banks]
        for gi, vecs in enumerate(banks):
            for i in range(0, len(vecs) - 1, 2):
                nxt[gi].append(next(sums))
            if len(vecs) % 2:
                nxt[gi].append(vecs[-1])  # odd vector rides up to the next level
        banks = nxt
    return [vecs[0] for vecs in banks]


def popcount(b: NetlistBuilder, sigs) -> list[int]:
    """Unsigned count of the given signals, as a little-endian bit vector.

    Balanced tree of half/full adders (XOR/AND/OR only); every stage is closed
    with ``b.end_layer()`` so the result is simulator-legal by construction."""
    return _popcount_bank(b, [list(sigs)])[0]


# -----------------------------------------------------------------------------------
# comparators
# -----------------------------------------------------------------------------------
def _vec_cmp(b: NetlistBuilder, abits, bbits, strict: bool) -> int:
    """Unsigned a > b (strict) or a >= b (non-strict), little-endian vectors.

    LSB→MSB ripple with   acc_i = (a_i > b_i) OR ((a_i == b_i) AND acc_{i-1}):
    each higher bit overrides everything below it, so after the MSB ``acc`` is the
    full unsigned comparison. The seed value of ``acc`` is the all-bits-equal
    verdict, i.e. CONST0 for ``>`` and CONST1 for ``>=``."""
    n = max(len(abits), len(bbits))
    if n == 0:
        return CONST0 if strict else CONST1
    bit = lambda v, i: v[i] if i < len(v) else CONST0  # noqa: E731  (CONST0 pad)
    gt = [b.add_gate(A_AND_NOT_B, bit(abits, i), bit(bbits, i)) for i in range(n)]
    eq = [b.add_gate(XNOR, bit(abits, i), bit(bbits, i)) for i in range(n)]
    b.end_layer()
    acc = CONST0 if strict else CONST1
    for i in range(n):
        keep = b.add_gate(AND, eq[i], acc)
        b.end_layer()
        acc = b.add_gate(OR, gt[i], keep)
        b.end_layer()
    return acc


def vec_gt(b: NetlistBuilder, abits, bbits) -> int:
    """1 iff unsigned(abits) > unsigned(bbits); shorter vector padded with CONST0."""
    return _vec_cmp(b, abits, bbits, strict=True)


def vec_ge(b: NetlistBuilder, abits, bbits) -> int:
    """1 iff unsigned(abits) >= unsigned(bbits); shorter vector padded with CONST0."""
    return _vec_cmp(b, abits, bbits, strict=False)


# -----------------------------------------------------------------------------------
# the GroupSum argmax-equals-shadow test
# -----------------------------------------------------------------------------------
def decode_ok(b: NetlistBuilder, head: tuple[int, int], state_sigs, shadow_sigs) -> int:
    """1 iff the GroupSum argmax over ``state_sigs`` names the class marked in the
    one-hot ``shadow_sigs``.

    ``head`` = (k, group_size). Class c's score s_c = popcount of the CONTIGUOUS
    slice ``state_sigs[c*gs:(c+1)*gs]`` — exactly sim.head_scores. torch/numpy
    argmax tie-break is FIRST max wins, so

        predicted == c   iff   (s_c > s_j  for all j < c)  AND
                               (s_c >= s_j for all j > c)

    and   decode_ok = OR_c ( shadow_c AND predicted==c ).

    If the shadow register is all-zero the result is constant 0 (no class claimed);
    callers gate their ``bad`` output with a legality signal that guarantees a
    one-hot shadow whenever the verdict matters, so this is safe."""
    k, gs = head
    assert len(state_sigs) == k * gs, (len(state_sigs), k, gs)
    assert len(shadow_sigs) == k, (len(shadow_sigs), k)
    scores = _popcount_bank(b, [state_sigs[c * gs:(c + 1) * gs] for c in range(k)])
    hits = []
    for c in range(k):
        wins = [vec_gt(b, scores[c], scores[j]) if j < c
                else vec_ge(b, scores[c], scores[j])
                for j in range(k) if j != c]
        pred_c = b.and_tree(wins)            # k=1 edge: and_tree([]) == CONST1
        hits.append(b.add_gate(AND, shadow_sigs[c], pred_c))
        b.end_layer()
    return b.or_tree(hits)


# -----------------------------------------------------------------------------------
# the GroupSum argmax as explicit outputs (deployable head, P3b T1)
# -----------------------------------------------------------------------------------
def argmax_onehot(b: NetlistBuilder, head: tuple[int, int], state_sigs) -> list[int]:
    """k signals, EXACTLY one high: pred_c = 1 iff the GroupSum argmax over
    ``state_sigs`` is c, with torch/numpy's first-max-wins tie-break:

        predicted == c   iff   (s_c > s_j  for all j < c)  AND
                               (s_c >= s_j for all j > c)

    The wins-loop duplicates ``decode_ok``'s predicted==c predicate verbatim
    (decode_ok itself stays untouched: its gate structure backs already-proved ABC
    theorems). Exactly one pred_c is high for any state: the first maximum always
    exists and is unique under this predicate."""
    k, gs = head
    assert len(state_sigs) == k * gs, (len(state_sigs), k, gs)
    scores = _popcount_bank(b, [state_sigs[c * gs:(c + 1) * gs] for c in range(k)])
    preds = []
    for c in range(k):
        wins = [vec_gt(b, scores[c], scores[j]) if j < c
                else vec_ge(b, scores[c], scores[j])
                for j in range(k) if j != c]
        preds.append(b.and_tree(wins))       # k=1 edge: and_tree([]) == CONST1
    return preds


def argmax_bits(b: NetlistBuilder, head: tuple[int, int], state_sigs) -> list[int]:
    """ceil(log2(k)) little-endian bits encoding the GroupSum argmax class
    (``bits[j]`` has weight 2**j). Exact because ``argmax_onehot`` is guaranteed
    exactly-one-high, so the per-bit OR over the one-hot preds cannot mix classes.
    k=1 edge: one constant-0 bit (class 0)."""
    preds = argmax_onehot(b, head, state_sigs)
    k = head[0]
    w = max(1, (k - 1).bit_length())         # w = 3 for k = 8
    return [b.or_tree([preds[c] for c in range(k) if (c >> j) & 1]) for j in range(w)]
