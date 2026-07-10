"""
ir.py — gate-level netlist IR for trained sequential LGNs.
==========================================================

The eval-time semantics of every seqlgn mechanism is an exact-binary clocked FSM:
``model.eval()`` argmaxes each LogicLayer neuron to ONE of the 16 two-input Boolean
gates, inputs are binarised, and the recurrent state is a vector of {0,1} bits carried
across steps — i.e. a register file. This module extracts that FSM as an explicit
netlist so it can be simulated, model-checked (ABC), and later synthesized (yosys).

Signal numbering (one flat space):
    0                  CONST0
    1                  CONST1
    2 .. 2+n_pi-1      primary inputs  x_0 .. x_{n_pi-1}   (one input vector per step)
    .. + n_state       latch outputs   q_0 .. q_{n_state-1} (current state, init per latch)
    .. + n_gates       gate outputs, in topological order

Gate ops use difflogic's 16-gate indexing (see difflogic/functional.py and
utils.GATE_NAMES): op g on inputs (a, b) with truth-table bit order AB=00,01,10,11.

Mechanism → next-state circuit (all EXACT at eval time):
    gated / combo / clatch : per bit  q' = MUX(s, q, c) = (s AND q) OR (NOT s AND c)
                             (combo's state-round and clatch's enable-round are identity
                             on binary values, so the deployed circuit is the same MUX)
    rddlgn                 : q' = update-MLP output (recompute)
    latch (sr)             : q' = S OR (NOT R AND q)
    latch (tff)            : q' = T XOR q
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

CONST0 = 0
CONST1 = 1

# op id -> function on numpy bool arrays (mirrors difflogic/functional.py bin_op at {0,1})
GATE_FN = {
    0: lambda a, b: np.zeros_like(a),
    1: lambda a, b: a & b,
    2: lambda a, b: a & ~b,
    3: lambda a, b: a,
    4: lambda a, b: ~a & b,
    5: lambda a, b: b,
    6: lambda a, b: a ^ b,
    7: lambda a, b: a | b,
    8: lambda a, b: ~(a | b),
    9: lambda a, b: ~(a ^ b),
    10: lambda a, b: ~b,
    11: lambda a, b: a | ~b,
    12: lambda a, b: ~a,
    13: lambda a, b: ~a | b,
    14: lambda a, b: ~(a & b),
    15: lambda a, b: np.ones_like(a),
}

# op id -> BLIF ON-set cover lines over (a, b). None entries are constants (no inputs).
BLIF_COVER = {
    0: [],                     # const 0  (empty cover)
    1: ["11 1"],
    2: ["10 1"],
    3: ["1- 1"],
    4: ["01 1"],
    5: ["-1 1"],
    6: ["10 1", "01 1"],
    7: ["1- 1", "-1 1"],
    8: ["00 1"],
    9: ["00 1", "11 1"],
    10: ["-0 1"],
    11: ["1- 1", "-0 1"],
    12: ["0- 1"],
    13: ["0- 1", "-1 1"],
    14: ["0- 1", "-0 1"],
    15: ["1"],                 # const 1  (zero-input cover)
}

AND, A_AND_NOT_B, NOT_A_AND_B, XOR, OR, NOR, NOT_A = 1, 2, 4, 6, 7, 8, 12


@dataclass
class Netlist:
    """A flat, topologically-ordered gate netlist with registers.

    ``layers`` holds (start, end) gate-index slices in which every gate's inputs come
    only from earlier signals — the unit of vectorization for the simulator.
    ``head`` is (num_classes, group_size): GroupSum class scores = per-group popcounts
    over the STATE bits (argmax = predicted class); tau only rescales, so it is dropped.
    """

    n_pi: int
    n_state: int
    init: list[int]
    ops: np.ndarray
    src_a: np.ndarray
    src_b: np.ndarray
    layers: list[tuple[int, int]]
    next_state: list[int]
    outputs: list[int] = field(default_factory=list)
    head: tuple[int, int] | None = None

    # -- signal-id helpers ---------------------------------------------------------
    def pi(self, i: int) -> int:
        return 2 + i

    def q(self, i: int) -> int:
        return 2 + self.n_pi + i

    def gate_sig(self, g: int) -> int:
        return 2 + self.n_pi + self.n_state + g

    @property
    def n_gates(self) -> int:
        return len(self.ops)

    @property
    def n_signals(self) -> int:
        return 2 + self.n_pi + self.n_state + self.n_gates


class NetlistBuilder:
    def __init__(self, n_pi: int, n_state: int, init: list[int] | None = None):
        self.n_pi = n_pi
        self.n_state = n_state
        self.init = list(init) if init is not None else [0] * n_state
        self.ops: list[int] = []
        self.src_a: list[int] = []
        self.src_b: list[int] = []
        self.layers: list[tuple[int, int]] = []
        self._layer_start = 0

    # signal ids (mirror Netlist)
    def pi(self, i: int) -> int:
        return 2 + i

    def q(self, i: int) -> int:
        return 2 + self.n_pi + i

    def add_gate(self, op: int, a: int, b: int) -> int:
        self.ops.append(op)
        self.src_a.append(a)
        self.src_b.append(b)
        return 2 + self.n_pi + self.n_state + len(self.ops) - 1

    def end_layer(self) -> None:
        """Close the current vectorization slice (gates added since the last call)."""
        n = len(self.ops)
        if n > self._layer_start:
            self.layers.append((self._layer_start, n))
        self._layer_start = n

    def or_tree(self, sigs: list[int]) -> int:
        """Balanced OR-reduction; returns CONST0 for an empty list. Each level is
        closed as its own layer slice (inputs of a level come from earlier levels)."""
        return self._tree(OR, sigs, CONST0)

    def and_tree(self, sigs: list[int]) -> int:
        return self._tree(AND, sigs, CONST1)

    def _tree(self, op: int, sigs: list[int], empty: int) -> int:
        if not sigs:
            return empty
        level = list(sigs)
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                nxt.append(self.add_gate(op, level[i], level[i + 1]))
            if len(level) % 2:
                nxt.append(level[-1])
            self.end_layer()
            level = nxt
        return level[0]

    def build(self, next_state: list[int], outputs: list[int] | None = None,
              head: tuple[int, int] | None = None) -> Netlist:
        self.end_layer()
        assert len(next_state) == self.n_state, (len(next_state), self.n_state)
        return Netlist(
            n_pi=self.n_pi,
            n_state=self.n_state,
            init=self.init,
            ops=np.asarray(self.ops, dtype=np.uint8),
            src_a=np.asarray(self.src_a, dtype=np.int32),
            src_b=np.asarray(self.src_b, dtype=np.int32),
            layers=self.layers,
            next_state=list(next_state),
            outputs=list(outputs or []),
            head=head,
        )


def _add_logic_mlp(b: NetlistBuilder, mlp, in_sigs: list[int]) -> list[int]:
    """Append one LogicMLP's argmax-locked gates; returns its output signals."""
    sigs = in_sigs
    for layer in mlp.net:
        ops = layer.weights.argmax(-1).cpu().numpy()
        ia = layer.indices[0].cpu().numpy()
        ib = layer.indices[1].cpu().numpy()
        assert layer.in_dim == len(sigs), (layer.in_dim, len(sigs))
        out = [b.add_gate(int(ops[i]), sigs[int(ia[i])], sigs[int(ib[i])])
               for i in range(layer.out_dim)]
        b.end_layer()
        sigs = out
    return sigs


def extract_netlist(model) -> Netlist:
    """SequenceClassifier (eval semantics) → Netlist. Single-tensor-state mechanisms only."""
    cell = model.cell
    mech = cell.mechanism
    n_pi, n_state = cell.input_dim, cell.hidden_dim
    b = NetlistBuilder(n_pi, n_state)  # state initialised to all-zeros, as in init_state()
    z = [b.pi(i) for i in range(n_pi)] + [b.q(i) for i in range(n_state)]

    if mech in ("gated", "combo", "clatch"):
        c = _add_logic_mlp(b, cell.candidate, z)
        s = _add_logic_mlp(b, cell.gate, z)
        keep = [b.add_gate(AND, s[i], b.q(i)) for i in range(n_state)]
        b.end_layer()
        write = [b.add_gate(NOT_A_AND_B, s[i], c[i]) for i in range(n_state)]
        b.end_layer()
        nxt = [b.add_gate(OR, keep[i], write[i]) for i in range(n_state)]
        b.end_layer()
    elif mech == "rddlgn":
        nxt = _add_logic_mlp(b, cell.update, z)
    elif mech == "latch" and cell.latch_kind == "sr":
        S = _add_logic_mlp(b, cell.set_net, z)
        R = _add_logic_mlp(b, cell.reset_net, z)
        hold = [b.add_gate(A_AND_NOT_B, b.q(i), R[i]) for i in range(n_state)]  # q AND NOT r
        b.end_layer()
        nxt = [b.add_gate(OR, S[i], hold[i]) for i in range(n_state)]
        b.end_layer()
    elif mech == "latch" and cell.latch_kind == "tff":
        T = _add_logic_mlp(b, cell.toggle_net, z)
        nxt = [b.add_gate(XOR, T[i], b.q(i)) for i in range(n_state)]
        b.end_layer()
    else:
        raise NotImplementedError(
            f"mechanism {mech!r} carries a tuple state (h, C); the exporter currently "
            f"supports single-state mechanisms (gated/combo/clatch/rddlgn/latch)."
        )

    k = model.num_classes
    return b.build(nxt, head=(k, n_state // k))


def copy_gates_with_remap(b: NetlistBuilder, net: Netlist, sigmap: dict[int, int]) -> dict[int, int]:
    """Replay ``net``'s gates into builder ``b`` with leaf signals remapped.

    ``sigmap`` must cover CONST0/CONST1 and every PI/latch signal of ``net``; gate
    signals are mapped as they are recreated. Layer boundaries are preserved. Returns
    the completed old→new signal map.
    """
    m = dict(sigmap)
    for (s, e) in net.layers:
        for g in range(s, e):
            m[net.gate_sig(g)] = b.add_gate(
                int(net.ops[g]), m[int(net.src_a[g])], m[int(net.src_b[g])]
            )
        b.end_layer()
    return m
