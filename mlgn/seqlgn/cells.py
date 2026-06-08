"""
cells.py — Recurrent cells for Logic Gate Networks.
====================================================

This module defines the core research object for Paper #1 (gating) and the
infrastructure for Paper #2 (latches, currently parked): a **recurrent cell built
entirely from differentiable logic gates**, with a *pluggable memory mechanism* so that
every variant differs in exactly one thing — how state is carried across time.

Memory mechanisms
-----------------
- ``"rddlgn"``  — the CONTROL. A faithful reimplementation of the recurrence used by
  Recurrent Deep Differentiable Logic Gate Networks (Bührer et al., arXiv:2508.06097)
  and by DiffLogic CA (Miotti et al., arXiv:2506.04912): plain *concat-recurrence*.
  The new state is RECOMPUTED from scratch each step:
      h_{t+1} = LogicMLP([x_t ; h_t])
  There is no explicit "keep" path — the network must relearn to copy state every step.

- ``"lstm"``    — Paper #1 (richer arm). A logic-native LSTM with a dedicated cell state
  ``C`` carried across time and independent forget / input / output stages:
      f  = LGN_forget(z);  i = LGN_input(z);  C̃ = LGN_candidate(z)
      C' = (C AND f) OR (i AND C̃)              # OR = logic stand-in for LSTM's '+'
      o  = LGN_outproj(z)                       # project z to hidden_dim
      h' = LGN_readout([o ; C'])                # 2H -> H
  The carousel lives in the CELL state: ``∂C'/∂C = f`` (≈1 on kept bits). ``h'`` is a
  readout. ~5 LGNs and TWO carried states (h, C) vs the gated cell's 2 LGNs / 1 state —
  use it as the "does the extra forget/input/output machinery earn its cost?" ablation.

- ``"gated"``   — Paper #1's contribution. A logic-native LSTM/GRU-style update where a
  learned **2:1 multiplexer per hidden bit** decides keep-vs-write:
      c_t = LogicMLP_candidate([x_t ; h_t])     # the candidate ("write") state
      s_t = LogicMLP_gate([x_t ; h_t])          # the select bit per unit (in [0,1])
      h_{t+1} = s_t * h_t + (1 - s_t) * c_t      # soft multiplexer  (== GRU update gate)
  At binary values this is *exactly* the boolean multiplexer
      MUX(s, h, c) = (s AND h) OR (NOT s AND c)
  (in hardware: 3 gates per bit). The "keep" branch s_t * h_t is a **constant-error
  carousel**: when the gate keeps a bit (s_t ≈ 1), dh_{t+1}/dh_t ≈ 1, so the gradient
  flows backward through time un-attenuated — the mechanism that lets LSTMs/GRUs beat
  vanilla RNNs, here realised in pure logic.

- ``"latch"``   — Paper #2 (PARKED). Will replace recomputed state with a *bistable
  memory primitive* (D-flip-flop / gated D-latch / SR latch) plus a custom
  straight-through gradient through the feedback. Stubbed here so the interface is ready;
  see ``docs/design.md`` §"Paper 2".

Notes on the difflogic API (see ``docs/api.md`` for the full list)
------------------------------------------------------------------
- ``LogicLayer(in_dim, out_dim)`` holds ``weights`` of shape ``[out_dim, 16]`` (the gate
  logits). In ``.train()`` it outputs a softmax mixture over the 16 gates in [0,1]; in
  ``.eval()`` it argmaxes to a single hard gate and outputs {0,1} (given binary inputs).
- Each ``LogicLayer`` requires a 2D input ``[batch, in_dim]``; we therefore process one
  timestep at a time (each ``x_t`` and ``h`` is ``[batch, dim]``).
- Connectivity constraint: ``out_dim * 2 >= in_dim``. For the first layer of a cell,
  ``in_dim = input_dim + hidden_dim`` and ``out_dim = hidden_dim`` ⇒ **hidden_dim must be
  >= input_dim**. We assert this with a helpful message.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflogic import LogicLayer


MECHANISMS = ("rddlgn", "gated", "lstm", "latch")


def _or(a, b):
    """Soft logic OR, OR(a,b) = a + b - a*b (difflogic gate id 7). Exact at {0,1};
    stays in [0,1] for a,b in [0,1]. Used as the Boolean stand-in for LSTM's additive
    cell-state update (you can't add bits: 1+1 would leave {0,1})."""
    return a + b - a * b


class LogicMLP(nn.Module):
    """A stack of ``num_layers`` LogicLayers mapping ``in_dim -> out_dim``.

    Layer 1 maps ``in_dim -> out_dim``; any further layers map ``out_dim -> out_dim``.
    This is the basic feed-forward logic block we reuse for the candidate/gate/update
    networks inside a cell.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 2,
        device: str = "cuda",
        grad_factor: float = 1.0,
        implementation: str | None = None,
        connections: str = "random",
    ):
        super().__init__()
        assert num_layers >= 1, num_layers
        # difflogic connectivity constraint for the first (widest-input) layer.
        assert out_dim * 2 >= in_dim, (
            f"LogicLayer needs out_dim*2 >= in_dim, got out_dim={out_dim}, in_dim={in_dim}. "
            f"For a recurrent cell this means hidden_dim must be >= input_dim."
        )
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(
                LogicLayer(
                    in_dim=d,
                    out_dim=out_dim,
                    device=device,
                    grad_factor=grad_factor,
                    implementation=implementation,
                    connections=connections,
                )
            )
            d = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LogicRecurrentCell(nn.Module):
    """One recurrent step built from logic gates, with a pluggable memory mechanism.

    forward(x_t, state) -> new_state      (all tensors are [batch, dim])

    State convention (so one model loop drives every mechanism):
      - ``rddlgn`` / ``gated``: state is a single tensor ``h``  [batch, hidden_dim].
      - ``lstm``: state is a tuple ``(h, C)`` — hidden state and cell state.
    Use the helpers ``init_state`` / ``readout_h`` / ``carousel_state`` rather than
    poking at the state shape directly.

    The only thing that changes between the scientific conditions is ``mechanism``; the
    capacity (number/size of logic layers) is kept comparable so the comparison isolates
    the memory mechanism (see ``docs/experiments.md``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mechanism: str = "gated",
        cell_layers: int = 2,
        device: str = "cuda",
        grad_factor: float = 1.0,
        implementation: str | None = None,
        connections: str = "random",
    ):
        super().__init__()
        assert mechanism in MECHANISMS, f"mechanism must be one of {MECHANISMS}, got {mechanism!r}"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mechanism = mechanism
        self.cell_layers = cell_layers

        cat_dim = input_dim + hidden_dim
        mlp_kwargs = dict(
            num_layers=cell_layers,
            device=device,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
        )

        if mechanism == "rddlgn":
            # CONTROL: recompute the full state each step from [x_t ; h_t].
            self.update = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)

        elif mechanism == "gated":
            # Paper #1: separate candidate and gate networks (mirrors GRU's separate
            # weight matrices for the candidate and the update gate).
            self.candidate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.gate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)

        elif mechanism == "lstm":
            # Paper #1 (richer arm): independent forget / input / candidate stages over z,
            # a dedicated cell state C with an OR-combine (logic stand-in for LSTM's '+'),
            # and an output path that projects z to hidden_dim then reads out from [o ; C].
            self.forget = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.input = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.candidate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.out_proj = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            # readout sees [o ; C] (2*hidden_dim) -> hidden_dim; 2H >= 2H satisfies the
            # difflogic out*2>=in rule exactly (this is why we project z->o first instead
            # of feeding [z ; C] directly, which would violate it).
            self.readout = LogicMLP(2 * hidden_dim, hidden_dim, **mlp_kwargs)

        elif mechanism == "latch":
            raise NotImplementedError(
                "mechanism='latch' is Paper #2 and is currently PARKED. "
                "The interface is reserved; see mlgn/seqlgn/docs/design.md (Paper 2) for "
                "the planned D-flip-flop / gated-latch primitive with custom STE backprop."
            )

    def forward(self, x_t: torch.Tensor, state):
        # Unpack the (possibly tuple) state into the hidden vector h used to form z.
        h = state[0] if self.mechanism == "lstm" else state
        z = torch.cat([x_t, h], dim=-1)  # [batch, input_dim + hidden_dim]

        if self.mechanism == "rddlgn":
            return self.update(z)

        if self.mechanism == "gated":
            c = self.candidate(z)              # candidate / "write" state, in [0,1]
            s = self.gate(z)                   # select bit per unit, in [0,1]
            # Soft 2:1 multiplexer == GRU update gate. At binary {0,1} values this is the
            # exact boolean MUX (s AND h) OR (NOT s AND c). The s*h branch is the
            # constant-error carousel: gradient ~1 through kept bits.
            return s * h + (1.0 - s) * c

        if self.mechanism == "lstm":
            _, C = state
            f = self.forget(z)                 # forget gate (keep-mask), in [0,1]
            i = self.input(z)                  # input gate (write-enable), in [0,1]
            c_tilde = self.candidate(z)        # candidate cell value, in [0,1]
            # C' = (C AND f) OR (i AND C̃): the OR is the logic stand-in for LSTM's
            # additive C_t = f⊙C_{t-1} + i⊙C̃. ∂C'/∂C = f (≈1 on kept, non-overwritten
            # bits) -> the cell-state carousel.
            C_new = _or(C * f, i * c_tilde)
            o = self.out_proj(z)               # project z to hidden_dim (output-gate-like)
            h_new = self.readout(torch.cat([o, C_new], dim=-1))  # [o ; C'] (2H) -> H
            return (h_new, C_new)

        raise RuntimeError(f"unhandled mechanism {self.mechanism!r}")

    # --- state helpers (keep the model loop mechanism-agnostic) -----------------------
    def init_state(self, batch: int, device, dtype=torch.float32):
        h = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        if self.mechanism == "lstm":
            C = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
            return (h, C)
        return h

    def readout_h(self, state) -> torch.Tensor:
        """The hidden vector used for classification (and fed back next step)."""
        return state[0] if self.mechanism == "lstm" else state

    def carousel_state(self, state) -> torch.Tensor:
        """The state carrying the long-range gradient highway: the cell state C for
        lstm, the hidden state h otherwise. Used by the grad-norm-through-time analysis."""
        return state[1] if self.mechanism == "lstm" else state

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"mechanism={self.mechanism!r}, cell_layers={self.cell_layers}"
        )
