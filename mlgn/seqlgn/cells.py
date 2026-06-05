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


MECHANISMS = ("rddlgn", "gated", "latch")


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

    forward(x_t, h) -> h_next      (all tensors are [batch, dim])

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

        elif mechanism == "latch":
            raise NotImplementedError(
                "mechanism='latch' is Paper #2 and is currently PARKED. "
                "The interface is reserved; see mlgn/seqlgn/docs/design.md (Paper 2) for "
                "the planned D-flip-flop / gated-latch primitive with custom STE backprop."
            )

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
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

        raise RuntimeError(f"unhandled mechanism {self.mechanism!r}")

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"mechanism={self.mechanism!r}, cell_layers={self.cell_layers}"
        )
