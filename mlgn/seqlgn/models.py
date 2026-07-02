"""
models.py — Sequence models built on a LogicRecurrentCell.
==========================================================

``SequenceClassifier`` unrolls a :class:`~cells.LogicRecurrentCell` over time and reads
out a class prediction from the final hidden state via difflogic's ``GroupSum`` head.

Input convention
----------------
``forward(x)`` expects ``x`` of shape ``[batch, seq_len, input_dim]`` with values in
``[0, 1]`` (soft during training, binarised at eval — see ``data.py`` and
``train.py.evaluate``). The hidden state is initialised to all-zeros (a valid binary
state), matching ``mlgn/secuential.py``.

GroupSum head
-------------
``GroupSum(k, tau)`` partitions the ``hidden_dim`` output bits into ``k`` groups (one per
class) and sums each group, dividing by ``tau``. This requires ``hidden_dim % k == 0`` —
we assert it with a helpful message.

Gradient-flow analysis
----------------------
``forward(x, return_hidden=True)`` additionally returns the list of per-timestep hidden
states with ``retain_grad()`` enabled, so that after a backward pass one can read
``h_t.grad`` and measure the gradient norm reaching each timestep. This is how we
demonstrate the constant-error-carousel benefit of gating vs. the rddlgn control
(see ``utils.grad_norm_through_time`` and ``docs/experiments.md``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflogic import GroupSum

from .cells import LogicRecurrentCell


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        mechanism: str = "gated",
        cell_layers: int = 2,
        keep_bias: float = 3.0,
        latch_kind: str = "sr",
        hard_state: bool = True,
        hard_control: bool = False,
        tau: float = 30.0,
        device: str = "cuda",
        grad_factor: float = 1.0,
        implementation: str | None = None,
        connections: str = "random",
    ):
        super().__init__()
        assert hidden_dim % num_classes == 0, (
            f"GroupSum needs hidden_dim divisible by num_classes; "
            f"got hidden_dim={hidden_dim}, num_classes={num_classes}. "
            f"Pick e.g. hidden_dim={hidden_dim - (hidden_dim % num_classes)}."
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mechanism = mechanism

        self.cell = LogicRecurrentCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            mechanism=mechanism,
            cell_layers=cell_layers,
            keep_bias=keep_bias,
            latch_kind=latch_kind,
            hard_state=hard_state,
            hard_control=hard_control,
            device=device,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
        )
        self.head = GroupSum(k=num_classes, tau=tau, device=device)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """x: [batch, seq_len, input_dim] -> logits: [batch, num_classes]."""
        assert x.ndim == 3, f"expected [batch, seq_len, input_dim], got {tuple(x.shape)}"
        batch, seq_len, in_dim = x.shape
        assert in_dim == self.input_dim, (in_dim, self.input_dim)

        state = self.cell.init_state(batch, device=x.device, dtype=x.dtype)

        carousel_states = []
        for t in range(seq_len):
            state = self.cell(x[:, t, :], state)
            if return_hidden:
                # track the gradient highway (cell state C for lstm, hidden h otherwise)
                cs = self.cell.carousel_state(state)
                cs.retain_grad()
                carousel_states.append(cs)

        logits = self.head(self.cell.readout_h(state))
        if return_hidden:
            return logits, carousel_states
        return logits

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_classes={self.num_classes}, mechanism={self.mechanism!r}"
        )
