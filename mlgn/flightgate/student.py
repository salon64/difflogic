"""
student.py — the three student arms, built from the seqlgn / difflogic APIs.
============================================================================

TORCH-ONLY module (the only one in flightgate besides trainer's training half):
import it lazily so the numpy-only collection path works in the sim venv.

Arms (matched sizes — same ``hidden`` width; gate counts recorded per run via
``utils.count_gates`` for the honest capacity comparison):

* ``gated``  — :class:`mlgn.seqlgn.cells.LogicRecurrentCell(mechanism='gated')`
* ``clatch`` — same cell, ``mechanism='clatch'`` (input-clocked latch; the
  write-enable hardening is annealed per training step via ``cell.hard_alpha`` —
  a PLAIN attribute, NOT in the state_dict, silently 1.0 on fresh construction)
* ``ff``     — feedforward control: :class:`LogicMLP` + ``GroupSum`` (models.py
  has no feedforward classifier class; this is the sanctioned building-block path).
  It sees ONLY the current encoded frame — the degradation arm of the POMDP gate.

Action head: the observation bits map to ``hidden`` state bits; ONE parameterless
``GroupSum(k = n_act * n_bins)`` partitions them into ``n_act * n_bins`` disjoint
groups whose sums are the per-motor bin logits (reshape to [..., n_act, n_bins]).
GroupSum has no weights, so 4 "separate heads" over the same state would emit
identical logits — the disjoint-slice head is the correct construction. Requires
``hidden % (n_act * n_bins) == 0`` (asserted); difflogic additionally requires
``hidden >= input_bits`` on the first cell layer (asserted upstream with a message).

Device discipline (recon-verified): ``device`` is a CONSTRUCTOR argument threaded
through every class — LogicLayer allocates weights on it at ``__init__``; we pass it
down AND ``.to(device)``, mirroring train.py. Locally device MUST be 'cpu'
(implementation auto-selects 'python'); CUDA is DUST-only.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mlgn.seqlgn  # noqa: F401,E402  (injects the difflogic_cuda CPU stub FIRST)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from difflogic import GroupSum  # noqa: E402
from mlgn.seqlgn.cells import LogicMLP, LogicRecurrentCell  # noqa: E402

ARMS = ("gated", "clatch", "ff")


class _ActionHead(nn.Module):
    """GroupSum over disjoint hidden slices -> [..., n_act, n_bins] logits."""

    def __init__(self, hidden: int, n_act: int, n_bins: int, tau: float, device: str):
        super().__init__()
        assert hidden % (n_act * n_bins) == 0, (
            f"hidden ({hidden}) must be divisible by n_act*n_bins ({n_act * n_bins}) "
            f"for the GroupSum action head; pick e.g. hidden="
            f"{hidden - hidden % (n_act * n_bins) or n_act * n_bins}."
        )
        self.n_act, self.n_bins = n_act, n_bins
        self.group_sum = GroupSum(k=n_act * n_bins, tau=tau, device=device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.group_sum(h)  # [batch, n_act*n_bins]
        return logits.view(*h.shape[:-1], self.n_act, self.n_bins)


class RecurrentActionStudent(nn.Module):
    """LogicRecurrentCell unrolled over encoded observation bits, per-step action
    logits from the hidden state. forward_sequence(x [B,T,bits]) -> [B,T,n_act,n_bins]."""

    is_recurrent = True

    def __init__(
        self,
        input_bits: int,
        hidden: int,
        n_act: int = 4,
        n_bins: int = 9,
        mechanism: str = "gated",
        cell_layers: int = 2,
        keep_bias: float = 3.0,
        tau: float = 30.0,
        device: str = "cpu",
        grad_factor: float = 1.0,
    ):
        super().__init__()
        assert mechanism in ("gated", "clatch"), mechanism  # the two D1 arms
        self.input_bits, self.hidden = input_bits, hidden
        self.mechanism = mechanism
        self.cell = LogicRecurrentCell(
            input_dim=input_bits,
            hidden_dim=hidden,
            mechanism=mechanism,
            cell_layers=cell_layers,
            keep_bias=keep_bias,
            device=device,
            grad_factor=grad_factor,
        )
        self.head = _ActionHead(hidden, n_act, n_bins, tau, device)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3 and x.shape[-1] == self.input_bits, (x.shape, self.input_bits)
        batch, seq_len, _ = x.shape
        state = self.cell.init_state(batch, device=x.device, dtype=x.dtype)
        logits = []
        for t in range(seq_len):
            state = self.cell(x[:, t, :], state)
            logits.append(self.head(self.cell.readout_h(state)))
        return torch.stack(logits, dim=1)  # [B, T, n_act, n_bins]

    # single-step API for closed-loop rollouts (state carried by the caller)
    def step(self, x_t: torch.Tensor, state):
        state = self.cell(x_t, state)
        return self.head(self.cell.readout_h(state)), state


class FeedforwardActionStudent(nn.Module):
    """LogicMLP over the CURRENT encoded frame only (no state) — the control arm
    that must degrade under blackout. forward(x [N,bits]) -> [N,n_act,n_bins]."""

    is_recurrent = False
    mechanism = "ff"

    def __init__(
        self,
        input_bits: int,
        hidden: int,
        n_act: int = 4,
        n_bins: int = 9,
        num_layers: int = 4,
        tau: float = 30.0,
        device: str = "cpu",
        grad_factor: float = 1.0,
    ):
        super().__init__()
        self.input_bits, self.hidden = input_bits, hidden
        self.net = LogicMLP(input_bits, hidden, num_layers=num_layers, device=device,
                            grad_factor=grad_factor)
        self.head = _ActionHead(hidden, n_act, n_bins, tau, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, bits = x.shape
        out = self.forward(x.reshape(batch * seq_len, bits))
        return out.view(batch, seq_len, self.head.n_act, self.head.n_bins)


def build_student(
    arm: str,
    input_bits: int,
    hidden: int,
    n_act: int = 4,
    n_bins: int = 9,
    cell_layers: int = 2,
    keep_bias: float = 3.0,
    tau: float = 30.0,
    device: str = "cpu",
    grad_factor: float = 1.0,
) -> nn.Module:
    """One student per run (mirrors train.py's one-mechanism-per-run discipline).

    'ff' gets num_layers = 2*cell_layers so its LogicLayer count matches the
    recurrent cell's candidate+gate stacks (matched capacity; exact gate counts are
    recorded per run via utils.count_gates — never assume, always report)."""
    assert arm in ARMS, f"arm must be one of {ARMS}, got {arm!r}"
    if arm == "ff":
        model = FeedforwardActionStudent(
            input_bits, hidden, n_act=n_act, n_bins=n_bins,
            num_layers=2 * cell_layers, tau=tau, device=device, grad_factor=grad_factor)
    else:
        model = RecurrentActionStudent(
            input_bits, hidden, n_act=n_act, n_bins=n_bins, mechanism=arm,
            cell_layers=cell_layers, keep_bias=keep_bias, tau=tau, device=device,
            grad_factor=grad_factor)
    return model.to(device)


class StudentPolicy:
    """numpy <-> torch bridge exposing the actor contract for closed-loop rollouts:
    ``reset()`` then ``action(env, bits) -> (n_act,) float`` (env is IGNORED — the
    student only ever sees the masked, encoded observation).

    Deployed-consistent by default: ``discrete=True`` puts the model in ``.eval()``
    (argmax gates) and forces ``cell.hard_alpha = 1.0`` — the same circuit the
    netlist exporter would emit. Input bits are already {0,1} so no rounding is
    needed (asserted). The emitted action is the bin CENTER of the argmax bin.
    """

    def __init__(self, model: nn.Module, discretizer, device: str = "cpu",
                 discrete: bool = True):
        self.model = model
        self.discretizer = discretizer
        self.device = device
        self.discrete = discrete
        self._state = None

    def reset(self) -> None:
        self._state = None

    @torch.no_grad()
    def action(self, env, bits: np.ndarray) -> np.ndarray:
        del env  # the student never touches the env (no privileged access)
        assert np.all((bits == 0.0) | (bits == 1.0)), "encoded obs must be binary"
        was_training = self.model.training
        self.model.eval() if self.discrete else self.model.train()
        if hasattr(self.model, "cell"):
            old_alpha = self.model.cell.hard_alpha
            self.model.cell.hard_alpha = 1.0  # deployed-consistent enable/restore
        x = torch.as_tensor(np.asarray(bits, dtype=np.float32),
                            device=self.device).unsqueeze(0)
        if self.model.is_recurrent:
            if self._state is None:
                self._state = self.model.cell.init_state(1, device=self.device,
                                                         dtype=x.dtype)
            logits, self._state = self.model.step(x, self._state)
        else:
            logits = self.model(x)
        bins = logits.argmax(-1).squeeze(0).cpu().numpy()  # [n_act]
        if hasattr(self.model, "cell"):
            self.model.cell.hard_alpha = old_alpha
        self.model.train(was_training)
        return self.discretizer.to_values(bins).astype(np.float64)
