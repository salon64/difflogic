"""
utils.py — small helpers for the seqlgn experiments.
====================================================

Device selection, reproducibility, gate-distribution analysis (which of the 16 gates each
neuron settled on), and the gradient-norm-through-time measurement used to demonstrate the
constant-error-carousel effect of gating.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from difflogic import LogicLayer


# The 16 two-input Boolean gates, indexed as difflogic orders them (weights.argmax(-1)).
GATE_NAMES = [
    "FALSE", "AND", "A AND NOT B", "A",
    "NOT A AND B", "B", "XOR", "OR",
    "NOR", "XNOR", "NOT B", "A OR NOT B",
    "NOT A", "NOT A OR B", "NAND", "TRUE",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> str:
    if prefer in ("cuda", "cpu"):
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_gates(model: torch.nn.Module) -> int:
    """Total number of logic gates (= sum of out_dim over all LogicLayers)."""
    return sum(m.out_dim for m in model.modules() if isinstance(m, LogicLayer))


@torch.no_grad()
def gate_distribution(model: torch.nn.Module) -> dict[str, np.ndarray]:
    """Per-LogicLayer histogram over the 16 gate types (argmax of the learned logits)."""
    dist = {}
    for name, module in model.named_modules():
        if isinstance(module, LogicLayer):
            ids = module.weights.argmax(-1).cpu()
            dist[name] = torch.bincount(ids, minlength=16).numpy()
    return dist


def format_gate_distribution(dist: dict[str, np.ndarray]) -> str:
    lines = []
    for name, counts in dist.items():
        total = int(counts.sum()) or 1
        lines.append(f"  {name}:")
        for idx, c in enumerate(counts.tolist()):
            if c > 0:
                lines.append(f"    [{idx:2d}] {GATE_NAMES[idx]:<14s} {c:>7,} ({100 * c / total:4.1f}%)")
    return "\n".join(lines)


def grad_norm_through_time(model, x, y, loss_fn) -> list[float]:
    """Return the L2 norm of dL/dh_t at each timestep t.

    A *flat* profile (gradient norm roughly constant across t) indicates gradients reach
    early timesteps — the constant-error-carousel property we expect from the gated cell.
    A profile that decays toward early t indicates vanishing gradients through time, which
    we expect to hurt the rddlgn control on long sequences.

    Run on a single batch with the model in ``.train()`` mode.
    """
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    logits, hidden_states = model(x, return_hidden=True)
    loss = loss_fn(logits, y)
    loss.backward()

    norms = []
    for h in hidden_states:
        if h.grad is None:
            norms.append(float("nan"))
        else:
            norms.append(h.grad.detach().norm().item())

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return norms
