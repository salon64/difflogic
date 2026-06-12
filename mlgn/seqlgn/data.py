"""
data.py — sequential benchmark tasks for recurrent LGNs.
========================================================

Every task yields batches of shape ``[batch, seq_len, input_dim]`` with values in
``[0, 1]`` plus integer class labels, so the same training loop drives all of them. The
tasks are chosen to *isolate memory*: how well does a cell carry information across time?

Tasks
-----
- ``smnist``   — sequential MNIST, one ROW per timestep (T=28, input_dim=28). The mild
  warm-up (also what ``mlgn/secuential.py`` used). ~98% is reachable; not very
  memory-stressing.
- ``smnist-pixel`` — sequential MNIST, one PIXEL per timestep (T=784, input_dim=1). Long
  sequence ⇒ a real test of gradient flow through time.
- ``psmnist`` — permuted sequential MNIST: ``smnist-pixel`` with a FIXED random pixel
  permutation. The classic long-range memory benchmark; destroys local structure so the
  model must integrate over the whole 784-step sequence.
- ``parity``  — stream of L random bits; label = XOR(all bits) = parity. A pure 1-bit
  running-state task. Note: parity is *exactly what a T flip-flop computes* — a clean
  discriminator between "recompute state" (rddlgn) and "hold state" (gated / latch), and a
  direct tie-in to Paper #2. Difficulty scales with L.
- ``copy``    — present a one-hot symbol at t=0 (with a cue bit), then L-1 blank steps;
  label = the original symbol. Pure long-range recall; difficulty scales with the delay L.

Adding-problem (regression) is intentionally omitted for now: it needs a regression head
instead of GroupSum+cross-entropy. See ``docs/benchmarks.md`` (TODO).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data-mnist")


@dataclass
class TaskSpec:
    name: str
    input_dim: int
    num_classes: int
    seq_len: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


# --------------------------------------------------------------------------------------
# Synthetic memory tasks (generated on the fly, no download)
# --------------------------------------------------------------------------------------
def _make_parity(n: int, seq_len: int, gen: torch.Generator):
    bits = torch.randint(0, 2, (n, seq_len), generator=gen).float()
    x = bits.unsqueeze(-1)                       # [n, L, 1]
    y = (bits.sum(dim=1) % 2).long()             # [n]
    return x, y


def _make_copy(n: int, seq_len: int, alphabet: int, gen: torch.Generator):
    symbols = torch.randint(0, alphabet, (n,), generator=gen)
    x = torch.zeros(n, seq_len, alphabet + 1)    # last channel = cue bit
    x[torch.arange(n), 0, symbols] = 1.0         # one-hot symbol at t=0
    x[:, 0, alphabet] = 1.0                      # cue bit marks the symbol step
    y = symbols.long()
    return x, y


def _synthetic_task(
    name: str,
    seq_len: int,
    num_classes: int,
    input_dim: int,
    maker,
    batch_size: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> TaskSpec:
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = maker(n_train, g)
    xva, yva = maker(n_val, g)
    xte, yte = maker(n_test, g)

    def loader(x, y, shuffle):
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

    return TaskSpec(
        name=name,
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        train_loader=loader(xtr, ytr, True),
        val_loader=loader(xva, yva, False),
        test_loader=loader(xte, yte, False),
    )


# --------------------------------------------------------------------------------------
# Sequential MNIST family
# --------------------------------------------------------------------------------------
class _SeqMNIST(Dataset):
    """Wraps a torchvision MNIST dataset and emits sequences [seq_len, input_dim].

    pixel mode with ``chunk=k`` feeds k pixels per timestep (seq_len = 784//k, input_dim
    = k) — a knob to make MNIST a *feasible* long sequence (e.g. k=14 → 56 steps) instead
    of the full 784-step version that's beyond the cell's frontier and ~40h/run.
    """

    def __init__(self, base, mode: str, permutation: torch.Tensor | None = None, chunk: int = 1):
        self.base = base
        self.mode = mode
        self.permutation = permutation
        self.chunk = chunk

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]            # img: [1, 28, 28] in [0,1]
        if self.mode == "row":
            seq = img.view(28, 28)           # [T=28, input_dim=28]
        else:  # pixel / permuted-pixel, optionally chunked
            flat = img.view(784)
            if self.permutation is not None:
                flat = flat[self.permutation]
            c = self.chunk
            n = (784 // c) * c               # trim remainder so it reshapes cleanly
            seq = flat[:n].view(n // c, c)   # [T=784//c, input_dim=c]
        return seq, label


def _mnist_task(name: str, mode: str, batch_size: int, val_frac: float, seed: int,
                chunk: int = 1) -> TaskSpec:
    import torchvision

    tfm = torchvision.transforms.ToTensor()
    train_full = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_base = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)

    permutation = None
    if name == "psmnist":
        permutation = torch.randperm(784, generator=torch.Generator().manual_seed(1234))

    train_set = _SeqMNIST(train_full, mode, permutation, chunk=chunk)
    test_set = _SeqMNIST(test_base, mode, permutation, chunk=chunk)

    val_size = int(len(train_set) * val_frac)
    train_size = len(train_set) - val_size
    train_split, val_split = random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    input_dim = 28 if mode == "row" else chunk
    seq_len = 28 if mode == "row" else (784 // chunk)
    return TaskSpec(
        name=name,
        input_dim=input_dim,
        num_classes=10,
        seq_len=seq_len,
        train_loader=DataLoader(train_split, batch_size=batch_size, shuffle=True, drop_last=True),
        val_loader=DataLoader(val_split, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def get_task(
    name: str,
    batch_size: int = 128,
    seq_len: int | None = None,
    alphabet: int = 8,
    chunk: int = 1,
    val_frac: float = 0.1,
    n_train: int = 50_000,
    n_val: int = 5_000,
    n_test: int = 10_000,
    seed: int = 0,
) -> TaskSpec:
    """Build a :class:`TaskSpec`. ``seq_len`` overrides synthetic-task length; ``chunk``
    sets pixels-per-step for the pixel-MNIST tasks (seq_len = 784//chunk)."""
    name = name.lower()

    if name in ("smnist", "smnist-row"):
        return _mnist_task("smnist", "row", batch_size, val_frac, seed)
    if name == "smnist-pixel":
        return _mnist_task("smnist-pixel", "pixel", batch_size, val_frac, seed, chunk=chunk)
    if name == "psmnist":
        return _mnist_task("psmnist", "pixel", batch_size, val_frac, seed, chunk=chunk)

    if name == "parity":
        L = seq_len or 64
        return _synthetic_task(
            "parity", L, num_classes=2, input_dim=1,
            maker=lambda n, g: _make_parity(n, L, g),
            batch_size=batch_size, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,
        )
    if name == "copy":
        L = seq_len or 64
        return _synthetic_task(
            "copy", L, num_classes=alphabet, input_dim=alphabet + 1,
            maker=lambda n, g: _make_copy(n, L, alphabet, g),
            batch_size=batch_size, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,
        )

    raise ValueError(
        f"unknown task {name!r}. options: smnist, smnist-pixel, psmnist, parity, copy"
    )


AVAILABLE_TASKS = ("smnist", "smnist-pixel", "psmnist", "parity", "copy")
