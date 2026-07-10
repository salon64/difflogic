"""
extract.py — rebuild a trained SequenceClassifier from a checkpoint, exactly.
=============================================================================

Problem: ``LogicLayer`` wiring (``indices``) is created by two ``torch.randperm``
calls on the GLOBAL CPU generator at construction time and is NOT saved in the
state_dict (it is a plain attribute, not a buffer). A checkpoint alone therefore
cannot reconstruct the circuit — the wiring must be replayed from the run's seed.

Why the replay is exact for the runs in ``run_queue.sh``:
  * ``train.py`` calls ``utils.set_seed(seed)`` and then ``get_task(...)`` BEFORE
    constructing the model. Every task builder consumes ZERO draws from the global
    CPU generator (synthetic tasks use a dedicated ``torch.Generator``; the MNIST
    tasks use dedicated generators for the permutation and the split, and
    DataLoader/RandomSampler only draw at iteration time).
  * During construction on CUDA, each LogicLayer consumes the CUDA stream for its
    weight init (``torch.randn(..., device='cuda')``) and the CPU stream ONLY for
    ``get_connections``: ``randperm(2*out_dim)`` then ``randperm(in_dim)``.
  * Submodules are registered in construction order, so iterating
    ``model.modules()`` visits LogicLayers in exactly the order they consumed the
    CPU stream.

So: seed the global generator, then regenerate each layer's two randperms in module
order. NOTE this replays a CUDA-constructed model on a CPU host precisely BECAUSE the
weight inits went to the CUDA stream; rebuilding with a plain
``SequenceClassifier(device='cpu')`` constructor call would interleave the randn draws
into the CPU stream and desync the wiring. Correctness is then GATED, not assumed:
``check_accuracy`` must reproduce the run's recorded discrete ``test_acc`` on the
regenerated test set (same seed ⇒ bit-identical tensors), and the netlist simulator
must match the torch model bit-for-bit (see sim.equivalence_check). A wrong wiring
collapses accuracy to ~chance, so the gate is unambiguous.

(The forward-looking fix — registering ``indices`` as persistent buffers — changes the
checkpoint format and is deliberately left out of this v0; see README.)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.seqlgn import utils  # noqa: E402  (installs the CPU shim for difflogic)
from mlgn.seqlgn.data import get_task  # noqa: E402
from mlgn.seqlgn.models import SequenceClassifier  # noqa: E402
from mlgn.seqlgn.train import evaluate  # noqa: E402
from difflogic import LogicLayer  # noqa: E402


@dataclass
class RunSpec:
    """The construction-relevant arguments of a training run (from its results JSON +
    the run_queue.sh command line for fields the JSON does not record)."""

    task: str
    mechanism: str
    seq_len: int
    hidden: int
    cell_layers: int
    keep_bias: float
    tau: float
    seed: int
    alphabet: int = 8          # NOT in the JSON — confirm against run_queue.sh
    latch_kind: str = "sr"
    hard_state: bool = True
    hard_control: bool = False
    batch_size: int = 128
    n_distractors: int = 8     # distcopy only; NOT in the JSON
    sel_flag: bool = False     # selcopy only; NOT in the JSON
    test_acc: float | None = None   # recorded discrete accuracy (the replay gate)
    test_soft: float | None = None

    @property
    def input_dim(self) -> int:
        if self.task in ("copy", "selcopy", "distcopy"):
            return self.alphabet + 1
        if self.task == "parity":
            return 1
        raise ValueError(f"input_dim unknown for task {self.task!r} (add it here)")

    @property
    def num_classes(self) -> int:
        if self.task in ("copy", "selcopy", "distcopy"):
            return self.alphabet
        if self.task == "parity":
            return 2
        raise ValueError(f"num_classes unknown for task {self.task!r} (add it here)")


def spec_from_json(json_path: str, alphabet: int = 8, n_distractors: int = 8,
                   sel_flag: bool = False) -> RunSpec:
    with open(json_path) as f:
        r = json.load(f)
    return RunSpec(
        task=r["task"],
        mechanism=r["mechanism"],
        seq_len=r["seq_len"],
        hidden=r["hidden"],
        cell_layers=r["cell_layers"],
        keep_bias=r["keep_bias"],
        tau=r["tau"],
        seed=r["seed"],
        alphabet=alphabet,
        latch_kind=r.get("latch_kind") or "sr",
        hard_state=r.get("hard_state") if r.get("hard_state") is not None else True,
        hard_control=bool(r.get("hard_control")),
        batch_size=r["batch_size"],
        n_distractors=n_distractors,
        sel_flag=sel_flag,
        test_acc=r.get("test_acc"),
        test_soft=r.get("test_soft"),
    )


def _replay_connections(in_dim: int, out_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact replay of LogicLayer.get_connections('random') on the global CPU RNG."""
    c = torch.randperm(2 * out_dim) % in_dim
    c = torch.randperm(in_dim)[c]
    c = c.reshape(2, out_dim)
    return c[0].to(torch.int64), c[1].to(torch.int64)


def rebuild_model(spec: RunSpec, ckpt_path: str) -> SequenceClassifier:
    """Construct the model on CPU, replay the training-time wiring, load the weights."""
    model = SequenceClassifier(
        input_dim=spec.input_dim,
        hidden_dim=spec.hidden,
        num_classes=spec.num_classes,
        mechanism=spec.mechanism,
        cell_layers=spec.cell_layers,
        keep_bias=spec.keep_bias,
        latch_kind=spec.latch_kind,
        hard_state=spec.hard_state,
        hard_control=spec.hard_control,
        tau=spec.tau,
        device="cpu",
    )
    # Replay the training run's CPU-stream draws: set_seed, then (randperm, randperm)
    # per LogicLayer in construction order. get_task consumed nothing (see docstring).
    utils.set_seed(spec.seed)
    for layer in (m for m in model.modules() if isinstance(m, LogicLayer)):
        layer.indices = _replay_connections(layer.in_dim, layer.out_dim)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.cell.hard_alpha = 1.0  # deployed-consistent (train.py sets this before final eval)
    return model


def build_task(spec: RunSpec):
    """Regenerate the run's task; same seed ⇒ bit-identical train/val/test tensors."""
    return get_task(
        spec.task,
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
        alphabet=spec.alphabet,
        write_flag=spec.sel_flag,
        n_distractors=spec.n_distractors,
        seed=spec.seed,
    )


def check_accuracy(model, spec: RunSpec, task=None, max_batches: int | None = None) -> dict:
    """The replay gate: discrete test accuracy of the rebuilt model vs the recorded one."""
    task = task or build_task(spec)
    if max_batches is None:
        acc = evaluate(model, task.test_loader, "cpu", discrete=True)
    else:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(task.test_loader):
                if i >= max_batches:
                    break
                preds = model(x.round()).argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
    ok = spec.test_acc is None or max_batches is not None or abs(acc - spec.test_acc) < 1e-9
    return {"rebuilt_test_acc": acc, "recorded_test_acc": spec.test_acc,
            "full_test_set": max_batches is None, "gate_passed": ok}
