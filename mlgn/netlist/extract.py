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

Task coverage
-------------
Besides copy/selcopy/distcopy/parity, this module also rebuilds and exports the CAN-bus
IDS checkpoints (``can`` / ``can-syn``; research/23 §C0.g) — the CAN-as-verification-
carrier bridge. Those are self-contained new-format checkpoints (wiring in ``conn_a``/
``conn_b`` buffers, so the RNG replay is a harmless no-op), and their ``input_dim`` /
``num_classes`` + fitted encoder config ride along in the results JSON so ``build_task``
regenerates the exact (seed-independent, bit-reproducible for ``can-syn``) dataset. Two
deployment shapes are handled: the recurrent arms (clatch/gated) → the MUX-register FSM
(``ir.extract_netlist``); the stateless ``ff`` arm → a COMBINATIONAL netlist, no latches
(``extract_netlist_ff`` + ``ff_equivalence_check`` at the bottom of this file). Use the
``extract_model_netlist`` / ``netlist_equivalence_check`` dispatchers to cover both.
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

import numpy as np  # noqa: E402

from mlgn.seqlgn import utils  # noqa: E402  (installs the CPU shim for difflogic)
from mlgn.seqlgn.data import get_task  # noqa: E402
from mlgn.seqlgn.models import SequenceClassifier  # noqa: E402
from mlgn.seqlgn.train import evaluate  # noqa: E402
from mlgn.netlist.ir import GATE_FN, NetlistBuilder, _add_logic_mlp  # noqa: E402
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
    keep_bias: float | None    # None for the stateless 'ff' arm (no gate MLP)
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
    # Explicit dims: the CAN tasks record input_dim/num_classes in the JSON (train.py) so
    # the exporter need not re-derive them from a task formula. None => derive from alphabet.
    input_dim_override: int | None = None
    num_classes_override: int | None = None
    # CAN task-rebuild knobs (from the results JSON). build_task replays get_task with the
    # SAME encoder config so the regenerated dataset is bit-identical (same input_dim, same
    # seed-independent split) — the accuracy gate needs an exactly-reproduced test set.
    can_window: int | None = None     # sliding-window length (== recorded can_window; the
                                      # recorded seq_len is 1 for the flattened 'ff' arm)
    can_source: str = "syn"
    can_file: str = ""
    can_attack: str = "all"
    can_stride: int = 1
    can_eval_stride: int = 1
    can_id_enc: str = "onehot"
    can_top_ids: int = 20
    can_dt_bins: int = 8
    can_dt_global: bool = False
    can_payload_bytes: int = 0
    can_per_step: bool = False
    can_flatten: bool = False
    can_ambient: bool = False

    @property
    def input_dim(self) -> int:
        if self.input_dim_override is not None:
            return self.input_dim_override
        if self.task in ("copy", "selcopy", "distcopy"):
            return self.alphabet + 1
        if self.task == "parity":
            return 1
        raise ValueError(f"input_dim unknown for task {self.task!r} (add it here)")

    @property
    def num_classes(self) -> int:
        if self.num_classes_override is not None:
            return self.num_classes_override
        if self.task in ("copy", "selcopy", "distcopy"):
            return self.alphabet
        if self.task == "parity":
            return 2
        raise ValueError(f"num_classes unknown for task {self.task!r} (add it here)")

    @property
    def is_can(self) -> bool:
        return self.task in ("can", "can-syn")


def spec_from_json(json_path: str, alphabet: int = 8, n_distractors: int = 8,
                   sel_flag: bool = False) -> RunSpec:
    with open(json_path) as f:
        r = json.load(f)
    common = dict(
        task=r["task"],
        mechanism=r["mechanism"],
        seq_len=r["seq_len"],
        hidden=r["hidden"],
        cell_layers=r["cell_layers"],
        keep_bias=r.get("keep_bias"),   # None for 'ff' (recorded null; no gate MLP)
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
    if r["task"] in ("can", "can-syn"):
        # CAN carries its input_dim/num_classes + the fitted encoder config in the JSON
        # (train.py persists them precisely so the exporter needs no task formula). The
        # window length lives in `can_window` (the recorded `seq_len` is 1 for the
        # flattened 'ff' arm, so it CANNOT be reused as the window here).
        return RunSpec(
            **common,
            input_dim_override=r["input_dim"],
            num_classes_override=r["num_classes"],
            can_window=r.get("can_window") or r["seq_len"],
            can_source=r.get("can_source") or ("syn" if r["task"] == "can-syn" else "road"),
            can_file=r.get("can_file") or "",
            can_attack=r.get("can_attack") or "all",
            can_stride=r.get("can_stride") or 1,
            can_eval_stride=r.get("can_eval_stride") or 1,
            can_id_enc=r.get("can_id_enc") or "onehot",
            can_top_ids=r["can_top_ids"] if r.get("can_top_ids") is not None else 20,
            can_dt_bins=r["can_dt_bins"] if r.get("can_dt_bins") is not None else 8,
            can_dt_global=bool(r.get("can_dt_global")),
            can_payload_bytes=r.get("can_payload_bytes") or 0,
            can_per_step=bool(r.get("can_per_step")),
            can_flatten=bool(r.get("can_flatten")),
            can_ambient=bool(r.get("can_ambient")),
        )
    return RunSpec(**common)


def _replay_connections(in_dim: int, out_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact replay of LogicLayer.get_connections('random') on the global CPU RNG."""
    c = torch.randperm(2 * out_dim) % in_dim
    c = torch.randperm(in_dim)[c]
    c = c.reshape(2, out_dim)
    return c[0].to(torch.int64), c[1].to(torch.int64)


def rebuild_model(spec: RunSpec, ckpt_path: str) -> SequenceClassifier:
    """Construct the model on CPU, replay the training-time wiring, load the weights.

    For the self-contained NEW-format checkpoints (wiring saved as ``conn_a``/``conn_b``
    buffers — every CAN checkpoint, and every DUST run since 2026-07-10) the RNG replay
    below is a harmless no-op: ``load_state_dict`` restores the wiring from the buffers,
    overriding whatever the replay wrote (verified by ``test_indices_buffers``). It is
    kept only so PRE-fix (replay-only) copy/parity checkpoints still reconstruct here.
    """
    model = SequenceClassifier(
        input_dim=spec.input_dim,
        hidden_dim=spec.hidden,
        num_classes=spec.num_classes,
        mechanism=spec.mechanism,
        cell_layers=spec.cell_layers,
        # 'ff' has no gate MLP so keep_bias is unused (recorded null) — pass a valid float
        # anyway; the SequenceClassifier ctor stores it but never applies it for 'ff'.
        keep_bias=spec.keep_bias if spec.keep_bias is not None else 0.0,
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
    """Regenerate the run's task; same config ⇒ bit-identical train/val/test tensors.

    CAN uses a seed-INDEPENDENT time/capture split and a dedicated (fixed-seed) generator
    for the synthetic stream, so `seed` is deliberately NOT passed (mirrors train.py) — the
    regenerated dataset is identical across arms/seeds and reproduces the recorded test set.
    """
    if spec.is_can:
        return get_task(
            spec.task,
            batch_size=spec.batch_size,
            seq_len=spec.can_window,   # the WINDOW length (recorded seq_len==1 when flattened)
            can_source=spec.can_source,
            can_file=spec.can_file,
            can_attack=spec.can_attack,
            can_stride=spec.can_stride,
            can_eval_stride=spec.can_eval_stride,
            can_id_enc=spec.can_id_enc,
            can_top_ids=spec.can_top_ids,
            can_dt_bins=spec.can_dt_bins,
            can_dt_global=spec.can_dt_global,
            can_payload_bytes=spec.can_payload_bytes,
            can_per_step=spec.can_per_step,
            can_flatten=spec.can_flatten,
            can_ambient=spec.can_ambient,
        )
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
    """The replay gate: discrete test accuracy of the rebuilt model vs the recorded one.

    For the CAN tasks the recorded ``test_acc`` is discrete detection accuracy; the
    synthetic ``can-syn`` test set is bit-reproducible so this gate is exact. For REAL
    ``can`` the test set is data-dependent (raw captures may be absent at export time), so
    callers gate on netlist bit-exactness alone and defer this accuracy gate (see the test).
    """
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
                if y.ndim == 2:            # CAN per-step labels: score the last (target) frame
                    y = y[:, -1]
                preds = model(x.round()).argmax(-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
    ok = spec.test_acc is None or max_batches is not None or abs(acc - spec.test_acc) < 1e-9
    return {"rebuilt_test_acc": acc, "recorded_test_acc": spec.test_acc,
            "full_test_set": max_batches is None, "gate_passed": ok}


# =====================================================================================
# 'ff' (stateless) export — a COMBINATIONAL netlist (no latches)
# =====================================================================================
# The recurrent arms (gated/clatch/combo/latch/rddlgn) go through ir.extract_netlist,
# which builds the MUX-register FSM and is checked by sim.equivalence_check. The 'ff'
# control arm has NO recurrence: h = LogicMLP(x_last), so it deploys to a pure
# combinational netlist (``n_state == 0``, no registers). ir.extract_netlist/sim assume
# >= 1 state register (the head popcounts STATE bits, run_sequence clocks the FSM), so the
# combinational case is exported + gated here instead. A strictly-latch-free emission all
# the way to BLIF/Verilog would additionally need an ``n_state==0`` path in ir/sim/blif —
# out of the extract.py lane; that is the (narrow, documented) remaining blocker.

def extract_netlist_ff(model):
    """SequenceClassifier(mechanism='ff') → COMBINATIONAL Netlist (no latches).

    The hidden bits are combinational OUTPUTS (``net.outputs``) feeding the GroupSum head;
    there is nothing to register. Uses the public ir.NetlistBuilder (ir.py is untouched).
    """
    cell = model.cell
    assert cell.mechanism == "ff", f"extract_netlist_ff needs mechanism='ff', got {cell.mechanism!r}"
    n_pi = cell.input_dim
    b = NetlistBuilder(n_pi, n_state=0)              # no latches
    z = [b.pi(i) for i in range(n_pi)]               # ff reads x ALONE (no state feedback)
    out = _add_logic_mlp(b, cell.update, z)          # combinational hidden bits
    k = model.num_classes
    assert cell.hidden_dim % k == 0, (cell.hidden_dim, k)
    return b.build(next_state=[], outputs=out, head=(k, cell.hidden_dim // k))


def ff_forward(net, x: np.ndarray):
    """Evaluate a combinational (n_state==0) Netlist. x: [B, n_pi] bool ->
    (preds [B], out_bits [B, hidden]). Mirrors sim.step's vectorized gate eval; kept here
    because sim.run_sequence/head_scores assume >= 1 state register (see the note above)."""
    assert net.n_state == 0, "ff_forward is for combinational netlists only"
    B = x.shape[0]
    sig = np.empty((B, net.n_signals), dtype=bool)
    sig[:, 0] = False
    sig[:, 1] = True
    sig[:, 2:2 + net.n_pi] = x
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        a = sig[:, net.src_a[s:e]]
        bb = sig[:, net.src_b[s:e]]
        out = np.empty_like(a)
        ops = net.ops[s:e]
        for code in np.unique(ops):
            m = ops == code
            out[:, m] = GATE_FN[int(code)](a[:, m], bb[:, m])
        sig[:, base + s:base + e] = out
    out_bits = sig[:, net.outputs]
    k, gs = net.head
    return out_bits.reshape(B, k, gs).sum(-1).argmax(-1), out_bits


@torch.no_grad()
def ff_equivalence_check(model, net, loader, max_batches: int | None = None,
                         trajectory_batches: int = 2) -> dict:
    """Bit-exact gate for the combinational ff netlist vs the torch model (the
    sim.equivalence_check contract: identical predictions on every batch + identical
    pre-head hidden bits). ff is stateless, so the hidden-bit vector plays the role of the
    per-timestep state trajectory."""
    model.eval()
    n = mismatched_preds = 0
    bits_checked = bits_equal = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xb = x.round()
        x_last = xb[:, -1, :]                         # ff output depends only on the last frame
        h = model.cell.readout_h(model.cell(x_last, None))   # ff ignores `state`
        hbits = h.numpy().astype(bool)
        torch_preds = model.head(h).argmax(-1).numpy()
        sim_preds, out_bits = ff_forward(net, x_last.numpy().astype(bool))
        mismatched_preds += int((sim_preds != torch_preds).sum())
        if i < trajectory_batches:
            bits_checked += hbits.size
            bits_equal += int((hbits == out_bits).sum())
        n += len(torch_preds)
    return {
        "samples": n,
        "mismatched_predictions": mismatched_preds,
        "trajectory_bits_checked": bits_checked,
        "trajectory_bits_mismatched": bits_checked - bits_equal,
        "bit_exact": mismatched_preds == 0 and bits_checked == bits_equal,
    }


def extract_model_netlist(model):
    """Dispatch to the right extractor: 'ff' → combinational (extract_netlist_ff), every
    other (recurrent) mechanism → ir.extract_netlist (the MUX-register FSM)."""
    if model.cell.mechanism == "ff":
        return extract_netlist_ff(model)
    from mlgn.netlist.ir import extract_netlist
    return extract_netlist(model)


def netlist_equivalence_check(model, net, loader, **kwargs) -> dict:
    """Dispatch the bit-exact gate: 'ff' → ff_equivalence_check (combinational),
    recurrent → sim.equivalence_check (clocked FSM)."""
    if model.cell.mechanism == "ff":
        return ff_equivalence_check(model, net, loader, **kwargs)
    from mlgn.netlist.sim import equivalence_check
    return equivalence_check(model, net, loader, **kwargs)
