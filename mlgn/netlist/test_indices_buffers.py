"""
test_indices_buffers.py — tests for persistent LogicLayer wiring (conn_a/conn_b).
=================================================================================

``LogicLayer.indices`` is now a property backed by two persistent int64 buffers
``conn_a`` / ``conn_b`` (see difflogic/difflogic.py), fixing research/20 footgun #2:
the random wiring travels inside the checkpoint instead of having to be replayed
from the construction-time RNG state.

Semantics covered here:

1. OLD-format checkpoints (no conn_* keys) still load under strict=True; the layer
   KEEPS the wiring it was constructed with (exactly the pre-change behaviour — this
   is why the extract.py RNG-replay path keeps working: replay wiring, then load an
   old checkpoint, and the replayed wiring survives).
2. NEW-format checkpoints carry the wiring and RESTORE it on load, overriding
   whatever wiring the receiving model was constructed with (self-contained
   checkpoints; also means --init-from with a new-format checkpoint now warm-starts
   the wiring, not just the weights).
3. state_dict of any model contains conn_a/conn_b for every LogicLayer, and shape
   mismatches on those buffers still error normally.
4. End-to-end regression: the falsify pipeline (old checkpoint + RNG replay) is
   bit-exact unchanged.
5. Training (including --save-model / --init-from warm-start) still runs.

Run from the repo root:

    python -m mlgn.netlist.test_indices_buffers            # everything
    python -m mlgn.netlist.test_indices_buffers --fast     # skip the slow subprocess tests (4/5/6)
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import subprocess
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.seqlgn import utils  # noqa: E402,F401  (installs the CPU shim for difflogic)
from mlgn.seqlgn.models import SequenceClassifier  # noqa: E402
from difflogic import LogicLayer  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "mlgn", "seqlgn", "results")


def tiny_model(seed: int) -> SequenceClassifier:
    """The tiny reference model: input 4, hidden 32, 2 classes, gated, cpu."""
    torch.manual_seed(seed)
    return SequenceClassifier(input_dim=4, hidden_dim=32, num_classes=2,
                              mechanism="gated", device="cpu")


def logic_layers(model):
    return [m for m in model.modules() if isinstance(m, LogicLayer)]


def wiring_of(model):
    return [(l.conn_a.clone(), l.conn_b.clone()) for l in logic_layers(model)]


def wiring_equal(w1, w2) -> bool:
    return all(torch.equal(a1, a2) and torch.equal(b1, b2)
               for (a1, b1), (a2, b2) in zip(w1, w2, strict=True))


def strip_conn_keys(sd):
    """Simulate an OLD-format checkpoint: drop every conn_a/conn_b entry."""
    return {k: v.clone() for k, v in sd.items()
            if not k.rsplit(".", 1)[-1].startswith("conn_")}


# --- 1. old-format load under strict=True -------------------------------------------

def test_old_format_load_strict():
    m1 = tiny_model(seed=10)
    old_sd = strip_conn_keys(m1.state_dict())
    assert not any("conn_" in k for k in old_sd), "strip failed"

    m2 = tiny_model(seed=11)                      # different wiring AND weights
    own = wiring_of(m2)
    assert not wiring_equal(own, wiring_of(m1)), "seeds 10/11 gave identical wiring?!"

    m2.load_state_dict(old_sd)                    # strict=True (the default) must succeed

    # weights came from the checkpoint...
    for k, v in m2.state_dict().items():
        if k in old_sd:
            assert torch.equal(v, old_sd[k]), f"{k} not loaded"
    # ...but the wiring stays the RECEIVING model's own (old checkpoints carry none).
    assert wiring_equal(wiring_of(m2), own), "old-format load must keep constructed wiring"
    assert not wiring_equal(wiring_of(m2), wiring_of(m1))
    print("[PASS] 1. old-format checkpoint loads under strict=True; constructed wiring kept")


# --- 2. new-format roundtrip ---------------------------------------------------------

def test_new_format_roundtrip():
    m1 = tiny_model(seed=1)
    buf = io.BytesIO()
    torch.save(m1.state_dict(), buf)

    m2 = tiny_model(seed=2)
    assert not wiring_equal(wiring_of(m2), wiring_of(m1)), "seeds 1/2 gave identical wiring?!"

    # simulate an extract.py-style replay FIRST: the checkpoint must still win.
    lay = logic_layers(m2)[0]
    torch.manual_seed(123)
    lay.indices = (torch.randint(0, lay.in_dim, (lay.out_dim,)),
                   torch.randint(0, lay.in_dim, (lay.out_dim,)))

    buf.seek(0)
    m2.load_state_dict(torch.load(buf, weights_only=True))

    assert wiring_equal(wiring_of(m2), wiring_of(m1)), \
        "new-format load must restore the checkpoint's wiring"

    # eval-mode outputs bit-identical on random binary inputs
    m1.eval(), m2.eval()
    torch.manual_seed(99)
    x = torch.randint(0, 2, (16, 6, 4)).float()
    with torch.no_grad():
        y1, y2 = m1(x), m2(x)
    assert torch.equal(y1, y2), "outputs differ after new-format roundtrip"
    print("[PASS] 2. new-format roundtrip: wiring restored (overriding a replay), outputs bit-identical")


# --- 3. state_dict contents + setter sync + shape errors ------------------------------

def test_state_dict_and_setter():
    m = tiny_model(seed=3)
    sd = m.state_dict()
    n = 0
    for name, mod in m.named_modules():
        if isinstance(mod, LogicLayer):
            assert f"{name}.conn_a" in sd, f"missing {name}.conn_a"
            assert f"{name}.conn_b" in sd, f"missing {name}.conn_b"
            assert sd[f"{name}.conn_a"].dtype == torch.int64
            assert sd[f"{name}.conn_b"].dtype == torch.int64
            n += 1
    assert n == 4, f"expected 4 LogicLayers in the tiny gated model, found {n}"

    # the setter (extract.py path: layer.indices = (a, b)) must sync the buffers
    lay = logic_layers(m)[0]
    a = torch.randint(0, lay.in_dim, (lay.out_dim,), dtype=torch.int64)
    b = torch.randint(0, lay.in_dim, (lay.out_dim,), dtype=torch.int64)
    lay.indices = (a, b)
    assert torch.equal(lay.conn_a, a) and torch.equal(lay.conn_b, b)
    ia, ib = lay.indices
    assert torch.equal(ia, a) and torch.equal(ib, b)
    key = [k for k in m.state_dict() if k.endswith("conn_a")][0]
    assert torch.equal(m.state_dict()[key], a), "setter did not sync the persistent buffer"

    # shape mismatches on conn buffers must still error normally
    bad = {k: v.clone() for k, v in m.state_dict().items()}
    bad[key] = torch.zeros(5, dtype=torch.int64)
    try:
        tiny_model(seed=4).load_state_dict(bad)
    except RuntimeError as e:
        assert "size mismatch" in str(e) or "shape" in str(e), str(e)
    else:
        raise AssertionError("shape-mismatched conn_a loaded without error")

    # the SETTER must also reject shape mismatches atomically: no silent scalar
    # broadcast, and no half-updated (a written, b not) wiring after the error.
    snap_a, snap_b = lay.conn_a.clone(), lay.conn_b.clone()
    for bad_pair in [
        (torch.zeros(5, dtype=torch.int64), torch.zeros(5, dtype=torch.int64)),   # wrong length
        (torch.tensor(1, dtype=torch.int64), torch.tensor(2, dtype=torch.int64)),  # scalar broadcast
        (torch.zeros(lay.out_dim, dtype=torch.int64), torch.zeros(3, dtype=torch.int64)),  # good a, bad b
    ]:
        try:
            lay.indices = bad_pair
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"setter accepted mismatched shapes {tuple(t.shape for t in bad_pair)}")
        assert torch.equal(lay.conn_a, snap_a) and torch.equal(lay.conn_b, snap_b), \
            "failed setter assignment mutated the buffers"

    # a HALF-present checkpoint (conn_a without conn_b) is corrupt, not old-format:
    # strict=True must report the absent key instead of silently mixing wirings.
    half = {k: v.clone() for k, v in m.state_dict().items()
            if k.rsplit(".", 1)[-1] != "conn_b"}
    try:
        tiny_model(seed=4).load_state_dict(half)
    except RuntimeError as e:
        assert "conn_b" in str(e), str(e)
    else:
        raise AssertionError("half-present (conn_a only) checkpoint loaded under strict=True")
    res = tiny_model(seed=4).load_state_dict(half, strict=False)
    assert all(k.rsplit(".", 1)[-1] == "conn_b" for k in res.missing_keys), res.missing_keys
    print("[PASS] 3. conn_a/conn_b in state_dict for all 4 LogicLayers; setter syncs; "
          "shape mismatches (load AND setter) error atomically; half-present ckpt errors")


# --- 3b. CUDA backward-helper indices stay in sync with the wiring --------------------

def _expected_given_x(layer):
    """Reference recompute of the CUDA backward helper from the current wiring."""
    given = [[] for _ in range(layer.in_dim)]
    a, b = layer.indices[0].tolist(), layer.indices[1].tolist()
    for y in range(layer.out_dim):
        given[a[y]].append(y)
        given[b[y]].append(y)
    start = [0]
    for g in given:
        start.append(start[-1] + len(g))
    flat = [i for g in given for i in g]
    return torch.tensor(start, dtype=torch.int64), torch.tensor(flat, dtype=torch.int64)


def test_cuda_helper_recompute():
    """implementation='cuda' precomputes given_x_indices_of_y from the wiring; it must be
    refreshed when the wiring changes (setter or new-format load), else CUDA training
    after a warm-start would backprop through stale scatter indices. The precompute is
    pure torch/numpy, so it is testable on this CPU-only host."""
    torch.manual_seed(5)
    l1 = LogicLayer(8, 16, device="cpu", implementation="cuda")
    s, f = _expected_given_x(l1)
    assert torch.equal(l1.given_x_indices_of_y_start, s) and torch.equal(l1.given_x_indices_of_y, f)

    # setter path
    torch.manual_seed(6)
    l1.indices = (torch.randint(0, 8, (16,)), torch.randint(0, 8, (16,)))
    s, f = _expected_given_x(l1)
    assert torch.equal(l1.given_x_indices_of_y_start, s) and torch.equal(l1.given_x_indices_of_y, f)

    # new-format load path
    torch.manual_seed(7)
    l2 = LogicLayer(8, 16, device="cpu", implementation="cuda")
    l2.load_state_dict(l1.state_dict())
    assert torch.equal(l2.conn_a, l1.conn_a) and torch.equal(l2.conn_b, l1.conn_b)
    s, f = _expected_given_x(l2)
    assert torch.equal(l2.given_x_indices_of_y_start, s) and torch.equal(l2.given_x_indices_of_y, f)
    print("[PASS] 3b. cuda backward-helper indices refreshed on wiring reassign and new-format load")


# --- 4/5/6. subprocess tests ----------------------------------------------------------

def _run(cmd, timeout):
    print(f"       $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:])
        raise AssertionError(f"exit code {res.returncode}: {' '.join(cmd)}")
    return res.stdout + res.stderr


def _cleanup_results(out: str):
    """Delete the results JSON a train run printed ('results -> path')."""
    m = re.search(r"results -> (\S+)", out)
    if m and os.path.exists(m.group(1)):
        os.remove(m.group(1))


def test_falsify_regression():
    """The critical end-to-end regression: old checkpoint + RNG replay, bit-exact."""
    out = _run([sys.executable, "-m", "mlgn.netlist.falsify",
                "--ckpt", "mlgn/seqlgn/results/ckpt_cp50A_curr_c35.pt",
                "--json", "mlgn/seqlgn/results/copy_combo_cp50A_curr_c35_20260704-023800.json",
                "--eval-batches", "8", "--equiv-batches", "2", "--skip-abc",
                "--out", "mlgn/netlist/out/a3_regression"], timeout=540)
    assert "rebuilt=1.0000" in out, "accuracy gate did not print rebuilt=1.0000"
    assert "bit_exact=True" in out, "equivalence check did not print bit_exact=True"
    print("[PASS] 4. falsify regression: rebuilt=1.0000, bit_exact=True (old ckpt + replay path intact)")


def test_train_smoke():
    out = _run([sys.executable, "-m", "mlgn.seqlgn.train",
                "--task", "parity", "--seq-len", "8", "--mechanism", "gated",
                "--hidden", "20", "--iters", "30", "--eval-freq", "10",
                "--batch-size", "32", "--device", "cpu"], timeout=300)
    assert "results ->" in out, "train did not write a results record"
    _cleanup_results(out)
    print("[PASS] 5. train smoke: parity/gated 30 iters on cpu ran to completion")


def test_init_from_smoke():
    ckpt = os.path.join(RESULTS_DIR, "ckpt_buffertest.pt")
    base = [sys.executable, "-m", "mlgn.seqlgn.train",
            "--task", "parity", "--seq-len", "8", "--mechanism", "gated",
            "--hidden", "20", "--iters", "30", "--eval-freq", "10",
            "--batch-size", "32", "--device", "cpu", "--tag", "buffertest"]
    try:
        out1 = _run(base + ["--save-model"], timeout=300)
        _cleanup_results(out1)
        assert os.path.exists(ckpt), "ckpt_buffertest.pt was not saved"
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert any(k.endswith("conn_a") for k in sd), "saved checkpoint is not new-format"

        out2 = _run(base + ["--init-from", ckpt], timeout=300)
        _cleanup_results(out2)
        assert "[init] warm-started from" in out2, "--init-from did not warm-start"
        print("[PASS] 6. --save-model + --init-from smoke: new-format ckpt saved and warm-started cleanly")
    finally:
        if os.path.exists(ckpt):
            os.remove(ckpt)
        for junk in glob.glob(os.path.join(RESULTS_DIR, "*buffertest*.json")):
            os.remove(junk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the slow subprocess tests (4/5/6)")
    args = ap.parse_args()

    test_old_format_load_strict()
    test_new_format_roundtrip()
    test_state_dict_and_setter()
    test_cuda_helper_recompute()
    if not args.fast:
        test_falsify_regression()
        test_train_smoke()
        test_init_from_smoke()
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
