"""
test_can_extract.py — CAN checkpoint → netlist export + bit-exact gate.
=======================================================================

Covers the CAN-as-verification-carrier bridge (research/23 §C0.g; the P3b hook in
netlist/README's "Next steps"): a trained CAN checkpoint must rebuild from its results
JSON and export to a gate netlist that matches the torch model BIT-FOR-BIT — the same
house correctness gate the copy/distcopy exporter already passes, now for the two CAN
deployment shapes:

  * clatch (recurrent)  → the MUX-register FSM (ir.extract_netlist + sim.equivalence_check),
  * ff     (stateless)  → a COMBINATIONAL netlist, no latches (extract.extract_netlist_ff +
                          extract.ff_equivalence_check).

Because ``can-syn`` is a zero-file, fixed-seed synthetic stream with a seed-INDEPENDENT
time/capture split, the regenerated test set is bit-identical to training, so the FULL
accuracy gate runs (rebuilt discrete test_acc == recorded). For REAL ``can`` the test set
is data-dependent (raw captures may be absent at export time), so that path gates on
netlist bit-exactness over arbitrary input batches only — recorded here as a NOTE, not a
skipped assertion, because we have no real capture in-tree.

Each bit-exact gate is paired with a MUTATION control (flip one output/next-state gate to
its truth-table complement) that MUST break bit-exactness — otherwise a vacuous checker
would "pass" trivially.

Prereq: the two tiny checkpoints exist under mlgn/seqlgn/results/. Regenerate with:

    python -m mlgn.seqlgn.train --task can-syn --mechanism clatch --seq-len 8 \
        --hidden 16 --cell-layers 2 --iters 300 --eval-freq 100 --batch-size 64 \
        --device cpu --seed 0 --save-model --tag can_syn_clatch
    python -m mlgn.seqlgn.train --task can-syn --mechanism ff --can-flatten --seq-len 8 \
        --hidden 64 --cell-layers 2 --iters 300 --eval-freq 100 --batch-size 64 \
        --device cpu --seed 0 --save-model --tag can_syn_ff

Run from the repo root:

    python -m mlgn.netlist.test_can_extract
"""

from __future__ import annotations

import copy
import glob
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.seqlgn import utils  # noqa: E402,F401  (installs the CPU shim for difflogic)
from mlgn.netlist.extract import (  # noqa: E402
    build_task, check_accuracy, extract_model_netlist, netlist_equivalence_check,
    rebuild_model, spec_from_json,
)

RESULTS_DIR = os.path.join(_ROOT, "mlgn", "seqlgn", "results")


def _find(ckpt_name: str, json_glob: str):
    """Locate the deterministic checkpoint path + the newest matching results JSON."""
    ckpt = os.path.join(RESULTS_DIR, ckpt_name)
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, json_glob)), key=os.path.getmtime)
    if not os.path.exists(ckpt) or not matches:
        raise FileNotFoundError(
            f"missing CAN test artifacts: ckpt={ckpt} (exists={os.path.exists(ckpt)}), "
            f"json glob {json_glob!r} matched {len(matches)}. Train the tiny checkpoints "
            f"first — see the command block in this file's docstring.")
    return ckpt, matches[-1]


def _mutate(net):
    """Return a deep copy of ``net`` with ONE output-feeding gate flipped to its
    truth-table complement (op -> 15-op), which inverts that bit for EVERY input — a
    guaranteed break so the bit-exact gate is provably non-vacuous."""
    m = copy.deepcopy(net)
    base = 2 + m.n_pi + m.n_state
    driver = (m.next_state[0] if m.n_state else m.outputs[0])   # a signal that reaches the head
    g = int(driver) - base
    assert 0 <= g < m.n_gates, (driver, g, m.n_gates)   # a gate (not a PI/latch/const)
    m.ops[g] = 15 - int(m.ops[g])
    return m


def _run_arm(name: str, ckpt_name: str, json_glob: str, expect_mech: str,
             expect_latches: bool, full_accuracy_gate: bool):
    ckpt, jpath = _find(ckpt_name, json_glob)
    spec = spec_from_json(jpath)
    print(f"\n=== {name}: {os.path.basename(jpath)} ===")
    print(f"    task={spec.task} mech={spec.mechanism} hidden={spec.hidden} "
          f"input_dim={spec.input_dim} num_classes={spec.num_classes} "
          f"can_window={spec.can_window} flatten={spec.can_flatten} "
          f"recorded_test_acc={spec.test_acc}")
    assert spec.mechanism == expect_mech, (spec.mechanism, expect_mech)
    assert spec.is_can, spec.task

    model = rebuild_model(spec, ckpt)
    task = build_task(spec)

    # regeneration sanity: the rebuilt task reproduces the recorded input_dim exactly
    # (same encoder config => same feature width => the checkpoint's first layer fits).
    assert task.input_dim == spec.input_dim, (task.input_dim, spec.input_dim)
    assert model.cell.input_dim == spec.input_dim, (model.cell.input_dim, spec.input_dim)

    # --- accuracy gate (can-syn only; real can defers this — see module docstring) -----
    if full_accuracy_gate:
        gate = check_accuracy(model, spec, task=task)
        print(f"    accuracy gate: rebuilt={gate['rebuilt_test_acc']:.6f} "
              f"recorded={gate['recorded_test_acc']} passed={gate['gate_passed']}")
        assert gate["gate_passed"], f"accuracy gate FAILED: {gate}"
    else:
        print("    accuracy gate: DEFERRED (real can test set is data-dependent; "
              "gating on bit-exactness only)")

    # --- netlist extraction + shape sanity ---------------------------------------------
    net = extract_model_netlist(model)
    if expect_latches:
        assert net.n_state == model.cell.hidden_dim, (net.n_state, model.cell.hidden_dim)
        assert net.n_pi == spec.input_dim, (net.n_pi, spec.input_dim)
        print(f"    netlist: FSM  n_pi={net.n_pi} n_state={net.n_state} "
              f"gates={net.n_gates} head={net.head}")
    else:
        assert net.n_state == 0, f"ff must be latch-free, got n_state={net.n_state}"
        assert len(net.outputs) == model.cell.hidden_dim, (len(net.outputs), model.cell.hidden_dim)
        assert net.n_pi == spec.input_dim, (net.n_pi, spec.input_dim)
        print(f"    netlist: COMBINATIONAL (no latches)  n_pi={net.n_pi} "
              f"outputs={len(net.outputs)} gates={net.n_gates} head={net.head}")

    # --- bit-exact gate (the house rule) -----------------------------------------------
    eq = netlist_equivalence_check(model, net, task.test_loader, trajectory_batches=4)
    print(f"    bit-exact gate: samples={eq['samples']} "
          f"mism_preds={eq['mismatched_predictions']} "
          f"traj_bits={eq['trajectory_bits_checked']} "
          f"traj_mism={eq['trajectory_bits_mismatched']} bit_exact={eq['bit_exact']}")
    assert eq["bit_exact"], f"bit-exact gate FAILED: {eq}"
    assert eq["samples"] > 0 and eq["trajectory_bits_checked"] > 0, eq

    # --- mutation control: a flipped gate MUST break the gate (non-vacuity) -------------
    bad = netlist_equivalence_check(model, _mutate(net), task.test_loader, trajectory_batches=4)
    print(f"    mutation control: bit_exact={bad['bit_exact']} "
          f"(mism_preds={bad['mismatched_predictions']} traj_mism={bad['trajectory_bits_mismatched']})")
    assert not bad["bit_exact"], "mutation control did NOT break bit-exactness — gate is vacuous!"

    print(f"[PASS] {name}: rebuild + accuracy + bit-exact + non-vacuity all green")
    return eq


def test_clatch_can_syn():
    """Recurrent arm: MUX-register FSM, full accuracy + bit-exact gates."""
    return _run_arm(
        "clatch-can-syn", "ckpt_can_syn_clatch.pt", "can-syn_clatch_*.json",
        expect_mech="clatch", expect_latches=True, full_accuracy_gate=True)


def test_ff_can_syn():
    """Stateless arm: COMBINATIONAL netlist (no latches), full accuracy + bit-exact gates."""
    return _run_arm(
        "ff-can-syn", "ckpt_can_syn_ff.pt", "can-syn_ff_*.json",
        expect_mech="ff", expect_latches=False, full_accuracy_gate=True)


def main() -> int:
    test_clatch_can_syn()
    test_ff_can_syn()
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
