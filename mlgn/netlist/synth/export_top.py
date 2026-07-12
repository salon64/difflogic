"""
export_top.py — the head-in-fabric deployable top (P3b T1): lgn_top = FSM + head.
==================================================================================

Composes the verified copy-FSM (``ckpt_cp50A_curr_c35``, see export_fsm.py and
../README.md) with the in-netlist GroupSum argmax head (head.argmax_bits) into ONE
netlist whose only outputs are the 3-bit predicted class — killing the 1,035-port
I/O blocker of the fsm-only export (ports drop to clk + rst + x[8:0] + class_out[2:0]
= 14 bits). Emits into this directory:

    top.v                  synthesizable Verilog-2001, module lgn_top; q is INTERNAL
                           (expose_q=False), outputs are one class_out[2:0] bus
    top.blif / top_clk.blif BLIF twin (3 class outputs; every state bit stays
                           observable THROUGH the head) + the re-clk'd yosys variant
    golden_cls_sym<k>.txt  40 frames of the per-frame DECODED CLASS (one hex digit
                           per line) per legal write, from the bit-exact python
                           simulator — the transient's wrong decodes are kept
                           verbatim; only the settled tail must equal the symbol
    tb_top.v               iverilog TB: drives the 8 writes and compares class_out
                           cycle-exactly (ports only, so it re-runs unchanged on the
                           post-synthesis netlist); `ifdef RTL_Q_CHECK additionally
                           compares the full 1024-bit dut.q against golden_sym<k>.txt
                           (pre-synthesis only)
    top_mut_obs.v          mutation controls (single op 2->4 flip each), searched for
    top_mut_head.v         CLASS-observability by IR-level re-simulation: FSM-gate
    top_mut_masked.v       mutant + head-gate mutant must FAIL the TB, masked-gate
                           mutant must PASS (deleted by run_top.sh after evidence)

Gates, all before any RTL is written (abort on any failure):
    1. accuracy gate >= 0.999 (rebuilt model vs recorded, as export_fsm);
    2. netlist shape asserts: 9 PIs / 1024 latches / 7168 gates / head (8,128) /
       init all-zeros;
    3. emit_verilog regression: fsm.v regenerated with DEFAULT parameters must be
       byte-identical to the checked-in synth/fsm.v (the expose_q/out_bus params
       must not perturb the verified golden record);
    4. composition gate: over 8 legal writes x 40 frames AND 2 random-input
       sequences, the composed top's next-state trajectory must equal
       sim.run_sequence(net_base) bit-for-bit and its 3 output bits must equal
       sim.head_scores(net_base).argmax(-1) (np.argmax = first-max-wins) at every
       frame; plus the TORCH pipeline cross-check: per-frame model.head(h_t).argmax
       on the same 8 writes must match the netlist class trajectory.

Run from the repo root (Windows side), then bash run_top.sh inside WSL:

    python -m mlgn.netlist.synth.export_top            # quick accuracy gate (8 batches)
    python -m mlgn.netlist.synth.export_top --full-gate
"""

from __future__ import annotations

import argparse
import dataclasses
import filecmp
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.netlist import blif, sim  # noqa: E402
from mlgn.netlist.extract import (build_task, check_accuracy, rebuild_model,  # noqa: E402
                                  spec_from_json)
from mlgn.netlist.head import argmax_bits  # noqa: E402
from mlgn.netlist.ir import (CONST0, CONST1, Netlist, NetlistBuilder,  # noqa: E402
                             copy_gates_with_remap, extract_netlist)
from mlgn.netlist.synth.export_fsm import CKPT, FRAMES, JSON, derive_clk_blif  # noqa: E402
from mlgn.netlist.test_head import bits_to_int, eval_netlist  # noqa: E402
from mlgn.netlist.verilog import emit_verilog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHABET = 8
N_FSM_GATES = 7168          # the verified base netlist's gate count (asserted)


# -----------------------------------------------------------------------------------
# composition
# -----------------------------------------------------------------------------------
def compose_top(net: Netlist) -> Netlist:
    """FSM + argmax head in one netlist. The FSM gates are replayed FIRST in
    identical order (g0..g7167 keep their ids — the mutation precedent transfers);
    the head reads the CURRENT q latches in plain order (GroupSum group c = the
    contiguous slice q[c*gs..(c+1)*gs-1], exactly sim.head_scores' reshape), so
    class_out at frame t decodes state h_t — registered-q semantics identical to
    tb_fsm. Outputs = ceil(log2(k)) argmax bits, nothing else (no verification
    latches: props.protocol_decode's warm/legal/shadow machinery is NOT wanted in
    the deployable top)."""
    b = NetlistBuilder(net.n_pi, net.n_state, init=net.init)
    m = {CONST0: CONST0, CONST1: CONST1}
    m.update({net.pi(i): b.pi(i) for i in range(net.n_pi)})
    m.update({net.q(i): b.q(i) for i in range(net.n_state)})
    m = copy_gates_with_remap(b, net, m)
    bits = argmax_bits(b, net.head, [b.q(i) for i in range(net.n_state)])
    return b.build(next_state=[m[ns] for ns in net.next_state], outputs=bits,
                   head=net.head)


def write_seqs(n_pi: int, alphabet: int, frames: int) -> np.ndarray:
    """The 8 legal write sequences: cue + one-hot symbol at t=0, then blanks."""
    x = np.zeros((alphabet, frames, n_pi), dtype=bool)
    for s in range(alphabet):
        x[s, 0, s] = True
        x[s, 0, alphabet] = True
    return x


def run_top(top: Netlist, x_seq: np.ndarray):
    """Frame loop over the composed top. Returns (classes [B, T], states [T][B, n]):
    classes[:, t] is the output DURING frame t, i.e. the decode of the PRE-edge
    state h_t (h_0 = init); states[t] is the post-edge state h_{t+1}."""
    B, T, _ = x_seq.shape
    state = np.tile(np.asarray(top.init, dtype=bool), (B, 1))
    classes = np.zeros((B, T), dtype=np.int64)
    states = []
    for t in range(T):
        outs, state = eval_netlist(top, x_seq[:, t], state)
        classes[:, t] = bits_to_int(outs)
        states.append(state)
    return classes, states


def base_class_traj(net_base: Netlist, x_seq: np.ndarray) -> np.ndarray:
    """[B, T] per-frame decode of the base netlist: class of states[t] (post-edge),
    i.e. exactly what golden line t / class_out after edge t+1 must show."""
    _, states = sim.run_sequence(net_base, x_seq, return_trajectory=True)
    return np.stack([sim.head_scores(net_base, st).argmax(-1) for st in states], axis=1)


def composition_gate(net_base: Netlist, top: Netlist, model) -> None:
    """Bit-exactness gate (docstring at module top, gate 4). Raises on mismatch."""
    rng = np.random.default_rng(11)
    x_legal = write_seqs(net_base.n_pi, ALPHABET, FRAMES)
    x_rand = rng.random((2, FRAMES, net_base.n_pi)) < 0.5
    for name, x in (("8 legal writes", x_legal), ("2 random-input seqs", x_rand)):
        B = x.shape[0]
        _, gold_states = sim.run_sequence(net_base, x, return_trajectory=True)
        classes, states = run_top(top, x)
        prev = np.tile(np.asarray(net_base.init, dtype=bool), (B, 1))
        for t in range(FRAMES):
            want_cls = sim.head_scores(net_base, prev).argmax(-1)
            assert (classes[:, t] == want_cls).all(), \
                (name, t, classes[:, t], want_cls)
            assert (states[t] == gold_states[t]).all(), (name, t, "state mismatch")
            prev = gold_states[t]
        print(f"      composition gate [{name}]: {B}x{FRAMES} frames bit-exact "
              f"(state + class)")

    # torch pipeline cross-check: per-frame head decode on the same 8 writes
    import torch
    model.eval()
    cell = model.cell
    gold_cls = base_class_traj(net_base, x_legal)          # [8, T], netlist truth
    with torch.no_grad():
        xt = torch.from_numpy(x_legal.astype(np.float32))
        state = cell.init_state(ALPHABET, device=xt.device, dtype=xt.dtype)
        for t in range(FRAMES):
            state = cell(xt[:, t, :], state)
            torch_cls = model.head(cell.readout_h(state)).argmax(-1).numpy()
            assert (torch_cls == gold_cls[:, t]).all(), (t, torch_cls, gold_cls[:, t])
    print(f"      torch cross-check: per-frame model.head decode == netlist class, "
          f"8x{FRAMES} frames")


# -----------------------------------------------------------------------------------
# golden class vectors + testbench
# -----------------------------------------------------------------------------------
def dump_golden_cls(net_base: Netlist, alphabet: int) -> tuple[list[int], int]:
    """Per legal write: FRAMES per-frame decoded classes as one hex digit per line
    (golden line t = decode of states[t], same alignment as golden_sym<k>.txt).
    The transient's wrong decodes are kept verbatim. Returns (final decodes,
    reset-state class)."""
    cls_traj = base_class_traj(net_base, write_seqs(net_base.n_pi, alphabet, FRAMES))
    for sym in range(alphabet):
        path = os.path.join(HERE, f"golden_cls_sym{sym}.txt")
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            for t in range(FRAMES):
                f.write(format(int(cls_traj[sym, t]), "x") + "\n")
    init_row = np.asarray(net_base.init, dtype=bool)[None, :]
    reset_cls = int(sim.head_scores(net_base, init_row).argmax(-1)[0])
    return [int(c) for c in cls_traj[:, -1]], reset_cls


def write_tb_top(net_base: Netlist, alphabet: int, reset_cls: int) -> None:
    """tb_fsm.v's skeleton with class_out in place of q. Ports-only references, so
    the SAME TB re-runs on top_synth.v; the optional full-state check (hierarchical
    dut.q vs golden_sym<k>.txt) lives behind `ifdef RTL_Q_CHECK — pre-synthesis
    only, where the internal q register still exists."""
    n, npi = net_base.n_state, net_base.n_pi
    w = max(1, (alphabet - 1).bit_length())
    total = alphabet * FRAMES
    init_bits = "".join(str(int(net_base.init[i])) for i in range(n - 1, -1, -1))
    L = [
        "// generated by export_top.py — class-level golden-vector testbench",
        "`timescale 1ns/1ps",
        "module tb_top;",
        "    reg clk = 1'b0;",
        "    reg rst;",
        f"    reg  [{npi - 1}:0] x;",
        f"    wire [{w - 1}:0] class_out;",
        "    lgn_top dut (.clk(clk), .rst(rst), .x(x), .class_out(class_out));",
        "    always #5 clk = ~clk;",
        "",
        f"    reg [3:0] golden_cls [0:{total - 1}];",
        "`ifdef RTL_Q_CHECK",
        f"    reg [{n - 1}:0] golden_q [0:{total - 1}];",
        "    integer q_errors;",
        "`endif",
        "    integer sym, t, errors, sym_errors, pass_count;",
        "",
        "    initial begin",
    ]
    for sym in range(alphabet):
        L.append(f'        $readmemh("golden_cls_sym{sym}.txt", golden_cls, '
                 f"{sym * FRAMES}, {sym * FRAMES + FRAMES - 1});")
    L.append("`ifdef RTL_Q_CHECK")
    for sym in range(alphabet):
        L.append(f'        $readmemh("golden_sym{sym}.txt", golden_q, '
                 f"{sym * FRAMES}, {sym * FRAMES + FRAMES - 1});")
    L += [
        "        q_errors = 0;",
        "`endif",
        "        errors = 0;",
        "        pass_count = 0;",
        f"        for (sym = 0; sym < {alphabet}; sym = sym + 1) begin",
        "            sym_errors = 0;",
        "            // synchronous reset: load net.init",
        "            rst = 1'b1;",
        f"            x = {npi}'b0;",
        "            @(posedge clk); #1;",
        "            // reset-state decode (simulator-computed, first-max tie-break)",
        f"            if (class_out !== {w}'d{reset_cls}) begin",
        '                $display("SYM %0d: FAIL at reset (class_out=%0d)",'
        " sym, class_out);",
        "                sym_errors = sym_errors + 1;",
        "            end",
        "`ifdef RTL_Q_CHECK",
        f"            if (dut.q !== {n}'b{init_bits}) begin",
        '                $display("SYM %0d: Q FAIL at reset", sym);',
        "                q_errors = q_errors + 1;",
        "            end",
        "`endif",
        "            rst = 1'b0;",
        "            // t=0: the write frame (cue + one-hot symbol)",
        f"            x = ({npi}'b1 << sym) | ({npi}'b1 << {alphabet});",
        "            @(posedge clk); #1;",
        f"            x = {npi}'b0;                 // forced blanks afterwards",
        f"            for (t = 0; t < {FRAMES}; t = t + 1) begin",
        "                if (t > 0) begin @(posedge clk); #1; end",
        f"                if (class_out !== golden_cls[sym * {FRAMES} + t][{w - 1}:0])"
        " begin",
        "                    sym_errors = sym_errors + 1;",
        "                    if (sym_errors <= 3)",
        '                        $display("SYM %0d: CLASS MISMATCH frame %0d '
        '(got %0d want %0d)", sym, t, class_out,'
        f" golden_cls[sym * {FRAMES} + t][{w - 1}:0]);",
        "                end",
        "`ifdef RTL_Q_CHECK",
        f"                if (dut.q !== golden_q[sym * {FRAMES} + t]) begin",
        "                    q_errors = q_errors + 1;",
        "                    if (q_errors <= 3)",
        '                        $display("SYM %0d: Q MISMATCH frame %0d", sym, t);',
        "                end",
        "`endif",
        "            end",
        "            if (sym_errors == 0) begin",
        f'                $display("SYM %0d: PASS ({FRAMES}/{FRAMES} frames)", sym);',
        "                pass_count = pass_count + 1;",
        "            end else",
        '                $display("SYM %0d: FAIL (%0d bad frames)", sym, sym_errors);',
        "            errors = errors + sym_errors;",
        "        end",
        "`ifdef RTL_Q_CHECK",
        '        $display("QCHECK: %0d q-frame mismatches", q_errors);',
        "        errors = errors + q_errors;",
        "`endif",
        f'        $display("EQUIV: %0d/{alphabet} symbols PASS, %0d frame mismatches",'
        " pass_count, errors);",
        f"        if (pass_count == {alphabet} && errors == 0)",
        '            $display("RESULT: PASS");',
        "        else",
        '            $display("RESULT: FAIL");',
        "        $finish;",
        "    end",
        "endmodule",
    ]
    with open(os.path.join(HERE, "tb_top.v"), "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# -----------------------------------------------------------------------------------
# mutation controls (CLASS-observable, searched by IR-level re-simulation)
# -----------------------------------------------------------------------------------
def _flip(net: Netlist, g: int, to_op: int = 4) -> Netlist:
    mut = dataclasses.replace(net, ops=net.ops.copy())
    mut.ops[g] = to_op
    return mut


def find_fsm_mutation(net_base: Netlist, gold_cls: np.ndarray,
                      candidates) -> tuple[int, tuple[int, int]]:
    """First candidate FSM gate (op 2 -> 4) whose flip changes the CLASS trajectory
    over the 8x40 protocol frames. q-observability (report.md §2's g46) is NOT
    sufficient — the 3-bit argmax may mask it, so we re-derive it here."""
    x = write_seqs(net_base.n_pi, ALPHABET, FRAMES)
    for g in candidates:
        assert int(net_base.ops[g]) == 2, (g, int(net_base.ops[g]))
        traj = base_class_traj(_flip(net_base, g), x)
        diff = np.argwhere(traj != gold_cls)
        if len(diff):
            sym, frame = int(diff[0][0]), int(diff[0][1])
            return g, (sym, frame)
    raise AssertionError("no class-observable FSM mutation found among candidates")


def find_head_mutation(top: Netlist, states_all: np.ndarray,
                       gold_cls_flat: np.ndarray, candidates) -> tuple[int, int]:
    """First candidate HEAD gate (op 2 -> 4: the comparator gt gates) whose flip
    changes the decode of at least one of the 321 protocol-visited states (init +
    8x40 trajectory). Head gates don't affect the trajectory, so one batched
    combinational evaluation per candidate suffices."""
    x0 = np.zeros((len(states_all), top.n_pi), dtype=bool)
    for g in candidates:
        assert int(top.ops[g]) == 2, (g, int(top.ops[g]))
        outs, _ = eval_netlist(_flip(top, g), x0, states_all)
        got = bits_to_int(outs)
        diff = np.argwhere(got != gold_cls_flat)
        if len(diff):
            return g, int(diff[0][0])
    raise AssertionError("no class-observable head mutation found among candidates")


def check_masked_mutation(net_base: Netlist, g: int) -> bool:
    """True iff flipping gate g (op 2 -> 4) changes NEITHER the state trajectory
    NOR the class trajectory over the 8x40 protocol frames."""
    x = write_seqs(net_base.n_pi, ALPHABET, FRAMES)
    _, gold_states = sim.run_sequence(net_base, x, return_trajectory=True)
    mut = _flip(net_base, g)
    _, mut_states = sim.run_sequence(mut, x, return_trajectory=True)
    states_same = all((a == b).all() for a, b in zip(gold_states, mut_states))
    cls_same = (base_class_traj(net_base, x) == base_class_traj(mut, x)).all()
    return bool(states_same and cls_same)


def emit_mutant(top: Netlist, g: int, path: str) -> None:
    """Emit the composed top with gate g's op flipped 2 -> 4, and assert the file
    differs from top.v in EXACTLY the one 'wire g<id> = ...;' line."""
    emit_verilog(_flip(top, g), path, module="lgn_top",
                 expose_q=False, out_bus="class_out")
    ref = open(os.path.join(HERE, "top.v"), encoding="utf-8").read().splitlines()
    mut = open(path, encoding="utf-8").read().splitlines()
    assert len(ref) == len(mut)
    diffs = [i for i, (a, b) in enumerate(zip(ref, mut)) if a != b]
    assert len(diffs) == 1 and f"wire g{g} " in mut[diffs[0]], (path, diffs[:5])


# -----------------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--json", default=JSON)
    ap.add_argument("--alphabet", type=int, default=ALPHABET)
    ap.add_argument("--full-gate", action="store_true",
                    help="accuracy gate on the FULL test set (default: 8 batches)")
    args = ap.parse_args()

    spec = spec_from_json(args.json, alphabet=args.alphabet)
    print(f"[1/6] rebuild {os.path.basename(args.ckpt)}: {spec.task}/{spec.mechanism} "
          f"hidden={spec.hidden} seed={spec.seed} (recorded test_acc={spec.test_acc})")
    model = rebuild_model(spec, args.ckpt)
    task = build_task(spec)
    gate = check_accuracy(model, spec, task=task,
                          max_batches=None if args.full_gate else 8)
    print(f"      accuracy gate: rebuilt={gate['rebuilt_test_acc']:.4f} "
          f"recorded={gate['recorded_test_acc']} full={gate['full_test_set']}")
    if gate["rebuilt_test_acc"] < 0.999:
        print("[FAIL] rebuild gate — aborting export.")
        return 2

    net = extract_netlist(model)
    assert (net.n_pi, net.n_state, net.n_gates) == (9, 1024, N_FSM_GATES), \
        (net.n_pi, net.n_state, net.n_gates)
    assert net.head == (ALPHABET, net.n_state // ALPHABET), net.head
    assert all(v == 0 for v in net.init), "init must be all-zeros"
    print(f"[2/6] base netlist: {net.n_pi} PIs, {net.n_state} latches, "
          f"{net.n_gates} gates, head={net.head}")

    # emitter regression: DEFAULT parameters must reproduce synth/fsm.v byte-for-byte
    fsm_ref = os.path.join(HERE, "fsm.v")
    fsm_chk = os.path.join(HERE, "fsm_regen_check.v")
    emit_verilog(net, fsm_chk, module="lgn_fsm")
    same = filecmp.cmp(fsm_ref, fsm_chk, shallow=False)
    os.remove(fsm_chk)
    if not same:
        print("[FAIL] emit_verilog default-parameter regression: regenerated fsm.v "
              "is NOT byte-identical to the checked-in synth/fsm.v — aborting.")
        return 2
    print("      emit_verilog regression: regenerated fsm.v byte-identical (defaults "
          "unperturbed)")

    top = compose_top(net)
    n_head = top.n_gates - net.n_gates
    print(f"[3/6] composed lgn_top: {top.n_pi} PIs, {top.n_state} latches, "
          f"{top.n_gates} gates ({net.n_gates} FSM + {n_head} head), "
          f"{len(top.layers)} layers, {len(top.outputs)} class bits")
    composition_gate(net, top, model)

    emit_verilog(top, os.path.join(HERE, "top.v"), module="lgn_top",
                 expose_q=False, out_bus="class_out")
    blif.emit_blif(top, os.path.join(HERE, "top.blif"), model="lgn_top")
    derive_clk_blif(os.path.join(HERE, "top.blif"), os.path.join(HERE, "top_clk.blif"))
    print("[4/6] wrote top.v, top.blif, top_clk.blif")

    decodes, reset_cls = dump_golden_cls(net, args.alphabet)
    ok = sum(int(d == s) for s, d in enumerate(decodes))
    print(f"[5/6] golden class vectors: {args.alphabet} files x {FRAMES} frames; "
          f"final decode correct {ok}/{args.alphabet} {decodes}; "
          f"reset-state class {reset_cls}")
    if ok != args.alphabet:
        print("[FAIL] final decode != symbol for some write — aborting.")
        return 1
    write_tb_top(net, args.alphabet, reset_cls)
    print("      wrote tb_top.v")

    print("[6/6] mutation controls (class-observability by IR-level re-simulation)")
    x_legal = write_seqs(net.n_pi, args.alphabet, FRAMES)
    gold_cls = base_class_traj(net, x_legal)
    op2_fsm = [int(g) for g in np.where(net.ops == 2)[0]]
    fsm_cands = ([46] if 46 in op2_fsm else []) + [g for g in op2_fsm if g != 46]
    g_obs, (sym, frame) = find_fsm_mutation(net, gold_cls, fsm_cands)
    print(f"      FSM mutant: g{g_obs} (op 2->4) diverges first at "
          f"sym={sym} frame={frame}")

    # all 321 protocol-visited states (init + 8x40 trajectory) in one batch
    _, gold_states = sim.run_sequence(net, x_legal, return_trajectory=True)
    states_all = np.concatenate(
        [np.asarray(net.init, dtype=bool)[None, :]] + gold_states)
    gold_cls_flat = sim.head_scores(net, states_all).argmax(-1)
    op2_head = [int(g) for g in np.where(top.ops == 2)[0] if g >= net.n_gates]
    g_head, row = find_head_mutation(top, states_all, gold_cls_flat, op2_head)
    print(f"      head mutant: g{g_head} (op 2->4, comparator gt) diverges on "
          f"visited state row {row} (0 = reset state)")

    g_masked = None
    for g in ([7] if int(net.ops[7]) == 2 else []) + op2_fsm:
        if g not in (g_obs, g_head) and check_masked_mutation(net, g):
            g_masked = g
            break
    assert g_masked is not None, "no masked op-2 gate found"
    print(f"      masked mutant: g{g_masked} (op 2->4) changes NO state/class over "
          f"8x{FRAMES} frames (negative control, TB must PASS)")

    emit_mutant(top, g_obs, os.path.join(HERE, "top_mut_obs.v"))
    emit_mutant(top, g_head, os.path.join(HERE, "top_mut_head.v"))
    emit_mutant(top, g_masked, os.path.join(HERE, "top_mut_masked.v"))
    print("      wrote top_mut_obs.v / top_mut_head.v / top_mut_masked.v "
          "(each = top.v with exactly one flipped gate line)")
    print("\nNext: wsl -e bash -lc "
          "'cd /mnt/c/Users/malco/projects/difflogic/mlgn/netlist/synth "
          "&& bash run_top.sh'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
