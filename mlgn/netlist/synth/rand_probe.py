"""rand_probe.py — random-stimulus equivalence probe (adversarial review add-on).

The golden vectors in tb_fsm.v only exercise the copy protocol (one legal write,
then blanks). This probe drives fsm.v with RANDOM 9-bit inputs every cycle —
multi-hot symbols, cue toggling mid-sequence, inputs the protocol never reaches —
and checks q cycle-exactly against the bit-exact python simulator.

Deliberately does NOT rebuild the torch model: it parses fsm.blif back into an
ir.Netlist (covers -> ops via ir.BLIF_COVER) and simulates that, so the expected
values come through an independent path from the one that wrote fsm.v. As a
sanity anchor it first replays golden_sym0.txt through the parsed netlist.

    python -m mlgn.netlist.synth.rand_probe        # writes rand_stim/rand_gold/tb_rand
    # then in WSL:  iverilog -g2001 -o tb_rand.vvp fsm.v tb_rand.v && vvp tb_rand.vvp
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.netlist import sim  # noqa: E402
from mlgn.netlist.ir import BLIF_COVER, Netlist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R, T, SEED = 6, 30, 20260710
COVER2OP = {tuple(v): k for k, v in BLIF_COVER.items()}


def blif_to_netlist(path: str) -> Netlist:
    """Parse an emit_blif file (gates g<i> in topological order) back to a Netlist."""
    lines = open(path, encoding="utf-8").read().splitlines()
    inputs: list[str] = []
    latches: list[tuple[str, str, int]] = []
    names: dict[str, tuple[list[str], tuple[str, ...]]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith(".inputs"):
            inputs = ln.split()[1:]
        elif ln.startswith(".latch"):
            t = ln.split()[1:]
            assert len(t) == 3, ln
            latches.append((t[0], t[1], int(t[2])))
        elif ln.startswith(".names"):
            sig = ln.split()[1:]
            cover = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and \
                    not lines[j].strip().startswith("."):
                cover.append(lines[j].strip())
                j += 1
            names[sig[-1]] = (sig[:-1], tuple(cover))
            i = j - 1
        i += 1
    n_pi, n_state = len(inputs), len(latches)
    sid = {"c0": 0, "c1": 1}
    for k, x in enumerate(inputs):
        sid[x] = 2 + k
    for k, (_, qn, _) in enumerate(latches):
        assert qn == f"q{k}"
        sid[qn] = 2 + n_pi + k
    n_gates = sum(1 for k in names if re.fullmatch(r"g\d+", k))
    ops = np.zeros(n_gates, dtype=np.uint8)
    src_a = np.zeros(n_gates, dtype=np.int32)
    src_b = np.zeros(n_gates, dtype=np.int32)
    for g in range(n_gates):
        ins, cover = names[f"g{g}"]
        op = COVER2OP[cover]
        ops[g] = op
        if op not in (0, 15):
            assert len(ins) == 2, (g, ins)
            src_a[g], src_b[g] = sid[ins[0]], sid[ins[1]]
        sid[f"g{g}"] = 2 + n_pi + n_state + g
    # greedy contiguous topological slices for sim.step
    layers: list[tuple[int, int]] = []
    start, gbase = 0, 2 + n_pi + n_state
    for g in range(n_gates):
        if max(src_a[g], src_b[g]) >= gbase + start:
            layers.append((start, g))
            start = g
    layers.append((start, n_gates))
    return Netlist(n_pi=n_pi, n_state=n_state,
                   init=[iv for _, _, iv in latches], ops=ops,
                   src_a=src_a, src_b=src_b, layers=layers,
                   next_state=[sid[din] for din, _, _ in latches])


def state_hex(state: np.ndarray, n: int) -> str:
    bits = np.asarray(state, dtype=np.uint8).ravel()
    val = int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")
    return format(val, f"0{(n + 3) // 4}x")


def main() -> int:
    net = blif_to_netlist(os.path.join(HERE, "fsm.blif"))
    print(f"parsed fsm.blif: {net.n_pi} PIs, {net.n_state} latches, "
          f"{net.n_gates} gates, {len(net.layers)} layers")

    # sanity anchor: the parsed netlist must replay golden_sym0 exactly
    x0 = np.zeros((1, 40, net.n_pi), dtype=bool)
    x0[0, 0, 0] = x0[0, 0, 8] = True
    _, states = sim.run_sequence(net, x0, return_trajectory=True)
    golden0 = open(os.path.join(HERE, "golden_sym0.txt")).read().split()
    assert all(state_hex(s[0], net.n_state) == g
               for s, g in zip(states, golden0)), "parsed netlist != golden_sym0"
    print("golden_sym0 replay through parsed netlist: MATCH")

    rng = np.random.default_rng(SEED)
    x = rng.integers(0, 2, size=(R, T, net.n_pi)).astype(bool)
    _, traj = sim.run_sequence(net, x, return_trajectory=True)
    with open(os.path.join(HERE, "rand_stim.txt"), "w", newline="\n") as f:
        for r in range(R):
            for t in range(T):
                f.write(format(sum(int(x[r, t, i]) << i
                                   for i in range(net.n_pi)), "03x") + "\n")
    with open(os.path.join(HERE, "rand_gold.txt"), "w", newline="\n") as f:
        for r in range(R):
            for t in range(T):
                f.write(state_hex(traj[t][r], net.n_state) + "\n")

    n = net.n_state
    init_bits = "".join(str(net.init[i]) for i in range(n - 1, -1, -1))
    tb = f"""// generated by rand_probe.py — random-stimulus cycle-exact equivalence
`timescale 1ns/1ps
module tb_rand;
    reg clk = 1'b0;
    reg rst;
    reg  [{net.n_pi - 1}:0] x;
    wire [{n - 1}:0] q;
    lgn_fsm dut (.clk(clk), .rst(rst), .x(x), .q(q));
    always #5 clk = ~clk;
    reg [{net.n_pi - 1}:0] stim [0:{R * T - 1}];
    reg [{n - 1}:0] gold [0:{R * T - 1}];
    integer r, t, errors;
    initial begin
        $readmemh("rand_stim.txt", stim);
        $readmemh("rand_gold.txt", gold);
        errors = 0;
        for (r = 0; r < {R}; r = r + 1) begin
            rst = 1'b1; x = {net.n_pi}'b0;
            @(posedge clk); #1;
            if (q !== {n}'b{init_bits}) begin
                $display("RAND seq %0d: FAIL at reset", r);
                errors = errors + 1;
            end
            rst = 1'b0;
            for (t = 0; t < {T}; t = t + 1) begin
                x = stim[r * {T} + t];
                @(posedge clk); #1;
                if (q !== gold[r * {T} + t]) begin
                    errors = errors + 1;
                    if (errors <= 5)
                        $display("RAND seq %0d: MISMATCH frame %0d", r, t);
                end
            end
        end
        if (errors == 0)
            $display("RAND: {R} seqs x {T} frames PASS");
        else
            $display("RAND: FAIL (%0d mismatches)", errors);
        $finish;
    end
endmodule
"""
    with open(os.path.join(HERE, "tb_rand.v"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(tb)
    print(f"wrote rand_stim.txt / rand_gold.txt / tb_rand.v "
          f"({R} seqs x {T} frames, seed {SEED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
