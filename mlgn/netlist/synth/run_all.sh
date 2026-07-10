#!/usr/bin/env bash
# run_all.sh — reproduce the full hardware flow (run INSIDE WSL from this directory).
# Prereqs: OSS CAD Suite in ~/oss-cad-suite (yosys + iverilog); artifacts fsm.v,
# fsm.blif, fsm_clk.blif, golden_sym*.txt, tb_*.v generated on the Windows side by
#     python -m mlgn.netlist.synth.export_fsm
#     python -m mlgn.netlist.synth.rand_probe     # optional random-stimulus probe
set -euo pipefail
export PATH="$HOME/oss-cad-suite/bin:$PATH"
mkdir -p log

echo "== [1/4] gate-op micro-test (all 16 ops, exhaustive) =="
iverilog -g2001 -o /tmp/tb_ops.vvp ops_test.v tb_ops.v
vvp /tmp/tb_ops.vvp | tee log/iverilog_equiv.log

echo "== [2/4] golden-vector equivalence (8 legal writes x 40 frames x 1024 bits) =="
iverilog -g2001 -o /tmp/tb_fsm.vvp fsm.v tb_fsm.v
vvp /tmp/tb_fsm.vvp | tee -a log/iverilog_equiv.log
if [ -f tb_rand.v ]; then
    echo "-- random-stimulus probe (non-protocol inputs) --"
    iverilog -g2001 -o /tmp/tb_rand.vvp fsm.v tb_rand.v
    vvp /tmp/tb_rand.vvp | tee -a log/iverilog_equiv.log
fi

echo "== [3/4] synthesis =="
yosys -p 'read_verilog fsm.v; synth_xilinx -top lgn_fsm; stat; write_verilog fsm_synth.v' \
    > log/yosys_xilinx_verilog.log 2>&1
yosys -p 'read_blif fsm_clk.blif; synth_xilinx; stat' \
    > log/yosys_xilinx_blif_clk.log 2>&1
yosys -p 'read_blif fsm.blif; synth_xilinx; stat' \
    > log/yosys_xilinx_blif.log 2>&1
yosys -p 'read_verilog fsm.v; synth -top lgn_fsm; stat' \
    > log/yosys_generic_verilog.log 2>&1
grep -A 40 'Printing statistics' log/yosys_xilinx_verilog.log | tail -40

echo "== [4/4] post-synthesis gate-level re-simulation (Xilinx cell models) =="
CELLS="$(yosys-config --datdir)/xilinx/cells_sim.v"
iverilog -g2001 -o /tmp/tb_ps.vvp fsm_synth.v "$CELLS" tb_fsm.v
vvp /tmp/tb_ps.vvp | tee log/iverilog_postsynth.log
if [ -f tb_rand.v ]; then
    iverilog -g2001 -o /tmp/tb_psr.vvp fsm_synth.v "$CELLS" tb_rand.v
    vvp /tmp/tb_psr.vvp | tee -a log/iverilog_postsynth.log
fi
