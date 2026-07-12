#!/usr/bin/env bash
# run_top.sh — hardware flow for the head-in-fabric deployable top (P3b T1).
# Run INSIDE WSL from this directory. Prereqs: OSS CAD Suite in ~/oss-cad-suite;
# artifacts top.v, top.blif, top_clk.blif, golden_cls_sym*.txt, tb_top.v and the
# three mutation controls top_mut_{obs,head,masked}.v generated on the Windows
# side by
#     python -m mlgn.netlist.synth.export_top
#
# The three mutants are single-gate op flips (2 -> 4) searched for CLASS
# observability by IR-level re-simulation; the TB must FAIL on the first two and
# PASS on the masked one (negative control) — a TB that cannot fail is vacuous.
# The mutant files are DELETED after their evidence is logged (regenerate with
# export_top.py).
set -euo pipefail
export PATH="$HOME/oss-cad-suite/bin:$PATH"
mkdir -p log

expect() {  # expect <want PASS|FAIL> <log-file of the run just appended>
    local want="$1" file="$2"
    if grep -q "RESULT: $want" "$file"; then
        echo "-- expected RESULT: $want — confirmed --"
    else
        echo "!! expected RESULT: $want but got:" && grep "RESULT:" "$file" || true
        exit 1
    fi
}

echo "== [1/5] RTL golden class equivalence (8 legal writes x 40 frames x 3 bits) =="
iverilog -g2001 -o /tmp/tb_top.vvp top.v tb_top.v
vvp /tmp/tb_top.vvp | tee log/iverilog_top_equiv.log
expect PASS log/iverilog_top_equiv.log

echo "== [2/5] RTL + full-state check (RTL_Q_CHECK: dut.q vs golden_sym*.txt) =="
iverilog -g2001 -DRTL_Q_CHECK -o /tmp/tb_topq.vvp top.v tb_top.v
vvp /tmp/tb_topq.vvp | tee log/iverilog_top_equiv_q.log
expect PASS log/iverilog_top_equiv_q.log

echo "== [3/5] mutation controls (expect FAIL / FAIL / PASS) =="
iverilog -g2001 -o /tmp/tb_m1.vvp top_mut_obs.v tb_top.v
vvp /tmp/tb_m1.vvp | tee log/iverilog_top_mut.log
expect FAIL log/iverilog_top_mut.log
iverilog -g2001 -o /tmp/tb_m2.vvp top_mut_head.v tb_top.v
vvp /tmp/tb_m2.vvp | tee /tmp/mut2.log && cat /tmp/mut2.log >> log/iverilog_top_mut.log
expect FAIL /tmp/mut2.log
iverilog -g2001 -o /tmp/tb_m3.vvp top_mut_masked.v tb_top.v
vvp /tmp/tb_m3.vvp | tee /tmp/mut3.log && cat /tmp/mut3.log >> log/iverilog_top_mut.log
expect PASS /tmp/mut3.log
rm -f top_mut_obs.v top_mut_head.v top_mut_masked.v
echo "-- mutation evidence logged (log/iverilog_top_mut.log); mutant copies deleted --"

echo "== [4/5] synthesis (four flows, as run_all.sh) =="
yosys -p 'read_verilog top.v; synth_xilinx -top lgn_top; stat; write_verilog top_synth.v' \
    > log/yosys_xilinx_top_verilog.log 2>&1
yosys -p 'read_blif top_clk.blif; synth_xilinx; stat' \
    > log/yosys_xilinx_top_blif_clk.log 2>&1
yosys -p 'read_blif top.blif; synth_xilinx; stat' \
    > log/yosys_xilinx_top_blif.log 2>&1
yosys -p 'read_verilog top.v; synth -top lgn_top; stat' \
    > log/yosys_generic_top_verilog.log 2>&1
grep -A 40 'Printing statistics' log/yosys_xilinx_top_verilog.log | tail -40

echo "== [5/5] post-synthesis gate-level re-simulation (Xilinx cell models) =="
CELLS="$(yosys-config --datdir)/xilinx/cells_sim.v"
iverilog -g2001 -o /tmp/tb_ps_top.vvp top_synth.v "$CELLS" tb_top.v
vvp /tmp/tb_ps_top.vvp | tee log/iverilog_top_postsynth.log
expect PASS log/iverilog_top_postsynth.log

echo "== run_top.sh: ALL STAGES PASSED =="
