#!/usr/bin/env bash
# run_pnr.sh — open-flow place-and-route + post-P&R timing (Fmax) for the
# exported LGN circuits on Artix-7 (P3b T2). Board-independent: a P&R TIMING
# run, no bitstream programming.
#
# Toolchain: openXC7 prebuilt snaps (installed via ~/openXC7/toolchain-installer.sh):
#   yosys 0.38 (openXC7 snap), nextpnr-xilinx 0.8.2, prjxray-db (snap-bundled).
# Run INSIDE WSL from this directory (mlgn/netlist/synth). Prereq: the chipdb .bin
# built once by build_chipdb.sh (default ~/openXC7/pnr/<dbpart>.bin).
#
# Usage:
#   ./run_pnr.sh <design.v> <top_module> <period_ns> <outtag> [seed]
# e.g.
#   ./run_pnr.sh top.v          lgn_top       3.0 top
#   ./run_pnr.sh fsm_pnr_wrap.v fsm_pnr_wrap  3.0 fsm
#
# nextpnr reports the ACHIEVED "Max frequency" regardless of the target period;
# the create_clock target only drives the timing-driven placer effort and the
# PASS/FAIL verdict. Set period_ns slightly below the expected achievable period
# to push the placer. Sweep seed / period to reduce the (acknowledged) variance
# of nextpnr's timing-driven PnR.
set -euo pipefail

export PATH=/snap/bin:$PATH
SNAP=/snap/openxc7/current/opt/nextpnr-xilinx

DESIGN="${1:?design.v}"
TOP="${2:?top module}"
PERIOD="${3:?clock period ns}"
TAG="${4:?output tag}"
SEED="${5:-1}"

PART="${PART:-xc7a35tcsg324-1}"                     # override e.g. PART=xc7a100tcsg324-1
DBPART="$(echo "$PART" | sed -e 's/-[0-9]//g')"     # speed grade stripped (openXC7.mk convention)
CHIPDB_DIR="${CHIPDB_DIR:-$HOME/openXC7/pnr}"
CHIPDB="$CHIPDB_DIR/$DBPART.bin"
WORK="${WORK:-$HOME/openXC7/pnr}"          # scratch for large json/fasm
mkdir -p log "$WORK"

[ -f "$CHIPDB" ] || { echo "!! chipdb missing: $CHIPDB (run build_chipdb.sh)"; exit 1; }

JSON="$WORK/${TAG}.json"
XDC="$WORK/${TAG}.xdc"
ROUTED="$WORK/${TAG}_routed.json"
YLOG="log/pnr_${TAG}_yosys.log"
PLOG="log/pnr_${TAG}_nextpnr_p${PERIOD}_s${SEED}.log"
SDF="$WORK/${TAG}.sdf"

echo "== [1/3] yosys synth_xilinx -> json ($DESIGN, top=$TOP) =="
yosys -q -p "read_verilog $DESIGN; synth_xilinx -flatten -abc9 -nobram -arch xc7 -top $TOP; write_json $JSON" \
    > "$YLOG" 2>&1
echo "   json: $(ls -la "$JSON" | awk '{print $5}') bytes  (log: $YLOG)"

echo "== [2/3] XDC: per-port IOSTANDARD + create_clock ${PERIOD}ns on clk (pin E3) =="
# nextpnr-xilinx REQUIRES an IOSTANDARD on every top-level PAD (a bare port errors
# out), and its XDC parser does NOT expand get_ports wildcards — so emit one
# explicit line per port BIT, read straight from the synthesized JSON. LOC is
# optional (unplaced I/O is auto-assigned to free IOBs); only clk gets a real pin.
# Pin identity does not affect the reg-to-reg (core) Fmax we care about.
FREQ="$(awk "BEGIN{printf \"%.3f\", 1000.0/$PERIOD}")"   # target MHz = 1000/period(ns)
python3 - "$JSON" "$TOP" "$PERIOD" > "$XDC" <<'PYEOF'
import json, sys
jf, top, period = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(jf))
mods = d["modules"]
# pick the module flagged top=1, else the requested name, else the one with a clk port
tm = None
for name, m in mods.items():
    if str(m.get("attributes", {}).get("top", "0")).lstrip("0") in ("1",):
        tm = m; break
if tm is None: tm = mods.get(top)
if tm is None:
    tm = next(m for m in mods.values() if "clk" in m.get("ports", {}))
print("create_clock -period %s -name clk [get_ports clk]" % period)
print("set_property LOC E3 [get_ports clk]")
for pname, p in tm["ports"].items():
    w = len(p["bits"])
    if w == 1:
        print("set_property IOSTANDARD LVCMOS33 [get_ports %s]" % pname)
    else:
        for i in range(w):
            print("set_property IOSTANDARD LVCMOS33 [get_ports {%s[%d]}]" % (pname, i))
PYEOF
echo "   XDC: $(grep -c IOSTANDARD "$XDC") port bits constrained  (target ${FREQ} MHz)"

echo "== [3/3] nextpnr-xilinx P&R (part $PART, seed $SEED, target ${FREQ}MHz) =="
# --freq sets the timing-driven target by frequency (create_clock on the port
# does NOT bind after -abc9 renames the clock net, so it silently falls back to
# nextpnr's 12 MHz default; --freq is what actually pushes the placer).
# --timing-allow-fail: an aggressive target is EXPECTED to miss; nextpnr still
# fully routes and prints the ACHIEVED post-route "Max frequency" (what we want).
nextpnr-xilinx --chipdb "$CHIPDB" --xdc "$XDC" --json "$JSON" \
    --write "$ROUTED" --sdf "$SDF" --seed "$SEED" --freq "$FREQ" --timing-allow-fail \
    > "$PLOG" 2>&1 || { echo "!! nextpnr FAILED (see $PLOG)"; tail -25 "$PLOG"; exit 2; }

echo "---- nextpnr timing / utilisation summary ($PLOG) ----"
grep -E "Max frequency|Critical path|Max delay|utilisation|SLICE|LUT|FDRE|CARRY|MUX|clock 'clk'" "$PLOG" | tail -50 || true
echo "== run_pnr.sh done: $TAG @ target ${PERIOD}ns seed ${SEED} =="
