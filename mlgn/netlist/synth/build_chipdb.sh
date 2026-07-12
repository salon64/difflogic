#!/usr/bin/env bash
# build_chipdb.sh — build the nextpnr-xilinx chipdb .bin ONCE per part.
# The chipdb is NOT shipped with the openXC7 snap (only prjxray-db source is);
# it is generated from the snap-bundled prjxray-db via bbaexport.py + bbasm.
# Run INSIDE WSL. Output: $OUT/<dbpart>.bin (default ~/openXC7/pnr).
#
# xc7a35tcsg324-1 = Arty A7-35T part (open-flow Fmax reference; ~92 MB .bin, ~2 min).
# xc7a100tcsg324-1 = Arty A7-100T part (the PURCHASE target; larger DB, more time/RAM).
#   ./build_chipdb.sh xc7a35tcsg324-1
#   ./build_chipdb.sh xc7a100tcsg324-1
set -euo pipefail
export PATH=/snap/bin:$PATH
SNAP=/snap/openxc7/current/opt/nextpnr-xilinx

PART="${1:-xc7a35tcsg324-1}"
DBPART="$(echo "$PART" | sed -e 's/-[0-9]//g')"   # strip speed grade (openXC7.mk convention)
OUT="${OUT:-$HOME/openXC7/pnr}"
mkdir -p "$OUT"

echo "== bbaexport ($PART) -> $OUT/$DBPART.bba =="
python3 "$SNAP/python/bbaexport.py" --device "$PART" --bba "$OUT/$DBPART.bba"
echo "== bbasm -> $OUT/$DBPART.bin =="
bbasm --l "$OUT/$DBPART.bba" "$OUT/$DBPART.bin"
ls -la "$OUT/$DBPART.bin"
echo "== chipdb ready: $OUT/$DBPART.bin =="
