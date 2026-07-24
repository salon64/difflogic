#!/usr/bin/env bash
# run_ganak.sh — run ganak on exact_count CNFs, one saved log per instance (research/26).
#
# Usage (from repo root, on DUST):
#   nohup ./mlgn/netlist/run_ganak.sh > run_ganak.out 2>&1 &
#   nohup ./mlgn/netlist/run_ganak.sh ~/work/ganak path/to/other.cnf ... &   # custom set
#
# Instances run in PARALLEL (ganak is single-threaded; ~6 GB each). A log that
# already contains a final count is skipped, so re-running after a crash/restart
# only picks up what is missing. Logs: mlgn/netlist/out/exact_count/logs/<name>.ganak.log
# Interpretation: the "c s exact arb int N" line -> N / 2^120 = the exact quantity
# (volume for *_volume.cnf, average sensitivity for *_sensitivity.cnf).
set -u
GANAK="${1:-$HOME/work/ganak}"
shift || true
CNFS=("$@")
if [ ${#CNFS[@]} -eq 0 ]; then
  CNFS=(
    mlgn/netlist/out/exact_count/h512_volume.cnf
    mlgn/netlist/out/exact_count/h512_sensitivity.cnf
  )
fi
LOGDIR=mlgn/netlist/out/exact_count/logs
mkdir -p "$LOGDIR"

for cnf in "${CNFS[@]}"; do
  name=$(basename "$cnf" .cnf)
  log="$LOGDIR/$name.ganak.log"
  if grep -q "exact arb int" "$log" 2>/dev/null; then
    echo "[skip] $name already counted -> $log"
    continue
  fi
  if [ ! -f "$cnf" ]; then
    echo "[miss] $cnf not found — pull manifold or regenerate with exact_count --dimacs"
    continue
  fi
  echo "[run ] $name -> $log"
  {
    echo "=== start $(date -u +%FT%TZ) :: $GANAK $cnf ==="
    "$GANAK" "$cnf"
    echo "=== exit $? at $(date -u +%FT%TZ) ==="
  } > "$log" 2>&1 &
done
wait

echo "=== all runs finished $(date -u +%FT%TZ) ==="
grep -H "exact arb int" "$LOGDIR"/*.ganak.log 2>/dev/null || echo "(no counts found — inspect logs in $LOGDIR)"
