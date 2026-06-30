#!/usr/bin/env bash
# run_queue.sh — run a list of seqlgn training jobs across the two DUST GPUs.
#
# Each entry in JOBS is one train.py arg-string (NO `python -m …` prefix, NO
# --device, NO CUDA_VISIBLE_DEVICES — the script adds those). Jobs are dealt
# round-robin to GPU 0 / GPU 1; each GPU runs its share sequentially. A job whose
# results file already exists (matched by its --tag) is SKIPPED, so re-running
# after a disconnect or container restart resumes instead of redoing finished work
# (resume is per-job, not mid-job: an interrupted run restarts from scratch).
#
# Run detached so it survives a closed browser tab (mkdir first — the shell opens
# the redirect before the script's own mkdir runs):
#   cd ~/work/difflogic && mkdir -p logs
#   nohup bash mlgn/seqlgn/run_queue.sh > logs/queue.log 2>&1 &
#   tail -f logs/queue.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (holds difflogic/ + mlgn/)
LOGS="$ROOT/logs"
RESULTS="$ROOT/mlgn/seqlgn/results"
mkdir -p "$LOGS"
cd "$ROOT"

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
JOBS=(
  # Essential — 3rd seed at headline delays (these failed on GPU0)
  "--task smnist-pixel --chunk 784 --delay 50  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 1 --tag rec_d50_gated_s1"
  "--task smnist-pixel --chunk 784 --delay 100 --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 1 --tag rec_d100_gated_s1"
  # Recommended — recall keep-bias sweep (mirrors the psMNIST sweep)
  "--task smnist-pixel --chunk 784 --delay 50  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 0 --lr 0.003 --lr-min 0.0003 --tag rec_d50_gated_kb0"
  "--task smnist-pixel --chunk 784 --delay 50  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 3 --lr 0.003 --lr-min 0.0003 --tag rec_d50_gated_kb3"
  # Optional — fill intermediate delays to 3 seeds + the missing control point
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003          --tag rec_d25_gated"
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 2 --tag rec_d25_gated_s2"
  "--task smnist-pixel --chunk 784 --delay 75  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003          --tag rec_d75_gated"
  "--task smnist-pixel --chunk 784 --delay 75  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 2 --tag rec_d75_gated_s2"
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism rddlgn --lr 0.003 --lr-min 0.0003 --tag rec_d25_rddlgn"
)
# ─────────────────────────────────────────────────────────────────────────────

ts()     { date +%H:%M:%S; }
tag_of() { sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<<"$1"; }

run_worker() {
  local gpu="$1"; shift
  local job tag rc
  for job in "$@"; do
    tag="$(tag_of "$job")"
    # require a digit (the timestamp) right after the tag, so e.g. tag "rec_d25_gated"
    # is NOT matched by a sibling "rec_d25_gated_s1_<stamp>.json" file
    if compgen -G "$RESULTS/*_${tag}_[0-9]*.json" >/dev/null; then
      echo "$(ts) [gpu$gpu] SKIP  $tag (results already exist)"; continue
    fi
    echo "$(ts) [gpu$gpu] START $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python -m mlgn.seqlgn.train $job >"$LOGS/$tag.log" 2>&1
    rc=$?
    echo "$(ts) [gpu$gpu] DONE  $tag (exit $rc) -> logs/$tag.log"
  done
}

# deal jobs round-robin: even index → GPU0, odd index → GPU1
g0=(); g1=()
for i in "${!JOBS[@]}"; do
  if (( i % 2 == 0 )); then g0+=("${JOBS[$i]}"); else g1+=("${JOBS[$i]}"); fi
done

echo "$(ts) queue start: ${#g0[@]} job(s) on GPU0, ${#g1[@]} on GPU1"
run_worker 0 "${g0[@]}" &
run_worker 1 "${g1[@]}" &
wait
echo "$(ts) ALL DONE"
