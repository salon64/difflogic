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
  # ── delayed-MNIST error bars: gated seeds 1 & 2 at the existing delays (seed 0 already done) ──
  "--task smnist-pixel --chunk 784 --delay 0   --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 1 --tag rec_d0_gated_s1"
  "--task smnist-pixel --chunk 784 --delay 0   --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 2 --tag rec_d0_gated_s2"
  "--task smnist-pixel --chunk 784 --delay 50  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 1 --tag rec_d50_gated_s1"
  "--task smnist-pixel --chunk 784 --delay 50  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 2 --tag rec_d50_gated_s2"
  "--task smnist-pixel --chunk 784 --delay 100 --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 1 --tag rec_d100_gated_s1"
  "--task smnist-pixel --chunk 784 --delay 100 --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 2 --tag rec_d100_gated_s2"
  # ── intermediate delays 25 & 75: gated (seeds 0/1/2) + control, to densify the curve ──
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003          --tag rec_d25_gated"
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 1 --tag rec_d25_gated_s1"
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min 0.0003 --seed 2 --tag rec_d25_gated_s2"
  "--task smnist-pixel --chunk 784 --delay 75  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min         --tag rec_d75_gated"
  "--task smnist-pixel --chunk 784 --delay 75  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min--seed 1 --tag rec_d75_gated_s1"
  "--task smnist-pixel --chunk 784 --delay 75  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 6 --lr 0.003 --lr-min--seed 2 --tag rec_d75_gated_s2"
  "--task smnist-pixel --chunk 784 --delay 25  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism rddlgn --lr 0.003 --lr-min 0.0003 --tag rec_d25_rddlgn"
  "--task smnist-pixel --chunk 784 --delay 75  --hidden 1000 --iters 20000 --eval-freq 1000 --mechanism rddlgn --lr 0.003 --lr-min 0.0003 --tag
rec_d75_rddlgn"
)
# ─────────────────────────────────────────────────────────────────────────────

ts()     { date +%H:%M:%S; }
tag_of() { sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<<"$1"; }

run_worker() {
  local gpu="$1"; shift
  local job tag rc
  for job in "$@"; do
    tag="$(tag_of "$job")"
    if compgen -G "$RESULTS/*${tag}_*.json" >/dev/null; then
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
