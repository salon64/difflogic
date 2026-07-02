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

# Physical GPUs to use (round-robin). GPU 0 is currently FAULTY — intermittent CUDA
# error 700 (illegal memory access → exit 188). Route everything to GPU 1 for now;
# restore GPUS=(0 1) once the admin fixes/replaces GPU 0.
GPUS=(1)

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
# P2 FIRST SLICE — the copy-50 discretization-gap GO/NO-GO (see research/04_experiment_log
# 2026-07-02 + workmap §F). Decisive question: does the bistable `latch` close the discrete
# gap at L=50 where soft-multiply `gated` cannot? If latch DISCRETE (test_acc) >> gated
# DISCRETE here, P2 is real → expand to the full length sweep {8,20,35,50} + Tier 2. Common:
# copy alphabet 8 (chance .125), hidden 1024, 20k iters, lr 0.003→0.0003 cosine, kb 3 (copy =
# hold/recall). NOTE: if the SOFT acc is capacity-bound (P1 saw copy-50 needs ~2048), bump
# --hidden to 2048 and re-tag. ~10 runs × ~20-50 min on the one GPU ≈ an overnight queue.
# (Prior P1 recall/delay runs are complete — results already in results/, so they'd SKIP.)
JOBS=(
  # Primary — the headline comparison, 3 seeds each
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3          --tag cp50_gated_s0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 1 --tag cp50_gated_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 2 --tag cp50_gated_s2"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6          --tag cp50_latch_s0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6 --seed 1 --tag cp50_latch_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6 --seed 2 --tag cp50_latch_s2"
  # Control (recompute-recurrence — expect dead at chance on copy-50)
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --tag cp50_rddlgn"
  # Ablations (1 seed) — isolate the mechanism
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --soft-state  --tag cp50_latch_softstate"  # v0 (no bistable restore) → expect a gated-like gap
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3               --tag cp50_latch_noanneal"   # hard-from-scratch (no --anneal) → tests anneal necessity
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6 --entropy-reg 0.02 --entropy-ramp 0.5 --tag cp50_latch_ent"  # + entropy-reg for the gate-selection tail
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

# deal jobs round-robin across GPUS, one worker per GPU
n=${#GPUS[@]}
echo "$(ts) queue start: ${#JOBS[@]} job(s) across GPU(s): ${GPUS[*]}"
for k in "${!GPUS[@]}"; do
  share=()
  for ((i=k; i<${#JOBS[@]}; i+=n)); do share+=("${JOBS[$i]}"); done
  run_worker "${GPUS[$k]}" "${share[@]}" &
done
wait
echo "$(ts) ALL DONE"
