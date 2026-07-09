#!/usr/bin/env bash
# run_queue_p1.sh — P1 pre-submission hardening queue (2026-07-09).
#
# STATUS: nothing here BLOCKS submission — every table/figure cell in p1_draft1.md is
# already backed by a results JSON (validated in research/20_program_validation.md §B).
# These runs harden the cells a reviewer would poke: single-seed controls sitting next to
# 3-seed treatment arms, the single-seed ends of the keep-bias sweeps, and one missing
# equal-gates control on the headline task. Tiers A+B ≈ 23 jobs ≈ 12–15 GPU-hours total
# → one overnight queue on the two DUST 2080 Tis. Delete any block you're happy to argue
# instead of measure; Tier C at the bottom is commented out (truly optional).
#
# Mechanics are identical to mlgn/seqlgn/run_queue.sh: jobs are dealt round-robin to the
# GPUs, each --tag must be unique, and a job whose results JSON already exists is SKIPPED
# (safe to re-run after a disconnect; resume is per-job). All tags here are prefixed p1f_
# so they can't collide with, or be skipped because of, any earlier run.
#
# Run detached on DUST (survives a closed browser tab):
#   cd ~/work/difflogic && mkdir -p logs
#   nohup bash mlgn/paper/p1/run_queue_p1.sh > logs/queue_p1.log 2>&1 &
#   tail -f logs/queue_p1.log
#
# AFTER THE QUEUE — update the draft (p1_draft1.md):
#   * §5.3 table: control rows become mean ± s.d. of 3 seeds (drop the "single seed" marks).
#   * §5.1 dMNIST: add the equal-gates-control-at-d50 sentence (expect exactly 0.1135 again,
#     grad_profile[0] == 0 — check logs/p1f_dm50_rddlgn_eqgates.log).
#   * §5.2 / Fig 2B: kb0/kb3 recall points and the kb4 psMNIST point get error bars.
#   * App. A.1: extend the per-seed lists.
#   * Figures: plot.py must FILTER before regenerating (deep_sup in {0,None}, no anneal,
#     mechanism in {rddlgn,gated,lstm,gru_cell}) — see draft App. A.3 caveat. The p1f_ copy
#     controls land in the same (task,hidden,lr_min,iters) group as the originals, which is
#     what we want (they're seeds of the same config).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root (this file lives in mlgn/paper/p1/)
LOGS="$ROOT/logs"
RESULTS="$ROOT/mlgn/seqlgn/results"
mkdir -p "$LOGS"
cd "$ROOT"

# Physical GPUs to use (round-robin). Both cards healthy as of 2026-07-07; if one starts
# throwing CUDA error 700 / exit 188 again, drop the bad index (e.g. GPUS=(1)).
GPUS=(0 1)

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
JOBS=(
  # ═══ TIER A — highest claim-repair value per GPU-minute ══════════════════════════════
  #
  # (A1) psMNIST-28 equal-gates control, seeds 1,2  → draft §5.3 (the honest-boundary table).
  #      The 0.655 that carries "gating does not help classification" is 1 seed vs gated's 3.
  #      Existing: psm28_rddlgn_eqgates (s0, h2000 = 4,000 gates, disc 0.655). ~20 min each.
  "--task psmnist --chunk 28 --hidden 2000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_psm28_rddlgn_eqgates_s1"
  "--task psmnist --chunk 28 --hidden 2000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_psm28_rddlgn_eqgates_s2"
  # (A2) psMNIST-28 equal-WIDTH control (h1000 = 2,000 gates), seeds 1,2 — same table, 2nd row.
  #      Existing: psm28_rddlgn (s0, disc 0.620). ~15 min each.
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_psm28_rddlgn_s1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_psm28_rddlgn_s2"
  #
  # (A3) delayed-MNIST d50 EQUAL-GATES control (h2000 = 4,000 gates) — MISSING ENTIRELY.
  #      Converts §5.1's "no amount of gate count fixes a zero write-step gradient" from
  #      argument (backed only by copy data) to data ON THE HEADLINE TASK. Expect exactly
  #      0.1135 (majority class) again with grad_profile[0] == 0 — that's the point.
  #      --grad-analysis so the zero-gradient claim has a JSON to cite. ~1.5–3 h, 1 run.
  "--task smnist-pixel --chunk 784 --delay 50 --hidden 2000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --grad-analysis --tag p1f_dm50_rddlgn_eqgates"
  #
  # (A4) delayed-MNIST d50 keep-bias sweep, seeds 1,2 for kb0 and kb3 → draft §5.2 / Fig 2B.
  #      The recall side of the task-dependence dial is single-seed (kb0 0.177 / kb3 0.293);
  #      kb6 already has 3 seeds via the headline runs. ~45–60 min each.
  "--task smnist-pixel --chunk 784 --delay 50 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 0 --seed 1 --tag p1f_dm50_gated_kb0_s1"
  "--task smnist-pixel --chunk 784 --delay 50 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 0 --seed 2 --tag p1f_dm50_gated_kb0_s2"
  "--task smnist-pixel --chunk 784 --delay 50 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 1 --tag p1f_dm50_gated_kb3_s1"
  "--task smnist-pixel --chunk 784 --delay 50 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 2 --tag p1f_dm50_gated_kb3_s2"
  #
  # (A5) psMNIST-28 gated kb4, seeds 1,2 → the integration-side ENDPOINT of the same sweep
  #      (kb0 already has 3 seeds; kb1/kb2 mid-points stay 1-seed — their ordering is noise
  #      and the draft says so). Existing: psm28_gated (s0, kb4, disc 0.389). ~25 min each.
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 4 --seed 1 --tag p1f_psm28_gated_kb4_s1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 4 --seed 2 --tag p1f_psm28_gated_kb4_s2"

  # ═══ TIER B — cheap insurance (delete if you'd rather argue "~0 variance on a dead
  #     control" than measure it; research/08 called these optional for exactly that reason) ═
  #
  # (B1) copy control (h1024 = 2,048 gates), seeds 1,2 at L ∈ {20,35,50} → §5.1 table, so the
  #      control column is 3-seed like the gated column. Expect soft ≈ chance every time.
  #      ~10–40 min each.
  "--task copy --seq-len 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_cp_rddlgn_L20_s1"
  "--task copy --seq-len 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_cp_rddlgn_L20_s2"
  "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_cp_rddlgn_L35_s1"
  "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_cp_rddlgn_L35_s2"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_cp_rddlgn_L50_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_cp_rddlgn_L50_s2"
  #
  # (B2) copy EQUAL-GATES control (h2048 = 4,096 gates), seeds 1,2 at L ∈ {20,35,50}.
  #      Bonus: resolves the unanalyzed L20 anomaly (eqgates s0 = 0.126 vs narrow s0 = 0.247
  #      — seed noise between two dead circuits, or something odd?). Flagged in
  #      research/20 §E.3. ~15–60 min each.
  "--task copy --seq-len 20 --alphabet 8 --hidden 2048 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_cp_rddlgn_eqgates_L20_s1"
  "--task copy --seq-len 20 --alphabet 8 --hidden 2048 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_cp_rddlgn_eqgates_L20_s2"
  "--task copy --seq-len 35 --alphabet 8 --hidden 2048 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_cp_rddlgn_eqgates_L35_s1"
  "--task copy --seq-len 35 --alphabet 8 --hidden 2048 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_cp_rddlgn_eqgates_L35_s2"
  "--task copy --seq-len 50 --alphabet 8 --hidden 2048 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag p1f_cp_rddlgn_eqgates_L50_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 2048 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag p1f_cp_rddlgn_eqgates_L50_s2"

  # ═══ TIER C — optional; uncomment what you want ═══════════════════════════════════════
  #
  # (C1) LSTM keep-bias 6 at L20/L35 → firms §5.4's "the two-gate design needs ever stronger
  #      coordinated init as length grows" (kb4 works at L20, dies at L35 — does kb6 rescue
  #      L35 or not? Either answer strengthens the sentence). Old checklist item, never run.
  # "--task copy --seq-len 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism lstm --keep-bias 6 --tag p1f_cp_lstm_kb6_L20"
  # "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism lstm --keep-bias 6 --tag p1f_cp_lstm_kb6_L35"
  #
  # (C2) copy-50 capacity run (gated, h2048), seeds 1,2 → the "doubling capacity lifts the
  #      deployed circuit to 0.76" parenthetical (§5.1/§5.5) is single-seed. Config: 30k
  #      iters + LR decay, NO entropy reg (cleaner than either original run). ~80 min each.
  # "--task copy --seq-len 50 --alphabet 8 --hidden 2048 --iters 30000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 1 --tag p1f_cp50_gated_h2048_s1"
  # "--task copy --seq-len 50 --alphabet 8 --hidden 2048 --iters 30000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 2 --tag p1f_cp50_gated_h2048_s2"
  #
  # (C3) plain row-sMNIST credibility pair (research/20 §B.3: "plain sMNIST never run") — a
  #      one-line "both mechanisms are equivalent where memory isn't stressed" sanity row.
  #      Not referenced by any current draft table; add only if a reviewer asks. ~10 min each.
  # "--task smnist --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --tag p1f_smnist_gated_kb0"
  # "--task smnist --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn                --tag p1f_smnist_rddlgn"
  #
  # (C4) dMNIST d100 equal-gates control — same argument as (A3) at the longest delay; only
  #      if you want the full eqgates curve. LONG (~4–6 h).
  # "--task smnist-pixel --chunk 784 --delay 100 --hidden 2000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --tag p1f_dm100_rddlgn_eqgates"
  #
  # (C5) Float GRU reference (research/20 §A3: "price the accuracy-for-hardware trade before
  #      a reviewer does") — CANNOT run from this queue: train.py has no float-RNN arm yet
  #      (needs a small nn.GRU harness addition first). The draft currently owns this in
  #      §7 Limitations instead; build the harness only if the chosen workshop's reviewers
  #      are likely to demand the number.
)
# ─────────────────────────────────────────────────────────────────────────────

ts()     { date +%H:%M:%S; }
tag_of() { sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<<"$1"; }

run_worker() {
  local gpu="$1"; shift
  local job tag rc
  for job in "$@"; do
    tag="$(tag_of "$job")"
    # require a digit (the timestamp) right after the tag, so a tag is never matched by a
    # sibling run whose tag merely extends it (e.g. ..._s1 vs ..._s1_retry)
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
