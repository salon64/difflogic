#!/usr/bin/env bash
# run_queue_c0g.sh — P3b GATE C0.g: recurrence-earns-its-keep on CAN-bus IDS (ROAD).
#
# Same harness as run_queue.sh: each JOBS entry is one train.py arg-string (NO
# `python -m …` prefix, NO --device / CUDA_VISIBLE_DEVICES — the script adds those).
# Jobs are dealt round-robin to GPU 0 / GPU 1; a job whose results file already exists
# (matched by its --tag) is SKIPPED, so re-running resumes. Run detached on DUST:
#   cd ~/work/difflogic && mkdir -p logs
#   nohup bash mlgn/seqlgn/run_queue_c0g.sh > logs/queue_c0g.log 2>&1 &
#   tail -f logs/queue_c0g.log
#
# PREREQ: ROAD raw logs under data-can/road/ (zenodo.org/records/10462796, MD5
# cab184cfc2fe12c0834bc46188c0f330) with per-capture metadata JSON — see
# mlgn/seqlgn/data/can/README.md for the exact layout the loader consumes.
#
# ── THE GATE (research/23 §C0.g) ─────────────────────────────────────────────
# Question: does a RECURRENT logic circuit (gated / clatch) beat an information-matched
# WINDOWED-FEEDFORWARD logic circuit (mechanism ff, same 32-frame window flattened to one
# step) on timing-opaque attacks — ROAD's simulated-MASQUERADE suite? Flooding/fuzzing
# are excluded BY DESIGN (any frequency window solves them; they cannot decide the gate).
#
# WIN = recurrent (gated and/or clatch) discrete test recall/F1 >> ff at matched FPR on
# the masquerade captures across >=3 seeds (read test_fpr/test_recall/test_f1 from the
# JSONs: `python -m mlgn.seqlgn.collate --task can`). Every arm must first beat the
# always-normal floor (recall 0, FPR 0 — see can_attack_frac_windows_test in the JSON).
# ff MATCHING the recurrent arms => the C0 flagship falls to the research/23 §C2
# fallback chain (bearings). REPORT EITHER OUTCOME HONESTLY.
#
# Matched-construction discipline:
#   * gate parity (utils.count_gates): gated/clatch @ hidden 1024, cell_layers 2
#     = 2*2*1024 = 4096 gates; ff @ hidden 2048, cell_layers 2 = 4096 gates. Exact.
#   * SAME keep-bias (kb 1) on gated AND clatch (research/20 finding #1: the selcopy
#     kb3-vs-kb1 "matched pair" was a confound). kb is N/A for ff (no gate MLP;
#     recorded as null).
#   * SAME window (--seq-len 32), stride, encoding, split (time/capture-based, seed-
#     independent) for every arm — --seed only moves init + train-window shuffling.
#   * masquerade is frequency-preserving => payload bits are load-bearing
#     (--can-payload-bytes 8; ID one-hot top-20 + 8 log-spaced Δt bins => F = 93).
#   * per-message supervision: recurrent arms use --can-per-step + --deep-sup 0.2
#     (each state supervised against its own frame label); deep-sup is a no-op for the
#     single-step ff arm, so it is omitted there (information window still identical).
#   * --can-eval-stride 16 keeps the per-eval cost sane on ~1e6-frame captures; the
#     recorded metrics are a stride-16 subsample of per-message scoring. For the paper
#     row, re-run the winner's final eval at --can-eval-stride 1.
#   * --can-ambient is OFF here (masquerade captures carry their own normal frames);
#     the held-out-ambient FPR calibration row is a follow-up, not gate-deciding.
#
# Attacks (web-recon difficulty ranking): max_speedometer + reverse_light_on masquerade
# = the HARD set (gate-deciding, 3 seeds x 3 arms each); correlated_signal masquerade =
# easier control (1 seed per arm); rddlgn = recompute-recurrence control; ff window/depth
# sweep rows make the FF baseline honestly strong (web-recon: compare vs the BEST
# windowed model — note their gate budgets differ from 4096, recorded in the JSON).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (holds difflogic/ + mlgn/)
LOGS="$ROOT/logs"
RESULTS="$ROOT/mlgn/seqlgn/results"
mkdir -p "$LOGS"
cd "$ROOT"

# Physical GPUs to use (round-robin). Drop an index if a card acts up (see run_queue.sh).
GPUS=(0 1)

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
CMN="--task can --can-source road --seq-len 32 --can-id-enc onehot --can-top-ids 20 --can-dt-bins 8 --can-payload-bytes 8 --can-stride 4 --can-eval-stride 16 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --batch-size 128"
REC="--hidden 1024 --deep-sup 0.2 --can-per-step"       # recurrent arms: 4096 gates
FF="--mechanism ff --can-flatten --hidden 2048"         # stateless arm:  4096 gates

JOBS=(
  # ── HARD SET 1: max_speedometer masquerade (3 seeds x 3 arms) ──────────────
  "$CMN --can-attack max_speedometer+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 0 --tag c0g_speedo_gated_s0"
  "$CMN --can-attack max_speedometer+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 1 --tag c0g_speedo_gated_s1"
  "$CMN --can-attack max_speedometer+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 2 --tag c0g_speedo_gated_s2"
  "$CMN --can-attack max_speedometer+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 0 --tag c0g_speedo_clatch_s0"
  "$CMN --can-attack max_speedometer+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 1 --tag c0g_speedo_clatch_s1"
  "$CMN --can-attack max_speedometer+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 2 --tag c0g_speedo_clatch_s2"
  "$CMN --can-attack max_speedometer+masquerade $FF                                                    --seed 0 --tag c0g_speedo_ff_s0"
  "$CMN --can-attack max_speedometer+masquerade $FF                                                    --seed 1 --tag c0g_speedo_ff_s1"
  "$CMN --can-attack max_speedometer+masquerade $FF                                                    --seed 2 --tag c0g_speedo_ff_s2"

  # ── HARD SET 2: reverse_light_on masquerade (3 seeds x 3 arms) ─────────────
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 0 --tag c0g_rlon_gated_s0"
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 1 --tag c0g_rlon_gated_s1"
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 2 --tag c0g_rlon_gated_s2"
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 0 --tag c0g_rlon_clatch_s0"
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 1 --tag c0g_rlon_clatch_s1"
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 2 --tag c0g_rlon_clatch_s2"
  "$CMN --can-attack reverse_light_on+masquerade $FF                                                    --seed 0 --tag c0g_rlon_ff_s0"
  "$CMN --can-attack reverse_light_on+masquerade $FF                                                    --seed 1 --tag c0g_rlon_ff_s1"
  "$CMN --can-attack reverse_light_on+masquerade $FF                                                    --seed 2 --tag c0g_rlon_ff_s2"

  # ── EASIER CONTROL: correlated_signal masquerade (1 seed per arm) ──────────
  "$CMN --can-attack correlated_signal+masquerade $REC --mechanism gated  --keep-bias 1                 --seed 0 --tag c0g_corr_gated_s0"
  "$CMN --can-attack correlated_signal+masquerade $REC --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 0 --tag c0g_corr_clatch_s0"
  "$CMN --can-attack correlated_signal+masquerade $FF                                                    --seed 0 --tag c0g_corr_ff_s0"

  # ── rddlgn recompute-recurrence control (1 seed per hard attack) ───────────
  #    rddlgn has ONE update MLP (not candidate+gate), so $REC's hidden 1024 would give
  #    2*1024 = 2048 gates — half the matched budget. Override to hidden 2048 for exact
  #    4096-gate parity (the copy_rddlgn_rddlgn_eqgates precedent; argparse last-wins).
  "$CMN --can-attack max_speedometer+masquerade  $REC --mechanism rddlgn --keep-bias 0 --hidden 2048 --seed 0 --tag c0g_speedo_rddlgn_s0"
  "$CMN --can-attack reverse_light_on+masquerade $REC --mechanism rddlgn --keep-bias 0 --hidden 2048 --seed 0 --tag c0g_rlon_rddlgn_s0"

  # ── FF strength sweep (speedo; make the stateless baseline honestly strong).
  #    Budgets differ from 4096 (recorded in the JSON) — these rows only matter if the
  #    matched ff loses: does MORE window/depth/gates close its gap?
  "--task can --can-source road --seq-len 16 --can-id-enc onehot --can-top-ids 20 --can-dt-bins 8 --can-payload-bytes 8 --can-stride 4 --can-eval-stride 16 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --batch-size 128 --can-attack max_speedometer+masquerade --mechanism ff --can-flatten --hidden 1024 --seed 0 --tag c0g_speedo_ff_w16"
  "--task can --can-source road --seq-len 64 --can-id-enc onehot --can-top-ids 20 --can-dt-bins 8 --can-payload-bytes 8 --can-stride 4 --can-eval-stride 16 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --batch-size 128 --can-attack max_speedometer+masquerade --mechanism ff --can-flatten --hidden 4096 --seed 0 --tag c0g_speedo_ff_w64"
  "$CMN --can-attack max_speedometer+masquerade --mechanism ff --can-flatten --hidden 2048 --cell-layers 4 --seed 0 --tag c0g_speedo_ff_deep"
)
# ─────────────────────────────────────────────────────────────────────────────

ts()     { date +%H:%M:%S; }
tag_of() { sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<<"$1"; }

run_worker() {
  local gpu="$1"; shift
  local job tag rc
  for job in "$@"; do
    tag="$(tag_of "$job")"
    # require a digit (the timestamp) right after the tag, so a tag that prefixes a
    # sibling tag is not falsely matched by the sibling's results file
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
