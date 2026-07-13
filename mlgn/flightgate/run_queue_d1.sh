#!/usr/bin/env bash
# run_queue_d1.sh — P3b GATE D1: recurrence-earns-its-keep on a HOVER-MEMORY POMDP.
#
# Same harness shape as mlgn/seqlgn/run_queue_c0g.sh: each JOBS entry is one
# `mlgn.flightgate.cli` arg-string (NO `python -m …` prefix, NO --device — the worker
# adds `--device cuda` and CUDA_VISIBLE_DEVICES). Jobs are dealt round-robin to GPU 0 /
# GPU 1; a job whose results file already exists (matched by its --tag) is SKIPPED, so
# re-running RESUMES. Run detached on DUST:
#   cd ~/work/difflogic && mkdir -p logs
#   nohup bash mlgn/flightgate/run_queue_d1.sh > logs/queue_d1.log 2>&1 &
#   tail -f logs/queue_d1.log
# Read the gate table when it finishes (or any time — it tolerates missing cells):
#   python -m mlgn.flightgate.gate_eval --results-dir mlgn/flightgate/results
#
# PREREQ (see mlgn/flightgate/README.md 'Install'): the interpreter this launches
# (`python`) MUST have BOTH torch+CUDA AND gym-pybullet-drones @ e712698a
# (record it: .venv-flight/GPD_COMMIT, cli picks it up). Student training needs torch;
# the closed-loop sim needs pybullet — SAME process, so ONE env with both. On DUST use
# the image env (or a venv) with the README pins; the mock backend needs neither and is
# NOT what this queue runs (this is the real hover gate).
#
# ── THE GATE (research/23 §D1) ───────────────────────────────────────────────
# Question: does a RECURRENT logic cell (gated / clatch) fly the occluded hover task
# BETTER than a gate-count-MATCHED STATELESS logic net (ff) — closed-loop, DAgger-
# distilled from a stock PID — when a SensorBlackout wrapper zeroes the observation for
# 0.2–0.4 s bursts (memory REQUIRED). The non-vacuity CONTROL (--no-blackout) proves the
# gap is MEMORY not capacity: there ff must MATCH the recurrent arms.
#
# WIN = under blackout, recurrent (gated and/or clatch) closed-loop return >> ff by more
# than the cross-seed spread, AND ff ≈ recurrent under the no-blackout control, across
# >=3 seeds. Read it with `python -m mlgn.flightgate.gate_eval` (prints the verdict).
# FAIL => workmap D1 pre-committed pivot ("verified logic flight controller": feedforward
# + P3a verification + board; memory claims scoped). REPORT EITHER OUTCOME HONESTLY.
#
# Reference rows the table needs (auto-included below):
#   * a' CEILING (--teacher-masked, blackout only): the PID oracle fed the SAME masked
#     obs the student sees (obs->state shim, VERIFIED bit-exact on unmasked frames) —
#     upper-bounds every student under occlusion. Smoke (0.03,8,16): raw ret 480 ->
#     a' 143 (occlusion costs the ORACLE ~336 return); the gate's blackout below
#     (0.02,6,12) is milder so students have room to separate.
#   * teacher raw / teacher-through-bins are recorded in EVERY run JSON (the quantization
#     gate: teacher-through-9-bins must still solve the task; smoke ratio = 1.00).
#
# Matched-construction discipline (VERIFIED with utils.count_gates, 2026-07-12):
#   * gate parity: gated / clatch / ff ALL = 1,728 logic gates @ hidden 432, cell_layers 2
#     (build_student gives ff num_layers = 2*cell_layers so L*H_ff == 2*L*H_rec at EQUAL
#     hidden — so every arm shares --hidden 432; do NOT double ff's hidden here).
#   * hidden 432 = 12*36 is divisible by n_act*n_bins = 4*9 = 36 (GroupSum head) and
#     >= input_bits (193 with the blackout valid-bit / 192 without).
#   * SAME keep-bias (3.0) on gated AND clatch (matched pair; research/20 finding #1).
#     kb is N/A for ff (no gate MLP).
#   * SAME encoder (16-bit thermometer), bins (9), jitter (env.HOVER_JITTER, TEMPERED
#     default — deliverable 4: raw-PID envelope-exit 0.65 -> 0.00, see report), frozen
#     eval seeds (seed_base = seed*1e5+9e5) and blackout mask seeds (seed*7919) for EVERY
#     arm at a given --seed => bit-identical eval envs across arms.
#   * --bptt 0 (full-episode BPTT, the DUST setting); --rounds 4 DAgger.
#
# NOTE (velocity thermometer ranges are PLACEHOLDERS until calibrated): the first job is
# a --teacher-only run that writes `calibrated_ranges_preview` (0.5/99.5 pct, x1.5) from
# 50 jittered teacher episodes. Inspect it and, if it departs from encode.HOVER_RANGES,
# freeze the calibrated ranges before trusting the gate numbers (README gate 4).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (holds difflogic/ + mlgn/)
LOGS="$ROOT/logs"
RESULTS="$ROOT/mlgn/flightgate/results"
mkdir -p "$LOGS" "$RESULTS"
cd "$ROOT"

# Physical GPUs to use (round-robin). Drop an index if a card acts up.
GPUS=(0 1)

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
# ROUND-2 RECALIBRATION (2026-07-13, tags d1v2_*). Round 1 (d1_*) was INCONCLUSIVE:
# every student arm exited the envelope 100% in BOTH conditions (returns 15-51 vs
# teacher 440) and even the a' oracle exited 97% -> the policy never learned to hover
# at all (undertraining, visible in the no-blackout control), so the gate had no room
# to measure memory. This round: (1) SCALE the distillation hard (hidden 432->864,
# iters 2000->8000, rounds 4->6, episodes 50->100, batch 16->32; GPU time is free) so
# the students first learn to FLY the control; (2) SOFTEN the blackout (0.2-0.4s ->
# 0.13-0.27s bursts) so a memory policy can bridge it and the a' ceiling stays flyable.
# Round 1's d1_* results are preserved as the undertrained baseline.
# READ-BACK: python -m mlgn.flightgate.gate_eval --results-dir mlgn/flightgate/results \
#            --glob 'flightgate_hover_*_d1v2_*.json'
# Success signal to watch FIRST: control-condition students' envelope_exit << 100%
# (they can hover); only then does the blackout comparison mean anything. If the
# control still exits ~100% at this scale, the bottleneck is architectural (action
# discretization / DAgger / head), not compute -- diagnose before adding seeds.
CMN="--backend hover --hidden 864 --bits 16 --n-bins 9 --cell-layers 2 --rounds 6 --episodes 100 --eval-episodes 50 --iters 8000 --batch-size 32 --bptt 0 --lr 0.01 --settle-frac 0.25 --save-traj"
REC="--keep-bias 3.0"                     # gated/clatch: matched gates @ hidden 864 (2x round 1)
BK="--blackout 0.02,4,8"                  # blackout-ON: 0.13–0.27 s bursts @ 30 Hz (softened)
NB="--no-blackout"                        # non-vacuity CONTROL

JOBS=(
  # ── velocity-range calibration snapshot (run first; inspect the preview) ────
  "--backend hover --episodes 50 --eval-episodes 10 $BK --teacher-only --seed 0 --tag d1v2_calib_s0"

  # ── a' CEILING (PID on the masked obs; blackout only, 3 seeds) ─────────────
  "--backend hover --eval-episodes 50 $BK --teacher-masked --seed 0 --tag d1v2_aprime_bk_s0"
  "--backend hover --eval-episodes 50 $BK --teacher-masked --seed 1 --tag d1v2_aprime_bk_s1"
  "--backend hover --eval-episodes 50 $BK --teacher-masked --seed 2 --tag d1v2_aprime_bk_s2"

  # ── BLACKOUT-ON (gate-deciding): 3 arms x 3 seeds ──────────────────────────
  "$CMN --arm gated  $REC                 $BK --seed 0 --tag d1v2_gated_bk_s0"
  "$CMN --arm gated  $REC                 $BK --seed 1 --tag d1v2_gated_bk_s1"
  "$CMN --arm gated  $REC                 $BK --seed 2 --tag d1v2_gated_bk_s2"
  "$CMN --arm clatch $REC --anneal 0.1,0.6 $BK --seed 0 --tag d1v2_clatch_bk_s0"
  "$CMN --arm clatch $REC --anneal 0.1,0.6 $BK --seed 1 --tag d1v2_clatch_bk_s1"
  "$CMN --arm clatch $REC --anneal 0.1,0.6 $BK --seed 2 --tag d1v2_clatch_bk_s2"
  "$CMN --arm ff                          $BK --seed 0 --tag d1v2_ff_bk_s0"
  "$CMN --arm ff                          $BK --seed 1 --tag d1v2_ff_bk_s1"
  "$CMN --arm ff                          $BK --seed 2 --tag d1v2_ff_bk_s2"

  # ── NO-BLACKOUT CONTROL (non-vacuity): 3 arms x 3 seeds — ff MUST match here ─
  "$CMN --arm gated  $REC                 $NB --seed 0 --tag d1v2_gated_nb_s0"
  "$CMN --arm gated  $REC                 $NB --seed 1 --tag d1v2_gated_nb_s1"
  "$CMN --arm gated  $REC                 $NB --seed 2 --tag d1v2_gated_nb_s2"
  "$CMN --arm clatch $REC --anneal 0.1,0.6 $NB --seed 0 --tag d1v2_clatch_nb_s0"
  "$CMN --arm clatch $REC --anneal 0.1,0.6 $NB --seed 1 --tag d1v2_clatch_nb_s1"
  "$CMN --arm clatch $REC --anneal 0.1,0.6 $NB --seed 2 --tag d1v2_clatch_nb_s2"
  "$CMN --arm ff                          $NB --seed 0 --tag d1v2_ff_nb_s0"
  "$CMN --arm ff                          $NB --seed 1 --tag d1v2_ff_nb_s1"
  "$CMN --arm ff                          $NB --seed 2 --tag d1v2_ff_nb_s2"
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
    CUDA_VISIBLE_DEVICES="$gpu" python -m mlgn.flightgate.cli $job --device cuda \
      >"$LOGS/$tag.log" 2>&1
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
echo "$(ts) gate table: python -m mlgn.flightgate.gate_eval --results-dir mlgn/flightgate/results --glob 'flightgate_hover_*_d1v2_*.json'"
