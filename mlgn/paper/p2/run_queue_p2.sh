#!/usr/bin/env bash
# run_queue_p2.sh — the REMAINING runs for Paper #2 ("Clock the Enable, Not the Value").
#
# Copy of the mlgn/seqlgn/run_queue.sh harness (same conventions: one train.py arg-string
# per JOBS entry, round-robin over the DUST GPUs, per-tag SKIP if a results JSON already
# exists). The historical queue with every completed P2 run stays at mlgn/seqlgn/run_queue.sh;
# THIS file holds only what is still worth running, per:
#   - research/11_paper2_workmap.md §A0'  — "no GPU runs required before writing"; parity-L16
#     mechanism panel listed as the one optional nice-to-have. Do NOT run distcopy-d40
#     (accuracy tied, gap-sign axis refuted) and do NOT resume separator hunts.
#   - research/20_program_validation.md §F — cheap GPU backfills in order of claim-repair value
#     (items 7-9, 11), which map to the draft's inline [TODO]s (mlgn/paper/p2/p2_draft1.md).
#
# TIER 1 repairs claims already in the draft (distcopy ≥3-seed rule; kb-matched selcopy-L100
# stability pair). TIER 2 is optional hardening. Nothing here uses --init-from, so round-robin
# ordering is safe. Logs are written INSIDE mlgn/paper/p2/logs/ so the provenance of paper
# numbers can be committed with the draft (fixes the "queue logs not archived" gap from the
# validation report).
#
# NOT in this queue (blocked on code, tracked in the draft's TODOs / §F):
#   - float GRU baseline on copy-50 / distcopy-d20 / psMNIST-28 (train.py has no float-RNN
#     mechanism yet — add a small nn.GRU harness first; §F item 10)
#   - n-bit memorization insurance pair (needs a new data.py task; §F item 11 / doc 18)
#   - netlist/AIGER exporter + ABC smoke + yosys report (code artifact, not a training run —
#     the §8 build-or-descope decision in the draft; §F item 4)
#   - P1 backfills (delayed-MNIST re-basing etc.) — P1's checklist, not P2's.
#
# Run detached on DUST (mkdir first — the shell opens the redirect before the script runs):
#   cd ~/work/difflogic && mkdir -p mlgn/paper/p2/logs
#   nohup bash mlgn/paper/p2/run_queue_p2.sh > mlgn/paper/p2/logs/queue.log 2>&1 &
#   tail -f mlgn/paper/p2/logs/queue.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root (script lives in mlgn/paper/p2/)
LOGS="$ROOT/mlgn/paper/p2/logs"
RESULTS="$ROOT/mlgn/seqlgn/results"
mkdir -p "$LOGS"
cd "$ROOT"

# Physical GPUs to use (round-robin). Both cards healthy as of 2026-07-07.
GPUS=(0 1)

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
JOBS=(
  # ══ TIER 1 — claim repair (do these; ~6 runs × ~40 min ≈ 2 h wall on 2 GPUs) ══════════════

  # (1.1) distcopy SEED-2 BACKFILL (validation §F item 8; draft §7.2 TODO).
  #       The d8/d20 accuracy rows are n=2 — one more seed per cell meets the project's own
  #       ≥3-seed headline rule. Same config as dc_* round 5 (matched kb3, ds 0.2).
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2 --seed 2                 --tag dc_gated_d8_s2"
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --seed 2 --tag dc_clatch_d8_s2"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2 --seed 2                 --tag dc_gated_d20_s2"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --seed 2 --tag dc_clatch_d20_s2"

  # (1.2) selcopy-L100 KB-MATCHED PAIR (validation §F item 7; draft §7.3 TODO).
  #       The flagship stability pair (gated 2082 skips vs clatch 0) is kb-mismatched (3 vs 1).
  #       Run each mechanism at the OTHER's keep-bias so a matched pair exists at BOTH kb values
  #       → upgrades §7.3's "task-tuned settings" wording into a controlled comparison. Everything
  #       else identical to bh_selcopy_*_L100. (Accuracy from these is still census-only — the
  #       K=1 task stays excluded from accuracy claims per §7.2.)
  "--task selcopy --seq-len 100 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 1 --deep-sup 0.2                 --tag sc_gated_L100_kb1"
  "--task selcopy --seq-len 100 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --tag sc_clatch_L100_kb3"

  # ══ TIER 2 — optional hardening (skip any block freely; queue order ≈ priority) ═══════════

  # (2.1) psMNIST-28 kb0, +2 SEEDS PER ARM (validation §F item 9; draft §7.4 TODO).
  #       Seeds 3-4 for gated/clatch/combo stabilize the non-overlapping-gap claim and let the
  #       paper apply a symmetric outlier policy (gated-s2 0.519 currently does double duty).
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --seed 3                  --tag psm_gated_kb0_s3"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --seed 4                  --tag psm_gated_kb0_s4"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --seed 3 --tag psm_clatch_kb0_s3"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --seed 4 --tag psm_clatch_kb0_s4"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo  --keep-bias 0 --anneal 0.1,0.6 --seed 3 --tag psm_combo_kb0_s3"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo  --keep-bias 0 --anneal 0.1,0.6 --seed 4 --tag psm_combo_kb0_s4"

  # (2.2) psMNIST-28 rddlgn EQUAL-GATES control, seeds 1-2 (validation A2.9: the 4-way table's
  #       rddlgn row traces to a single June seed-0 run). hidden 2000 ⇒ 4,000 gates = matched to
  #       the gated/clatch arms; keep-bias is a no-op for rddlgn (no gate net). Upgrades the
  #       Table-3 "own config †" footnote row into a seeded control.
  "--task psmnist --chunk 28 --hidden 2000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 1 --tag psm28_rddlgn_eqgates_s1"
  "--task psmnist --chunk 28 --hidden 2000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --seed 2 --tag psm28_rddlgn_eqgates_s2"

  # (2.3) PARITY L=16 MECHANISM PANEL (workmap §A0' optional nice-to-have; ~15 min/run).
  #       Track-B *mechanism* figure: with the dense running-XOR target at a SHORT length, the
  #       primitive whose inductive bias matches the task (tff) should be the sole mover — the
  #       "deep-sup enables; the matched primitive moves" panel for §7.2/[FIG 5]. It is about tff
  #       on its home task and cannot change the accuracy verdict; skippable by A0'.
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target          --tag pd16_tff_s0"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 1 --tag pd16_tff_s1"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 2 --tag pd16_tff_s2"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --deep-sup 0.3 --running-target          --tag pd16_gated_s0"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --deep-sup 0.3 --running-target --seed 1 --tag pd16_gated_s1"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --deep-sup 0.3 --running-target --seed 2 --tag pd16_gated_s2"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target          --tag pd16_clatch_s0"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 1 --tag pd16_clatch_s1"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 2 --tag pd16_clatch_s2"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --deep-sup 0.3 --running-target --tag pd16_rddlgn_ctrl"

  # (2.4) psMNIST-28 kb0 + DEEP-SUP (validation §F item 11): does the method claim survive an
  #       integration task? §6.1 scopes deep-sup to holds; this prices it off its home turf
  #       (per-step supervision against the final label is only approximately valid here — that
  #       tension is the point of the probe). gated arm ×3 seeds; add clatch mirrors if the
  #       result is interesting.
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 0 --deep-sup 0.2          --tag psm_gated_kb0_ds_s0"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 0 --deep-sup 0.2 --seed 1 --tag psm_gated_kb0_ds_s1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 0 --deep-sup 0.2 --seed 2 --tag psm_gated_kb0_ds_s2"
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
    echo "$(ts) [gpu$gpu] DONE  $tag (exit $rc) -> mlgn/paper/p2/logs/$tag.log"
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
