#!/usr/bin/env bash
# run_queue_p3a.sh — training runs whose CHECKPOINTS feed the netlist-verification pipeline.
#
# Same harness as mlgn/paper/p2/run_queue_p2.sh (one train.py arg-string per JOBS entry,
# round-robin over the DUST GPUs, per-tag SKIP if a results JSON already exists). Every job
# here carries --save-model because the checkpoint IS the deliverable — the accuracy JSONs
# are a byproduct. NOTE the SKIP check keys on the JSON: if a run finished but you lost the
# ckpt, delete its results JSON to force a re-run.
#
# Why these runs (see mlgn/netlist/README.md + out/distractor_study/report.md +
# out/ladder_summary.md for the round-1/2 verification results they extend):
#
#   TIER 1a — clatch copy checkpoints. Rounds 1-2 proved the write→settle→readout theorem
#     on combo/gated circuits only (no clatch ckpt was ever saved). clatch is P2's hero
#     mechanism: these give the P2-native theorems (and the P2 beat-4 demo, if malcolm
#     decides demo-in-P2). 2 seeds on purpose — the ladder study showed solution FAMILIES
#     (fixed-point vs limit-cycle) vary by seed at identical accuracy.
#
#   TIER 1b — distcopy-trained checkpoints, the P3a HEADLINE experiment: the copy-trained
#     combo register is provably NOT distractor-robust ("correct iff the channel stays
#     silent" — report.md). Question: does training WITH distractors buy machine-checkable
#     robustness (distractor_hold / distractor_decode PROVED), or only empirical accuracy?
#     Configs match the existing dc_* accuracy rows exactly (kb3, ds 0.2) so the numbers
#     stay comparable; fresh v_ tags keep the accuracy census untouched.
#
#   TIER 2 — family-variance seeds for 1b; combo-on-distcopy (the pipeline's most-verified
#     mechanism); the cp50A ladder's missing c50 rung (line 79 of the old queue had no
#     --save-model — this completes the limit-cycle-crystallization curve 48→32→0→? at c50);
#     T-FF parity checkpoints (a verified "this circuit computes parity for ALL inputs,
#     forever" theorem = the cleanest possible P2 demo; needs a ~10-line parity_decode
#     property builder on the laptop side, running-XOR shadow + head — not yet written).
#
# Code-version note (difflogic.py wiring buffers, changed 2026-07-10 on the laptop, not yet
# committed): it does NOT matter which version DUST runs. Old code → old-format ckpts → the
# laptop's RNG-replay path reconstructs them (validated 11/11 so far). New code → ckpts are
# self-contained. The only FORBIDDEN mix is loading a NEW-format ckpt with OLD code
# (unexpected conn_a/conn_b keys) — so if you later --init-from one of these ckpts on DUST,
# make sure DUST's difflogic.py is at least as new as the code that saved it.
#
# Run detached on DUST:
#   cd ~/work/difflogic && mkdir -p mlgn/netlist/logs
#   nohup bash mlgn/netlist/run_queue_p3a.sh > mlgn/netlist/logs/queue.log 2>&1 &
#   tail -f mlgn/netlist/logs/queue.log
#
# Sync back to the laptop afterwards (checkpoints + JSONs + logs):
#   mlgn/seqlgn/results/ckpt_v_*.pt  mlgn/seqlgn/results/*_v_*.json  mlgn/netlist/logs/
#
# Laptop follow-up per checkpoint:
#   python -m mlgn.netlist.falsify --ckpt ... --json ...           (gates + hold theorems)
#   python -m mlgn.netlist.run_distractor_study                    (distcopy robustness campaign)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (script lives in mlgn/netlist/)
LOGS="$ROOT/mlgn/netlist/logs"
RESULTS="$ROOT/mlgn/seqlgn/results"
mkdir -p "$LOGS"
cd "$ROOT"

# Physical GPUs to use (round-robin).
GPUS=(0 1)

# ── EDIT ME: one line per run; keep each --tag unique ────────────────────────
JOBS=(
  # ══ TIER 1a — clatch copy checkpoints (P2-native theorems; config = cpB_clatch_ds, hit 1.000) ══
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --save-model          --tag v_cp50_clatch_s0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --seed 1 --tag v_cp50_clatch_s1"

  # ══ TIER 1b — distcopy-trained checkpoints (does distractor training buy PROVABLE robustness?)
  #    configs = the dc_* accuracy rows (kb3, ds 0.2), fresh seeds under v_ tags ══════════════════
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --tag v_dc_clatch_d8_s0"
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --save-model --tag v_dc_gated_d8_s0"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --tag v_dc_clatch_d20_s0"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --save-model --tag v_dc_gated_d20_s0"

  # ══ TIER 2 — family-variance seeds + extras (skip any block freely) ══════════════════════════

  # (2.1) second seed for the key distcopy cells — the ladder showed solution families flip by seed
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --seed 1 --tag v_dc_clatch_d20_s1"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --save-model --seed 1 --tag v_dc_gated_d20_s1"

  # (2.2) combo on distcopy (kb1 like the verified combo-copy circuits) — links the new results
  #       to the mechanism all round-1/2 theorems were proved on
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --tag v_dc_combo_d8_s0"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --tag v_dc_combo_d20_s0"

  # (2.3) cp50A ladder c50 rung WITH a checkpoint (the old c50 run never saved one) — completes
  #       the crystallization curve; needs ckpt_cp50A_curr_c35.pt present in results/ (it is, if
  #       this repo copy ran the original ladder; train.py falls back to fresh-start if missing,
  #       which would NOT be the ladder continuation — check the log line "[init] warm-started")
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --save-model --init-from mlgn/seqlgn/results/ckpt_cp50A_curr_c35.pt --tag v_cp50A_curr_c50"

  # (2.4) T-FF parity checkpoints (config = the pd16 panel) — enables the "computes parity for
  #       ALL inputs, forever" theorem, the cleanest P2 demo candidate
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --save-model          --tag v_pd16_tff_s0"
  "--task parity --seq-len 16 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --save-model --seed 1 --tag v_pd16_tff_s1"

  # (2.5) clatch copy-35 — direct same-task comparison against the fully-verified combo c35 circuit
  "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --save-model --tag v_cp35_clatch_s0"
)
# ─────────────────────────────────────────────────────────────────────────────

ts()     { date +%H:%M:%S; }
tag_of() { sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<<"$1"; }

run_worker() {
  local gpu="$1"; shift
  local job tag rc
  for job in "$@"; do
    tag="$(tag_of "$job")"
    # require a digit (the timestamp) right after the tag, so e.g. tag "v_dc_clatch_d20_s0"
    # is NOT matched by a sibling "v_dc_clatch_d20_s1_<stamp>.json" file
    if compgen -G "$RESULTS/*_${tag}_[0-9]*.json" >/dev/null; then
      echo "$(ts) [gpu$gpu] SKIP  $tag (results already exist)"; continue
    fi
    echo "$(ts) [gpu$gpu] START $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python -m mlgn.seqlgn.train $job >"$LOGS/$tag.log" 2>&1
    rc=$?
    echo "$(ts) [gpu$gpu] DONE  $tag (exit $rc) -> mlgn/netlist/logs/$tag.log"
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
