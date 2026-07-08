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

# Physical GPUs to use (round-robin). Both cards healthy as of 2026-07-07 (fresh
# GPUs, bus 1A/1B). If one starts throwing CUDA error 700 / exit 188 again, drop the
# bad index here (e.g. GPUS=(1)) to route around it.
GPUS=(0 1)

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
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3          --tag cp50_gated_s0_gate_distribution"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 1 --tag cp50_gated_s1_gate_distribution"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --seed 2 --tag cp50_gated_s2_gate_distribution"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6          --tag cp50_latch_s0_gate_distribution"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6 --seed 1 --tag cp50_latch_s1_gate_distribution"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6 --seed 2 --tag cp50_latch_s2_gate_distribution"
  # Control (recompute-recurrence — expect dead at chance on copy-50)
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --tag cp50_rddlgn_gate_distribution"
  # Ablations (1 seed) — isolate the mechanism
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --soft-state  --tag cp50_latch_softstate_gate_distribution"  # v0 (no bistable restore) → expect a gated-like gap
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3               --tag cp50_latch_noanneal_gate_distribution"   # hard-from-scratch (no --anneal) → tests anneal necessity
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 3 --anneal 0.1,0.6 --entropy-reg 0.02 --entropy-ramp 0.5 --tag cp50_latch_ent_gate_distribution"  # + entropy-reg for the gate-selection tail

  # ── ROUND 2 (2026-07-03): pure SR latch was DEAD at chance on copy-50 (write-value/enable
  # entangled in S/R → can't learn the write). NEW go/no-go = `combo` (gated write + bistable
  # restore): should train like gated (soft ~0.83) AND close the gap (discrete ~= soft, not 0.33).
  # Compare cp50_combo_* discrete vs cp50_gated_* discrete (0.33). + one pure-latch long shot.
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 3 --anneal 0.1,0.6          --tag cp50_combo_s0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 3 --anneal 0.1,0.6 --seed 1 --tag cp50_combo_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 3 --anneal 0.1,0.6 --seed 2 --tag cp50_combo_s2"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 3                        --tag cp50_combo_noanneal"  # does combo even need the anneal? (gated trains, so maybe not)
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 40000 --eval-freq 2000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 6 --anneal 0.1,0.6 --tag cp50_latch_kb6"  # pure-latch long shot: stronger hold-bias + 2x iters

  # ── PATH A (2026-07-03): rescue attempts for the "never-write collapse" (round 2 diagnosis:
  # hard-state -> latch set/reset collapse to FALSE, combo gate collapses to TRUE=always-keep;
  # keep-bias DRIVES it, kb6->FALSE 100%). Two justified fixes: (A1) LOW keep-bias (kb0/1) +
  # SLOW anneal (soft until 85%) so writes can form before hardening; (A2) LENGTH CURRICULUM
  # (warm-start copy-8->20->35->50 — hard-state trains fine at L=8, so transfer the solution up).
  # WIN = combo/latch reaches gated-like SOFT (~0.83) AND keeps DISCRETE high (gap closed).
  # A1 — keep-bias x slow-anneal, direct at copy-50
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 0 --anneal 0.05,0.85 --tag cp50A_combo_kb0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.05,0.85 --tag cp50A_combo_kb1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 0 --anneal 0.05,0.85 --tag cp50A_latch_kb0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind sr --keep-bias 1 --anneal 0.05,0.85 --tag cp50A_latch_kb1"
  # A2 — combo LENGTH CURRICULUM (must run in listed order; GPUS=(1) => sequential, so ordering holds).
  #      each stage warm-starts from the previous ckpt (results/ckpt_<tag>.pt). Fresh 1st run only.
  "--task copy --seq-len 8  --alphabet 8 --hidden 1024 --iters 10000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --save-model --tag cp50A_curr_c8"
  "--task copy --seq-len 20 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --save-model --init-from mlgn/seqlgn/results/ckpt_cp50A_curr_c8.pt  --tag cp50A_curr_c20"
  "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --save-model --init-from mlgn/seqlgn/results/ckpt_cp50A_curr_c20.pt --tag cp50A_curr_c35"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6              --init-from mlgn/seqlgn/results/ckpt_cp50A_curr_c35.pt --tag cp50A_curr_c50"

  # ── ROUND 3 (2026-07-03, workflow-informed): the KEY INSIGHT — plain `gated` ALREADY deploys an
  # exactly-binary state; hard-rounding the STATE was unneeded and CAUSED the collapse. Two arms
  # HEAD-TO-HEAD (keep whichever closes the gap): (1) `clatch` = round the write-ENABLE not the value
  # (learnable write-enabled register: exact hold, no moat, no collapse — and PRESERVES the latch
  # primitive C1); (2) DROP the round, close the drift gap on plain gated with --margin-reg (push state
  # values to {0,1}) + --deep-sup (per-step recall loss). WIN = discrete test_acc >> gated's 0.33.
  # First run the ORACLE (state-drift histogram + saves a gated ckpt).
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --state-hist --save-model --tag cpB_gated_oracle"
  # Arm 1 — input-clocked latch (round the enable), 3 seeds
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6          --state-hist --tag cpB_clatch_s0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 1 --tag cpB_clatch_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --seed 2 --tag cpB_clatch_s2"
  # Arm 2 — drop-round: margin + deep-sup on gated, 3 seeds
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --margin-reg 0.1 --deep-sup 0.2          --tag cpB_gated_md_s0"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --margin-reg 0.1 --deep-sup 0.2 --seed 1 --tag cpB_gated_md_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --margin-reg 0.1 --deep-sup 0.2 --seed 2 --tag cpB_gated_md_s2"
  # Ablations (1 seed) — isolate margin vs deep-sup, + clatch+deep-sup booster
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --margin-reg 0.1 --state-hist --tag cpB_gated_marginonly"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                --tag cpB_gated_dsonly"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --tag cpB_clatch_ds"

  # ── ROUND 4 (2026-07-04): CONFIRM the round-3 winners at MULTI-SEED before locking the headline.
  # Round 3 showed the copy-50 gap IS closeable, but the only 3-seed-robust win was gated+deep-sup;
  # the two configs we'd actually headline (clatch+deep-sup, combo+curriculum) were SINGLE-SEED. So:
  #   (1) clatch + deep-sup  x3 seeds  — is the primitive+deep-sup route robust, or was s?=0 lucky?
  #   (2) gated  + deep-sup  x3 seeds  — the FOIL, confirm 1.000 holds without --margin (drop margin: it
  #       exploded standalone; cpB_gated_dsonly already =1.000 at s0, so this pins the deep-sup-only foil).
  #   (3) combo  + curriculum x2 more seeds (c8->50 ladders) — is the warm-start route robust?
  # WIN CRITERION IS NO LONGER "reach 1.000" (copy-50 saturates). It's: which route is 3-seed-STABLE at
  # disc 1.000 with the fewest knobs. (Separating clatch>gated needs a HARDER task — queue that next,
  # once we pick the surviving routes here.)
  # (1) clatch + deep-sup, seeds 1-2 (seed 0 = cpB_clatch_ds above)
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --seed 1 --tag cp4_clatch_ds_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --seed 2 --tag cp4_clatch_ds_s2"
  # (2) gated + deep-sup ONLY (no margin), seeds 1-2 (seed 0 = cpB_gated_dsonly above)
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --deep-sup 0.2 --seed 1 --tag cp4_gated_ds_s1"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated --keep-bias 3 --deep-sup 0.2 --seed 2 --tag cp4_gated_ds_s2"
  # (3) combo length curriculum, 2 more seeds. Each ladder must run in order (GPUS=(1) => sequential).
  #     seed 1 ladder
  "--task copy --seq-len 8  --alphabet 8 --hidden 1024 --iters 10000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 1 --save-model --tag cp4_curr_s1_c8"
  "--task copy --seq-len 20 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 1 --save-model --init-from mlgn/seqlgn/results/ckpt_cp4_curr_s1_c8.pt  --tag cp4_curr_s1_c20"
  "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 1 --save-model --init-from mlgn/seqlgn/results/ckpt_cp4_curr_s1_c20.pt --tag cp4_curr_s1_c35"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 1              --init-from mlgn/seqlgn/results/ckpt_cp4_curr_s1_c35.pt --tag cp4_curr_s1_c50"
  #     seed 2 ladder
  "--task copy --seq-len 8  --alphabet 8 --hidden 1024 --iters 10000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 2 --save-model --tag cp4_curr_s2_c8"
  "--task copy --seq-len 20 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 2 --save-model --init-from mlgn/seqlgn/results/ckpt_cp4_curr_s2_c8.pt  --tag cp4_curr_s2_c20"
  "--task copy --seq-len 35 --alphabet 8 --hidden 1024 --iters 15000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 2 --save-model --init-from mlgn/seqlgn/results/ckpt_cp4_curr_s2_c20.pt --tag cp4_curr_s2_c35"
  "--task copy --seq-len 50 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 1 --anneal 0.1,0.6 --seed 2              --init-from mlgn/seqlgn/results/ckpt_cp4_curr_s2_c35.pt --tag cp4_curr_s2_c50"

  # ── BENCHMARK-HUNT FOLLOW-UPS (2026-07-04, see research/18_sequential_benchmarks.md). copy-50
  # SATURATES (all routes -> disc 1.000) so it cannot rank clatch vs gated. Two tasks that CAN:
  #  (1) PARITY = a self-referential recurrence (state = state XOR bit). An exact T flip-flop / register
  #      should beat the soft-MUX as L grows; this is the IN-DISTRIBUTION length probe (find where gated
  #      cracks and tff/clatch hold). keep-bias 0 (parity TOGGLES -- a hold-bias is wrong here).
  #      NB: the true train-short/test-long length-GENERALIZATION eval (the money figure) needs a small
  #      train.py change (--test-seq-len, build test loader at a different L); this block is the precursor.
  #  (2) psMNIST = an integration/recall task (NOT a separator -- expect clatch ~= gated = "no regression
  #      on a real hard sequential benchmark"). Credibility row, matched to the existing psm28 runs
  #      (hidden 1000, 20k iters, chunk 28); gated/rddlgn psm28 already in results/ for the head-to-head.
  # (1) parity in-distribution length probe: gated (foil) / tff (exact toggle primitive) / clatch, L in {32,128}
  "--task parity --seq-len 32  --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0                                 --tag bh_parity_gated_L32"
  "--task parity --seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0                                 --tag bh_parity_gated_L128"
  "--task parity --seq-len 32  --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --tag bh_parity_tff_L32"
  "--task parity --seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --tag bh_parity_tff_L128"
  "--task parity --seq-len 32  --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6                 --tag bh_parity_clatch_L32"
  "--task parity --seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6                 --tag bh_parity_clatch_L128"
  "--task parity --seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn                                               --tag bh_parity_rddlgn_L128"  # control, expect chance
  # (2) psMNIST credibility rows (chunk 28 => 28 steps, matched to existing psm28 gated/rddlgn)
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6                 --tag bh_psm28_clatch_kb1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 1 --anneal 0.1,0.6 --tag bh_psm28_tff_kb1"

  # ── PARITY LENGTH-GENERALIZATION (the money figure; needs the --test-seq-len support just added).
  # Train SHORT (L=32) where every mechanism can learn parity, then eval the SAME model on LONGER test
  # sequences. Model selection stays on train-length (L=32) val; test_acc is the length-gen number.
  # Together with bh_parity_*_L32 above (test=train=32) this gives a curve test in {32,128,256}: an exact
  # T flip-flop / register (tff/clatch) should stay flat; the soft-MUX (gated) should ROLL OFF as it drifts.
  # (If a mechanism can't even learn parity in-dist at L=32 -- see bh_parity_*_L32 -- its curve is chance
  # at every length, which is itself the finding: bump --hidden/--iters and re-tag.)
  "--task parity --seq-len 32 --test-seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0                                 --tag bh_pargen_gated_t128"
  "--task parity --seq-len 32 --test-seq-len 256 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0                                 --tag bh_pargen_gated_t256"
  "--task parity --seq-len 32 --test-seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --tag bh_pargen_tff_t128"
  "--task parity --seq-len 32 --test-seq-len 256 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --tag bh_pargen_tff_t256"
  "--task parity --seq-len 32 --test-seq-len 128 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6                 --tag bh_pargen_clatch_t128"
  "--task parity --seq-len 32 --test-seq-len 256 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6                 --tag bh_pargen_clatch_t256"

  # ── SELECTIVE COPY (the HEADLINE separator, research/18). One data symbol at a random early position
  # among blanks, NO cue bit (content-based) -> must detect-and-hold across a variable gap. copy saturated
  # for both; this should SEPARATE (soft-MUX leaks per blank step, rounded write-enable register holds).
  # Uses --deep-sup 0.2 (length needs it, per cp4/round-3). Head-to-head gated(kb3) vs clatch(kb1,anneal)
  # at L=50 (copy-matched) and L=100 (longer gap -> sharper leak). WIN = clatch discrete >> gated discrete.
  # Plus the flag ablation: --sel-flag re-adds the cue bit -> expect gated to re-saturate (proves the gap
  # is created by content-selection, not the mechanism alone).
  "--task selcopy --seq-len 50  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --tag bh_selcopy_gated_L50"
  "--task selcopy --seq-len 50  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --tag bh_selcopy_clatch_L50"
  "--task selcopy --seq-len 100 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --tag bh_selcopy_gated_L100"
  "--task selcopy --seq-len 100 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 1 --anneal 0.1,0.6 --deep-sup 0.2 --tag bh_selcopy_clatch_L100"
  "--task selcopy --seq-len 50  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2 --sel-flag       --tag bh_selcopy_gated_L50_flag"  # ablation: flag -> expect re-saturation

  # ── ROUND 5 (2026-07-07, workflow-verified): the two prior separators BROKE (parity dead = final-only
  # supervision on the flat XOR gradient; selcopy K=1 = OR-solvable, never tests hold-vs-overwrite). Two
  # CORRECTED experiments decide the P2 headline. See research/04_experiment_log 2026-07-07 + doc 18.
  #
  # (1) PARITY-DENSE = the TRACK-A GATE. --running-target makes deep-sup supervise the per-step running-XOR
  #     (dense gradient) -> breaks the flat-parity wall (CPU-smoke validated: tff soft 0.50->0.75, XOR gates
  #     emerge). WIN = tff and/or clatch reach >0.9 disc with ~0 gap while gated LAGS/leaks => a clean,
  #     mechanistically-motivated separator (parity is the toggle primitive's home turf). If gated matches =>
  #     Track A exhausted, pivot to Track B. L=32, kb0, hidden 512, 20k, deep-sup 0.3, 3 seeds each.
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target          --tag pd_tff_s0"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 1 --tag pd_tff_s1"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism latch --latch-kind tff --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 2 --tag pd_tff_s2"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target          --tag pd_clatch_s0"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 1 --tag pd_clatch_s1"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --deep-sup 0.3 --running-target --seed 2 --tag pd_clatch_s2"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --deep-sup 0.3 --running-target          --tag pd_gated_s0"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --deep-sup 0.3 --running-target --seed 1 --tag pd_gated_s1"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --deep-sup 0.3 --running-target --seed 2 --tag pd_gated_s2"
  "--task parity --seq-len 32 --hidden 512 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism rddlgn --deep-sup 0.3 --running-target --tag pd_rddlgn_ctrl"  # memoryless control: should stay low

  # (2) DISTCOPY = the CORRECTED HOLD SEPARATOR. cued target at t=0 (deep-sup valid) + N non-cued distractor
  #     tokens to hold THROUGH (not OR-solvable). MATCHED keep-bias (both kb3, a hold task) removes the selcopy
  #     confound. WIN = clatch disc > gated disc AND clatch gap < gated gap, and the gap GROWS with #distractors
  #     for gated only (leak) but not clatch (exact hold). L=50, deep-sup 0.2, distractors in {8,20}, 2 seeds.
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --tag dc_gated_d8_s0"
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2 --seed 1        --tag dc_gated_d8_s1"
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2          --tag dc_clatch_d8_s0"
  "--task distcopy --seq-len 50 --distractors 8  --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --seed 1 --tag dc_clatch_d8_s1"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2                 --tag dc_gated_d20_s0"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 3 --deep-sup 0.2 --seed 1        --tag dc_gated_d20_s1"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2          --tag dc_clatch_d20_s0"
  "--task distcopy --seq-len 50 --distractors 20 --alphabet 8 --hidden 1024 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 3 --anneal 0.1,0.6 --deep-sup 0.2 --seed 1 --tag dc_clatch_d20_s1"

  # (3) psMNIST FAIR credibility match (kb0 = gated's best; clatch was only run at kb1). Sanity/no-regression,
  #     NOT a separator. clatch kb0 x3 + gated kb0 x2 more (s0 exists: disc 0.632). Win = clatch within ~1-2pt.
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6          --tag psm_clatch_kb0_s0"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --seed 1 --tag psm_clatch_kb0_s1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism clatch --keep-bias 0 --anneal 0.1,0.6 --seed 2 --tag psm_clatch_kb0_s2"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --seed 1 --tag psm_gated_kb0_s1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism gated  --keep-bias 0 --seed 2 --tag psm_gated_kb0_s2"

  # (4) psMNIST `combo` (= gated + hard-STATE bistable restore; the copy-50 collapse mechanism) x3, matched to
  #     the gated/clatch kb0 rows -> completes the mechanism table (gated/clatch/combo/rddlgn). NB: `combo` is
  #     NOT gated+clatch; the true gated(+)clatch per-bit LEARNED hybrid is logged as post-P2 future work
  #     ("latches as trainable params, not architectural"; see research/17 §2b). Insight: does the bistable
  #     restore survive at psm28's short length (28 steps) where the never-write collapse may not bite?
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 0 --anneal 0.1,0.6          --tag psm_combo_kb0_s0"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 0 --anneal 0.1,0.6 --seed 1 --tag psm_combo_kb0_s1"
  "--task psmnist --chunk 28 --hidden 1000 --iters 20000 --eval-freq 1000 --lr 0.003 --lr-min 0.0003 --mechanism combo --keep-bias 0 --anneal 0.1,0.6 --seed 2 --tag psm_combo_kb0_s2"
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
