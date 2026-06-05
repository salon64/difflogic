# Experiment protocol — Paper #1 (gating)

The claim: **a logic-native gate (2:1 MUX per bit) lets a recurrent LGN carry state over
long sequences where the recompute-every-step control (`rddlgn`) fails**, because the
kept branch is a constant-error carousel. This doc says exactly what to run and report.

## The controlled comparison

Hold *everything* fixed and flip one variable: `--mechanism rddlgn` vs `--mechanism gated`.

```bash
# realistic long-range task
python -m mlgn.seqlgn.train --task psmnist --mechanism rddlgn --hidden 2000 --iters 50000 --seed 0
python -m mlgn.seqlgn.train --task psmnist --mechanism gated  --hidden 2000 --iters 50000 --seed 0

# controllable difficulty axis (the headline plot)
for L in 16 32 64 128 256; do
  python -m mlgn.seqlgn.train --task parity --seq-len $L --mechanism rddlgn --grad-analysis
  python -m mlgn.seqlgn.train --task parity --seq-len $L --mechanism gated  --grad-analysis
done
```

Always run **≥3 seeds** (`--seed 0,1,2`) and report mean ± std — discrete LGNs are noisy.

## Capacity fairness (important)

`gated` has two `LogicMLP`s (candidate + gate); `rddlgn` has one. At equal
`--hidden`/`--cell-layers` the gated cell uses ~2× the gates. To avoid "gating wins
because it's bigger," do **both**:

1. **Equal-width** comparison (same `--hidden`) — the natural ablation, but note the gate
   count difference.
2. **Equal-gates** comparison — give `rddlgn` more width/layers so total
   `logic gates` (printed each run, and in the results JSON) roughly match `gated`.

Report accuracy **vs gate count**, not just vs `--hidden`.

## What to measure / report

Per run, `train.py` prints and saves to `results/*.json`:
- `best_val`, `test_acc` (discrete-locked), `train_minutes`, `logic_gates`.
- with `--grad-analysis`: `grad_profile` = `[‖dL/dh_t‖ for t in 0..T-1]`.

Headline artifacts for the paper:
1. **Accuracy vs sequence length** (`parity`/`copy`, L on x-axis), `rddlgn` vs `gated`.
   Expected: `rddlgn` falls to chance earlier; `gated` holds longer.
2. **psMNIST test accuracy** (+ `smnist-pixel` reference), equal-width and equal-gates.
3. **Gradient-norm-through-time** curves: plot `grad_profile` for both. Expected: `gated`
   is roughly flat (carousel); `rddlgn` decays sharply toward early `t`. This is the
   *mechanistic* evidence that ties the accuracy gap to gradient flow.
4. (Optional) **gate distribution** (`--show-gates`): does the gate network concentrate on
   pass-through/keep-style gates? Interpretability bonus.

> Instrument check (2026-06-04, CPU, untrained, seq=8): `rddlgn` already shows
> earliest/latest grad ratio ~9e-12 (severe decay). Confirms the measurement works; the
> real plots come from trained GPU runs.

## Training notes

- Optimizer Adam, `--lr 0.01` (difflogic default), `CrossEntropyLoss`, iteration-based.
- For long unrolls, try raising `--grad-factor` (e.g. 2) for the `rddlgn` control — it's
  the difflogic knob for vanishing gradients, and using it makes the control a *fair,
  tuned* baseline (don't beat a hobbled baseline; see research/06_paper_plan.md).
- `--tau` (GroupSum temperature) ~30 is a sane default; sweep if val is unstable.
- Bake in Gumbel+STE / IWP later as shared training infra (research backlog) so both arms
  benefit equally.

## Logging results back

Copy the printed `LOG-LINE:` into [`../../research/04_experiment_log.md`](../../research/04_experiment_log.md)
with a one-line takeaway. Keep failures too.

## Definition of "P1 is working"

The minimum result worth writing up: on at least one memory-stressing task (psMNIST or
parity/copy at large L), **`gated` beats `rddlgn` at matched gate count**, and the
gradient-norm-through-time plot shows the carousel (flat `gated`, decaying `rddlgn`). If
even `gated` can't carry state, that's a finding too — report it and dig into why.
