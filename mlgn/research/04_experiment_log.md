# Experiment Log

Newest first. One block per run/idea. Keep it terse and honest (record failures too).

Template:
```
## YYYY-MM-DD — <short title>
- Hypothesis:
- Setup: (script, dataset, model, key hyperparams, seed, hardware)
- Result: (val/test acc, gates, train time)
- Read: (what it tells me / next step)
```

---

## 2026-06-08 — First GPU validation (copy task): cold-start found, keep-bias fix added
Hardware: RTX 2080S. Task: copy/recall (chance = 12.5%, alphabet 8), hidden 1024,
cell_layers 2, 20k iters. **No keep-bias yet (effectively keep_bias=0).**

| run | seq | mech | best_val | test | grad ratio (early/late) | read |
|---|---|---|---|---|---|---|
| sanity | 8 | rddlgn | 0.868 | 0.874 | — | learns, but only after a long plateau (breaks ~iter 18k) |
| sanity | 8 | gated | **1.000** | **1.000** | — | learns **instantly** (val=1.0 by iter 1k) |
| val | 50 | rddlgn | 0.258 | 0.251 | 4e-20 | dead (catastrophic vanishing) |
| val | 50 | gated | 0.252 | 0.246 | 7e-8 | **never started** — loss flat at log(8)=2.08 |

**Reads:**
- **Positive:** at seq-8 gating *dominates* (instant 100% vs rddlgn's struggling 87%);
  gated gradient flow is **~12 orders of magnitude** better than the control (7e-8 vs
  4e-20). The carousel works mechanically.
- **Negative:** at seq-50 BOTH fail. gated's flat loss = a **cold start**, not slow
  learning: the gate isn't keep-biased at init, so the symbol decays before the gate can
  *learn* to keep it (chicken-and-egg), and there's no gradient signal to bootstrap.
- **Diagnosis = the known LSTM cold-start.** Fix = positive forget/keep-gate bias at init
  (Gers et al. 2000) ≡ difflogic residual init (Petersen 2024).

**Action taken:** implemented `keep_bias` (adds to the TRUE-gate logit of the gate's final
layer → carousel ON at init, write path preserved). CLI `--keep-bias` (default 3.0),
applies to `gated` (gate) and `lstm` (forget). `bias_gate_keep` in `seqlgn/cells.py`.

**Next (GPU):** re-run copy-50 `gated` with `--keep-bias 3` (sweep {2,3,5} if needed);
expect it to break the plateau. Then rddlgn with `--grad-factor 2` for a fair control.

## 2026-06-04 — seqlgn infra built + smoke-tested (Paper #1 ready to run)
- Built `mlgn/seqlgn/`: pluggable recurrent cell (`rddlgn` control / `gated` Paper#1 /
  `latch` stub), `SequenceClassifier`, benchmarks (smnist/smnist-pixel/psmnist/parity/
  copy), CLI `train.py` w/ discrete-locked eval + grad-norm-through-time, docs.
- **CPU enablement:** difflogic does `import difflogic_cuda` at module top → unimportable
  on this CPU-only laptop. Added `seqlgn/_cpu_compat.py` (stub injection) so dev/debug
  works on CPU (`device cpu`, python impl, slow). Also removed a dead debug-print block in
  `difflogic/difflogic.py` `forward_python` that spammed stdout every forward.
- **Smoke test (CPU, parity seq=8, hidden=20):** both mechanisms run end-to-end; results
  JSON + LOG-LINE emitted; `--grad-analysis` + `--show-gates` work.
- **Early instrument signal (untrained, tiny — NOT a result):** `rddlgn` control grad
  ratio earliest/latest ≈ 9e-12 → severe vanishing through time, exactly what `gated`'s
  carousel should fix. Confirm with trained GPU runs.
- Next: real runs on a GPU box — psmnist + parity/copy sweep, `gated` vs `rddlgn`,
  equal-width AND equal-gates. Protocol: `seqlgn/docs/experiments.md`.

## 2026-06-04 — Baselines reproduced (pre-research-program)
- **`mnist_test.py`** — paper-style FC DLGN, 6×`LogicLayer(64_000)`, `GroupSum(k=10,
  tau=30)`, Adam lr 0.01, 100k iters, bs 128, inputs `.round()` at eval.
  - Result: **val 98.04% / test 98.24%** (discrete/locked gates).
- **`secuential.py`** — `LogicRNNCell` (2-layer logic cell) over **28 MNIST rows** as
  timesteps, hidden 16k, `GroupSum(k=10, tau=30)`, Adam lr 0.01, 100k iters.
  - Result: best **val 98.04%** (test ~98.2% region). First working **recurrent LGN** on
    sequential-MNIST in this fork.
  - Note: this is the prototype overlapping Recurrent DDLGN [5] but on
    sequential-image classification rather than translation — see
    [03_open_problems.md](03_open_problems.md) §B.

## Standing setup notes
- difflogic needs CUDA + CUDA Toolkit; `implementation='cuda'` only on GPU.
- Inference modes: `PackBitsTensor` (GPU) / `CompiledLogicNet` (compile to C/.so).
- Eval must binarize inputs (`.round()`) + `model.eval()` to lock argmax gates, or the
  reported number is the soft (cheating) accuracy.
- For deep nets raise `grad_factor` (~2) to fight vanishing gradients.

## Backlog — aligned to the 3-paper plan ([06_paper_plan.md](06_paper_plan.md))
Scout done (2026-06-04): #1 gating = CONDITIONAL GO (race), #2 latch = GO (anchor).

Shared infra (do first):
- [x] Generalize `LogicRNNCell` → pluggable memory mechanism (rddlgn/gated; latch stub). → `mlgn/seqlgn/`
- [x] Sequential benchmark harness: sMNIST, smnist-pixel, psMNIST, parity(L), copy(L). (adding=TODO, regression head)
- [ ] Bake in Gumbel+STE [3] and IWP [4] as default training infra.
- [x] Logging: acc / gates / train-time / grad-norm-through-time per variant (`train.py`).
- [x] CPU-dev enablement (`_cpu_compat.py`) + smoke-tested.

P1 — gating (fast, plant flag):
- [ ] MUX-gated cell vs rddlgn control on psMNIST + parity/copy(L). ← next, NEEDS GPU.
      Equal-width AND equal-gates; ≥3 seeds; +`--grad-analysis`. (infra ready)

P2 — latch (anchor):
- [ ] D-flip-flop primitive (trivial delay) vs baseline on copy/parity.
- [ ] gated D-latch / SR latch + custom STE backprop (feedback stability is the risk).
- [ ] 4-way comparison: sequential / just-latch / gated / combo.

Deferred:
- [ ] P3 FPGA synthesis (future work, like RDDLGN & DiffLogic CA).
- [ ] #3 Fourier — analysis/cite only (method layer occupied by arXiv:2601.13953).
- [ ] #4 hard-attention/CAM — scout before committing.
