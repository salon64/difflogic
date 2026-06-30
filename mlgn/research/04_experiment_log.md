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

## 2026-06-21 — STORY FLIPS POSITIVE: keep-bias is task-dependent; gating wins recall AND classification
B + C results (fig: `results/curves_bc.png`). The psMNIST "loss" was a keep-bias artifact.

**B — psMNIST-28, gated, keep-bias sweep (vs rddlgn 0.620 / soft 0.652):**
| kb | test | soft | gap |
|---|---|---|---|
| 0 | **0.632** | **0.709** | +0.077 |
| 1 | 0.547 | 0.660 | +0.112 |
| 2 | 0.541 | 0.668 | +0.126 |
| 4 (old) | 0.389 | 0.659 | +0.270 |
→ **low keep-bias rescues integration:** soft 0.66→0.71, gap 0.27→0.08, and gated **kb0
beats the control** (0.632 > 0.620; soft 0.709 > 0.652). The earlier "gating loses
psMNIST" was keep-bias=4 *over-holding* (under-writing) on a task that needs to absorb
inputs. (Caveat: gated kb0 = 4k gates vs rddlgn 2k → firm up with an equal-gates control;
but soft superiority + the recall win below make the claim solid.)

**C — delayed-MNIST recall (1-step encode + delay), gated kb6 vs rddlgn:**
| delay | gated | rddlgn |
|---|---|---|
| 0 | 0.700 | 0.554 |
| 50 | **0.369** | **0.114 (chance)** |
| 100 | **0.339** | **0.114 (chance)** |
→ **decisive recall win:** the control collapses to chance at any delay (can't even learn
softly; grad ratio 0 = total vanishing), gated holds ~3× chance through 100 blank steps
(grad ratio 1.4e4). Clean real-data demonstration of the carousel.

**KEY INSIGHT:** keep-bias is **task-dependent** — HIGH for *recall* (hold state), LOW for
*integration* (absorb inputs).

**CORRECTION (equal-gates control, same day):** rddlgn at hidden 2000 = **4,000 gates**
(matching gated kb0) gets psMNIST-28 **test 0.655 > gated 0.632** (soft 0.694 vs 0.709).
So **gating does NOT win classification at equal gates** — my mid-day "gated beats control
on psMNIST" was a gate-count artifact (gated had 2×). low-keep-bias makes gated
*competitive* (0.63 vs 0.66) but the MUX discretization gap (0.077 vs 0.038) keeps it
behind. **Honest scope: gating helps long-range RECALL, not classification.**

**FINAL P1 story (scoped, airtight):** the gated carousel enables **long-range recall**
where concat-recurrence *completely fails* — copy (0.96 vs dead) and **delayed-MNIST
(control → chance at any delay; gated holds ~3×chance through 100 steps).** On
classification (psMNIST) gating gives no benefit at equal gates (control ties/wins).
Secondary: keep-bias is task-dependent; + the training recipe (keep-bias/lr-decay/skip-step).
This is the workshop contribution. Experiments DONE.
Full sweeps done (fig: `mlgn/seqlgn/results/curves.png`, `plot.py`).

**copy (synthetic recall), test acc:** gated **0.96 / 0.79 / 0.33** (L20/35/50, 3 seeds);
rddlgn dead ~0.25; lstm & gru_cell cold-start to chance at L≥35 (only work at L20, < gated).
→ on pure recall, **gated wins decisively**, and GRU > LSTM/gru_cell (simpler is better).

**psMNIST (real, chunked), test acc:** **rddlgn (control) BEATS gated at EVERY length:**
0.62/0.61/0.60/0.52 (L28/49/56/98) vs gated 0.39/0.38/0.30/0.28/0.32 (L28..112). NO crossover.

**Diagnosis — two effects, both against gated on psMNIST:**
1. **MUX discretization gap.** Soft models ≈ tied (~0.65 short), but gated gap +0.18..0.27
   vs rddlgn +0.03..0.05. The MUX `s·h+(1−s)·c` blends → analog hidden state → discretizes
   badly when the task isn't fully solvable; rddlgn's logic-recompute stays near-binary.
2. **keep-bias ⊥ integration.** gated's *soft* also degrades faster (0.66→0.50 by L98 vs
   rddlgn 0.65→0.57): keep-bias 4 makes it KEEP (under-write), which helps recall but hurts
   classification (needs to absorb every input). And at 28–98 steps rddlgn's vanishing
   (grad 1e-3) isn't fatal → gating's gradient-flow edge buys nothing here.

**Conclusion:** **the carousel helps long-range RECALL, not classification/integration.**
"Gating wins for recurrent LGNs" is NOT supported in general — only on pure-recall tasks.
This confirms the old scout read: P1 (gating) is an honest *workshop characterization*, not
a main-conf "our method wins"; **P2 (latch) is the real anchor.** Options + decision in chat.
copy, hidden 1024, lr 0.003→3e-4, 20k iters. Discrete test acc:

| seq | rddlgn (control) | **gated (GRU)** | lstm (kb4, fixed) |
|---|---|---|---|
| 20 | 0.25 | **1.00** | 0.76 |
| 35 | 0.26 | **0.88** | 0.13 ✗ cold-start |
| 50 | 0.26 | 0.38 (soft 1.0) | 0.13 ✗ cold-start |

(gated@50 = **0.75 at hidden 2048** — capacity extends the frontier; control `soft`≈0.12 =
genuinely dead at all lengths.)

**Findings:**
1. **GRU ≫ control at every length** (1.0/0.88/0.38 vs flat ~0.25). The headline figure.
2. **LSTM input-gate-closed fix WORKED at seq-20** (0.13→0.76; grad ratio 1.2e-3→1.1e+2 =
   carousel engaged). But **cold-starts again at seq-35/50** (kb4 carousel≈0.84, 0.84⁵⁰≈2e-4
   → vanish). LSTM needs ever-stronger keep-bias as length grows.
3. **GRU > LSTM cleanly** — even where LSTM works (seq-20) it's worse (0.76 vs 1.00) and far
   more finicky. → **ablation justifies the GRU**: one complementary MUX gate beats two
   independent gates needing coordinated init.

**Paper message locked:** the simple logic-GRU cell is the sweet spot — robust long-range
memory, beats the recompute control decisively, beats the more complex LSTM.

**Next (§4 rigor):** 3 seeds on the GRU points + equal-gates control (size rddlgn up to
gated's gate count). Optional: lstm kb6 @20/35 to firm up "LSTM harder even w/ strong init."

## 2026-06-11 (pm) — LSTM cold-starts (worse than GRU); fix = also close the input gate at init
First `lstm` run (copy-20, hidden 1024, keep-bias 3, lr-decay): **total failure** — flat at
chance (0.13), loss 2.08 all 20k, grad@t=0 = 8e-7 (**vanishing**, ratio 1.2e-3).

**Diagnosis:** LSTM carousel `∂C'/∂C = f·(1 − i·C̃)`. We keep-biased `f` (≈0.78) but `i`
(input gate) + `C̃` are random at init → `i·C̃ ≈ 0.25` **eats the carousel** → ∂C'/∂C ≈ 0.58
(vs GRU's `∂h'/∂h = s ≈ 0.78`). 0.58²⁰ → vanish → cold-start. So the LSTM's *separate*
forget/input gates make init harder than the GRU's single MUX gate.

**Fix:** standard LSTM init — keep-bias forget AND **close the input gate** (`bias_gate_closed`
→ FALSE logit) so `i·C̃≈0`, ∂C'/∂C ≈ f (strong), write path preserved. Applied in `cells.py`
lstm branch (both scaled by `keep_bias`). At keep_bias 4: f≈0.89, i≈0.11, carousel≈0.84.

**Note for the LSTM-vs-GRU framing:** even with the fix, this already shows the **GRU is
more robust to train** (single complementary gate vs LSTM's two independent gates needing
coordinated init) — a legitimate ablation point *for* the GRU as the recommended cell.

**Next:** re-run copy-20 lstm with `--keep-bias 4` (sweep 6 if still vanishing). Watch
grad@t=0 stops vanishing + loss drops below 2.08.

## 2026-06-11 — §1b entropy reg = NEGATIVE result; LR decay fixes stability; 0.75 is a discrete ceiling
copy-50, gated, hidden 2048, lr 0.003→3e-4 cosine, entropy-reg 0.05, 30k iters.
- ✅ **LR decay fixed stability:** skip=0 all 30k, no NaN (vs L50cap NaN @19k). Keep this.
- ✅ entropy reg committed gates: `ent` 1.6→0.03 (near one-hot).
- ❌ **Gap did NOT close.** val **0.75** (test 0.754), soft dropped **1.0→0.886**. Discrete
  unchanged vs L50cap (0.75); entropy reg pulled *soft DOWN* to discrete, not discrete up.

**Finding:** entropy/commitment reg is the WRONG tool — soft's 100% solution uses the gate
*mixture*; forcing one-hot destroys it and lands on the same 0.75 discrete circuit. We need
discrete→soft, not soft→discrete. **0.75 discrete is a capacity-bound ceiling** at hidden
2048 for copy-50 (recalls ~6/8 symbols), not a regularizable soft/hard mismatch.

**To lift discrete:** (a) more capacity (hidden 4096 / cell_layers 3), or (b) **STE / Mind
the Gap (§1c)** — hard forward + soft backward, optimizes the discrete circuit directly.

**Strategic call:** mechanism is VALIDATED (seq-20 100%, seq-50 75% discrete / soft-solvable,
control ~12.5%). Closing seq-50 to ~100% is orthogonal plumbing w/ diminishing returns.
**Recommend: bank this, pivot to core paper experiments** (lstm arm, length sweep, ≥3 seeds,
equal-gates). STE/capacity = optional polish, do only if a clean seq-50 is wanted.
Keep `--lr-min` (decay) as default-on infra going forward; drop `--entropy-reg`.

## 2026-06-10 (pm4) — §1a capacity bump: discrete 0.37→0.76; isolates a true residual gap
copy-50, gated, kb=3, lr 0.003, **hidden 2048** (2×), 30k iters.
**Result: test 0.757** (was 0.380 at hidden 1024) — chance 0.125, control ~0.125. Big win.

Two clean splits:
- **Capacity closed the *under-solving*:** discrete 0.37→0.75. Confirms §1a hypothesis.
- **A genuine ~0.25 discretization gap remains:** soft → **1.000** (iter 15k+) while
  discrete **plateaus at 0.75** (stable iters 12k–18k). So now it's a *true* gap (soft
  solved, discrete lags), not under-solving → **§1b entropy reg now properly motivated.**
- **NaN returned @ iter 19k** (loss had dropped to 0.07 → sharp gates → gradients spike
  again even at lr 0.003). Dead-weights early-stop fired (best ckpt 0.75 kept). So
  stability must hold through the *confident* late phase.

**Standing result for P1:** copy-50 gated **75.7%** vs control ~12.5%. Decisive; path to
~100% = close the residual gap (entropy reg) + stabilize the sharp phase.

**Next:** (i) §1b gate-entropy reg to close the 0.25 gap; (ii) late-phase stability — LR
decay (0.003→~3e-4) or `--grad-factor 0.5`. These pair (entropy reg sharpens gates →
needs the stability). Bigger capacity (4096) is a fallback but the gap, not under-solving,
is now the wall.

## 2026-06-10 (pm3) — lr=0.003 FIXES stability; only the discretization gap remains
copy-50, gated, kb=3, **lr 0.003**, 30k iters: **skip=0, no NaN** the whole run (explosion
fully fixed). **soft reaches 0.876 (hit 1.000 @ iter 13k)** but **discrete stuck at 0.37,
gap +0.50**, discrete flat from iter 14k → more training won't help.

**So all three bottlenecks resolved in order:** vanishing→keep-bias, exploding→lower LR,
**discretization gap = the sole remaining wall.** The gap is a *general difflogic* property
(orthogonal to our gating contribution), NOT specific to recurrence — and at seq-20 it was
**0** (fully solved discretizes perfectly). So the seq-50 gap is partly under-solving at
that length.

**Next steps to close it (cheapest first — NOT yet done):**
1. **More capacity** (bigger `--hidden`, `--cell-layers`) so seq-50 fully solves like
   seq-20 did → gap likely closes on its own (no special method).
2. **Gate-entropy regularizer** (push gate distributions to one-hot; `utils.gate_entropy`
   sketch noted) — cheap, CUDA/CPU-agnostic.
3. **Gumbel+STE (Mind the Gap, 2506.07500)** — proven heavy-artillery; only if 1–2 fail.
Likely don't *need* (3); it's borrowed plumbing, not our contribution. Try 1, then 2.

## 2026-06-10 (pm2) — skip-step shows failure is "tip into dead region" → prevention (lower LR)
Re-ran seq 35/50 with skip-step. It **learns then dies**: seq-50 soft hit **0.875** @ iter
6k, then one update poisons the weights → from there ~100% of steps skip and it spins on
NaN. Skip-step catches the aftermath but can't un-poison. seq-35: best_val 0.511 (soft
0.76) then dead @ ~4k; seq-50: best_val 0.489 (soft 0.875) then dead @ ~6k.

**Mechanism:** a finite-but-huge gradient slips through one step; Adam rescales it (small
2nd-moment → giant effective step) → a weight overflows → `softmax(inf)=nan` → dead. So
clipping/skip (magnitude safety nets) can't fix it; the fix is **prevention** — keep the
weights out of that region. Standard lever: **lower LR** (0.01 is aggressive for a 50-step
unroll). Also added a **dead-weights early-stop** (whole window skipped → break) so failed
runs die in seconds, not 30 min.

**Status for the paper:** seq-20 fully solved (100%, gap 0); seq-50 *soft* reaches 87.5% —
gating clearly enables 50-step memory; only the optimization is unstable at long lengths.

**Next:** seq-50 with `--lr 0.003` (then 0.001 / `--grad-factor 0.5` / `--keep-bias 2` if
needed). Watch `skip=` stays low and soft/val climb to convergence.

## 2026-06-10 (pm) — clip 1.0 insufficient; soft model hits 87%@seq50 → skip-step fix
Re-ran seq 35/50 with `--grad-clip 1.0`. **Clip did NOT prevent the NaN** — both still blew
up (NaN guard stopped them early: L35clip 6.2 min, L50clip 17.5 min).

| seq | best_val (disc) | test | **test_soft** | gap | grad@t=0 | outcome |
|---|---|---|---|---|---|---|
| 35 | 0.511 | 0.510 | 0.631 | 0.12 | 0.88 | NaN'd again (~iter 4k) |
| 50 | 0.363 | 0.373 | **0.867** | 0.49 | **341** | NaN'd (~iter 6k) — but soft was at 87%! |

**Key insight: the soft model nearly SOLVES copy-50 (87%) before exploding.** So gating
*can* do 50-step memory; the only blocker is numerical stability. grad@t=0 = 341 confirms a
real exploding gradient (~17000× early-vs-late).

**Why clip 1.0 failed:** `clip_grad_norm_` runs *after* backward — once a single backward
overflows to inf, clipping it yields nan (post-hoc clip can't rescue an overflowed grad).

**Better fix added: SKIP the optimizer step when the global grad norm is non-finite.** The
blow-up batch never touches the weights, and the model is kept OUT of the NaN basin (the
steps that would push it unstable are exactly the ones skipped). Tracks `skip=`/`skipped=`;
auto-suggests lower `--lr`/`--grad-factor 0.5` if >20% skipped. `train.py`.

**Next:** re-run seq 35/50 (skip-step is automatic). If it stalls (high skip count),
add `--lr 0.003` and/or `--grad-factor 0.5`. Expect seq-50 to finish solving (soft was 87%).

## 2026-06-10 — gated SOLVES copy-20 (gap=0); seq≥35 NaNs (exploding grad) → added clipping
copy, gated, keep_bias 3, hidden 1024 (RTX 2080S).

| seq | best_val | test | gap (soft−disc) | outcome |
|---|---|---|---|---|
| **20** | **1.000** | **1.000** | **0.000** | **SOLVED** — perfect, zero discretization gap |
| 35 | 0.511 | 0.510 | +0.12 | learned to 51%/76%-soft then **loss=NaN @ iter 4000** |
| 50 (50k it) | 0.124 | 0.125 | +0.26 | soft 0.38 @ 2k then **NaN @ iter 4000** (wasted 149 min) |

**Two findings:**
1. **Clean win at seq-20:** a logic GRU holds a symbol over 20 blank steps at 100%, and
   **gap→0** — the earlier 37%@seq-50 wasn't a fundamental gap; a fully-solved length
   discretizes perfectly. Quotable result.
2. **seq≥35 blocker is NaN = exploding gradients, not the gap.** keep-bias makes the
   recurrence Jacobian ≈ s ≈ 1 (fixes vanishing) but over 35–50 steps it creeps >1 →
   explodes. No gradient clipping was in the loop. We've now hit BOTH classic RNN
   pathologies: vanishing (→ keep-bias) and exploding (→ clipping). Coherent "how to train
   recurrent LGNs" story.

**Fix added:** `--grad-clip` (default 1.0, global grad-norm clip) + NaN early-stop guard +
`gnorm` logged each eval + `grad_clip` in JSON. `train.py`.

**Next:** re-run seq 35 & 50 with clipping (now default); watch `gnorm` (if pinned at 1.0
while loss still high → raise to 5–10). Expect the frontier to extend like seq-20.

## 2026-06-09 — keep-bias fixes the cold-start; bottleneck moves to discretization gap
copy, seq 50, hidden 1024, 20k iters (RTX 2080S).

| run | mech | keep_bias / gradf | best_val (discrete) | test | train loss | grad ratio | read |
|---|---|---|---|---|---|---|---|
| keepbias | gated | kb=3 | **0.379** | 0.373 | ~0.52 | 2e+4 | cold-start GONE; learns; beats control |
| fair | rddlgn | gradf=2 | 0.258 | 0.251 | 2.08 (flat) | 4e-4 | still DEAD — grad-factor 2 didn't save it |

**Verdict: directional claim validated.** gated (37%) ≫ fair control (25% ≈ chance) on a
50-step memory task; cold-start fixed (loss 2.08→0.52, gradient now reaches t=0, ratio 2e4
vs 7e-8 before). The control is genuinely dead even with grad-factor 2 → clean contrast.

**But not *solved* (37%, not 80%+).** Split: training **loss ~0.5** (soft model learning)
vs **discrete val ~0.37** → the **difflogic discretization gap** is now the dominant
bottleneck, plus discrete val was still noisy-climbing at 20k (under-trained). Cold-start
✅ → bottleneck is now (a) discretization gap, (b) under-training.

**Instrument added:** `evaluate(..., discrete=False)` → every eval now prints `soft` acc
and `gap = soft − discrete`; `test_soft`/`discretization_gap` in results JSON. Lets us
quantify the gap directly next run.

**Next:** (1) length sweep gated kb=3 {20,35,50} to find the clean-win regime + map the
frontier (the headline acc-vs-length plot); (2) more iters (50k) at seq50 to separate
"gap" from "under-trained"; (3) if soft ≫ discrete, implement Gumbel+STE (Mind the Gap,
arXiv:2506.07500) — the principled gap fix. Also probe keep_bias=2.

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
