# Paper #1 — Remaining Experiments → Submission Checklist

_Standing plan as of 2026-06-10. The "how to train recurrent Logic Gate Networks" paper:
a logic-native gated cell (constant-error carousel via a Boolean MUX) that does long-range
memory where the recompute-recurrence control (RDDLGN-style) fails. Pairs with
[06_paper_plan.md](06_paper_plan.md) (3-paper arc) and [07_venues_timeline.md](07_venues_timeline.md)._

## Where we are (validated)
- ✅ `gated` cell **solves copy-20 end-to-end: 100%, discretization gap = 0.**
- ✅ Control `rddlgn` **dead at chance** on long-range (even with `--grad-factor 2`).
- ✅ Training recipe characterised: **vanishing → keep-bias**, **exploding → lower LR /
  skip-step**, both implemented + explained, with gradient-flow analysis.
- ✅ `lstm` arm built (5 LGNs, dedicated cell state). `latch` (#2) parked.
- ⏳ Open: seq-50 discrete (soft solves it at 0.88/1.0; discrete 0.37 — see gap stance).

## Remaining experiments (ordered)

### 1. Close the long-sequence frontier (seq-50+) cleanly
**Status (2026-06-11): de-prioritized — orthogonal plumbing, diminishing returns.**
§1a (capacity) lifted discrete 0.37→0.75; §1b (entropy reg) was a **negative result** (it
drags soft *down* to discrete, doesn't lift discrete). 0.75 is a **capacity-bound discrete
ceiling** at hidden 2048 for copy-50. LR decay (§stability) is a keeper. The mechanism is
already validated (seq-20 100%, seq-50 75% discrete / soft-solvable, control ~12.5%), so
**recommend banking this and moving to §2–§4**; only return for STE/capacity if a clean
seq-50 number is wanted. The gap is a *general difflogic* property, orthogonal to our
contribution, and was **0 at seq-20**:
- [ ] **1a. Capacity bump** — re-run copy-50 with bigger `--hidden` (e.g. 2048) and/or
  `--cell-layers 3`. If soft→confident-100% the discrete likely follows (as at seq-20).
- [x] **1b. Gate-entropy regularizer** — BUILT. `--entropy-reg` (coeff, ramped via
  `--entropy-ramp`) adds `lambda*utils.gate_entropy(model)` to push gates one-hot. Pairs
  with **cosine LR decay** (`--lr-min`) to absorb the late-phase explosion once gates
  sharpen. Both opt-in (default off). _Run pending._
  - _Infra TODO (future, not blocking): make the LR schedule first-class & pluggable —
    explicit `--lr-schedule {none,cosine,linear,step}` set independently, not inferred from
    `--lr-min`. (TODO comment is in `train.py` at the scheduler.)_
- [ ] **1c. (Fallback only) Gumbel+STE** — Mind the Gap (arXiv:2506.07500). Proven but
  touches the CUDA `LogicLayer` forward; only if 1a/1b fail. **Borrowed plumbing — cite,
  don't over-invest.** Likely NOT needed.

### 2. The `lstm` arm (richer-mechanism ablation)
- [ ] Run `lstm` (keep-bias on forget gate) across the copy length sweep {20, 35, 50}.
- [ ] Question to answer: does the dedicated cell state + forget/input/output (5 LGNs,
  ~2.5× gates) beat the single-gate `gated` cell? Report acc-vs-gates.

### 3. A harder / real benchmark
- [ ] **psMNIST** (the original target, 784 steps) now that the stability recipe exists —
  or a shorter real sequence task if 784-step wall-clock is prohibitive on the 2080S.
- [ ] Optional high-value: reproduce **RDDLGN's own task** (WMT'14) to claim "beats the
  prior recurrent LGN on its turf" — heavier lift, decide by venue (see 07_venues).

### 4. Rigor / fairness (do before writing)
- [ ] **≥3 seeds** on every headline number (mean ± std — discrete LGNs are noisy).
- [ ] **Equal-gates control:** size `rddlgn` UP (more width/layers) to match `gated`'s gate
  count, so "gating wins" can't be dismissed as "more gates." Report acc-vs-gates curves.
- [ ] **Headline figure:** accuracy vs sequence length, `rddlgn` vs `gated` (vs `lstm`),
  showing the control flat at chance while gated holds — plus the grad-norm-through-time
  carousel plot.

## Discretization-gap stance (for the write-up)
Not our contribution; orthogonal to gating. Frame as: "the train(soft)→inference(hard) gap
is a known difflogic limitation; seq-20 shows our cell discretizes losslessly when it fully
solves; for longer sequences we [bump capacity / apply an off-the-shelf gap-reducer] and
note this is independent of the gating mechanism." Cite Mind the Gap. Do **not** present a
gap method as a contribution.

## Write-up framing — free wins (cost nothing, strengthen the paper)

**Carousel ↔ residual-init equivalence (positions `keep_bias` as a principled
generalization, not an ad-hoc trick).** Our `keep_bias` sits exactly where two established
techniques coincide:
- **LSTM forget-gate bias** (Gers et al. 2000) — a *temporal* keep bias.
- **Petersen's residual initialisation** (LogicTreeNet, **NeurIPS 2024 Oral**) — biasing
  gate logits toward the pass-through "A" gate to get differentiable residual/skip
  connections in *deep feedforward* LGNs.

The constant-error carousel (`keep_bias` + MUX) is the **temporal twin of a feedforward
skip/residual connection**: a learnable identity that, applied *through time* instead of
*through depth*, is the carousel. So frame `keep_bias` as **the temporal generalization of
the NeurIPS-Oral residual-init technique** — same mechanism (a learnable keep/skip),
recurrent axis. `bias_gate_keep` already cites both.

**Caveat (don't overclaim):** the *feedforward* residual-init idea is Petersen's
(occupied — see [[lgn-recurrent-scout-verdicts]] Angle #5). Our contribution is the
**recurrent/temporal** use + the **gated MUX** carousel + the empirical "it's *necessary*"
result (the cold-start ablation: `keep_bias=0` → dead, `keep_bias=3` → solves copy-20).
Credit residual-init; claim the temporal generalization, not the residual idea itself. The
standalone learned-gated/Highway skip angle stays **parked as a post-P1/P2 workshop
fast-follow** (scout `residual/dense/highway LGN` before any experiment there).

## Paper tier (honest)
- **Now:** solid **workshop** paper (NeurIPS/ICLR workshop) — essentially ready on the
  seq-20 result + the two-pathology recipe + dead control.
- **After §1–4:** workshop-strong → **main-conference borderline.** Realistic main-track
  homes: **AAAI / MLSys / DATE** (reward solid+useful recipes). **NeurIPS/ICLR main = reach**
  — needs a flashy real-task win; "gating + clipping for RNNs" reads as known tools in a
  new setting. The training-pathology characterisation is what lifted this above
  "we added a gate."
- **Arc:** ship the workshop now → fold #1 + #2 (latch) into the stronger main-track paper
  later (see 06_paper_plan.md). The latch is what reaches for top-tier.

## Immediate next action
Start **§1a (capacity bump on copy-50)** — cheapest test of "is the gap just
under-solving?" If yes, the frontier extends with no special method and §1b/1c are moot.
Then §2 (`lstm` sweep). Venues/deadlines: [07_venues_timeline.md](07_venues_timeline.md)
(NeurIPS-workshop submission ~Aug 29).
