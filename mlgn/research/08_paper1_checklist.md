# Paper #1 — Remaining Experiments → Submission Checklist

_Standing plan as of 2026-06-10. The "how to train recurrent Logic Gate Networks" paper:
a logic-native gated cell (constant-error carousel via a Boolean MUX) that does long-range
memory where the recompute-recurrence control (RDDLGN-style) fails. Pairs with
[06_paper_plan.md](06_paper_plan.md) (3-paper arc) and [07_venues_timeline.md](07_venues_timeline.md)._

---

## STATUS UPDATE (2026-07-10) — EXPERIMENTS CLOSED; draft v1.1 + figures + bib DONE
All remaining runs below were completed 06-24..07-01 (delayed-MNIST seeds/kb/delays) and
2026-07-10 (`p1f_*` hardening queue: all controls 3-seed, dMNIST eqgates control w/ grad≡0,
kb-sweep seeds — see 04_experiment_log 2026-07-10). Draft: `mlgn/paper/p1/p1_draft1.md`
(v1.1) + `references.bib` + canonical figure scripts `fig{1,2,3}_*.py` → `figs/`.
**Remaining is writing/admin only:** pick workshop (~Jul 11) → condense to 4pp → verify
bib author names → submit (~Aug 29).

## CURRENT STATUS (2026-06-21) — experiments ~complete; venue = NeurIPS 2026 workshop

**Result (scoped, honest):** the gated carousel enables long-range **recall** where
concat-recurrence goes to chance — copy (0.96 vs dead) + **delayed-MNIST (control→chance at
any delay, gated holds ~3× chance)**. **Gating does NOT help classification** (psMNIST equal
gates: control 0.655 ≥ gated 0.632). keep-bias is **task-dependent** (high=hold/recall,
low=absorb/integration). + training recipe (keep-bias vs vanishing; lr-decay/skip-step vs
exploding). Mechanism ablation: GRU > LSTM/gru_cell (cold-start).

**Venue decision: NeurIPS 2026 workshop** (~Aug 29 submit; pick workshop after the
list drops ~Jul 11). NOT AAAI main — result is workshop-tier (scoped, toy/constructed
benchmarks, a classification non-win) + AAAI Jul 28 is too tight. P1+P2 combined = the
later main-track shot. See [07_venues_timeline.md](07_venues_timeline.md).

**Draft:** `mlgn/paper/draft.md` (full-length skeleton; condense to ~4pp for workshop).
**Figures:** `mlgn/seqlgn/results/curves.png`, `curves_bc.png` (gen `plot.py`).

**REMAINING TO RUN (then done):**
- [ ] delayed-MNIST **seeds 1,2** for gated at delays {0,50,100} — error bars on the
  headline (control is at chance, ~0 variance; seed it only if you want bars on it too).
- [ ] delayed-MNIST **keep-bias sweep** at delay 50: kb {0,3} (have kb6) — completes the
  task-dependence story (recall needs HIGH kb; mirror of psMNIST needing LOW). Expect kb0
  to cold-start.
- [ ] (optional) intermediate delays {25,75} for a smoother recall curve.
- Everything else DONE (copy 3-seed; psMNIST kb sweep + equal-gates; mechanism ablation).

**THEN:** pick workshop (~Jul 11) → condense draft to 4pp + add seeds/citations → submit (~Aug 29).

---

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
contribution, and was **0 at seq-20** (precisely a *computation gap* that grows with length —
see §Discretization-gap stance):
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

### 2. Cell-mechanism ablation — the 2×2 (copy length sweep {20, 35, 50})
The four mechanisms isolate two variables (gate structure × state separation):

| | single MUX gate | two independent gates |
|---|---|---|
| **single state** | `gated` (GRU) — 2 LGNs | — |
| **separate state (h,C)** | `gru_cell` — 4 LGNs | `lstm` — 5 LGNs |
| (+ control) | `rddlgn` concat-recurrence | |

- [x] `rddlgn` / `gated` / `lstm` swept {20,35,50} (2026-06-12): **GRU wins** (1.0/0.88/0.38),
  control dead (~0.25), **LSTM cold-starts at ≥35** even with the input-gate-closed fix →
  GRU > LSTM (cleanly).
- [x] `lstm` fix: keep-bias forget **+ close input gate** (`bias_gate_closed`) — engages at
  seq-20 (0.13→0.76) but needs ever-higher keep-bias as length grows.
- [ ] **`gru_cell` sweep {20,35,50}** — the 2×2 completion: does a separate cell state help
  *with the robust MUX gate*? `gru_cell` vs `gated`; watch seq-50 (does it beat gated@0.38?).
- [ ] (opt) `lstm` keep-bias 6 @20/35 — firm up "LSTM harder even w/ strong init."

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

**Sharper, citable framing (Kim 2026, arXiv:2603.14157 — full read
[14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md)).** Name the gap
precisely instead of "general difflogic property." Kim splits the train/inference gap into a
**selection gap** (which-gate; killed by *Hard-ST forward*, any backward temperature) + a
**computation gap** (soft-vs-hard *values*; irreducible by training, =0 iff inputs binary).
Our copy-50 gap is a **computation gap** — the evidence is that it *grows with sequence
length* (0 at L20 → +0.50 at L50; a selection gap is length-independent for a shared cell):
the recurrent state carries continuous values that drift toward the mushy 0.5 region. Payoffs:
(a) makes P1's "orthogonal to gating" claim precise and falsifiable; (b) it is exactly the
wall **P2's bistable latch closes architecturally** — P1 *states* the wall, P2 *closes* it, a
clean P1→P2 handoff. **Consider adopting Hard-ST + CAGE** (Kim's recipe) over Gumbel/Mind-the-
Gap: Kim documents a **47-pt Gumbel-ST accuracy collapse at low τ** and *challenges* the
Hessian-regularization explanation of Mind-the-Gap (Yousefi/ETH) — so cite Mind-the-Gap's
*mechanism* cautiously. Caveat: at copy-50 soft is 0.88 (not 1.0), so ~0.12 is
capacity/under-solving and the +0.50 soft→hard is the computation gap — quantify the split
(§1 capacity note; workmap §Pre-commit sanity check).

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
