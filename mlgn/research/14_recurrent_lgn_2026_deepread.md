# 14 — Deep read: three 2026 papers vs P1/P2 claims

**Read:** 2026-07-01 (full text, from arXiv HTML — not summaries). **Why:** scout 13
([13_snn_hebbian_scout.md](13_snn_hebbian_scout.md)) surfaced three 2026 papers that touch
P1/P2 directly and flagged them "must-check." This doc is the full-text claim-by-claim
comparison. **Bottom line: none of the three scoops P2's core (C1 bistable primitives, C2
gradient obstruction). One is a gift (gives P1/P2 the exact gap vocabulary), one is the
closest neighbor (cite+distinguish on C3), one narrows the novel surface of C2's claim #2.**

| Paper | Group | P1 (gating) | C1 (latch prims) | C2 (grad obstruction) | C3 (gap/FSM) | claim #2 (multilinear reduction) |
|---|---|---|---|---|---|---|
| **2603.14157** Align Forward / CAGE | Kim, Inha (non-ETH) | **gift**: names the gap | — | — | **gift**: gap decomposition | — |
| **2605.24649** R-DTLGN | UMD Baras/Belta (non-ETH) | gating flagged as future work | safe (no bistable prim) | safe (diff. "stability") | **cite+distinguish (closest)** | — |
| **2605.08657** Multilinear LGN | Kim, Inha (non-ETH) | Angle #5 (skip) prior art | — | — | — | **narrows novel surface** |

Landscape update: recurrent-LGN now has **≥4 groups** — ETH (RDDLGN), Google (DiffLogic CA),
**UMD (R-DTLGN + PST)**, and you. Two of these three papers are by **Youngsung Kim (Inha
University, Korea)**, a prolific *non-ETH* LGN-training-methods author (also "Kim 2023" =
Gumbel-ST for LGNs; "logical skip connections"). The training/parametrization lane is no
longer ETH-only — but it's still not your sequential/latch lane.

---

## Paper 1 — arXiv:2603.14157, "Align Forward, Adapt Backward: Closing the Discretization Gap in LGNs" (Youngsung Kim, Mar 2026)

**What it is.** A *feedforward* LGN training-method paper. Its central move — and the gift —
is a **decomposition of the train/inference gap** (their Eq. 3):

> Total Gap = **Selection Gap** (`A^M − A^soft`, method-dependent, *reducible*) + **Computation
> Gap** (`A^soft − A^hard`, input-dependent, *irreducible*, method-independent).

- **Selection gap** = mismatch from the *selection mechanism* (soft mixture of 16 gates vs.
  argmax one gate). **Hard-ST forward → selection gap = 0 exactly, at every node, for any
  backward temperature** (their Prop 3). CAGE = adapt backward temperature by confidence to
  keep gradients flowing (avoids the Gumbel-ST low-τ **47-pt accuracy collapse** they document).
- **Computation gap** = soft vs. hard *gate evaluation* for the same selected gate. **Zero iff
  inputs are binary** (soft gate = hard gate on {0,1}²); for continuous inputs it is ~20% on
  Uniform[0,1] and **50–75% for values near the 0.5 decision boundary** (their App. I, Tables
  12–13). Training cannot remove it — it is a property of the *values being propagated*.

### Why this is a gift to P1 and P2 (not a scoop)

Kim is feedforward, no recurrence, no gating — **zero overlap with P1's recurrent gating or
P2's latch.** But his taxonomy hands you the precise vocabulary for the gap wall both papers
hit:

- **P1's copy-50 gap (soft 0.88 / discrete 0.37, +0.50) is a *computation* gap, not a
  selection gap.** The clincher is P1's own data: the gap **grows with sequence length** (0 at
  L20 → +0.50 at L50). A selection gap is *length-independent* for a shared recurrent cell (the
  argmax gate choice is the same at every step); only a **computation gap compounds over
  steps** as the continuous hidden state drifts toward the mushy 0.5 region — exactly Kim's
  "near decision boundary → 50–75% gap." This is a rigorous, falsifiable reframe of P1's "the
  gap is a general difflogic property, orthogonal to gating."
- **P2/C3 closes the *computation* gap architecturally — the one thing Kim says training
  cannot.** A bistable latch *re-binarizes the recurrent state to clean {0,1} each step*, which
  (by Kim's own Prop) drives the computation gap to 0 because the recurrent inputs become
  binary. So the razor-sharp C3 framing becomes: *"Kim 2026 shows the gap splits into a
  selection gap (closed by Hard-ST training) and a computation gap (irreducible by training,
  zero only for binary state). In a recurrent LGN the hidden state is continuous, so the
  computation gap compounds over timesteps (our copy-50: +0.50). Our bistable latch closes the
  computation gap **architecturally** by restoring the bit each step — training-side methods
  cannot."* This **pre-empts the "isn't the gap already solved?" reviewer** by showing the
  solved part (selection) is disjoint from the part P2 solves (computation-over-time).

### Actions
- **P1 checklist:** (a) rename the copy-50 gap a *computation gap* and use the
  length-dependence argument; (b) consider adopting **Hard-ST + CAGE** to kill the selection
  component and stabilize (cheaper/safer than Gumbel, which Kim shows can collapse at low τ —
  matches P1's "don't necessarily need Mind the Gap"); (c) note Kim *challenges* the
  Hessian-regularization explanation of Mind-the-Gap (Yousefi/ETH), so cite that mechanism
  cautiously.
- **P2 workmap C3:** adopt the selection/computation decomposition as the scaffold; cite Kim;
  state C3 = architectural closing of the *computation* gap over long sequences.

---

## Paper 2 — arXiv:2605.24649, "On the Stability and Realizability of Recurrent Polynomial Surrogate Ternary Logic Gate Networks (R-DTLGN)" (Damera, Matheu, Puranic, Baras, Belta; UMD; May 2026)

**What it is.** A **recurrent ternary** (Kleene {−1,0,+1}, 0 = "unknown") DLGN for **causal STL
runtime monitoring** in safety-critical systems, built on PST (Polynomial Surrogate Training,
2603.00302, same group — the ternary degree-(2,2) 9-coefficient surrogate). Recurrence =
**concat previous hidden state with predicate inputs → L combinational ternary-PST layers →
new state + verdict** (state initialized all-unknown). Contributions: (1) the architecture +
a **recurrent hardening routine (trajectory distillation)**; (2) **degradation guarantees**
from Kleene structure (principled abstention, input-certainty monotonicity — the "unknown"
value); (3) **realizability bound** `S ≥ B(φ)` sizing hidden state from STL formula structure
(Myhill–Nerode / Kalman analog); (4) PointMaze STL-monitoring experiments (competitive with a
vanilla Elman RNN).

### Claim-by-claim vs P2

- **C1 (bistable latch primitives + custom STE through cross-coupled feedback) — SAFE.**
  R-DTLGN has **no bistable primitive, no gating, no intra-step feedback**. Its "memory" is a
  plain **delay register** (state = previous combinational output fed back), i.e. the trivial
  D-FF-as-1-step-delay that P2 explicitly distinguishes from its *learnable bistable*
  contribution. It even says (§II-C) all prior DLGN work is "feedforward or binary" and that
  Bührer (RDDLGN) "do not address the hardness fragility that arises in recurrent settings."
  **→ Cite + distinguish:** R-DTLGN = ternary *combinational* concat-recurrence with delay
  registers; P2 = *learnable bistable latch primitives in the gate vocabulary* with custom STE
  through cross-coupled feedback.
- **C2 (gradient-degeneracy obstruction; the moat) — SAFE, but a terminology trap.**
  R-DTLGN's "stability" is a **completely different object**: forward-pass *dynamical* stability
  via **Tarski's theorem** (monotone maps → fixed-point existence + bounded orbits, for
  numerically-monotone gates) and **Kleene's fixed-point theorem** (Thm 1: ascending chain from
  ⊥ converges to a least fixed point in ≤S steps, for information-monotone gates). These are
  *inference-time state-convergence* properties under lattice orderings — **not** a statement
  about *gradient* well-posedness, `(I−J)` singularity, implicit/DEQ/Almeida–Pineda gradients,
  or why BPTT is used. And because R-DTLGN's recurrence is *combinational* (no intra-step
  cross-coupled feedback), the C2 object (`q*=F(q*;s,r)`, singular `(I−Jₚ)`) **does not even
  arise**. **→ C2 is not pre-empted**, but both papers say "stability / fixed point / recurrent
  LGN / realizable to ASIC," and R-DTLGN even uses the phrase **"degenerate memory"** (meaning
  state collapsing to unknown — *not* your gradient degeneracy). **C2 must explicitly
  contrast:** "unlike R-DTLGN's forward-dynamics stability (Tarski/Kleene fixed-point existence
  + graceful degradation), our obstruction is about *gradient* well-posedness at a *bistable*
  memory."
- **C3 (close the long-sequence gap + FSM induction + realizability) — CLOSEST OVERLAP,
  cite+distinguish prominently.** R-DTLGN independently identifies the **recurrent
  discretization gap** — "per-neuron errors amplify in the recurrent loop," "hardness fragility
  that arises in recurrent settings" (§III-D, §II-C) — *the same motivation as C3* — and fixes
  it with **trajectory distillation** (a *calibration/training-side* greedy re-fit), **not**
  architecturally. It also derives a **realizability/min-state bound** (Thm 2, Myhill–Nerode
  analog) and names **"DFA extraction for formal model checking"** and **"gated R-DTLGN
  variants"** as *future work* (§VII). **→ P2 must cite R-DTLGN as the nearest prior art and
  sharpen C3's distinct contribution:** (i) P2 closes the recurrent gap by an **architectural
  bistable restore** (re-binarize state each step), whereas R-DTLGN uses **post-hoc
  distillation**; (ii) P2 does **FSM *induction*** (learn/extract the machine), whereas R-DTLGN
  does **realizability *bounds*** (how big the state must be) and only *names* DFA extraction as
  future work. The gap between these is real but *narrow* — move deliberately.

### Note
- Not ETH — a *fourth* group in recurrent-LGN. Its **gated-variant future work** re-confirms
  gating (P1) is publicly flagged by yet another group (after Google/DiffLogic-CA), but still
  unbuilt — no new P1 race beyond what's known.
- The ternary "unknown" value is a *different* notion of state than P2's bistable bit — adjacent
  vocabulary, not a conflict.

---

## Paper 3 — arXiv:2605.08657, "Fitting Multilinear Polynomials for Logic Gate Networks" (Youngsung Kim, Inha; May 2026)

**What it is.** A **feedforward** LGN training-method paper. Observes every 2-input Boolean gate
has a unique **multilinear polynomial** `g(a,b) = c₀ + c_a·a + c_b·b + c_ab·ab` (4 coeffs), so
the 16 gates are 16 integer points in a rank-4 space → training = vector quantization in 4-D.
Trains directly in the 4-parameter coefficient space (Multilinear-STE, Multilinear-CovJac)
instead of a 16-way softmax; proves the 16-way softmax wastes 11/15 gradient directions and
that the **interaction coefficient `c_ab` is "starved" under STE** (gets gradient on only 25%
of samples); CovJac couples `c_ab` to the constant channel to fix it. **Zero recurrence,
feedback, latch, characteristic-equation, or sequential content (verified).**

### vs P2's claim #2 (the multilinear characteristic-equation reduction) — NOT anticipated, but it NARROWS the novel surface

P2 claim #2 = "replace the feedback relaxation with the multilinear **characteristic next-state
equation** `Q⁺ = S + R̄·Q = s + (1−r)q − s(1−r)q`, unifying all tiers + the 16 gates into one
relaxation, turning a fixed-point problem into a feedforward neuron + STE." Split it:

- **The sequential/feedback move IS novel and unanticipated.** Writing the *SR-latch
  characteristic next-state equation* in the multilinear basis, and using it to collapse the
  *intra-step cross-coupled feedback fixed-point* into a feedforward neuron + STE — paper 3
  never touches feedback/latches/sequential at all. Safe.
- **But the multilinear *representation* and *coefficient-space STE training* are NOT new —
  they are now doubly occupied.** The multilinear extension of the 16 gates is the field's
  founding relaxation (Petersen; O'Donnell Boolean analysis); training in the 4-param
  multilinear/corner-basis coefficient space with STE is **Kim 2605.08657** *and* **IWP
  (Rüttgers et al.)**. P2's own line — "the surrogate lives in the same multilinear algebra as
  difflogic's 16 gates" — must therefore **cite these and claim *only* the
  characteristic-equation / feedback-reduction**, never the multilinear representation or
  coefficient-space training itself. (P2 already says "the gradient estimators are imported —
  say so"; this makes that concrete: import the multilinear-STE, cite Kim + IWP.)
- **Bonus methodological hook:** P2's latch surrogate gradients (`∂Q⁺/∂s = 1−q+rq`, `∂Q⁺/∂r =
  −q(1−s)`, `∂Q⁺/∂q = (1−r)(1−s)`) are multilinear-STE gradients and may suffer the **same
  `c_ab`/interaction-coefficient starvation** Kim analyzes (the `q·(...)` interaction terms get
  gradient on a minority of state configurations). Worth (a) citing Kim's gradient analysis as
  the reason the latch surrogate needs care, and (b) trying **CovJac-style coupling** if the
  latch surrogate shows starvation in practice.

### vs P1 / Angle #5
- **Angle #5 (skip/Highway LGN) prior art:** paper 3 (ref [16]) cites Kim's **"logical skip
  connections"** for LGNs. So the skip-connection-for-LGN space now has *three* occupants
  (Petersen residual init; Kim logical skip connections; + Light DLGN's additive-residual
  negative result) — Angle #5's surviving sliver (a *learned gated* skip) narrows further.
  **Scout Kim's logical-skip-connections paper before any Angle-#5 experiment.**
- Paper 3's finding that **Soft-Mix collapses at depth** (−37pp on CIFAR-10 at L=12) while the
  4-param methods hold is a general deep-LGN datapoint; not directly P1/P2 but useful context.

---

## Consolidated action items

**P2 workmap ([11_paper2_workmap.md](11_paper2_workmap.md)):**
1. **C3:** adopt Kim's **selection/computation gap decomposition** (2603.14157); frame C3 as
   *architectural closing of the computation gap over long sequences*; use the
   gap-grows-with-length argument. Cite prominently.
2. **C3:** cite **R-DTLGN (2605.24649)** as nearest prior art on the *recurrent discretization
   gap*; distinguish **architectural bistable restore** (P2) vs **trajectory distillation**
   (R-DTLGN), and **FSM induction** (P2) vs **realizability bounds + DFA-extraction-as-future-
   work** (R-DTLGN).
3. **C2:** add the terminology-contrast sentence vs R-DTLGN's Tarski/Kleene forward-dynamics
   "stability" (different object); keep the scout-13 non-uniqueness reframe.
4. **claim #2:** cite **Kim 2605.08657 + IWP** for the multilinear representation and
   coefficient-space STE; scope the *new* claim to the characteristic-equation/feedback
   reduction only. Consider CovJac if the latch surrogate shows `c_ab` starvation.

**P1 checklist ([08_paper1_checklist.md](08_paper1_checklist.md)):**
5. Rename copy-50 gap a **computation gap**; length-dependence = the evidence. Cite 2603.14157.
6. Consider **Hard-ST + CAGE** as the training recipe (avoids Gumbel low-τ collapse); note Kim
   challenges the Hessian-regularization story of Mind-the-Gap.

**Angle #5 ([05_my_angles.md](05_my_angles.md)):** add Kim "logical skip connections" to the
prior-art to clear before any gated-skip experiment.

*All three papers are non-ETH (Kim/Inha ×2; UMD ×1). The genuine race pressure on P2 remains
ETH (RDDLGN + BitLogic, shared first author Bührer) per [10_fpga_scout.md](10_fpga_scout.md);
these three are citable neighbors, not scoopers — provided C1/C2/C3 and claim #2 are scoped and
distinguished as above.*
