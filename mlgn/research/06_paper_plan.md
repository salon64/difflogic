# Paper Plan — 3-paper roadmap

_Created 2026-06-04, from my decomposition + the novelty scout
([05_my_angles.md](05_my_angles.md))._

Three contributions, sharing one codebase. Prior art they must beat/cite: RDDLGN
(arXiv:2508.06097, plain concat-recurrence, translation) and DiffLogic CA
(arXiv:2506.04912, stateful CA, no gating/latch). Both **defer FPGA** — so do I (P3).

```
        shared infra (build once)
        ├─ recurrent LGN base cell (extends difflogic LogicLayer)
        ├─ sequential benchmark: sMNIST, psMNIST, copy, adding, parity
        └─ training infra: Gumbel+STE [3], IWP [4], grad_factor
              │
   P1 ──────┘ gating ──────► reused as the "combo" arm of P2
   P2 (latch primitives) ──► uses P1's gate for combo
   P3 (FPGA synthesis) ────► future; realizes P1/P2 in hardware
```

---

## Paper 1 — MUX gating for recurrent LGNs ("LogicGRU")
**Status: CONDITIONAL GO — race-prone "easy points." Move fast, aim workshop.**

- **Claim:** first to build & evaluate *logic-native gating* — a learnable 2:1
  multiplexer per hidden bit (`c_new = (sel∧keep) ∨ (¬sel∧write)`) as a forget/update
  gate with a constant-error-carousel keep-path — vs the plain concat-recurrence of
  RDDLGN / DiffLogic CA.
- **Why it's only "easy points":** Google (DiffLogic CA) **publicly flagged exactly this**
  as future work, citing Hochreiter & Schmidhuber. So: low moat, real scoop risk.
  Reviewers will call a bolt-on gate incremental **unless** it carries (a) a long-range
  benchmark where vanilla recurrence *fails* and gating *fixes* it (copy/adding/psMNIST),
  and (b) a gradient-flow analysis (show the carousel path kills vanishing gradients).
- **Verdict on doing it standalone:** OK as a **fast workshop flag-plant**, not a
  top-tier solo paper. If P2 is close behind, consider folding gating into P2 as a
  baseline arm instead of a separate paper — decide based on how fast P1 can ship.
- **Min experiments:** {plain concat-recurrence} vs {MUX-gated} on sMNIST, psMNIST,
  copy(T=10..100), adding(T). Report acc + train-time + gradient-norm-through-time.

## Paper 2 — Latch/flip-flop primitives ("Sequential Logic Gate Networks") ⭐ anchor
**Status: GO (~78%). The differentiated contribution with a moat. Make this the strong one.**

- **Claim:** add **bistable sequential-logic primitives** (D flip-flop, gated D-latch,
  SR latch) to the learnable gate vocabulary with a **custom STE backprop through the
  feedback**, turning the trained net from a combinational circuit (unrolled) into a
  **true clocked sequential circuit**.
- **The comparison — RDDLGN is THE control.** The head-to-head is *our latch-based cell*
  vs. a **faithful reimplementation of the RDDLGN recurrence cell** (their
  `[h_t^(DN); k_{t-1}^(DK)]` concat-recurrence: carry hidden bits, re-concat with current
  features, recompute through logic layers — no explicit state element). #1's gating folds
  in as extra arms. All variants = modifications to the *same* RDDLGN base cell, so the
  only changed variable is the memory mechanism:
  | Variant | Memory mechanism | Role |
  |---|---|---|
  | **RDDLGN design** (faithful reimpl.) | concat-recurrence, recompute state each step | **the control to beat** |
  | **+ gated** | RDDLGN + MUX gating on the update | P1 arm |
  | **just latch** | state *held* by latch/flip-flop primitives (not recomputed) | **P2 core** |
  | **combo** | latch primitives + MUX gating | P1 ⊕ P2 |
  - **The scientific question P2 isolates:** does *explicitly holding* state in a latch
    beat *recomputing* it from concat (RDDLGN)? — especially on long-range recall.
  - On-ramp inside P2: start with the **D-flip-flop** (1-step delay, gradient 1 = trivially
    differentiable) → if even that beats the RDDLGN control, escalate to **gated D-latch /
    SR latch** (cross-coupled feedback = where the custom STE lives + training-stability
    risk).
- **Benchmark-fairness decision (flag):** the controlled comparison runs all four cells on
  *my* sequential benchmark (sMNIST/psMNIST/copy/adding/parity), which isolates the cell
  design. To claim "we beat RDDLGN" with maximum external validity, optionally also run on
  **RDDLGN's own task (WMT'14 translation)** to anchor against their published BLEU —
  heavier lift; decide if reviewers will demand it. See open decision below.
- **Cite-and-differentiate (from scout):** flip-flop *neurons* (real-valued NNs, not
  LGNs); DiffLogic CA "dynamic gating" (gating ≠ bistable primitive); DeepDFA / FSM
  learning (abstract automata, not gate-level); Recurrent CircuitSAT Sampling
  (arXiv:2502.21226 — flip-flop-as-hidden-state for *SAT on fixed netlist*, not learned).
- **Risk = execution** (feedback-loop training stability), not novelty. Good kind.
- **Tasks:** the benchmark above + a memory-stress task (long copy / sequential parity)
  where a *real* state element should decisively beat combinational recurrence.

## Paper 3 — FPGA / ASIC synthesis of sequential LGNs (FUTURE, deferred)
**Status: scouted 2026-06-11 → GO ~80% but only P2-coupled; urgency ↑ (ETH has RDDLGN +
BitLogic RTL pipeline in one lab). See [10_fpga_scout.md](10_fpga_scout.md): plan shift =
minimal D-FF FPGA demo goes INTO P2; full toolchain/eval paper → FCCM/FPL'27.**

- **Claim (later):** synthesize the learned sequential circuit (registers + logic) to
  FPGA/ASIC; measure ns-latency / energy; this is where the "true sequential circuit →
  hardware flip-flops" story pays off as *measured* results, not motivation.
- **Why defer:** lets P1/P2 stand on simulation; hardware is a labor-heavy follow-up.
  Revisit once P2 lands. Keep a one-line "future work" pointer to it in P1 & P2.

---

## Shared infrastructure to build first (do once, before any paper)
1. **Recurrent LGN base cell** in this fork (generalize `LogicRNNCell` from
   [../secuential.py](../secuential.py); make the memory mechanism pluggable:
   **RDDLGN-design** (control) / gated / latch / combo). Reimplement the RDDLGN
   recurrence faithfully — it's the control, so it must be a fair, tuned baseline.
2. **Sequential benchmark harness:** sMNIST, psMNIST (fixed permutation seed),
   copy-task(T), adding-problem(T), sequential-parity. One eval script, discrete-locked.
3. **Training infra as defaults (not research bets):** Gumbel+STE [3] and IWP [4] —
   strictly beneficial; bake in so every variant trains on equal footing.
4. **Logging:** extend [04_experiment_log.md](04_experiment_log.md); record acc, gate
   count, train-time, gradient-norm-through-time per variant.

## Order of operations
1. Build shared infra (1–3 above).
2. **P1 gating** — fastest, plants the flag; reuse as the "combo/gated" arms of P2.
3. **P2 latch** — D-flip-flop first, then bistable latch + custom STE; run the 4-way
   comparison. This is the anchor — spend the most rigor here.
4. **P3** — defer; note as future work in P1/P2.

## Open decisions
- **P1 standalone vs folded into P2?** If P1 can ship to a workshop in weeks, plant the
  flag (scoop risk is real). If not, fold gating into P2's ablation and publish one
  stronger paper. Revisit after the gating experiment runs.
- **P2 benchmark scope — my tasks only, or also RDDLGN's translation task?** Controlled
  comparison (all cells on sMNIST/psMNIST/copy/adding) is enough to isolate the memory
  mechanism. Adding WMT'14 anchors directly to RDDLGN's published BLEU but is a big lift
  (embeddings, vocab, BLEU pipeline). Default: start controlled; add translation only if
  the latch result is strong enough to be worth the head-to-head.
