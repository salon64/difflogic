# 19 — Scout: "learnable latches as trainable parameters" (per-neuron learned memory-type LGN)

Adversarial novelty + paper-worthiness scout (2026-07-08, 16-agent workflow, 6 kill-angles + scoping +
adversarial verification: 6 CONFIRMED / 2 PARTIAL). Question: is **malcolm's future-work idea — "add latches
as TRAINABLE PARAMETERS instead of architecturally": extend the per-neuron 16-gate softmax to also SELECT among
stateful primitives {combinational, write-enabled register/`clatch`, D-latch, D-FF, SR, T-FF} per bit, so each
neuron LEARNS which memory element it is, argmax-deploying to a heterogeneous LUT+FF netlist** — enough for a
whole paper? See [17_concepts_and_journey.md](17_concepts_and_journey.md) §2b for the idea; this is the verdict.

## VERDICT: CONDITIONAL — a **P4 / thesis chapter, gated on P3b tooling.** NOT a P2 section, NOT a near-term standalone.

**The pincer that caps it:**
1. **The MECHANISM is occupied** (per-unit differentiable op-selection + argmax-commit): **DARTS** (Liu, Simonyan,
   Yang, ICLR 2019, arXiv:1806.09055) + its RNN-cell search space; and the sharpest hit, **MuFuRU** (Weissenborn
   & Rocktäschel, 2016, arXiv:1606.03002) = per-dimension learned *soft selection over memory-update ops*
   {keep/replace/forget/max/min} — structurally this idea, in continuous space. ⇒ **cannot claim the method.**
2. **Accuracy can't carry it** — our own gated≈clatch ACCURACY TIE (psMNIST, distcopy) removes the DARTS-style
   "the searched thing wins" discriminator. A learned mix that ties both fixed types has no ML-legible win.

**The SURVIVING GAP (genuinely vacant, one sentence):** a per-neuron differentiable softmax whose candidate set
**mixes the 16 combinational gates with synthesizable stateful primitives, argmax-deploying to a heterogeneous
LUT+FF netlist** — the net learning its own memory allocation as raw sequential logic. Novelty = **substrate +
primitive-set + per-bit granularity, NOT the learning method.**

## Prior art (real cites, verified)
- **DARTS** — Liu et al., ICLR 2019 (arXiv:1806.09055): the differentiable op-selection mechanism. Cited on sight.
- **MuFuRU** — Weissenborn & Rocktäschel 2016 (arXiv:1606.03002): learned per-dim selection over memory ops. The
  "you didn't invent learned per-unit memory selection" hit.
- **Per-unit heterogeneous learnable memory (continuous scalar only, fixed cell):** Phased LSTM (Neil et al.,
  NeurIPS 2016), Liquid Time-Constant nets (Hasani et al., AAAI 2021), IndRNN, Clockwork RNN.
- **"Flip-Flop Neuron"** (Kanchana/Kumari et al., Neural Computing & Applications 2023 / bioRxiv 2021.11.16.468605)
  — a *fixed* JK-FF unit, no selection. ⚠ do NOT claim "we made a differentiable flip-flop" (refutable).
- **Recurrent LGN neighbors:** RDDLGN (Bührer/Wattenhofer, ETH-DISCO, arXiv:2508.06097, EdgeFM'25), DiffLogic-CA
  (Miotti/Mordvintsev, Google, ALIFE 2025, arXiv:2506.04912), R-DTLGN (arXiv:2605.24649). All fix memory as a
  GLOBAL concat-recurrence; per-neuron softmax ranges over the 16 COMBINATIONAL gates only. **None does per-neuron
  primitive selection** (fetch-verified).
- **SmartMixed** (arXiv:2510.22450) — per-neuron learned activation menu + commit; shows this shape lands at
  workshop/preprint tier (this idea is strictly more novel → the workshop floor is real).
- **EDA prior art:** classical FF-type selection during FSM state-assignment (GA-based area/power) exists → frame
  novelty as the *differentiable, gradient-learned, co-trained-with-the-Boolean-function* version, or a hardware
  reviewer cites it.
- **Scoop risk = ETH-DISCO:** RDDLGN explicitly names flip-flops/latches as future FPGA work; they hold the
  codebase + substrate + motivation to bolt "NAS over LGN primitives" on fastest.

## Paper-worthiness — two tiers
- **(a) Workshop / ALIFE floor (no hardware needed):** method (trains + argmax-commits) + **INTERPRETABILITY**
  (which bits become registers vs combinational, matching task structure). The bar RDDLGN/DiffLogic-CA cleared as
  standalone papers with zero synthesis. Frame: *"first LGN that learns its own memory allocation as raw sequential
  logic."*
- **(b) B-tier EDA standalone (DATE / FPL / FCCM / TRETS):** requires a **synthesis-measured FF/LUT Pareto at
  matched accuracy on ≥2 tasks, beating BOTH the best fixed primitive AND post-hoc register pruning.** The accuracy
  tie becomes the *premise* ("same accuracy, fewer flip-flops"), not the failure.
- **NOT A*-main-track** on current evidence (accuracy tie + off-the-shelf mechanism).
- **Do NOT lead with:** stability (a *liability* — the mix is harder to train; reopens the never-write-collapse
  problem the fixed clatch sidesteps), discretization gap (clatch already discretizes clean), or verification (a
  heterogeneous learned circuit is *harder* to verify — adds burden).

## THE ONE GATE (odds ~30–40%)
Does the argmax-deployed learned mix beat **post-hoc register pruning of the best fixed-primitive net** on FF/LUT
count at matched accuracy, through REAL synthesis? Headwinds: (i) if a learned mix can't beat a fixed primitive on
accuracy, unclear it beats pruning on FF count; (ii) **on FPGA each LUT ships with a near-free paired flip-flop**,
so "fewer FFs" doesn't cut area unless you also cut LUTs/slices or improve Fmax — plan for **ASIC standard-cell
numbers or a LUT-count (not FF-count) win.**

## RECOMMENDATION → **P4, soft-go, gated on P3b. Do not invest ahead of P2/P3b.**
- **Keep it OUT of P2** (workmap §A0' already parked it; folding in dilutes the locked deploy-a-verifiable-register
  story + reopens a training-stability problem the single fixed clatch avoids).
- **Depends on P3b infrastructure** (RTL emitter / real synthesis) — the tier-(b) carrier is gated on tooling not
  yet built.
- **Sequence:** ship P2 → build P3b RTL/synthesis flow → run the ONE GATE. Pareto lands → B-tier EDA standalone
  (anchor on interpretability given the free-FF problem); ties → collapses to a one-figure ablation in the thesis.
- **Race defense:** a short tier-(a) workshop paper (trains + interpretable, no synthesis) is the fast defensive
  move IF desired — but only *after* P2 is arXiv'd (shipping P2 first partially burns this novelty anyway).

**Bottom line:** genuine vacant gap, thin moat (menu not method), dead on accuracy, alive only on a
hardware/interpretability payoff unmeasurable until P3b exists. **Park as P4.**
