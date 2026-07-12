# P4 — Learning What to Remember: Per-Neuron Memory-Primitive Selection in Sequential LGNs

_Status: **pool — parked, soft-go, gated on P3b tooling.** Source scout:
[research/19](../../research/19_learnable_latch_scout.md) (2026-07-08, 16-agent, adversarially
verified: **CONDITIONAL**). The idea itself: [research/17 §2b](../../research/17_concepts_and_journey.md).
Depends on: P2 shipped (arXiv'd) + P3b RTL emitter / real synthesis flow. **Do not invest ahead
of P2/P3b.** No timeline — grab from [research/24](../../research/24_roadmap.md) when ready._

## What it is

Extend the per-neuron 16-gate softmax so the candidate set **mixes the combinational gates with
synthesizable stateful primitives** — {combinational, write-enabled register/`clatch`, D-latch,
D-FF, SR, T-FF} — per bit. Each neuron *learns which memory element it is* and argmax-deploys to
a **heterogeneous LUT+FF netlist**: the first LGN that learns its own memory allocation as raw
sequential logic.

## What it covers

- **Method:** the mixed combinational/stateful softmax, training recipe (expect the
  never-write-collapse problem the fixed clatch sidesteps to reopen — budget for it), and
  argmax-commit to a deployable heterogeneous netlist.
- **Interpretability result (tier-a carrier):** which bits become registers vs combinational,
  and whether that allocation matches task structure (e.g. copy-task cue bits → registers,
  decode bits → combinational). This is the claim that needs no hardware.
- **Hardware result (tier-b carrier):** synthesis-measured **FF/LUT Pareto at matched accuracy
  on ≥2 tasks**, beating BOTH the best fixed primitive AND post-hoc register pruning of the best
  fixed-primitive net. The known gated≈clatch accuracy tie becomes the *premise* ("same
  accuracy, fewer flip-flops"), not the failure.
- Reuses P2's cell zoo/benchmarks and P3b's emitter + synthesis flow 1:1.

## Scope — claim discipline

- **Novelty = substrate + primitive-set + per-bit granularity, NOT the learning method.** The
  mechanism (per-unit differentiable op-selection + argmax-commit) is occupied: DARTS and,
  sharpest, MuFuRU (per-dimension learned soft selection over memory-update ops). Cite on sight.
- **Accuracy cannot carry the paper** — our own gated≈clatch tie (psMNIST, distcopy) removes the
  "searched thing wins" discriminator.
- Do **NOT** lead with: stability (a liability — the mix is harder to train), discretization gap
  (clatch already discretizes clean), or verification (a heterogeneous learned circuit is
  *harder* to verify — added burden, not a selling point).
- Do **NOT** claim "we made a differentiable flip-flop" (Flip-Flop Neuron exists; refutable).
- Boundary vs [P11](../p11/init.md): stateful primitives live here; the combinational
  standard-cell vocabulary (MAJ/AOI/OAI) is P11's. P11's ASIC flow is what makes this paper's
  standard-cell numbers cheap.
- Frame vs EDA prior art: classical FSM state-assignment does GA-based FF-type selection —
  novelty is the *differentiable, gradient-learned, co-trained-with-the-Boolean-function*
  version, or a hardware reviewer will cite it first.

## Venue & tier (honest call)

- **Tier (a) — workshop / ALIFE floor** (no hardware needed): method + interpretability. The bar
  RDDLGN / DiffLogic-CA cleared as standalone papers with zero synthesis. Fast defensive
  flag-plant IF needed — but only *after* P2 is arXiv'd (shipping P2 first partially burns this
  novelty anyway).
- **Tier (b) — B-tier EDA main track: DATE / FPL / FCCM / TRETS** — requires the real-synthesis
  Pareto above.
- **NOT A\* main-track** on current evidence (accuracy tie + off-the-shelf mechanism).
- Fallback if the gate ties: collapses to a one-figure ablation in the thesis.

## The ONE gate (run before committing; odds ~30–40%)

Does the argmax-deployed learned mix beat **post-hoc register pruning of the best
fixed-primitive net** on FF/LUT count at matched accuracy, **through real synthesis**?
Headwinds: (i) if a learned mix can't beat a fixed primitive on accuracy, unclear it beats
pruning on FF count; (ii) **on FPGA every LUT ships with a near-free paired flip-flop**, so
"fewer FFs" cuts nothing unless LUTs/slices drop or Fmax improves — plan for **ASIC
standard-cell numbers or a LUT-count (not FF-count) win**.

## Read up on

Must-cites (verified in the scout):
- **DARTS** — Liu, Simonyan, Yang, ICLR 2019, arXiv:1806.09055 (the op-selection mechanism + its RNN-cell space).
- **MuFuRU** — Weissenborn & Rocktäschel 2016, arXiv:1606.03002 (learned per-dim memory-op selection — the "you didn't invent this" hit).
- Per-unit heterogeneous memory, continuous/fixed-cell: **Phased LSTM** (Neil et al., NeurIPS 2016), **Liquid Time-Constant nets** (Hasani et al., AAAI 2021), IndRNN, Clockwork RNN.
- **Flip-Flop Neuron** (Kanchana/Kumari et al., Neural Comput. & Applic. 2023 / bioRxiv 2021.11.16.468605) — fixed JK-FF, no selection; the do-not-claim marker.
- Recurrent-LGN neighbors: **RDDLGN** arXiv:2508.06097 (EdgeFM'25 — names FF/latch FPGA work as future work = the scoop vector), **DiffLogic CA** arXiv:2506.04912 (ALIFE 2025), **R-DTLGN** arXiv:2605.24649.
- **SmartMixed** arXiv:2510.22450 — per-neuron learned activation menu + commit; calibrates the workshop floor.

Background (added here, not from the scout — sanity-check before citing):
- NAS surveys for the selection-mechanism lineage (e.g. Elsken et al., JMLR 2019).
- Classical FSM state-assignment / FF-type selection: De Micheli, *Synthesis and Optimization of Digital Circuits*, for the EDA framing a hardware reviewer expects.

## Risks / kill conditions

- **Scoop = ETH-DISCO** (RDDLGN codebase + explicit FF future-work flag; they can bolt
  "NAS-over-LGN-primitives" on fastest). Mitigation: speed once P3b tooling exists; tier-(a)
  flag-plant if racing.
- Gate fails on FF/LUT Pareto → thesis ablation, no standalone paper.
- Training instability eats the budget → the fixed-clatch story (P2) already covers the
  deployable-register claim; this paper adds only the *learned allocation* delta.
