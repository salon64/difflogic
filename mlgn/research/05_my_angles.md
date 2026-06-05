# My Research Angles — analysis + novelty scout

_Created 2026-06-04. Angles #1 and #2 have been run through an adversarial novelty
scout (results below). #3 and #4 are preliminary reads, not yet scouted._

The single most important prior-art find for this whole direction:

> **Differentiable Logic Cellular Automata (DiffLogic CA)** — Miotti, Niklasson,
> Randazzo, Mordvintsev (**Google**, ALIFE 2025) — arXiv:2506.04912.
> A **stateful, recurrent difflogic** system: cell state = binary vector, recurrence =
> re-apply the *same combinational logic circuit* each step (concat previous state +
> perception → new state). **No gating. No latch/flip-flop primitives.** Tasks: Game of
> Life, checkerboard, lizard growth, colored-letter generation.
> **Crucially, their explicit future work names exactly my idea #1:** "specialized gates
> designed to facilitate state forgetting" and "integrating **LSTM-like gating
> mechanisms** into the state update process."

So the recurrent/stateful-LGN frontier now has **three groups circling it**: ETH
(RDDLGN, translation), Google (DiffLogic CA, cellular automata), and me. It's small but
**heating up** — novelty moat matters, pick the differentiated angle.

### Gate closure (2026-06-04) — both remaining gates checked, verdicts hold
- **DiffLogic CA full text read.** Gate set = only the 16 combinational gates; **no
  latch/flip-flop, no gating** anywhere; state = fresh combinational recompute each step.
  Future work explicitly: _"Dynamic gating: Incorporating learnable mechanisms for
  information forgetting and remembering (Hochreiter & Schmidhuber, 1997)"_ → confirms
  **#1 is publicly flagged by Google (race risk)** and **#2 is NOT named (still open)**.
  FPGA called "natural" but **not implemented** (same as RDDLGN).
- **EDA/hardware sweep, no kill for #2.** Two cite-and-differentiate papers:
  - _Recurrent CircuitSAT Sampling_ (arXiv:2502.21226) — flip-flops as differentiable
    hidden state, but to **solve SAT on a fixed netlist**, not learn a circuit-as-model.
  - _Differentiable Logic Synthesis: Sinkhorn spectral composition_ (arXiv:2601.13953,
    Jan 2026) — learns Boolean fns in the **Walsh–Hadamard basis** → partly occupies
    **angle #3's method layer**; combinational only.

See [06_paper_plan.md](06_paper_plan.md) for the 3-paper roadmap built on these verdicts.

---

## Angle #1 — Gated logic recurrent cell ("LogicGRU/LogicLSTM")

**The idea / key insight.** An LSTM/GRU gate = a learnable **2:1 multiplexer per hidden
bit**: `c_new = (sel ∧ keep) ∨ (¬sel ∧ write)`, select line = learned gate. The "keep"
path is a constant-error-carousel bit: copied forward unless overwritten → clean
gradient through time. Build this explicitly instead of hoping random 2-input wiring
learns to route state (which is all RDDLGN and DiffLogic CA did).

**Scout verdict: CONDITIONAL GO (confidence ~80% the method is unbuilt).**

- The *method+evaluation* is **unbuilt** — nobody has implemented logic-native gating.
- BUT the *idea* is **explicitly named as future work by Google (DiffLogic CA)** and
  motivated by RDDLGN's flip-flop framing. → low moat, **race risk**. You can't claim
  "nobody thought of it"; you claim "first to build & evaluate it, and here's where it
  matters."
- A bolt-on MUX gate + MNIST is **too thin** as a standalone paper. It needs: (a) a hard
  benchmark where gating *demonstrably* matters (copy / adding / psMNIST / long-range,
  where vanilla recurrence fails), (b) gradient-flow analysis vs concat-recurrence, (c)
  ideally the hardware/FPGA story.

**Remaining gate:** confirm no 2025–26 preprint already did "gated/MUX recurrence in
LGN/BNN" (searched; found only the DiffLogic-CA future-work mention). **Move with some
urgency** — Google flagged it publicly.

**Cheapest falsifying test:** does a MUX-gated cell beat plain `LogicRNNCell`
([../secuential.py](../secuential.py)) on **permuted-sequential-MNIST** + a copy/adding
toy? One day on one GPU.

---

## Angle #2 — Latch / flip-flop primitives inside difflogic (stronger novelty)

**The idea / key insight.** Add **bistable sequential-logic primitives** (D flip-flop,
gated D-latch, SR latch) to the learnable gate vocabulary, with a **custom STE backprop**
through the feedback. This turns the trained net from a *combinational circuit unrolled
in time* into a **true clocked sequential circuit** (registers + logic) that synthesizes
**directly to FPGA/ASIC flip-flops**. On-ramp: D flip-flop = 1-step delay, trivially
differentiable (gradient 1 = the constant-error-carousel as a primitive). Ambitious:
gated/SR latch with cross-coupled feedback = where the custom gradient lives.

**Scout verdict: GO (confidence ~75% open at the method layer).** This is the more
differentiated bet — it has a real moat.

Threats found, and why each does **not** kill it:
- **Flip-flop neurons** (FFNN/BiFFNN/ConvFFNN; JK-flip-flop neuron, Springer/bioRxiv) —
  real-valued memory *neurons* in **standard NNs**, NOT primitives in a differentiable
  **logic-gate** network, and they don't synthesize to hardware registers. Different
  substrate → distinguishable. **Must cite and explicitly differentiate.**
- **DiffLogic CA's "state-forgetting gates"** future work — that's *gating* the update
  (≈ idea #1), **not** adding bistable memory primitives. Distinct.
- **Gradient-based FSM/automata learning** (DeepDFA arXiv:2408.08622; neural-FSM line) —
  learns *abstract* transition systems/matrices, not gate-level circuits with physical
  latch primitives. Distinct.
- **DEMOTIC** (arXiv:2502.08086) — differentiable digital-circuit sampler for CircuitSAT;
  combinational, different problem.
- Context: extending the LGN gate set is an *active but unsaturated* vein (e.g. ternary
  Kleene-logic LGNs, arXiv:2603.00302) — adding *stateful* elements is unclaimed there.

**Surviving gap:** stateful latch/flip-flop as a **learnable primitive within difflogic's
gate vocabulary**, with custom STE gradient through the bistable feedback, yielding a
hardware-mappable sequential circuit. Appears genuinely open.

**Contribution shape:** a *method* paper — "Sequential Logic Gate Networks: learning
clocked sequential circuits end-to-end," with the hardware-synthesis story as the hook.

**Remaining gate:** read the DiffLogic CA full paper + appendix (arXiv:2506.04912) and
sweep EDA/hardware venues for "learned sequential circuit synthesis with state elements"
to be sure no one added latch primitives. Then it's an **execution-risk** project
(training stability of feedback loops) — which is the good kind of risk.

**Cheapest falsifying test:** add a **D-flip-flop primitive** to a small recurrent LGN;
does it beat concat-recurrence on a task needing exact recall (copy / sequential parity)?
If even the trivially-differentiable delay primitive doesn't help, the bistable version
probably won't either.

---

## Strategic synthesis of #1 + #2

They're complementary: **#2 is the memory substrate (latch primitives), #1 is the
controller (logic-native gating) for it.** The strongest single paper is likely the
**combination**: a "logic-native LSTM" that is a *true sequential circuit* — latch
primitives gated by learned MUX logic, trained end-to-end, synthesized to FPGA.
- **#2 is the novelty anchor** (has a moat).
- **#1 alone is race-prone** (Google flagged it) — better as a component of the #2 story
  or backed by a strong long-range result.

Recommended order: prove the **D-flip-flop primitive** helps (#2 on-ramp) → add **MUX
gating** (#1) → escalate to the **bistable latch** with custom backprop → write it up as
sequential-circuit learning, with psMNIST/copy/adding + an FPGA synthesis demo.

---

## Angle #3 — Fourier / analysis of Boolean functions (NOT yet scouted)

Preliminary read (see chat 2026-06-04): **theory companion, not an architecture.**
- Light DLGN's 4-term `{1,A,B,AB}` reparametrization is already a Boolean-basis change;
  the **Walsh–Hadamard (parity) basis** is its sibling and the principled way to
  parametrize **n-input / 6-LUT gates** (an open problem) — low-degree truncation = a
  built-in complexity limiter.
- Fourier **degree / noise-sensitivity** measures fragility under bit-flips → an
  analytical handle on the **discretization gap** and the **accuracy ceiling**.
- Action: use it to back #2/k-gate work + explain ceilings; don't build a standalone
  "Fourier architecture."
- **Update (gate-b sweep):** the method layer is **already being occupied** —
  _Differentiable Logic Synthesis: Sinkhorn-Constrained Spectral Composition_
  (arXiv:2601.13953, Jan 2026) learns Boolean fns directly in the Walsh–Hadamard basis.
  So treat Fourier as a **cite/companion**, not a novel method. Still useful as the
  analysis lens for the discretization gap / accuracy ceiling.

## Angle #4 — "Makeshift transformer" → reframe to hard-attention / CAM (NOT yet scouted)

Preliminary read: a **literal transformer is the worst fit** (softmax-weighted averaging
is analog). The tractable, logic-native core is **content-addressable memory (CAM) /
hard attention**: XNOR-based Hamming similarity + hard top-1 MUX select. CAMs are native
hardware → real efficiency story. Note "DiffLogic CAM"-style ideas may exist → **scout
the CAM angle before committing.** Separate (non-recurrent) bet from #1/#2.

---

### Status board
| Angle | Scouted | Verdict | Risk type |
|---|---|---|---|
| #1 Gated logic cell | ✅ | CONDITIONAL GO (race risk) | novelty/race |
| #2 Latch primitives | ✅ | **GO** (best moat) | execution |
| #3 Fourier theory | partial | companion/cite (method layer occupied by 2601.13953) | — |
| #4 Hard-attention / CAM | ❌ | reframe, then scout | novelty |
