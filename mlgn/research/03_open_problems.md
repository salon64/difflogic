# Open Problems & Candidate Directions

_Scout date 2026-06-04. This is the "where's the gap" doc — opinionated, meant to be
argued with. Bracketed IDs reference [02_papers.md](02_papers.md)._

## A. The field's standing open problems (consensus pain points)

1. **Accuracy ceiling beyond CIFAR-10.** Nobody has shown a plain/conv LGN doing well on
   ImageNet-scale or hard tasks. Light DLGN [4] argues the bottleneck is **information
   loss in input preprocessing** (real values → 4-value thermometer encoding), not the
   model — **unproven and wide open.** Better input encodings ≈ free win if true.
2. **Training cost & vanishing gradients.** Improving but not solved: Gumbel+STE [3],
   IWP reparametrization [4], residual init [2], `grad_factor` [1]. Deep stacks still
   fragile. No principled optimizer/init theory yet.
3. **Discretization gap.** [3] closes ~98% on CIFAR via Gumbel+STE, but it's CIFAR-only;
   general guarantees / when-it-fails unknown.
4. **Fixed random connectivity.** [6][7] show learned connectivity is a big lever; still
   early (≤256k gates, no large-scale, no conv+learned-connectivity combo published).
5. **n-input gates / 6-LUT.** IWP [4] makes 2ⁿ-param gates tractable and explicitly
   leaves **learning k-input gates (FPGA-native 6-LUTs)** as future work. Untaken.
6. **Generalization gap.** [4] reports persistent train/test gap; dropout / residual /
   random interventions "fail to improve test performance." No working regularizer for
   LGNs is established. Open.
7. **Theory.** Expressivity, depth/width trade-offs, what function classes LGNs learn
   efficiently — almost untouched.

## B. The recurrent / sequential frontier (my lane)

State of the art here is **one paper**: Recurrent DDLGN [5] (translation). That thinness
is the opportunity. What [5] did and did **not** do:

**Did:** a working recurrent LGN cell (concat current features + previous hidden bits),
seq2seq translation, a memorization probe. Beat RNN, trailed GRU/Transformer.

**Did NOT (→ open gaps):**
- **Sequential / streaming classification** (MNIST-by-rows, sequential-MNIST,
  permuted-MNIST, sMNIST/psMNIST, time-series UCR). ← exactly what
  [../secuential.py](../secuential.py) is doing. [10] does time-series but **non-recurrently**.
- **Long-range dependencies.** [5] capped sequences at **16 tokens** and flagged
  vanishing gradients for longer seqs. Long Range Arena / copy / adding-problem on
  recurrent LGNs = open.
- **Gated logic cells.** No LSTM/GRU-analog built from gates (learnable forget/keep gate
  as actual logic). [5] used a plain concat-recurrence; gating is the obvious next step
  and a clean "logic-native" story (a forget gate literally *is* an AND-with-a-mask).
- **Parallel-scan / associative recurrence.** [5] explicitly names **associative
  recurrent blocks (O(log n) training)** as future work — i.e. a logic-gate analog of
  linear-attention/SSM/S4-style parallel scans. Nobody has built it.
- **State discretization over time.** How a *hard, binary* hidden state behaves under
  many recurrent steps (error accumulation, attractors, fixed points) is unstudied. The
  hidden state being literal bits makes this analyzable — a possible theory angle.
- **Hardware story for recurrent LGNs.** [5] leaves FPGA synthesis as future work; a
  streaming, single-bit-state recurrent circuit is a strong edge/always-on pitch.

## C. Concrete candidate directions (ranked by my read)

> Ranking heuristic: (novelty gap) × (fits this fork's tooling) × (cheap to falsify on
> MNIST-scale). Validate novelty with the `research-scout` skill before committing.

### C1. Gated logic recurrent cell ("LogicLSTM/LogicGRU") ⭐ top pick
Build an explicit gated recurrent cell from logic gates: a learnable **forget/update
gate** (a logic layer producing a mask bit) controlling whether each hidden bit is kept
or overwritten. Contrast vs [5]'s plain concat-recurrence on sequential-MNIST /
permuted-MNIST and a long-range toy (copy/adding). 
- *Why promising:* gating is the single biggest reason GRU/LSTM beat vanilla RNN, and [5]
  showed vanilla recurrence trails GRU — so a logic-native gate is the obvious lever, and
  it has a clean interpretability story (the forget gate is a readable circuit).
- *Cheapest test:* does it beat the plain `LogicRNNCell` on psMNIST? Falsifiable in a day.

### C2. Parallel-scan / associative logic recurrence
A logic-gate SSM: make the recurrence associative so it trains via parallel scan
(O(log n)). This is the future-work item [5] named but didn't build.
- *Why promising:* if it works it fixes [5]'s biggest practical complaints (training
  time + long sequences) at once, and rides the SSM/linear-RNN wave.
- *Risk:* associativity + Boolean state is non-trivial; higher research risk.

### C3. Better input encoding to attack the accuracy ceiling
Test Light DLGN's [4] hypothesis directly: replace 1–3-threshold thermometer encoding
with learned / higher-resolution / structure-aware binarization and measure the CIFAR-10
(or sequential-task) ceiling shift.
- *Why promising:* if the ceiling really is preprocessing, this is a high-impact, widely
  cited result; cheap to run; orthogonal to everyone's model-side work.

### C4. Conv + learned connectivity (combine [2] and [6]/[7])
LogicTreeNet conv currently uses random tree wiring; LILogic [7] showed learned wiring is
a big win for FC. Nobody has published **convolutional + learned Top-K connectivity**.
- *Why promising:* obvious additive gain, clear baseline, but "combination" papers are
  lower-novelty; do only if it unlocks a new accuracy/gate point.

### C5. Regularization that actually works for LGNs
[4] reports standard regularizers fail. A working LGN-specific regularizer (e.g. on gate
entropy, connectivity sparsity, or via the Gumbel/Hessian view of [3]) addressing the
generalization gap would be broadly useful.

## D. What to NOT spend novelty on (likely occupied)
- Plain discretization-gap fixes → owned by [3] (Gumbel+STE).
- Plain reparametrization for training speed → owned by [4] (IWP).
- Plain "learn the connections" on tabular/MNIST → [6][7] + Yue&Jha already there.
- Interpretable tabular/regression LGNs → Yue & Jha [8][9][10] own this.
- LUT-on-FPGA efficiency races → crowded ([11] + the whole LogicNets/PolyLUT/NeuraLUT
  cluster).

## E. Immediate next actions
1. Read Recurrent DDLGN [5] **in full** (arXiv:2508.06097) — confirm the exact gaps above
   are real and unclaimed.
2. Run `research-scout` on **C1 (gated logic recurrent cell)** to get a go/no-go before
   building.
3. Set up sequential-MNIST + permuted-MNIST + one long-range toy (copy/adding) as the
   standard bench in [../](..) so every idea is falsifiable on the same yardstick.
4. Bake [3] (Gumbel+STE) and [4] (IWP) into the cell from the start — they're strictly
   beneficial training infra, not research bets.
