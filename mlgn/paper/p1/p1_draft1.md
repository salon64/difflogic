<!--
p1_draft1.md — Paper 1, draft v1 (2026-07-09). Supersedes mlgn/paper/draft.md (v0.1, 06-21).

Changes vs v0.1 (per research/20_program_validation.md §B):
  - delayed-MNIST re-based on 3-seed means (was seed-0 = best-of-3); added d25/d75; error bars.
  - Added the delay-50 keep-bias sweep -> task-dependence is now shown from BOTH sides (§5.2).
  - Added the equal-gates copy control (h2048 rddlgn, still dead) -> gate-count defense is data (§5.1).
  - psMNIST equal-gates comparison re-based on 3 gated seeds (0.602±0.072); "gated soft is better"
    claim dropped (does not survive 3 seeds); renamed psMNIST-28 (chunked) everywhere.
  - "Collapses to chance" wording made exact (copy control: soft at chance, discrete 2x chance;
    delayed-MNIST control: exactly the majority-class baseline).
  - Discretization gap reframed via Kim's selection/computation decomposition + length-dependence
    evidence (research/14); Mind-the-Gap mechanism cited cautiously.
  - RDDLGN memorization-probe contrast sentence added (priority claim now scoped to LONG-RANGE).
  - Stability-recipe claim scoped ("sufficient for every run reported here", not universal).
  - FPGA/ASIC -> FPGA only. Citations now real, keys match references.bib (pandoc syntax [@key]).

Remaining TODOs before submission (~Aug 29, workshop picked after Jul 11 list drops):
  [TODO-FIG] split Fig 1 (recall = money plot); grad-norm-through-time figure (data in 47 JSONs,
             see results/curves_gradnorm.png if already generated / appendix A.3).
  [TODO-BIB] verify author given names + final venue strings against arXiv (marked in .bib).
  [TODO-VENUE] condense to 4 pp for the chosen workshop template; affiliations; acknowledgements.
  [TODO-OPT] optional runs: none required (all tables have >=1 seed, headlines have 3).
-->

# Gating Enables Long-Range Recall in Recurrent Logic Gate Networks

**Malcolm Malle** — [affiliation TBD]
*Draft v1 (2026-07-09). Target: NeurIPS 2026 workshop (non-archival). Code & results:
`mlgn/seqlgn/` in the project repository.*

---

## Abstract

Logic Gate Networks (LGNs) learn a circuit of two-input Boolean gates by gradient descent
and run as raw logic at inference, making them extremely cheap on CPUs and FPGAs. The few
existing recurrent LGNs all carry state by *concat-recurrence* — recomputing the hidden
state from scratch at every step — and two of the three groups behind them explicitly name
gating as future work. We build it: a **logic-native gated recurrent cell** in which a
learned two-to-one multiplexer per hidden bit implements a *constant-error carousel* — the
LSTM/GRU trick, realized in pure Boolean logic. On two long-range recall tasks the gated
cell carries information across delays where the concat-recurrence control fails
completely: on a synthetic copy task the control never learns (relaxed accuracy at chance),
and on delayed-recall MNIST it collapses to the majority-class baseline at *any* delay —
its gradient at the write step is exactly zero — while the gated cell still recovers the
digit three times more often than chance after 100 blank steps (3-seed means). We show the
gate's *keep-bias* is a task-dependent dial (high to hold, low to absorb): the same sweep
that rescues recall inverts on a classification task. We give the training recipe recurrent
LGNs need (keep-biased initialization against vanishing gradients; learning-rate decay and
non-finite-step skipping against exploding ones), and we report an honest boundary: at
equal gate count, gating does **not** help classification (permuted-MNIST-28: control
0.655 vs. gated 0.602 ± 0.072), and the multiplexer's blended state discretizes worse than
logic-recompute, a *computation* gap that grows with sequence length. To our knowledge this
is the first gated recurrent LGN and the first demonstration of long-range (50–100-step)
recall in a learned logic circuit.

---

## 1. Introduction

Differentiable Logic Gate Networks [@petersen2022deep] learn, for every neuron, *which* of
the 16 two-input Boolean functions to apply, via a softmax relaxation that is discretized
to a hard circuit at inference. The resulting models need no multiplications and no
weights — inference is pure logic — which yields order-of-magnitude efficiency gains over
quantized networks on FPGAs (e.g., 9 ns per CIFAR-10 image) [@petersen2024convolutional].

Extending LGNs to *sequences* is recent and thin. Three groups have built recurrent or
stateful LGNs: Recurrent Deep DLGNs for machine translation [@buehrer2025recurrent],
DiffLogic Cellular Automata [@miotti2025difflogic], and recurrent ternary LGNs for runtime
monitoring [@damera2026rdtlgn]. All three carry state the same way, by
**concat-recurrence**: concatenate the previous hidden bits with the current input and
recompute the state from scratch through combinational logic layers. None has *gating* —
no learned mechanism that decides, per bit and per step, whether to keep the stored value
or overwrite it. This is a recognized gap, not an oversight: DiffLogic CA explicitly names
"LSTM-like gating mechanisms" and "specialized gates for state forgetting" as future work
[@miotti2025difflogic], and the R-DTLGN authors likewise name gated variants as future work
[@damera2026rdtlgn].

We ask whether the single idea that made recurrent neural networks trainable on long
dependencies — **gating with a constant-error carousel** [@hochreiter1997long; @cho2014learning]
— can be realized in pure logic, and what it buys. The answer is specific and, we think,
useful: gating buys **long-range recall** — carrying a value across many steps of
uninformative input — a regime in which concat-recurrence provably starves (its gradient to
the write step vanishes to exactly zero) and goes to chance. It does *not* buy better
classification.

**Contributions.**

1. **A logic-native gated recurrent cell** (§3): a learned 2:1 multiplexer per hidden bit,
   `h′ = s·h + (1−s)·c`, which at binary values is exactly the Boolean MUX
   `(s∧h) ∨ (¬s∧c)` and whose keep branch is a constant-error carousel
   (`∂h′/∂h = s ≈ 1` on kept bits).
2. **The first demonstration of long-range recall in a recurrent LGN** (§5.1): on copy
   (up to 50 blank steps) and delayed-recall MNIST (up to 100 blank steps) the
   concat-recurrence control fails completely — on delayed-MNIST it cannot even learn the
   relaxed task (gradient at the write step identically zero) — while the gated cell holds.
   All headline numbers are 3-seed means; the control failure is confirmed at *equal gate
   count*.
3. **Keep-bias is a task-dependent dial** (§5.2), shown from both sides: sweeping the
   gate's keep-bias moves recall and classification accuracy in *opposite* directions
   (recall wants the carousel on; integration wants it off).
4. **A training recipe for recurrent LGNs** (§3.4, validated §5.4): both classic RNN
   pathologies appear in the logic setting — vanishing gradients (fixed by keep-biased
   initialization, the temporal analog of difflogic's residual initialization) and
   exploding gradients (handled by cosine learning-rate decay plus skipping non-finite
   optimizer steps; post-hoc gradient clipping provably cannot help once a backward pass
   overflows).
5. **Honest boundaries** (§5.3, §5.5): at equal gates, gating gives no benefit on a
   classification task; and the multiplexer's convex-blend state discretizes worse than
   logic-recompute — a *computation* gap in the sense of @kim2026align that grows with
   sequence length. We state this wall precisely rather than claim to remove it.

**Relation to the closest prior work.** RDDLGN's memorization probe already shows
*short-shift* recall (reproducing tokens shifted by up to 12 positions within 16-step
sequences) [@buehrer2025recurrent], and its authors report vanishing gradients on longer
sequences. Our claim is therefore scoped to the **long-range** regime (50–100 steps of
*blank* input between write and read), where we show concat-recurrence does not degrade
gracefully but fails outright, and a gated carousel is sufficient to cross it.

---

## 2. Background and Related Work

**Differentiable LGNs.** A `LogicLayer` holds, per neuron, a categorical distribution over
the 16 two-input Boolean functions; inputs are relaxed to `[0,1]`, each gate is replaced by
its probabilistic surrogate (`AND → a·b`, `XOR → a+b−2ab`, …), and the soft output is the
softmax-weighted mixture [@petersen2022deep]. At inference the distribution is argmaxed to
one hard gate. Wiring (which two inputs feed each neuron) is random and fixed. A
`GroupSum` head partitions output bits into `k` class groups and sums each group (a
popcount per class, temperature `τ`). Follow-up work improves scaling and training:
convolutional gate trees with **residual initialization** — biasing gate logits toward the
pass-through gate so deep networks start near identity [@petersen2024convolutional] —
and reparametrizations of the 16-way softmax [@ruettgers2025light; @kim2026multilinear].
Weightless/LUT-based networks pursue the same deploy-as-logic goal with lookup tables as
the unit [@bacellar2024dwn].

**Recurrent and stateful LGNs.** RDDLGN [@buehrer2025recurrent] carries hidden bits across
16 timesteps by re-concatenation for neural machine translation (WMT'14 En–De, BLEU 5.00
soft / 4.39 discretized, between an RNN and a GRU); it reports vanishing gradients for
longer sequences and defers hardware. DiffLogic CA [@miotti2025difflogic] re-applies one
combinational difflogic circuit as a cellular-automaton update rule over a binary state
("working memory"), demonstrating learned self-organizing patterns and Conway's Game of
Life. R-DTLGN [@damera2026rdtlgn] is a recurrent *ternary* (Kleene logic) variant for
signal-temporal-logic runtime monitoring, with a delay-register concat-recurrence and a
training-side "trajectory distillation" to harden the recurrent loop. Separately,
interpretable Differentiable Logic Networks address time-series *classification* without
recurrence [@yue2025dlnts]. None of these learns a keep/write decision per state bit; all
recompute (or delay-register) the full state each step.

**Gating in RNNs.** Vanilla recurrence suffers vanishing/exploding gradients
[@bengio1994learning; @pascanu2013difficulty]. LSTM's constant error carousel — an
approximately-identity self-connection fenced by multiplicative gates
[@hochreiter1997long] — and its GRU simplification [@cho2014learning] fixed this;
initializing the forget gate open is standard practice [@gers2000learning]. Our cell is
the Boolean-minimal version of this machinery: a single learned MUX per bit; the
keep-bias (§3.3) is exactly the forget-gate bias, and simultaneously the *temporal* twin of
difflogic residual initialization [@petersen2024convolutional].

**The discretization gap.** Trained (soft) and deployed (hard) LGNs differ.
@yousefi2025mindthegap reduce the gap in feedforward LGNs with Gumbel noise plus a
straight-through estimator; @kim2026align decompose the gap into a **selection gap**
(soft mixture vs. argmaxed gate; removable by hard-forward training) and a **computation
gap** (soft vs. hard evaluation of the *same* gate; zero iff the propagated values are
binary, irreducible by training otherwise).¹ We adopt this vocabulary in §5.5: the
recurrent state of our gated cell is exactly the kind of continuous value that makes the
computation gap compound over timesteps. We use neither method — we report the gap and
characterize its interaction with gating.

¹ @kim2026align also challenge the Hessian-regularization explanation given for the Gumbel
method's success, so we cite the mechanism cautiously and rely only on the decomposition.

---

## 3. Method

### 3.1 A recurrent logic cell with a pluggable memory mechanism

We process one timestep at a time. Let `z = [x_t ; h]` be the concatenation of the current
input and the previous hidden state (all values in `[0,1]`; at deployment, in `{0,1}`).
Every mechanism below is built from the same block — a `LogicMLP`, a stack of two
`LogicLayer`s mapping `z` to `hidden` bits — and they differ **only** in how state is
carried, so the comparison isolates the memory mechanism:

| Mechanism | Update | Carried state | Learned gates |
|---|---|---|---|
| `rddlgn` (control) | `h′ = LogicMLP(z)` | `h`, recomputed | `2H` |
| `gated` (**ours**) | `c = cand(z)`, `s = gate(z)`, `h′ = s·h + (1−s)·c` | `h`, multiplexed | `4H` |
| `lstm` (ablation) | `C′ = (C∧f) ∨ (i∧C̃)`; `h′ = readout([o ; C′])` | `(h, C)` | `10H` |
| `gru_cell` (ablation) | `C′ = s·C + (1−s)·C̃`; `h′ = readout([o ; C′])` | `(h, C)` | `8H` |

`H` is the hidden width. The `rddlgn` control is a faithful reimplementation of the
concat-recurrence design shared by RDDLGN and DiffLogic CA. The two dual-state ablations
complete a 2×2 over (gate structure × state separation): `lstm` uses independent
forget/input gates with a Boolean OR standing in for LSTM's addition; `gru_cell` uses the
single complementary MUX gate on a dedicated cell state. A `GroupSum(k, τ)` head reads the
final hidden state. The hidden state is initialized to zeros (a valid binary state).

*Gate accounting.* "Learned gates" counts LogicLayer neurons (the objects that consume
parameters and training compute). At deployment the MUX itself costs 3 additional *fixed*
gates per bit — `(s∧h) ∨ (¬s∧c)` — and the GroupSum head is a popcount tree; neither is
learned. Equal-gates comparisons in §5 match learned gates (e.g., control widened from
`H=1000` to `H=2000` to match the gated cell's `4H = 4000`).

### 3.2 The carousel, in logic

The gated update `h′ = s·h + (1−s)·c` equals, at binary `s, h, c`, the Boolean multiplexer
`(s∧h) ∨ (¬s∧c)`. Its recurrence Jacobian is `∂h′/∂h = diag(s)`: when the gate keeps a bit
(`s ≈ 1`), the gradient flows backward through that step **unattenuated** — the constant
error carousel of LSTMs and GRUs [@hochreiter1997long], realized in pure logic. The
control has no such path: its `∂h′/∂h` is a product of logic-layer Jacobians whose norm
shrinks multiplicatively with every layer and every timestep. Empirically this is not a
soft degradation but a hard one: on the recall tasks of §5.1 the control's gradient norm at
the write step is *exactly zero* (below float precision) — the training signal never
reaches the only timestep that matters.

A property we use later (§5.5): at deployment the gated cell's state is **exactly binary
by construction** — gates argmax to hard Boolean functions, the MUX of binary values is
binary, inputs are binarized, and `h₀ = 0` — so the deployed model is a clean clocked
sequential circuit. The *soft training-time* state, in contrast, drifts into `(0,1)`
through the convex blend; that mismatch is the computation gap of §5.5.

### 3.3 Keep-bias: switching the carousel on at initialization

The carousel only helps if the gate actually *keeps* early in training. With an unbiased
gate (`s ≈ 0.5` at initialization) on a long sequence, the stored value decays by half
each step, so no gradient survives to teach the gate to keep — a chicken-and-egg cold
start (§5.4 shows it empirically: flat loss at chance). We therefore add `keep_bias` to
the TRUE-gate logit of the gate network's final LogicLayer, so `s` defaults high (carousel
on) while a write path (`s < 1`) remains learnable. This is the logic-native form of the
LSTM forget-gate bias [@gers2000learning] and the *temporal* generalization of difflogic's
residual initialization [@petersen2024convolutional] — the same trick, applied through
time rather than through depth. Crucially it is **necessary, not cosmetic**: `keep_bias = 0`
reproduces the cold start on every recall task we ran (§5.2, §5.4). The dual-state cells
get the analogous standard initialization (keep-bias the forget gate *and* close the input
gate; §5.4 explains why they need both and still cold-start at longer lengths).

### 3.4 Training recipe: both classic pathologies, handled

Training recurrent LGNs surfaces both classic RNN pathologies, and both must be handled
explicitly:

- **Vanishing → keep-bias** (§3.3). Without it the gated cell never starts on long
  sequences.
- **Exploding → learning-rate decay + step-skipping.** Keep-biased near-identity Jacobians
  can creep above 1 across many steps; late in training (as gates sharpen and the loss
  drops) a single backward pass can overflow. Post-hoc gradient clipping **cannot** rescue
  this: once a backward pass produces `inf`, normalizing the gradient yields `nan`, which
  poisons the weights (we verified this failure mode directly; §5.4). We instead
  (i) cosine-decay the learning rate (3e-3 → 3e-4) so the confident late phase takes small
  steps, and (ii) **skip any optimizer step whose global gradient norm is non-finite**, so
  an exploding batch never touches the weights; a dead-run early-stop aborts if an entire
  evaluation window is skipped. With this recipe every headline run in §5 completed with
  **zero skipped steps**. We scope the claim honestly: the recipe was sufficient for every
  configuration reported here, but in later experiments outside this paper's scope (longer
  sequences, larger learning rates, auxiliary losses) heavy step-skipping can reappear;
  skipping is a symptom to fix via the learning rate, not a cure.

---

## 4. Experimental Setup

**Tasks.** All tasks emit sequences with values in `[0,1]` and an integer label, and are
chosen to isolate *memory* — how well a cell carries information across time:

- **copy(L)** — synthetic recall. A one-hot symbol from an alphabet of 8 (plus a cue bit
  marking the write step) is presented at `t = 0`, followed by `L−1` blank steps; the model
  must classify the symbol at the end. Uniform chance 12.5%. Because the write step is
  cue-flagged, the task tests pure *holding*, not detection; `L` is the difficulty dial.
  Lineage: the classic copy/memorization probes [@hochreiter1997long; @arjovsky2016unitary].
- **delayed-recall MNIST (dMNIST-D)** — real-data recall. The full 784-pixel MNIST image is
  presented in a *single* timestep (input width 784), followed by `D` blank steps, then
  classification. Uniform chance 10%; the majority-class baseline on the MNIST test set is
  **11.35%**, which matters below. Isolates *holding* real data from encoding it.
- **psMNIST-28 (chunked)** — classification/integration control-task. Permuted sequential
  MNIST [@le2015simple] with a fixed pixel permutation, fed **28 pixels per step for 28
  steps**. We flag clearly: this is *not* the standard 784-step, 1-pixel benchmark (which
  is beyond our wall-clock budget; §7) — published LSTM numbers on the standard task
  (~0.90) are not comparable. A length sweep feeds 784/k pixels per step, k ∈ {28…7}
  (28–112 steps).

**Protocol.** We evaluate **discrete-locked**: gates argmaxed, inputs binarized — the
accuracy of the actual deployed logic circuit (`disc`). We also report the relaxed
training-mode accuracy (`soft`) and the **discretization gap** `= soft − disc`, both on
the same best-validation checkpoint. Model selection uses discrete validation accuracy;
the test set is touched once. Headline numbers are means ± s.d. over 3 seeds with paired
seeds across arms; single-seed cells are marked. Fairness controls: the concat-recurrence
control is run both at equal *width* (2H learned gates) and at equal *learned gates*
(width doubled, 4H); we also verified the control's failure is not a gradient-scaling
artifact (`grad_factor = 2` does not revive it).

**Hyperparameters.** Adam, learning rate 3e-3 cosine-decayed to 3e-4, 20k iterations,
batch 128, `τ = 30`, two LogicLayers per LogicMLP, hidden width 1000 (dMNIST, psMNIST-28)
or 1024 (copy); keep-bias 3 (copy), 6 (dMNIST), swept on psMNIST-28 and dMNIST-50.
Runs take ~25 min (psMNIST-28) to ~4.5 h (dMNIST-100) on RTX-2080-class GPUs.

---

## 5. Results

![**Figure 1** — Length sweeps (`results/curves.png`). Left: copy — the gated cell (blue)
solves long-range recall while the concat-recurrence control (red) never learns and the
dual-state cells (`lstm`/`gru_cell`) cold-start past L20. Right: psMNIST-28 length sweep at
a fixed recall-tuned keep-bias — see §5.2 for why this keep-bias is wrong for
classification (cautionary panel).](../../seqlgn/results/curves.png)

![**Figure 2** — (`results/curves_bc.png`) Left (B): one dial, opposite slopes — the
keep-bias sweep *rises* on recall (dMNIST-50, blue) and *falls* on integration
(psMNIST-28, orange). Right (C): real-data recall vs. delay, mean ± s.d. over 3 seeds —
the control (red) sits exactly at the majority-class baseline at every delay ≥ 25 while
the gated cell (blue) holds ~3× chance through 100 blank
steps.](../../seqlgn/results/curves_bc.png)

### 5.1 Gating enables long-range recall; concat-recurrence fails outright

**Copy.** Discrete test accuracy, mean ± s.d. over 3 seeds (per-seed values in App. A.1):

| L (blank steps) | gated (4,096 gates) | control (2,048) | control, equal-gates (4,096) | lstm (10,240) | gru_cell (8,192) |
|---:|:---:|:---:|:---:|:---:|:---:|
| 20 | **0.96 ± 0.07** | 0.25 | 0.13 | 0.76 | 0.75 |
| 35 | **0.79 ± 0.26** | 0.26 | 0.25 | 0.13 | 0.13 |
| 50 | **0.33 ± 0.08** | 0.26 | 0.25 | 0.13 | 0.13 |

The control **never learns the task**: its *soft* accuracy is pinned at chance (≈ 0.12) at
every length — the relaxed model, with every advantage of continuous optimization, extracts
nothing — and this does not change at equal gate count or with `grad_factor = 2`. (Its
discretized circuit reaches ≈ 0.25, i.e., 2× chance; rounding a chance-level soft optimum
recovers a weak heuristic — we note the curiosity and do not analyze it further.) The
gated cell solves L = 20 essentially perfectly and degrades with length; the L = 35 spread
is bimodal ({0.50, 0.88, 1.00} — seeds either solve or half-solve). At L = 50 the *soft*
model still reaches 0.83 ± 0.08 while the deployed circuit gets 0.33 — a gap we return to
in §5.5; doubling capacity (hidden 2048) lifts the deployed circuit to 0.76.

The dual-state cells are strictly worse than the single-MUX cell: they reach ~0.75 at
L = 20 but **cold-start to chance at L ≥ 35** — their initialization carousel is weaker
(§5.4) — so the simplest gate structure is also the most robust.

**Delayed-recall MNIST (the headline).** Discrete test accuracy, mean ± s.d. (3 seeds);
control single-seed (its variance is degenerate — see below):

| delay D | gated (kb 6) | control |
|---:|:---:|:---:|
| 0 | 0.715 ± 0.013 | 0.554 |
| 25 | 0.489 ± 0.028 | 0.1135 |
| 50 | 0.362 ± 0.045 | 0.1135 |
| 75 | 0.274 ± 0.035 | 0.1135 |
| 100 | **0.300 ± 0.061** | 0.1135 |

At **any** nonzero delay we tested (25–100), the control lands on **exactly 0.1135 — the
majority-class baseline of the MNIST test set — in both soft and discrete evaluation**. It
does not degrade; it fails completely, and the mechanism is visible in the gradients: the
norm of the loss gradient reaching the write step (`t = 0`) is **identically zero**, and
the gradient dies within ~3 steps of the readout (App. A.3). No amount of gate count,
capacity, or tuning fixes a model whose training signal never reaches the only informative
timestep. The gated cell, whose carousel carries gradient to `t = 0` at full strength
(early-step gradient norms exceed late-step norms by 2–3 orders of magnitude), holds the
digit at **3.0× uniform chance (2.6× the majority baseline) through 100 blank steps**.
(D = 75 dipping below D = 100 is within seed noise.)

### 5.2 Keep-bias is a task-dependent dial — shown from both sides

The same initialization that *enables* recall *hurts* integration, and vice versa. We
sweep `keep_bias` on one task of each kind (single seed; Fig. 2B):

| keep-bias | psMNIST-28, disc (soft) | dMNIST-50, disc (soft) |
|---:|:---:|:---:|
| 0 | **0.632** (0.709) | 0.177 (0.114 — cold start) |
| 1 | 0.548 (0.660) | — |
| 2 | 0.541 (0.668) | — |
| 3 | — | 0.293 (0.582) |
| 4 | 0.389 (0.659) | — |
| 6 | — | **0.369** (0.562) |

On classification, keep-bias 0 is best and the discretization gap shrinks monotonically as
the bias drops (0.27 at kb 4 → 0.08 at kb 0); the kb 1 vs. kb 2 ordering is within noise.
On recall the slope is *reversed*: kb 0 cold-starts (the soft model never leaves the
majority baseline — the §3.3 mechanism, observed on real data), and accuracy rises with
the bias. **One interpretable dial, opposite optima: high to hold, low to absorb.** This
also retroactively explains our own first psMNIST result (kb 4, chosen for recall) being
poor — Fig. 1 right shows the control beating the recall-tuned gated cell at *every*
length, a cautionary tale about porting a memory-task configuration to an integration
task.

### 5.3 Gating does not help classification at equal gates (honest boundary)

At matched learned-gate count (4,000: gated at H = 1000, control at H = 2000) and each
arm's best keep-bias, on psMNIST-28:

| model | gates | disc | soft | gap |
|---|:---:|:---:|:---:|:---:|
| gated, kb 0 (3 seeds) | 4,000 | 0.602 ± 0.072 | 0.689 ± 0.022 | +0.087 |
| control (H = 2000) | 4,000 | **0.655** | 0.694 | +0.038 |
| control (H = 1000) | 2,000 | 0.620 | 0.652 | +0.033 |

The control matches or beats the gated cell (the gated mean is dragged by one outlier
seed at 0.519, but no seed beats 0.655). Soft accuracies are a wash; the difference is the
**discretization gap**, which is 2–4× larger for the gated cell (§5.5). **Gating buys
nothing on integration tasks.** This is consistent with §5.1–5.2: the carousel matters
exactly when the task requires carrying information across many uninformative steps —
which classification at these lengths does not, since the control's vanishing gradient is
not fatal at 28 steps of informative input.

### 5.4 The two RNN pathologies, in logic

Training only works with the recipe of §3.4, and each ingredient maps to one observed
failure:

- **Cold start (vanishing).** Without keep-bias the gated cell sits at chance with a flat
  loss on copy-50 and dMNIST-50 (soft accuracy at the majority baseline; §5.2). With it,
  the gradient reaches `t = 0` at full strength.
- **Late-phase explosion.** With keep-bias but a flat learning rate, copy runs at L ≥ 35
  learn and then die: a single overflowing backward pass poisons the weights to NaN.
  Gradient clipping does not prevent this (clipping an `inf` gradient yields `nan` — we
  observed exactly this); skip-stepping alone catches the aftermath but the run is already
  poisoned (13–17k of 20k steps skipped). **Prevention** — cosine LR decay so the sharp,
  confident late phase takes small steps — fixed it: all reported runs finish with zero
  skipped steps and no NaN.
- **Why the dual-state cells cold-start.** LSTM's carousel is `∂C′/∂C = f·(1 − i·C̃)`;
  keep-biasing only the forget gate leaves a random input path that eats it
  (`f·(1−i·C̃) ≈ 0.58` at init vs. the GRU MUX's `s ≈ 0.78`, and `0.58^{35} ≈ 10⁻⁸`).
  Closing the input gate at initialization (the standard LSTM trick) rescues L = 20
  (0.13 → 0.76) but the cell re-collapses at L ≥ 35 — the two-gate design needs ever
  stronger coordinated initialization as length grows. The single complementary MUX gate,
  whose keep and write paths cannot disagree, is the robust choice.

### 5.5 The gap that remains: a computation gap that grows with length

Where the gated cell only partially solves a task, its deployed circuit underperforms its
soft model (copy-50: soft 0.83, disc 0.33; psMNIST-28: gap +0.09 vs. the control's +0.04).
Following @kim2026align, the train/deploy gap decomposes into a *selection* gap (which
gate; removable by hard-forward training) and a *computation* gap (soft vs. hard values
through the *same* gates; zero iff the propagated values are binary). Two observations
place our gap squarely in the second category:

1. **It grows with sequence length** (copy: 0.00 at L = 20 → +0.50 at L = 50, and the
   dMNIST gaps grow with delay). A selection gap is length-independent for a shared
   recurrent cell — the argmaxed gate choice is the same at every step; only a computation
   gap compounds, as the convex-blend state `s·h + (1−s)·c` drifts away from `{0,1}` a
   little more each step.
2. **Its size tracks how binary the state is.** The control, which recomputes its state
   through hard-saturating logic each step, keeps near-binary activations and shows small
   gaps (+0.03–0.04) even when *it* partially solves a task; the MUX's blended state is
   precisely the continuous-input case in which soft and hard gate evaluation diverge.

Consistent with this reading, a gate-*selection* regularizer does not help: an entropy
penalty that pushes gate distributions to one-hot dragged the *soft* accuracy down to the
discrete one (copy-50: soft 1.0 → 0.89, disc unchanged at 0.75) instead of lifting the
circuit — the mixture was doing real work, and committing it does not binarize the
*state*. Where the task is fully solved, the gap vanishes (copy-20: disc = soft = 1.0,
gap exactly 0): a fully-solved cell drives its state to saturation and discretizes
losslessly.

We deliberately do not claim a fix. The gap is a property of carrying *continuous* state
through time under a convex blend; training-side hard-forward methods
[@yousefi2025mindthegap; @kim2026align] target the selection component, and architectural
re-binarization of the *state* is a different design axis — we state the wall precisely
and leave closing it to follow-up work.

---

## 6. What We Learned

1. **Gating buys long-range *recall*, not general performance.** The carousel is decisive
   exactly when a task must carry information across many uninformative steps (copy,
   delayed recall); on integration tasks the simpler concat-recurrence is as good — and
   its failure on recall is total (zero gradient at the write step), not gradual.
2. **Keep-bias is the dial** that adapts one cell to both regimes — a single,
   interpretable, task-dependent hyperparameter (high = hold, low = absorb), shown by
   opposite-slope sweeps on a recall and an integration task.
3. **The MUX trades discretization-robustness for gradient flow.** Its convex-blend state
   is what carries gradients across 100 steps *and* what discretizes worse than
   logic-recompute on partially-solved tasks — one mechanism, both faces; the gap is a
   computation gap that compounds with length.
4. **Simpler is better.** One complementary MUX gate beats dual-gate cells
   (LSTM-style), which need coordinated initialization and still cold-start at length.
5. **Recurrent LGNs need both classic RNN fixes at once** — carousel-on initialization
   against vanishing, and prevention-style stability (LR decay; skip non-finite steps —
   clipping is provably too late) against exploding.

---

## 7. Limitations and Future Work

- **Scope: recall, not classification.** Our positive claim is scoped to recall; at equal
  gates, gating does not help integration tasks (§5.3).
- **Constructed benchmarks; no standard-benchmark or float baseline.** copy and dMNIST are
  synthetic/constructed probes; psMNIST-28 is a chunked, nonstandard variant (the
  standard 784-step task is ~40 h/run on our hardware and unattempted). We also do not
  report a floating-point RNN baseline: a small float GRU would likely solve these tasks —
  the LGN value proposition is deployment cost (gates, no multiplies, FPGA-native), not
  accuracy, and the comparisons here isolate memory *mechanisms* within the LGN family.
- **The computation gap is stated, not closed** (§5.5). Hard-forward training
  [@kim2026align] and architectural state re-binarization are the obvious candidates; we
  pursue the architectural route in follow-up work.
- **Dual-state cells were not exhaustively tuned** on recall; copy already shows the
  single-gate cell dominating, but a focused study could tighten that claim.
- **Hardware.** A core motivation for LGNs is FPGA deployment; the deployed gated cell is
  already an exactly-binary clocked sequential circuit (§3.2), and synthesizing it —
  registers plus learned combinational logic — is the natural next paper. We defer it, as
  did RDDLGN and DiffLogic CA.

---

## 8. Conclusion

We presented, to our knowledge, the first *gated* recurrent Logic Gate Network: a learned
per-bit Boolean multiplexer whose keep path is a constant-error carousel in pure logic.
The carousel converts long-range recall from impossible to solvable for recurrent LGNs —
the existing concat-recurrence design does not merely underperform but receives literally
zero gradient at the write step and collapses to baseline — while on classification at
equal gate count it buys nothing, and its blended state discretizes worse than
logic-recompute, a computation gap that grows with length. The cell costs 2× the learned
gates of the control, 3 fixed gates per bit at deployment, and one interpretable
hyperparameter with a clear rule (keep-bias: high to hold, low to absorb), plus a
two-line training recipe. We hope this establishes gating as a cheap, characterized
building block for sequential logic circuits — and the remaining state-binarization wall
as a crisp target for what comes next.

---

## Appendix A — Reproducibility

### A.1 Per-seed headline numbers

- **copy, gated, disc:** L20 {1.000, 0.878, 1.000}; L35 {0.878, 0.504, 1.000};
  L50 {0.380, 0.241, 0.378} (soft L50 {0.879, 0.743, 0.877}; an earlier seed-0 run
  reached soft 1.000 at the same disc — the gap, not the seed, is the story).
- **dMNIST, gated kb 6, disc:** D0 {0.700, 0.722, 0.722}; D25 {0.457, 0.511, 0.498};
  D50 {0.370, 0.403, 0.314}; D75 {0.313, 0.263, 0.245}; D100 {0.339, 0.230, 0.331}.
- **psMNIST-28, gated kb 0, disc:** {0.632, 0.654, 0.519} (soft {0.709, 0.693, 0.665}).
- **copy capacity run:** gated H = 2048, L50: disc 0.757 (soft 0.879).
- **Entropy-regularizer negative result:** copy-50, H = 2048, coeff 0.05: disc 0.754
  (unchanged), soft 1.0 → 0.886.

### A.2 Commands

All runs use `python -m mlgn.seqlgn.train` on the `difflogic` CUDA backend. Representative
configurations (seeds 0/1/2 via `--seed`):

```bash
# copy(L): gated vs control vs equal-gates control vs lstm/gru_cell
train --task copy --seq-len {20,35,50} --alphabet 8 --mechanism gated  --hidden 1024 \
      --keep-bias 3 --lr 0.003 --lr-min 0.0003 --iters 20000 --batch-size 128
train --task copy --seq-len {20,35,50} --mechanism rddlgn --hidden 1024   # control
train --task copy --seq-len {20,35,50} --mechanism rddlgn --hidden 2048   # equal-gates
train --task copy --seq-len {20,35,50} --mechanism {lstm,gru_cell} --hidden 1024 --keep-bias {4,3}

# delayed-recall MNIST: whole image at t=0 (chunk 784), D blank steps
train --task smnist-pixel --chunk 784 --delay {0,25,50,75,100} --mechanism gated \
      --hidden 1000 --keep-bias 6            # + kb {0,3} at delay 50 for the sweep
train --task smnist-pixel --chunk 784 --delay {0,25,50,75,100} --mechanism rddlgn --hidden 1000

# psMNIST-28 (chunked): kb sweep + equal-gates control + length sweep
train --task psmnist --chunk 28 --mechanism gated  --hidden 1000 --keep-bias {0,1,2,4}
train --task psmnist --chunk 28 --mechanism rddlgn --hidden {1000,2000}
train --task psmnist --chunk {16,14,8,7} --mechanism {gated,rddlgn} --hidden 1000
```

Result JSONs (accuracy, soft, gap, gate count, skipped steps, train minutes, gradient
profiles) live in `mlgn/seqlgn/results/`; the full run log is
`mlgn/research/04_experiment_log.md`.

### A.3 Figures

`curves.png` and `curves_bc.png` are generated by `python -m mlgn.seqlgn.plot` (last
regenerated 2026-07-02, i.e., from exactly the runs reported here). **Caveat for
regeneration:** the results directory now also contains later-project runs (mechanisms
`latch`/`clatch`/`combo`, and `gated` runs with `deep_sup > 0` / `margin_reg > 0` /
`anneal`); `plot.py` must exclude those (filter `deep_sup in {0, None}`, no `anneal`,
mechanism ∈ {rddlgn, gated, lstm, gru_cell}) before regenerating, or the copy panel will
mix in runs that use a different training signal. Gradient-norm-through-time profiles
(`grad_profile` arrays) are stored in 47 result JSONs — e.g., copy-50: gated
`[109.8, 98.2, …, 0.017]` vs. control `[0.0, 0.0, …, 0.031]`; dMNIST-50: gated
`[15.9, …, 0.026]` vs. control `[0.0, …, 0.028]` — and back the carousel figure
(`curves_gradnorm.png`). [TODO-FIG: promote the recall panels to Figure 1 for the
camera-ready; re-caption the psMNIST length panel as the cautionary keep-bias example.]
