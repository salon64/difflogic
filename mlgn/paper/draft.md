# Gating Enables Long-Range Recall in Recurrent Logic Gate Networks

**Rough draft — v0.1 (2026-06-21).** Workshop-length. `[TODO]`/`[CITE]` mark gaps.
Authors: Malcolm Malle, [advisors]. Code & data: `mlgn/seqlgn/`.

---

## Abstract

Logic Gate Networks (LGNs) learn a circuit of 2-input Boolean gates by gradient descent
and run as raw logic at inference, making them extremely cheap on CPU/FPGA/ASIC. Recurrent
LGNs have so far used *concat-recurrence* — recomputing the hidden state from scratch each
step — which we show **fails completely at long-range recall**: it cannot carry information
across even a modest delay, collapsing to chance. We introduce a **logic-native gated
recurrent cell** in which a learned 2:1 multiplexer per hidden bit implements a
*constant-error carousel* (the LSTM trick, in pure Boolean logic). On two long-range recall
tasks — a synthetic copy task and a delayed-recall variant of MNIST — the gated cell carries
information across long blank delays where the concat-recurrence control sits at chance. We
find the gate's *keep-bias* is a **task-dependent** knob: high to hold state (recall), low to
absorb inputs (classification). We further give a training recipe (keep-biased
initialization vs. vanishing gradients; learning-rate decay and step-skipping vs. exploding
gradients) needed to train recurrent LGNs at all. On a classification task (permuted,
chunked MNIST) gating offers no benefit at equal gate count — we scope our claim to recall
accordingly. To our knowledge this is the first *gated* recurrent LGN and the first
demonstration of long-range memory in a learned logic circuit.

---

## 1. Introduction

Differentiable Logic Gate Networks [Petersen et al., 2022] learn, for each neuron, which of
the 16 two-input Boolean functions to apply, via a softmax relaxation that is discretized to
a hard circuit at inference. The resulting networks need no multiplications and run as raw
logic gates, achieving order-of-magnitude efficiency gains over quantized neural nets on
FPGA/ASIC [Petersen et al., 2024].

Extending LGNs to *sequences* is recent and thin. Both prior recurrent approaches —
Recurrent Deep DLGN [Bührer et al., 2025] and DiffLogic Cellular Automata
[Miotti et al., 2025] — carry state by **concat-recurrence**: concatenate the current input
with the previous hidden bits and recompute the new state through logic layers. Neither uses
*gating*; DiffLogic CA explicitly names "LSTM-like gating" as future work.

We ask whether the single idea that made recurrent neural networks work — **gating with a
constant-error carousel** [Hochreiter & Schmidhuber, 1997] — can be realized in pure logic,
and whether it buys long-range memory that concat-recurrence lacks.

**Contributions.**
1. A **logic-native gated recurrent cell**: a learned 2:1 multiplexer per hidden bit
   (`h' = s·h + (1−s)·c`) that is, at binary values, the Boolean MUX `(s∧h)∨(¬s∧c)` and
   whose "keep" branch is a constant-error carousel (§3).
2. The first demonstration of **long-range recall** in a recurrent LGN: on copy and
   delayed-MNIST the control collapses to chance while the gated cell holds (§5.1).
3. The finding that **keep-bias is task-dependent** (recall vs. integration), and that
   gating helps recall but **not** classification at equal gate count (§5.2–5.3).
4. A **training recipe** for recurrent LGNs that addresses the vanishing- and
   exploding-gradient pathologies that otherwise prevent training (§4).

---

## 2. Background & Related Work

**Differentiable LGNs.** A `LogicLayer` holds, per neuron, a categorical distribution over
16 gates; inputs are relaxed to `[0,1]`, each gate has a probabilistic surrogate (`AND→ab`,
`XOR→a+b−2ab`, …), and the soft output is the softmax-weighted mixture. At inference the
distribution is argmaxed to one hard gate. Connections (which 2 inputs feed each neuron) are
random and fixed. A `GroupSum` head partitions output bits into `k` class groups
[Petersen et al., 2022].

**Recurrent LGNs.** RDDLGN [Bührer et al., 2025] applies concat-recurrence to neural machine
translation; DiffLogic CA [Miotti et al., 2025] applies it to cellular automata. Both
recompute state each step and report difficulty with long sequences. `[CITE]`

**Residual / forget-bias initialization.** Petersen et al. [2024] bias gate logits toward a
pass-through gate to obtain differentiable residual connections in *deep feed-forward* LGNs.
Our keep-bias is the *temporal* analog of this and of the LSTM forget-gate bias
[Gers et al., 2000] (§3.3).

**The discretization gap.** Soft (training) and hard (inference) LGNs differ; methods exist
to close the gap (Gumbel+STE [Yousefi et al., 2025]). We do not use them; we report the gap
and find it interacts with our gate (§5.3).

---

## 3. Method

### 3.1 Recurrent logic cell

We process one timestep at a time. Let `z = [x_t ; h]` be the concatenation of the current
input and previous hidden state. We compare four cell mechanisms, all built from stacks of
`LogicLayer`s ("LogicMLP"), differing only in how state is carried:

- **`rddlgn` (control)** — concat-recurrence, state recomputed each step:
  `h' = LogicMLP(z)`. This is the RDDLGN / DiffLogic-CA design.
- **`gated` (GRU-style, ours)** — a learned multiplexer per hidden bit:
  ```
  c = candidate(z);   s = gate(z)
  h' = s·h + (1−s)·c
  ```
- **`lstm`** — a dedicated cell state `C` with independent forget/input gates and an OR
  combine (`C' = (C∧f)∨(i∧C̃)`), output `h' = readout([o ; C'])`. (Ablation.)
- **`gru_cell`** — a dedicated cell state updated by the GRU MUX (`C' = s·C + (1−s)·C̃`),
  output `h' = readout([o ; C'])`. (Completes the gate-structure × state-separation 2×2.)

A `GroupSum(k, τ)` head reads the final hidden state.

### 3.2 The carousel

The gated update `h' = s·h + (1−s)·c` equals, at binary `s,h,c`, the Boolean multiplexer
`(s∧h)∨(¬s∧c)` (3 gates/bit in hardware). Its key property is the Jacobian of the recurrence:
`∂h'/∂h = s`. When the gate keeps a bit (`s≈1`), the gradient flows backward through time
**un-attenuated** — the constant-error carousel of LSTMs/GRUs, realized in pure logic. The
control has no such path: `∂h'/∂h` is a product of softmax-gate Jacobians whose norm shrinks
with depth/time → vanishing gradients on long sequences.

### 3.3 Keep-bias (initialization)

The carousel only helps if the gate *keeps* early in training. With a random gate (`s≈0.5`)
on a long sequence, state decays before the gate can learn to hold it — a cold start. We
bias the gate's final-layer logits toward the constant-`1` (`TRUE`) gate by a strength
`keep_bias`, so `s` defaults high (carousel on) while a write path (`s<1`) remains. This is
the logic-native form of the LSTM forget-gate bias [Gers et al., 2000] and of difflogic
residual initialization [Petersen et al., 2024]. **We find `keep_bias` must match the task**
(§5.2): high to hold (recall), low to absorb (integration).

### 3.4 Training recipe

Training recurrent LGNs surfaces both classic RNN pathologies; both must be handled:
- **Vanishing → keep-bias.** Without it, the gated cell cold-starts (flat loss at chance).
- **Exploding → LR decay + step-skipping.** keep-bias near-identity Jacobians can creep
  above 1 over many steps; gradients then overflow to NaN. We (i) cosine-decay the learning
  rate so the late, confident phase takes small steps, and (ii) **skip any optimizer step
  whose global grad-norm is non-finite**, which keeps the weights out of the NaN basin
  (post-hoc clipping cannot — clipping an `inf` grad yields `nan`). A dead-weights early-stop
  aborts a poisoned run.

---

## 4. Experimental setup

Built on the `difflogic` library (CUDA kernels). All runs: hidden 1000–1024, Adam, lr 0.003
cosine-decayed to 3e-4, 20k iterations, batch 128. **Discrete-locked evaluation**: gates
argmaxed and inputs binarized — we report the *real logic-circuit* accuracy (`test`), the
soft/relaxed accuracy (`soft`), and the gap. Hardware: RTX 2080S / dual-GPU server. `[TODO:
seeds — copy has 3, others 1]`. Tasks:

- **copy (synthetic recall).** Present a symbol (1-of-8) at `t=0`, then `L−1` blank steps;
  classify the symbol. Chance 12.5%. Pure long-range recall; `L` is the difficulty knob.
- **delayed-MNIST (real-data recall).** Encode an MNIST image in one step (784→hidden), then
  `D` blank steps, then classify the digit. Chance 10%. Isolates *holding* from encoding.
- **psMNIST (real-data classification).** Permuted MNIST fed `k` pixels/step (seq_len =
  784/k); an integration task. Chance 10%.

---

## 5. Results

![Length sweeps. Left: copy — the gated cell (blue) solves long-range recall while the
concat-recurrence control (red) sits at chance and the dual-state cells (lstm/gru_cell)
cold-start past L20. Right: psMNIST length sweep at a fixed (recall-tuned) keep-bias — see
§5.2 for why this keep-bias is wrong for classification.](../seqlgn/results/curves.png)

![Left (B): on psMNIST, lowering keep-bias rescues integration — gated accuracy and its soft
score rise as keep-bias drops, approaching the equal-gates control. Right (C): real-data
recall — the control collapses to chance at any delay while the gated cell holds ~3× chance
through 100 blank steps.](../seqlgn/results/curves_bc.png)

### 5.1 Gating enables long-range recall; concat-recurrence fails

**Copy.** The gated cell solves the task and degrades gracefully with length
(test 0.96 / 0.79 / 0.33 at L = 20 / 35 / 50, mean of 3 seeds), while the concat-recurrence
control never learns (≈0.25, with a soft score at chance — its gradients vanish). The
dual-state cells `lstm`/`gru_cell` reach ~0.75 at L=20 but **cold-start to chance at L≥35**:
their carousels are weaker at initialization (§5.4), so the simpler single-gate cell is best.

**Delayed-MNIST (the headline).** This is the cleanest evidence. With any delay, the control
**collapses to chance** — it cannot even learn the task softly (soft = 0.114, grad-norm at
`t=0` is 0, i.e. total vanishing). The gated cell **holds**:

| delay | gated (test) | control (test) |
|------:|:---:|:---:|
| 0 | 0.70 | 0.55 |
| 50 | **0.37** | 0.11 (chance) |
| 100 | **0.34** | 0.11 (chance) |

A control sitting at chance cannot be explained by gate count, capacity, or tuning — the
gated carousel is doing something the concat-recurrence architecture fundamentally cannot.

### 5.2 Keep-bias is task-dependent

The same keep-bias that *enables* recall *hurts* integration. On psMNIST (classification),
high keep-bias makes the cell over-hold (under-write), ignoring inputs it must absorb.
Sweeping keep-bias at fixed length (psMNIST, 28 steps):

| keep-bias | test | soft | gap |
|---:|:---:|:---:|:---:|
| 0 | **0.632** | **0.709** | 0.077 |
| 1 | 0.547 | 0.660 | 0.112 |
| 2 | 0.541 | 0.668 | 0.126 |
| 4 | 0.389 | 0.659 | 0.270 |

Lowering keep-bias raises soft accuracy (0.66→0.71) and shrinks the discretization gap
(0.27→0.08). **Takeaway:** keep-bias is a task knob — *high to hold, low to absorb*.

### 5.3 Gating does not help classification (honest boundary)

At equal gate count (4,000 gates: gated at hidden 1000, control at hidden 2000), on
psMNIST-28 the control matches or slightly beats the best gated cell:

| model | gates | test | soft |
|---|:---:|:---:|:---:|
| gated (keep-bias 0) | 4,000 | 0.632 | 0.709 |
| control (rddlgn) | 4,000 | **0.655** | 0.694 |

The gated cell's *soft* representation is marginally better, but its MUX discretization gap
(0.077 vs. 0.038) erases the advantage. **Gating gives no benefit on classification.** This
is consistent with §5.1–5.2: the carousel matters only when the task *requires* long-range
recall — which classification does not, since the control's vanishing gradients are not
fatal at these lengths.

### 5.4 The two RNN pathologies, in logic

Training only works with the recipe of §3.4. Without keep-bias, the gated cell cold-starts
(flat loss at chance). With keep-bias, long sequences instead *explode* to NaN; LR decay +
step-skipping fix it (0 skipped steps in the reported runs). The dual-state `lstm`/`gru_cell`
cold-start more readily because their separate input/forget paths weaken the
initialization carousel (`∂C'/∂C ≈ 0.58` vs. the GRU's `≈ s`), which we partly mitigate by
closing the input gate at init — but the single-gate GRU remains the most robust.

---

## 6. What we learned

1. **Gating buys long-range *recall*, not general performance.** The carousel is decisive
   exactly when a task needs to carry information across many steps with little intervening
   signal (copy, delayed-MNIST). When it does not (classification), the simpler
   concat-recurrence is as good.
2. **Keep-bias is the dial** that adapts the carousel to the task — a single, interpretable,
   task-dependent hyperparameter (high=hold, low=absorb).
3. **The MUX gate trades discretization-robustness for gradient flow.** Its convex-blend
   hidden state discretizes worse than plain logic-recompute on tasks it cannot fully solve
   — a real, mechanistic limitation.
4. **Simpler is better.** A single complementary gate (GRU) beats dual-gate cells
   (LSTM-style), which are harder to initialize and cold-start.
5. **Training recurrent LGNs requires handling both vanishing and exploding gradients**
   explicitly — neither is optional.

---

## 7. Limitations & Future work

- **Recall, not classification.** Our positive claim is scoped to recall; gating does not
  help integration tasks.
- **Discretization gap.** The MUX gap is unaddressed; Gumbel+STE [Yousefi et al., 2025] or
  state-binarization (STE on the hidden state) may close it — left to future work.
- **Sequence length.** Full 784-step psMNIST is beyond the cell's frontier (≈40h/run) and is
  not attempted; our chunked variants reach ~112 steps.
- **Dual-state cells on recall.** We did not exhaustively tune lstm/gru_cell on the recall
  tasks; copy already indicates the single-gate GRU dominates, but a focused study is
  possible.
- **Hardware.** A key motivation for LGNs is hardware efficiency; FPGA synthesis of the gated
  recurrent circuit (and *sequential* primitives such as latches) is the natural next paper.

---

## 8. Conclusion

We presented the first gated recurrent Logic Gate Network and showed that a logic-native
constant-error carousel enables long-range recall in a learned logic circuit — a regime
where the existing concat-recurrence approach goes to chance. The benefit is specific to
recall and governed by a task-dependent keep-bias, and training requires explicit handling
of both vanishing and exploding gradients. This establishes gating as a viable, cheap
building block for sequential logic-circuit models.

---

## Appendix / notes for revision
- `[TODO]` add seeds/error bars (copy has 3; add for delayed-MNIST headline points).
- `[TODO]` proper citations + bibtex; venue formatting.
- `[TODO]` figure polish: split the recall panel out as Fig. 1 (the money plot); the psMNIST
  length-sweep panel in `curves.png` uses the recall-tuned keep-bias and should be
  re-captioned or replaced by the keep-bias sweep.
- Data/figures regenerated by `mlgn/seqlgn/plot.py`; tables by `collate.py`. Full run log:
  `mlgn/research/04_experiment_log.md`.
