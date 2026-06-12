# Design — recurrent logic cells

How the cell is built, the math of each memory mechanism, and why the gated variant
should help. Code: [`../cells.py`](../cells.py), [`../models.py`](../models.py).

## 1. The substrate: difflogic in [0,1]

A `LogicLayer(in_dim, out_dim)` holds a learnable categorical distribution over the 16
two-input Boolean gates per output neuron (`weights` of shape `[out_dim, 16]`). Each
output neuron reads 2 inputs (random fixed wiring) and outputs:

- **train** (`.train()`): a softmax mixture of the 16 probabilistic gate surrogates
  (e.g. `AND→a·b`, `OR→a+b−ab`, `XOR→a+b−2ab`), all valued in `[0,1]`.
- **eval** (`.eval()`): the single argmax gate applied hard. Given binary inputs the
  output is binary.

`LogicMLP` just stacks `cell_layers` of these (`in→out`, then `out→out`). We use it as the
building block for the candidate / gate / update networks.

## 2. The recurrence

A `SequenceClassifier` initialises `h_0 = 0` (a valid binary state) and unrolls:

```
for t in range(T):
    h = cell(x_t, h)          # x_t, h are [batch, dim]
logits = GroupSum(h)          # partition h into num_classes groups, sum, / tau
```

Everything is processed one timestep at a time because `LogicLayer` wants 2D
`[batch, features]` inputs.

## 3. The memory mechanisms

Let `z = [x_t ; h]` (concatenation, dim = `input_dim + hidden_dim`).

### `rddlgn` — the control
```
h' = update(z)                # update = LogicMLP(input_dim+hidden_dim → hidden_dim)
```
State is **recomputed from scratch every step**. This is the recurrence used by Recurrent
DDLGN (Bührer et al., arXiv:2508.06097) and DiffLogic CA (Miotti et al.,
arXiv:2506.04912). There is no dedicated "keep" path, so to remember a bit the network
must *relearn to copy it* through fresh logic at every timestep — and the gradient that
would teach it must survive backprop through all those steps.

### `gated` — Paper #1
```
c = candidate(z)              # the "write" value, in [0,1]
s = gate(z)                   # the per-bit select / keep probability, in [0,1]
h' = s * h + (1 - s) * c      # soft 2:1 multiplexer  (== GRU update gate)
```
`candidate` and `gate` are separate `LogicMLP`s (mirroring GRU's separate weight matrices
for the candidate and the update gate).

**Why `s*h + (1-s)*c` is logic-native.** At binary values `s,h,c ∈ {0,1}` this is exactly
the Boolean multiplexer
```
MUX(s,h,c) = (s AND h) OR (NOT s AND c)
```
because the two product terms are mutually exclusive (`s` and `1−s` can't both be 1), so
the convex combination equals the OR of two ANDs. In hardware that's **3 gates per bit**
(plus the `gate`/`candidate` logic). The soft form is the standard, clean relaxation and
coincides with the GRU update gate `h = z⊙h + (1−z)⊙ĥ`.

**Why it should help — the constant-error carousel.** Differentiate the kept branch:
```
∂h'/∂h = s          (per element)
```
When the gate keeps a bit (`s ≈ 1`), the Jacobian of the recurrence is ≈ identity, so
`∂L/∂h_t ≈ ∂L/∂h_{t+1}` — the gradient passes backward through time **un-attenuated**.
This is the LSTM/GRU trick that beats vanilla RNNs, realised in pure logic. The `rddlgn`
control has no such path: `∂h'/∂h = J_update`, a product of softmax-gate Jacobians whose
norm shrinks with depth/time → vanishing gradients on long sequences.

**Keep-bias init (important — avoids cold-start).** The carousel only helps if the gate
*keeps* (`s≈1`) early on. With a random gate (`s≈0.5`) on a long sequence, state decays
before the network can learn to keep it, and there's no gradient signal to bootstrap — a
**cold start** (empirically: copy-50 sat at chance with flat loss; see
[experiments.md](experiments.md)). Fix: ``bias_gate_keep`` adds `keep_bias` to the
TRUE-gate logit of the gate's final layer, so `s` defaults high (carousel ON) while a
write path (`s<1`) remains. This is the logic-native form of the **LSTM forget-gate bias**
(Gers et al. 2000) and difflogic **residual init** (Petersen 2024). CLI `--keep-bias`
(default 3.0; `0` reproduces the cold-start). For `lstm` the same bias is applied to the
forget gate `f`.

> Early evidence (untrained, tiny, seq=8): the grad-norm-through-time instrument
> (`utils.grad_norm_through_time`) already shows the `rddlgn` control's gradient at the
> earliest timestep ~12 orders of magnitude below the latest. To be confirmed at scale,
> and contrasted with `gated`, during real training. See [experiments.md](experiments.md).

### `lstm` — Paper #1, richer arm
A logic-native LSTM with a **dedicated cell state `C`** carried across time (separate from
the hidden/output `h`), and **independent forget / input / output** stages:
```
z   = [x_t ; h]
f   = forget(z)            # keep-mask          in [0,1]^H
i   = input(z)             # write-enable       in [0,1]^H
C̃   = candidate(z)         # value to write     in [0,1]^H
C'  = (C AND f) OR (i AND C̃)        # soft: OR(C·f, i·C̃) = a+b−ab
o   = out_proj(z)          # project z to hidden_dim (output-gate-like)
h'  = readout([o ; C'])    # 2H -> H
state' = (h', C')
```

**Why OR (not multiply, not add).** Textbook LSTM is additive:
`C_t = f⊙C_{t-1} + i⊙C̃`. In a logic net you can't add bits (`1+1 ∉ {0,1}`), so we use
**OR as the stand-in for that addition** (`_or(a,b)=a+b−ab`). This is the crux:
- *multiply* the two terms → when you're not writing (`i·C̃≈0`) the cell collapses to 0
  and `∂C'/∂C = f·i·C̃ → 0`: memory and gradient both destroyed.
- *add / OR* the two terms → not-writing leaves `C' = C·f` and `∂C'/∂C = f`: the cell
  persists and the gradient flows. OR also saturates gracefully if both terms fire
  (`1 OR 1 = 1`) instead of overflowing to 2.

**The carousel is in `C`, not `h`.** `∂C'/∂C = f·(1 − i·C̃)`, which is `f` on bits you
keep and don't overwrite (and `=1` when `f=1`). So `C` is the gradient highway; `h` is a
readout of it. The grad-norm-through-time analysis therefore tracks **`C`** for lstm
(`cell.carousel_state` returns `C`), and `h` for the other mechanisms.

**Why project `z→o` before the readout.** The readout wants to see both `z` and `C'`, but
feeding `[z ; C']` (width `input_dim + 2H`) into a `→H` layer violates difflogic's
`out*2 ≥ in`. Projecting `z` to `o` (width `H`) first makes the readout `[o ; C'] (2H) → H`,
where `2H ≥ 2H` holds exactly. (Equivalent in spirit to LSTM's `h = o ⊙ tanh(C)`, but a
full LGN over `[o;C']` instead of a fixed AND — more expressive.)

**Cost.** 5 LGNs (forget, input, candidate, out_proj, readout) and **two** carried states
(`h`, `C`) — vs the gated cell's 2 LGNs / 1 state. So `lstm` is the "does the extra
forget/input/output machinery earn its ~2.5× gates?" ablation; `gated` stays the primary.

### `gru_cell` — the 2×2-completing ablation
A **dedicated cell state `C` (like LSTM) updated by the GRU's single complementary MUX
gate (like `gated`)**, with a separate output readout:
```
s = gate(z);  C̃ = candidate(z)
C' = s·C + (1−s)·C̃          # GRU MUX on the cell state; ∂C'/∂C = s (clean carousel, no leak)
o  = out_proj(z)
h' = readout([o ; C'])       # h is a learned readout of C
state = (h', C')
```
This isolates **"separate cell state"** from **"gate structure"** — the two things `gated`
and `lstm` change at once:

| | single MUX gate | two independent gates |
|---|---|---|
| **single state** (h) | `gated` (GRU) | — |
| **separate state** (h, C) | **`gru_cell`** | `lstm` |

Because the gate is a single complementary MUX, the carousel `∂C'/∂C = s` is clean (no
input-path leak) → it should be **robust to train**, unlike `lstm`. 4 LGNs (candidate,
gate, out_proj, readout) = 8,192 gates at hidden 1024 (2× `gated`, < `lstm`). Read it as
`gru_cell` vs `gated`: if ≈, the separate cell buys nothing (GRU wins); if it extends the
long-sequence frontier, decoupling memory from output helps. keep-bias goes on the gate.

## 4. Discreteness across time

For the discretised circuit to be exact, the hidden state must stay binary at eval. It
does: with binary `x_t` (inputs `.round()`) and hard argmax gates, `candidate` and `gate`
output `{0,1}`, and the MUX of binary values is binary. Inductively `h` stays in `{0,1}`
from `h_0 = 0`. The `rddlgn` update is likewise binary in→binary out. So `model.eval()` +
`.round()` measures the true logic-circuit accuracy, not the soft relaxation.

## 5. Capacity fairness

The comparison must isolate the *mechanism*, not capacity. Note `gated` has **two**
`LogicMLP`s (candidate + gate) vs `rddlgn`'s one, so at equal `--hidden`/`--cell-layers`
the gated cell has ~2× the gates (see the smoke test: 80 vs 40). When reporting, control
for this — either match total gate count (e.g. give `rddlgn` more width/layers) or report
accuracy-vs-gates curves. Documented as a protocol point in [experiments.md](experiments.md).

## 6. Paper #2 — `latch` (PARKED)

`mechanism='latch'` is reserved and currently raises `NotImplementedError`. The plan
(see [../../research/05_my_angles.md](../../research/05_my_angles.md) and
[../../research/06_paper_plan.md](../../research/06_paper_plan.md)):

- Add a **stateful primitive** that *holds* a bit rather than recomputing it:
  - **D flip-flop** (on-ramp): `h' = d` where `d` is a written value — a 1-step delay.
    Trivially differentiable (gradient 1) — the constant-error carousel *as a primitive*.
  - **gated D-latch / SR latch** (the real content): cross-coupled feedback with a
    **custom straight-through gradient** through the bistable element.
- This turns the unrolled network from a *combinational* circuit into a **true clocked
  sequential circuit** (registers + logic) → maps to FPGA/ASIC flip-flops.
- The 4-way comparison (rddlgn / gated / latch / combo) is Paper #2's core. The interface
  here is built so adding `latch` is a localised change in `cells.py`.
