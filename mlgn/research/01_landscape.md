# The Logic Gate Network Landscape

_Scout date: 2026-06-04. Maintained as I read more._

## 1. What an LGN is (method primer)

A **Logic Gate Network (LGN)** is a feed-forward network whose "neurons" are
**2-input Boolean logic gates** instead of weighted sums + nonlinearity. Each neuron:

1. takes **2 inputs** wired from the previous layer (in the original work, the wiring
   is **random and fixed** at init — not learned),
2. applies **one of the 16 possible 2-input Boolean functions** (FALSE, AND, XOR, …,
   NAND, …, TRUE),
3. outputs a single bit.

Inference is therefore just a circuit of gates — no multiplications, no floats. That
is why it is so fast/cheap on CPU/FPGA/ASIC.

### The differentiable relaxation (the key trick)

Gates are discrete, so you can't backprop through "which gate." Petersen et al. relax it:

- Each neuron holds a **learnable categorical distribution over the 16 gates**
  (softmax over 16 logits).
- Inputs are relaxed from `{0,1}` to **`[0,1]` (probabilities)**.
- Each gate gets a **real-valued (probabilistic) surrogate**, e.g.
  `AND(a,b) → a·b`, `OR(a,b) → a+b−a·b`, `XOR(a,b) → a+b−2ab`, `NOT a → 1−a`.
- A neuron's soft output is the **softmax-weighted mixture** of all 16 surrogates.
- Train with normal autodiff + Adam (recommended lr 0.01).
- At inference: **argmax** to pick one hard gate per neuron, binarize inputs → a real
  Boolean circuit.

Output head: **`GroupSum`** — partition the last layer's output bits into `k` groups
(one per class), sum each group, divide by temperature `τ`, feed to softmax/CE loss.

### The three structural pain points (the field's whole research agenda)

1. **Discretization gap** — the relaxed (train) net and the hard (inference) net
   differ; accuracy drops on the hard net. Half the neurons can end up unused.
2. **Training cost & vanishing gradients** — softmax-over-16 has redundant /
   self-cancelling parametrization; deep nets train slowly (days–weeks for CIFAR-10).
3. **Fixed random connectivity** — the wiring is not learned, so you "waste" gates and
   can't exploit input structure.

Almost every follow-up paper attacks one of these three.

## 2. The research lineages (who is doing what)

### A. Founders — Felix Petersen et al.
- **Deep DLGN** (NeurIPS 2022) — the founding method.
- **Convolutional DLGN / LogicTreeNet** (NeurIPS 2024 **Oral**) — gate-tree
  convolutions + OR-pooling + residual init; the big scaling/SOTA jump.

### B. ETH Zürich — Wattenhofer lab (the most active extenders, 2025)
Same author cluster (Plesner, Aczel, Wattenhofer + others) across three papers, each
hitting one pain point:
- **Mind the Gap** (NeurIPS 2025) → discretization gap (Gumbel + STE).
- **Light DLGN** (2025, under review) → training cost / vanishing gradients
  (input-wise reparametrization).
- **Recurrent DDLGN** (2025, edge/mobile workshop) → **sequence modeling** (first
  recurrent LGN). ← closest prior art to my direction.

### C. Connectivity-learning thread (pain point #3)
- **Mommen et al.** "Optimizing Connections in DLGNs" (2025).
- **LILogic Net** (2025) — Top-K differentiable connectivity, very gate-efficient.
- (Yue & Jha also remove the fixed-topology constraint, below.)

### D. Princeton — Yue & Jha (interpretable, tabular/time-series)
- **DLN** for tabular classification (TCAD 2024), **regression** (2025), and
  **time-series classification** (2025). Drops fixed topology + binary-only inputs;
  emphasis on **interpretability** and cheap inference rather than raw accuracy.

### E. Adjacent: Weightless / LUT-based networks (same goal, LUTs not gates)
- **DWN — Differentiable Weightless Neural Networks** (ICML 2024, Bacellar et al., UT
  Austin/França) — chained **lookup tables (LUT-3s)** trained via Extended Finite
  Difference. Strong on FPGA / microcontrollers / tabular.
- **DWC — Differentiable Weightless Controllers** (2025) — DWN idea for **RL / continuous
  control** (MuJoCo), single-clock-cycle FPGA policies.

### F. Adjacent: FPGA LUT-NN hardware lineage (HW community, pre-dates difflogic)
**LogicNets → PolyLUT → NeuraLUT → NullaNet**, plus optimizers **SparseLUT**,
**ReducedLUT**, **NeuraLUT-Assemble**, **AmigoLUT**. These map *whole neurons* into
LUTs for FPGA; conceptually cousins, different mechanism (quantize-then-map vs.
learn-the-logic). Useful baselines/competitors in the hardware-efficiency tables.

### G. Separate root: Truth-Table Net (TTnet)
"A Scalable, Interpretable, Verifiable & Differentiable Logic Gate CNN from Truth
Tables" (2022) — independent truth-table lineage; cited for context.

## 3. Timeline

```
2022-08  TTnet (truth-table CNN)                              [separate root]
2022-10  Deep DLGN (Petersen, NeurIPS'22)                     ← FOUNDATION
2024-06  DWN (Bacellar, ICML'24)                              [LUT cousin]
2024-09  Yue & Jha DLN (tabular classification, TCAD)
2024-11  Convolutional DLGN / LogicTreeNet (NeurIPS'24 Oral)  ← SOTA scaling
2025-05  DLN for tabular regression (Yue & Jha)
2025-06  Mind the Gap (ETH, NeurIPS'25)                       ← discretization gap
2025-07  Optimizing Connections in DLGNs (Mommen)
2025-08  Recurrent DDLGN (ETH)                                ← FIRST recurrent LGN
2025-08  DLN for time-series classification (Yue & Jha)
2025-10  Light DLGN (ETH, IWP reparametrization)              ← training cost
2025-11  LILogic Net (Top-K learnable connectivity)
2025-11  LGN anomaly detection in High-Energy Physics (CMS)   [application]
2025-12  DWC — weightless controllers (RL/control)
2026-01  Enhancing LUT-based DNN inference                    [LUT cousin]
```

## 4. Hardware framing (why anyone cares)

The selling point is **inference efficiency**, not accuracy-per-parameter:
- Original DLGN: >1M MNIST images/s on a single CPU core.
- LogicTreeNet on FPGA: **4–24 ns/image**, ~41.6M FPS on CIFAR-10; bottleneck is I/O,
  not compute.
- Gates map 1:1 to hardware (CPU bit-ops, FPGA LUTs, ASIC standard cells) → no
  multipliers, tiny energy, and the circuit is **formally inspectable/verifiable**.

The trade you accept: **expensive, slow training** and **lower accuracy ceiling** than
dense nets, in exchange for nearly free, inspectable inference.

## 5. Quick mental model vs. neighbors

| Approach | Learned unit | Inference primitive | Connectivity | Training |
|---|---|---|---|---|
| Dense NN | weighted sum + ReLU | MACs (float/int) | dense, learned | easy |
| BNN/XNOR-Net | sign(weighted sum) | XNOR + popcount | dense, learned | medium |
| **LGN (difflogic)** | 1-of-16 Boolean gate | logic gate | **random, fixed** | hard/slow |
| LUT-NN (LogicNets…) | quantized neuron→LUT | LUT read | sparse, fixed | medium |
| DWN (weightless) | LUT-3 entries | LUT read | sparse, (learnable map) | medium |

See [results_table.md](results_table.md) for the head-to-head numbers and
[02_papers.md](02_papers.md) for per-paper detail.
