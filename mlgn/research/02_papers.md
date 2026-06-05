# Annotated Bibliography — LGN & adjacent

Format per entry: **citation** · arXiv/venue · one-line TL;DR, then method / numbers /
limitations / why-it-matters. Tier = how central to the difflogic line.

Legend: ⭐ = must-read core · 🔧 = method improvement · 🔁 = sequential/recurrent ·
🔌 = connectivity · 🧠 = interpretable/tabular · 🧮 = LUT/weightless cousin ·
🛠 = application/hardware.

---

## ⭐ CORE — Petersen lineage

### [1] Deep Differentiable Logic Gate Networks ⭐ (FOUNDATION)
Petersen, Borgelt, Kuehne, Deussen — **NeurIPS 2022** — arXiv:2210.08277 ·
repo: `Felix-Petersen/difflogic` (this fork's base).

- **Idea:** learn a network of 2-input logic gates by gradient descent via a
  differentiable relaxation. Each neuron = softmax over 16 gates; inputs relaxed to
  [0,1]; probabilistic gate surrogates; `GroupSum` output head with temperature τ.
- **Connectivity:** random, fixed at init (NOT learned). Nets are very wide & sparse.
- **Results:** MNIST ~98% (and up to ~99% with very wide nets); CIFAR-10 ~57–62% with
  the largest models. >1M MNIST images/s on 1 CPU core. Inference via `PackBitsTensor`
  (GPU) or `CompiledLogicNet` (compiles to C/`.so`).
- **Limits:** slow training, vanishing gradients in deep nets (mitigated by
  `grad_factor`), discretization gap, modest accuracy ceiling, fixed wiring.
- **Why it matters:** defines the whole paradigm + the API I'm building on.

### [2] Convolutional Differentiable Logic Gate Networks ⭐ (SOTA scaling)
Petersen et al. — **NeurIPS 2024 (Oral)** — arXiv:2411.04732 · "LogicTreeNet".

- **3 contributions:**
  1. **Logic-gate tree convolutions** — conv "kernel" = a binary **tree of learnable
     gates** (depth d → 2^d−1 gates over 2^d inputs from a receptive field), weight-
     shared across spatial positions like a CNN.
  2. **OR-pooling** — pooling via logical OR (max t-conorm); cheap fwd/bwd, only
     backprop through the max activation.
  3. **Residual initialization** — init the gate logits to favor pass-through gate "A"
     (~90%), giving differentiable residual connections that fix gradient decay in
     deep nets.
- **Numbers:** CIFAR-10 **86.29%** with **61M gates** (LogicTreeNet-G); 84.99%/28.9M;
  80.17%/16M. **29×–61× fewer gates** than XNOR-Net/FINN/LUTNet at matched accuracy.
  Beats original DLGN: 71.0%/3.08M vs 62.1%/5.12M.
- **Hardware:** FPGA **9 ns/image** (-M), 24 ns (-B), ~41.6M FPS; I/O-bound not
  compute-bound. Fused CUDA kernels → up to 200× faster per gate; -L trains ~30 s/epoch
  on one RTX 4090 (vs 90 h on A6000 for the old 5M-gate net).
- **Limits / future:** connectivity in trees still random/fixed; suggests vision tasks
  with continuous decisions (object localization); custom ASIC.
- **Why it matters:** the architecture template if I go convolutional; the SOTA row.

---

## 🔧 ETH Zürich (Wattenhofer lab) — training/discretization fixes

### [3] Mind the Gap: Removing the Discretization Gap in DLGNs 🔧
Yousefi, Plesner, Aczel, Wattenhofer — **NeurIPS 2025** — arXiv:2506.07500 ·
"Gumbel LGN (GLGN)".

- **Problem:** train (soft) vs inference (hard) gap ≈3% acc; ~half the neurons unused;
  sharp loss landscape.
- **Method:** **Gumbel-noise** on gate logits + **straight-through estimator** — hard
  noisy-argmax gate in the forward pass, Gumbel-Softmax gradient in the backward pass,
  so training matches inference. Proves (Lemma 1) it adds an **implicit Hessian-trace
  regularizer** ∝ π²/(12τ²) → flatter minima.
- **Numbers:** **98% smaller** discretization gap; **4.5×** faster wall-clock; unused
  gates **49.8% → 0%**; ~5% per-iter overhead. Best τ ≈ 0.15–0.5. CIFAR-10/100, 12
  layers, 256k–2048k wide, 48 GPU-h budget.
- **Limits/future:** mostly CIFAR; τ needs tuning (adaptive schedule = open); width×depth
  interplay under-studied; fuller theory open.
- **Why it matters:** cheapest high-leverage add-on; I should adopt Gumbel+STE early.

### [4] Light Differentiable Logic Gate Networks 🔧
Rüttgers, Aczel, Plesner, Wattenhofer — 2025 (under review) — arXiv:2510.03250 ·
"Input-Wise Parametrization (IWP)".

- **Problem:** softmax-over-16 (original parametrization, OP) is **redundant** —
  negation-symmetric gate pairs (Gᵢ vs ¬Gᵢ) create **self-cancelling gradients**, and
  argmax can pick a gate the neuron doesn't actually behave like (discretization error).
- **Method:** decompose any 2-input Boolean fn in the **4-term basis** {E₀₀,E₀₁,E₁₀,E₁₁}
  with 4 learnable weights ωᵢⱼ∈[0,1]:
  `g(p,q) = (1−p)(1−q)ω₀₀ + (1−p)q·ω₀₁ + p(1−q)ω₁₀ + pq·ω₁₁`.
  → **2ⁿ params/gate instead of 2^(2ⁿ)** (4× fewer for 2-input). Gradients no longer
  cancel. **Residual init is essential** (heavy-tail, negation-asymmetric, biased to
  pass-through "A").
- **Numbers:** **4× less memory**, backward up to **1.86×** faster, **8.5× fewer steps**
  to reach OP's best acc; on CIFAR-100 IWP holds accuracy as depth scales where OP
  plateaus (~28% discretized) — CDLGN: 1.3× better test acc at 5× depth.
- **Big future hook:** because params are now 2ⁿ not 2^(2ⁿ), **n-input gates (e.g. 6-LUT)
  finally tractable** — flagged but not done. Also: encoding-aware / learned connections;
  preprocessing/thermometer-encoding info loss is the suspected accuracy ceiling.
- **Why it matters:** the cleanest reparametrization; the same 4-basis projection shows
  up in LILogic for speed. Likely a default going forward.

### [5] Recurrent Deep Differentiable Logic Gate Networks 🔁 (closest prior art to me)
Bührer, Plesner, Aczel, Wattenhofer — 2025, Edge & Mobile Foundation Models workshop
(MobiCom'25) — arXiv:2508.06097 · "RDDLGN".

- **Idea:** first **recurrent** LGN, for **seq2seq** (neural machine translation).
- **Method:** encoder = N-layers (feature) + **K-layers that carry a hidden state across
  16 timesteps** by concatenating current features with previous K-output
  `[h_t^(DN); k_{t-1}^(DK)]`; decoder = L/P/M layers with autoregressive P-recurrence.
  Embeddings sigmoid-relaxed to [0,1]; same softmax-over-16 gates; train then binarize
  + discretize ("collapse").
- **Numbers (WMT'14 En–De, seqs truncated to 16 tokens):** BLEU — Transformer 5.98 >
  GRU 5.41 > **RDDLGN 5.00 (uncollapsed) / 4.39 (collapsed)** > RNN 4.59. Beats RNN,
  trails GRU/Transformer. Collapse costs ~8.7% rel. BLEU. **Memorization task: RDDLGN
  dominates** (97%+ at shift ≤4; shift-12: 64.6% vs RNN 2.1%, GRU 28.1%). Gradients
  stable (Std/Mean ≈7.9 across groups).
- **Limits/future (read these closely):** needs **much larger embedding params**
  (16.4M vs 4.1M) eroding the efficiency win; **vanishing gradients for longer
  sequences / deeper stacks**; only 16-token seqs, small vocab. Future: weight
  reparametrization for gradients; **associative/parallel-scan recurrent blocks** (O(log n)
  training); FPGA synthesis; RBM-style binary variants.
- **Why it matters:** this is the paper my `secuential.py` direction overlaps with.
  They did **translation**; the **sequential-image / time-series classification** and
  **long-range** regimes are far less explored. See [03_open_problems.md](03_open_problems.md).

### [5b] Differentiable Logic Cellular Automata (DiffLogic CA) 🔁 (the other stateful LGN)
Miotti, Niklasson, Randazzo, Mordvintsev — **Google**, ALIFE 2025 — arXiv:2506.04912 ·
[project page](https://google-research.github.io/self-organising-systems/difflogic-ca/).

- **Idea:** Neural Cellular Automata where each cell's update rule is a **Petersen-style
  difflogic circuit**. Fully differentiable in training, fully discrete at inference.
- **State/recurrence:** cell state = **binary vector** ("working memory"); each step
  **re-applies the same combinational circuit** to `concat(prev_state, perception)` →
  new state. **No gating, no latch/flip-flop primitives** — pure combinational
  recurrence.
- **Tasks:** Conway's Game of Life (all 512 nbhds), checkerboard (noise/damage-resilient,
  self-healing), lizard growth (20×20→40×40), 8-color letter "G".
- **Limits/future (read closely — overlaps my angles):** hard to train (heavy HP tuning),
  limited expressiveness; **explicitly names as future work** "specialized gates designed
  to facilitate **state forgetting**" and "integrating **LSTM-like gating mechanisms**
  into the state update" — i.e. my [05_my_angles.md](05_my_angles.md) #1.
- **Why it matters:** alongside RDDLGN [5], this is the **second group doing stateful/
  recurrent difflogic** (Google). The frontier is small but heating up. It establishes
  the prior art my angles must differentiate from (it has neither gating nor latch
  primitives — both still open).

---

## 🔌 Connectivity learning (pain point #3)

### [6] A Method for Optimizing Connections in DLGNs 🔌
Mommen, Keuninckx, Hartmann, Wambacq — 2025 — arXiv:2507.06173.

- **Idea:** make the **wiring** learnable (differentiable connection selection) instead
  of fixed-random, optimized jointly with the gate choice.
- **Results:** improved acc/efficiency vs fixed-random on **MNIST / Fashion-MNIST**.
- **Limits:** scalability to large/deep nets; efficiency of the connection learning.

### [7] LILogic Net: Compact LGNs with Learnable Connectivity 🔌
Fojcik, Zioma, Armaitis — Nov 2025 — arXiv:2511.12340.

- **Method:** **Top-K differentiable connectivity** — each gate softmax-selects its
  inputs from K preselected candidates (K∈{2…128}); continuous in training, mode at
  inference → discrete circuit. An N-layer Top-K net reaches up to **Kᴺ input paths**
  (sparse connectivity ≈ compositional depth). Gate eval via the **4-basis projection
  {1,A,B,A·B}** (3.4–4.0× training speedup) — same trick as Light DLGN.
- **Numbers (very gate-efficient):** MNIST 98.95%/**32k gates** (vs DiffLogic-S
  97.69%/48k; LogicTreeNet-M 99.23%/566k). CIFAR-10 60.98%/256k gates, beating
  LogicTreeNet-S (60.38%/400k) at 1.6× fewer gates. Claims ≥2 orders of magnitude fewer
  ops than comparable nets. τ must scale with layer width.
- **Limits/future:** capped at 256k gates (kernel limits); regularization (dropout/weight
  decay/norm) untapped.
- **Why it matters:** strongest evidence that **learned connectivity >> random** for
  gate efficiency; the K^N "paths = depth" framing is useful intuition.

---

## 🧠 Princeton (Yue & Jha) — interpretable, tabular & time-series

### [8] Learning Interpretable Differentiable Logic Networks 🧠
Yue & Jha — TCAD 2024 (arXiv 2024) · "DLN".

- Reworks the LGN framework for **general tabular classification**: removes fixed
  topology and binary-only inputs; emphasis on **human-readable logic** + cheap
  inference. Two-stage training.

### [9] DLN for Tabular Regression 🧠
Yue & Jha — 2025 — arXiv:2505.23615.

- Redesigns the output **SumLayer** for continuous targets; **single-stage** training
  (two-stage was suboptimal for regression). Matches/beats NN & classical baselines on
  15 regression benchmarks while staying interpretable + fast.

### [10] DLN for Time-Series Classification 🔁🧠
Yue & Jha — 2025 — arXiv:2508.17512.

- Extends interpretable DLNs to **time-series classification** — relevant to my
  sequential interest, but via a (non-recurrent) DLN framing rather than a recurrent
  cell. Worth reading to contrast with RDDLGN's recurrence.

---

## 🧮 Adjacent: weightless / LUT networks (same goal, different unit)

### [11] Differentiable Weightless Neural Networks (DWN) 🧮
Bacellar, Susskind, Breternitz, John, Lima, França et al. — **ICML 2024** —
arXiv:2410.11112 · repo `alanbacellar/DWN`.

- **Unit:** chained **lookup tables (LUT-3s)**, not 2-input gates. Inputs via unary
  **thermometer** encoding → tuples → address the first LUT layer.
- **Tricks:** **Extended Finite Difference** (approx. gradient through binary LUT
  entries), **Learnable Mapping** (which inputs address which LUT), **Learnable
  Reduction**, **Spectral Regularization**.
- **Results:** beats SOTA FPGA accelerators on latency/throughput/energy/area; beats
  XGBoost on MCUs under tight memory; strong on tabular. The main LUT-side competitor
  to difflogic.
- **Contrast:** LUTs are more expressive per unit than a single 2-gate but cost more
  silicon; gates (difflogic) are the finest granularity.

### [12] Differentiable Weightless Controllers (DWC) 🧮🛠
2025 — arXiv:2512.01467.

- DWN idea for **RL / continuous control** (MuJoCo: Humanoid, HalfCheetah). Policies =
  sparsely-connected Boolean LUT layers + light action heads; trains by gradient,
  compiles to FPGA with **single-clock-cycle, nJ-per-action** inference. Matches
  weight-based nets on 5 benchmarks.
- **Why note it:** shows the paradigm reaching **control/RL** — an under-explored
  application surface for plain LGNs too.

---

## 🛠 Adjacent: FPGA LUT-NN hardware lineage (baselines/competitors)

These map *whole neurons* into FPGA LUTs (quantize-then-map), predating/parallel to
difflogic. Mostly relevant as **hardware-efficiency baselines**.

- **LogicNets** (Umuroglu et al., AMD/Xilinx) — neuron → single LUT, sparse fixed.
- **PolyLUT** — piecewise-polynomial neuron functions (fewer regions).
- **NeuraLUT** — composition of piecewise-linear functions per LUT.
- **NullaNet** — early neuron→logic mapping.
- **SparseLUT** (arXiv:2503.12829) — sparse-connectivity optimization for LUT-NNs.
- **ReducedLUT** (arXiv:2412.18579) — table decomposition w/ "don't-care" conditions.
- **NeuraLUT-Assemble** (arXiv:2504.00592) — assemble sub-NNs into LUTs.
- **AmigoLUT** (FPGA'25) — scaling LUT-NNs via ensembling.
- **Enhancing LUT-based DNN Inference** (arXiv:2601.09773, 2026) — arch + connectivity
  optimization.

## 🛠 Applications & tooling

- **eXpLogic** — interpretability for LGNs: saliency maps + gate pruning with minimal
  accuracy loss.
- **Rapid Inference of LGNs for Anomaly Detection in HEP** (arXiv:2511.01908, 2025) —
  convolutional LGN for the CMS Level-1 trigger (CICADA); matches/beats quantized NNs,
  strong FPGA characteristics → real on-detector deployment case.

## Separate root

- **TTnet** — "A Scalable, Interpretable, Verifiable & Differentiable Logic Gate CNN
  from Truth Tables" (arXiv:2208.08609, 2022). Independent truth-table lineage; context
  only.

---

### Reading priority for my recurrent/sequential angle
1. [5] Recurrent DDLGN (direct overlap — read in full, esp. limits/future).
2. [1] Deep DLGN + [2] Conv DLGN (the substrate I build on).
3. [3] Mind the Gap + [4] Light DLGN (training fixes I should bake in).
4. [10] DLN time-series + [7] LILogic (sequential framing & learned connectivity).
