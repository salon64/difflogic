# Clock the Enable, Not the Value: Deploy-Consistent Memory for Recurrent Logic Gate Networks

**Draft v0.1 (2026-07-09) — Paper #2 (Track B, per `research/11_paper2_workmap.md` §A0').**
Author: Malcolm Malle `[affiliation]`. Solo paper.
Primary venue target: **ICML 2027** (~Jan/Feb '27 deadline); slip valve NeurIPS 2027; arXiv preprint as soon as internally reviewed (the timestamp is the race defense).
Citations are pandoc keys into [`references.bib`](references.bib) (`[@key]`). `[TODO]`/`[FIG]`/`[VERIFY]` mark gaps.
Code & results: `mlgn/seqlgn/` (191 result JSONs; every number in this draft traces to a JSON — see the Evidence Map at the end).

---

## Abstract

Logic Gate Networks (LGNs) train a relaxed circuit of two-input Boolean gates by gradient descent and deploy the argmax-discretized circuit, which runs as raw logic at nanosecond scale on CPU/FPGA. Extending LGNs to sequences promises a *clocked* circuit — registers plus combinational logic — but existing recurrent LGNs suffer a discretization gap that *grows with sequence length*: the relaxed training trajectory and the deployed discrete circuit diverge as continuous state drifts over timesteps. We study how to make recurrent-LGN training *deploy-consistent*, and report four findings. **(1)** The obvious architectural fix — hard-rounding the recurrent state during training so it stays binary ("bistable restore") — fails at scale: on a 50-step copy task every state-rounded variant collapses to chance via a **never-write collapse**, which we diagnose at the gate level (write networks collapse to constant-FALSE/TRUE gates; an uncertain write is rounded into a corrupting write, so "never write" becomes the loss attractor). **(2)** The rounding was never needed: we show the argmax-deployed gated recurrent LGN *already* carries an exactly binary state, so the residual gap is a *training-time* trajectory divergence — the computation gap of Kim [-@kim2026align] — which our data isolates by its length-dependence (0 at 20 steps, +0.50 at 50). **(3)** The gap at 50 steps is closed not by any primitive but by **deep per-timestep supervision**, which repairs 50-step credit assignment and dissolves the never-write plateau: both the soft-gated cell and our register cell reach 100% *discrete* accuracy on copy-50 (a length curriculum closes it independently; an activation-margin loss alone destabilizes training and is dominated). **(4)** We introduce the **input-clocked latch** (`clatch`): apply the straight-through round to the write-*enable*, never to the value — a learnable write-enabled register whose training forward pass already has deployment semantics on the hold path. It is **accuracy-competitive but never superior** to the soft gate across five task families (we report this symmetric outcome in full), and it carries two verified advantages: it never triggered a non-finite training step across our 75-run register-family sweep, in which the soft-gated cell is the only mechanism that ever destabilizes; and at matched configuration on chunked psMNIST its discretization gap is bounded and non-overlapping with the gated cell's (mean +0.02 vs. +0.09 across seeds at indistinguishable accuracy) — a *by-construction* consequence of clocking the enable, which is precisely the design point. The result is a training recipe (supervise every step; round the control, never the state) and a register primitive that maps one-to-one to flip-flops with clock enables, moving learned circuits one step closer to trustworthy deployment on sequential hardware.

---

## 1. Introduction

Differentiable Logic Gate Networks learn, for each neuron, which of the 16 two-input Boolean functions to apply, by relaxing the choice to a softmax over probabilistic gate surrogates; at inference the choice is argmaxed and the network *is* a discrete circuit [@petersen2022deep]. The payoff is extreme inference efficiency — millions of images per second on one CPU core, single-digit-nanosecond FPGA latency for convolutional variants [@petersen2024conv] — because the deployed object contains no arithmetic at all.

For sequential problems the natural deployed object is a **clocked sequential circuit**: a register file holding the state, combinational logic computing the next state, one timestep per clock tick. Recurrent LGNs exist [@buehrer2025rddlgn; @miotti2025dlca; @damera2026rdtlgn], but all carry state by *concat-recurrence* — recompute the state from scratch each step — and prior work on the training/inference ("discretization") gap is feedforward-only [@yousefi2025mind; @kim2026align; @ruettgers2025light]. In the recurrent setting the gap acquires a new and worse character: it **compounds over time**. In our earlier study of logic-native gating (the P1 workshop paper, [@malle2026gating]), a gated recurrent LGN *solves* a 50-step copy task in relaxation (soft accuracy 0.88) while the deployed discrete circuit sits near chance (0.33) — a gap of +0.50 that was 0.00 at 20 steps. A model that ships as a circuit but was validated as a relaxation is not deployable; for the safety-adjacent, verification-friendly settings LGNs are pitched at [@kresse2025verification], this train/deploy divergence is the central obstacle.

This paper asks: **what does it take to train a recurrent LGN whose deployed clocked circuit inherits the trained behavior?** We answer with a four-part study, in the order we learned it — including the failures, which turn out to carry the mechanism.

**Contribution 1 — the never-write collapse (a negative result with a mechanism).**
The textbook fix suggests itself: if continuous state drift causes the gap, re-binarize the state each step during training (a straight-through round on the recurrent state — a "bistable restore", making each hidden bit an SR-latch-like element that snaps back to {0,1}). We implement three such primitives (state-rounded SR latch, T flip-flop, and a rounded gated cell) and show they all **fail at copy-50, at chance, across seeds and ablations** (anneal schedules, keep-bias settings, entropy regularization). Gate-usage forensics expose the mechanism: the write-path logic collapses to constant gates (set/reset networks → FALSE at 74–100%; the rounded gated cell's gate network → TRUE at 84%), i.e. the network learns to **never write**. Hard-rounding the state creates a moat: a partially-confident write gets rounded into a *corrupting* write, which scores worse than holding nothing, so "hold forever" becomes the loss attractor. This is a forward loss-landscape obstruction — gradients flow fine under BPTT. We scope it honestly: the collapse is a *hold-task/high-keep-bias regime* phenomenon; the same rounded cell trains normally on an integration task (psMNIST-28: 0.62, healthy gate distributions).

**Contribution 2 — the reframe: deployment is already exact; the gap is a training-trajectory artifact.**
The state-rounding was solving a non-problem. We observe (and verify in code) that an argmax-deployed gated recurrent LGN **already carries an exactly binary state**: hard gates on binary inputs emit binary outputs, a multiplexer of binary values is binary, and the state initializes at zero — so by induction the deployed state is in {0,1} at every step, with no rounding anywhere. Exact-binary deployment was never missing. What remains is that the *soft training trajectory* visits continuous states the deployed circuit never sees — precisely the **computation gap** of Kim [-@kim2026align] (irreducible by selection-side methods, zero iff inputs are binary), here compounding across timesteps. Our data isolates it: the gap grows with length (0.00 → +0.50 from L=20 to L=50) while a selection gap is length-independent for a shared recurrent cell; and a gate-entropy regularizer — a selection-side tool — leaves the discrete accuracy unchanged (0.75) while dragging the soft accuracy down to meet it. An oracle probe of the trained gated cell shows its soft state is only mildly "mushy" (0.8–3.1% of values in [0.4, 0.6]), which is why the remaining gap is closable at all.

**Contribution 3 — deep supervision, not the primitive, closes the gap.**
The actual barrier at 50 steps is **credit assignment**: copy supervises only at the final step, so the "write early, then hold" program earns no intermediate credit, and the never-write plateau is flat. Supervising the state through the (time-invariant) label at *every* timestep — deep per-timestep supervision — dissolves the plateau: the gated cell reaches discrete 1.000 on copy-50 (2/3 seeds exactly, third 0.882), and our register cell reaches discrete **1.000 on 3/3 seeds**, both with zero gap. A **length curriculum** (train copy-8, warm-start 20 → 35 → 50) closes it independently (3/3 seeds at every rung), confirming the diagnosis. An activation-margin penalty — the drift fix that stays entirely on the loss side — *destabilizes* training when used alone (903 skipped steps, dead run) and adds nothing over deep supervision. We scope the claim: this is evidence on hold/recall tasks where a valid per-step target exists; deep supervision does not universally erase gaps off-saturation (on distractor-copy the gated cell retains a +0.12 gap in one seed despite it).

**Contribution 4 — the input-clocked latch: round the control, never the state.**
The collapse analysis dictates the primitive. If rounding the *value* creates the moat and the *deployed* value is binary anyway, then the only thing worth rounding in training is the **write-enable**: `h′ = ŝ·h + (1−ŝ)·c` with `ŝ = round(s)` under an identity straight-through estimator (annealed). Holding is then an *exact* identity during training (unit-Jacobian carousel, zero drift by construction); writing stores the *unrounded* candidate (no corrupting round, no moat, no collapse). This is a **learnable write-enabled register** — functionally what a hardware flip-flop with a clock-enable is. We are explicit about what it is *not*: at inference, argmax already binarizes the gated cell's enable, so `clatch` and `gated` deploy to the same circuit class. The contribution is the *placement of the straight-through round*: it makes the training forward pass deploy-consistent on the hold path. Empirically the register is **competitive, never superior, on accuracy** — copy-50 saturates for every trainable route; distractor-copy ties at 20 distractors and favors the gated cell at 8; psMNIST-28 is a statistical tie — and we report all of it. Its two verified edges are: **training stability** (0 non-finite steps in all 75 register-family runs; the only 5 destabilized runs in our 191-run archive are all gated — we stratify this census by config and flag its observational limits), and **bounded, non-overlapping discretization gap at matched configuration** (psMNIST-28: clatch +0.009..+0.027 vs. gated +0.038..+0.146 across seeds at indistinguishable accuracy) — which we frame as the *design property* it is (the training forward already runs the hard enable) rather than a discovered empirical law.

**Integrity statement.** We found **no task on which the register beats the soft gate on accuracy**, and two purpose-built separator tasks failed to separate the primitives. This paper's claims are the collapse mechanism, the reframe, the training method, and the stability/deploy-consistency properties of the register — not an accuracy win. We believe stating this plainly is a feature: the deployment story for learned circuits needs load-bearing negative results as much as positive ones.

---

## 2. Related Work

**Differentiable logic gate networks.** Petersen et al. [-@petersen2022deep] introduced the softmax-over-16-gates relaxation with fixed random wiring; convolutional LGNs with residual initialization scale it to CIFAR-10 at nanosecond FPGA latency [@petersen2024conv]. Training-side improvements include Gumbel-noise + straight-through gates [@yousefi2025mind; @jang2017categorical], the input-wise 4-basis parametrization [@ruettgers2025light], learned connectivity [@mommen2025connections; @fojcik2025lilogic], and coefficient-space multilinear training [@kim2026multilinear]. LUT-based weightless networks are the coarser-grained cousin [@bacellar2024dwn]. All are feedforward.

**The discretization gap.** Yousefi et al. [-@yousefi2025mind] close most of the feedforward gap with hard-forward training; Kim [-@kim2026align] decomposes the gap into a *selection* component (which gate; removable by hard-ST forward) and a *computation* component (soft vs. hard evaluation of the chosen gate; provably zero iff inputs are binary, large near 0.5). We import this decomposition and locate the recurrent LGN's length-growing gap in the computation half — where training-side selection fixes cannot reach — and close it with per-step supervision plus an enable-clocked cell. Kim's analysis of interaction-coefficient gradient starvation [@kim2026multilinear] applies to our multilinear latch surrogates; we cite rather than re-derive.

**Recurrent LGNs.** RDDLGN [@buehrer2025rddlgn] carries state by concatenating the previous hidden bits into the next combinational recompute (seq2seq translation, 16 steps); DiffLogic CA [@miotti2025dlca] uses the same concat-recurrence for cellular automata and explicitly names gating/state-forgetting as future work; our P1 [@malle2026gating] added the logic-native gated (MUX) cell and the training recipe this paper inherits. Closest to our motivation, R-DTLGN [@damera2026rdtlgn] — recurrent *ternary* LGNs for STL runtime monitoring — independently identifies the recurrent hardness-fragility problem, but fixes it with **post-hoc trajectory distillation** (a calibration pass) and carries state as a plain delay register; it has no bistable or write-enabled primitive, and its Tarski/Kleene "stability" results concern forward-dynamics fixed points, a different object from our training-stability and deploy-consistency claims. Yue & Jha extend interpretable differentiable logic to time series without recurrence [@yue2025dlnts]; LG-TS-FFN simulates automata with logic gates plus recurrence for exact AFA execution [@dhayalkar2026lgtsffn]. Differentiable FSMs [@mordvintsev2022dfsm] and DeepDFA [@deepdfa2024] learn abstract transition systems, not gate-level circuits; recurrent CircuitSAT sampling uses flip-flops as hidden state on a *fixed* netlist [@sat2025recurrent]. **None of these places a learnable clocked register in the training loop**; to our knowledge the never-write collapse, the deploy-consistency framing, and the enable-clocked cell are new.

**Gating and memory in real-valued RNNs.** The constant-error carousel and forget-gate bias [@hochreiter1997long; @gers2000learning; @cho2014gru] motivate our keep-bias and hold path; vanishing/exploding analyses [@bengio1994learning; @pascanu2013difficulty] transfer wholesale to logic recurrences (P1). Flip-flop *neurons* in real-valued networks [@kumari2023flipflop] are memory units by architecture, not primitives in a discretizable gate vocabulary. Soft addressing of discrete memory (NTM [@graves2014ntm]) shares the relax-then-harden problem but not the substrate. Deep supervision echoes auxiliary per-step losses long used in deep nets; our use is specifically as *credit-assignment repair for holds*, and the running-XOR variant for parity follows the dense-target logic of Deletang et al.'s Chomsky-hierarchy analysis [@deletang2023chomsky].

**Hardware and verification context.** Feedforward LGN-to-FPGA is now commodity (measured ns-scale deployments and partial toolchains [@petersen2024conv; @gerlach2025cicada; @wormald2026fpga; @bitlogic2026]); quantized-RNN FPGA flows (FINN-L/GL [@finnl2018; @finngl2025]) hold state in registers as an implementation artifact of arithmetic RNNs, not as learned gate-level circuits. Kresse et al. [-@kresse2025verification] show feedforward LGNs are unusually amenable to formal verification; DWC compiles weightless policies to single-cycle FPGA control [@kresse2026dwc]. Our register cell is the training-side complement: a state element whose train-time semantics match the deployed flip-flop, which is what a *sequential* verification story will need (§8).

---

## 3. Preliminaries

### 3.1 Recurrent logic cells

We use the P1 harness (`mlgn/seqlgn/`). A `LogicMLP` is a stack of two `LogicLayer`s (per-neuron softmax over 16 gates, fixed random wiring; soft mixture in train mode, hard argmax gate in eval mode) [@petersen2022deep]. A recurrent cell consumes `z_t = [x_t ; h_{t−1}]` and emits `h_t`; a `GroupSum(k, τ)` head reads the final state. All mechanisms share this scaffold and differ **only** in how state is carried:

| mechanism | update | role |
|---|---|---|
| `rddlgn` | `h′ = U(z)` — recompute from scratch | concat-recurrence control [@buehrer2025rddlgn; @miotti2025dlca] |
| `gated` | `h′ = s·h + (1−s)·c`, `s = G(z)`, `c = C(z)` | soft 2:1 MUX per bit (P1); = gated D-latch relaxation |
| `latch` (SR / T-FF) | char. eq. `Q⁺ = S + (1−R)Q − S(1−R)Q` (or `Q⁺ = T ⊕ Q`), **state STE-rounded** each step | the "bistable restore" (§4) |
| `combo` | gated MUX write, **state STE-rounded** | gated write + restore (§4) |
| **`clatch`** | `h′ = ŝ·h + (1−ŝ)·c`, **`ŝ = round(s)` via identity STE** (annealed) | **input-clocked latch: round the enable, never the value (§6.2)** |

Notes: (i) the SR/T-FF cells use the closed-form *characteristic next-state equation*, not an iterated cross-coupled feedback loop — there is no within-step fixed point to settle, so no metastability and no implicit-gradient machinery is needed (Remark, §6.3); the multilinear surrogates live in the same algebra as the 16 gate relaxations [@petersen2022deep; @kim2026multilinear; @ruettgers2025light], and we claim only the characteristic-equation reduction, not the basis. (ii) At binary values the MUX is exactly `(s∧h)∨(¬s∧c)`, three gates per bit. (iii) The latch/register "primitive" is a *cell-level wiring pattern* around per-bit fixed combiners fed by learned control `LogicMLP`s — the 16-gate vocabulary itself is unchanged.

### 3.2 Discrete-locked evaluation and the gap

We report: **disc** — accuracy of the deployed object (argmax gates, inputs binarized, hard recurrent semantics): the number that matters; **soft** — the relaxed model's accuracy; **gap = soft − disc** on the same best-validation checkpoint. Model selection is on discrete validation accuracy; the test set is touched once. Training inherits P1's recipe: keep-bias initialization of the enable toward TRUE (carousel on at init; task-dependent strength), Adam at lr 0.003 cosine-decayed to 3e-4, global-grad-norm clip 1.0, and **skip-step**: any update whose global gradient norm is non-finite is skipped (`n_skipped` is recorded per run — it is our destabilization signal in §7.3).

### 3.3 Tasks

**copy(L)** — a 1-of-8 symbol at t=0, L−1 blank steps, classify the symbol (chance 0.125); pure hold [@hochreiter1997long; @le2015simple]. **distcopy(L, D)** — cued target at t=0 plus D *distractor* symbol tokens scattered in the sequence that must be held through without overwriting (the hold-vs-overwrite dial; per-step targets remain valid for deep supervision) — our tractable variant of selective copying [@gu2023mamba]. **selcopy(L)** — one un-cued symbol at a random early position (content-based write decision); we report it only in the stability census because the K=1 variant is structurally confounded (§7.2). **parity(L)** — running XOR of a bit stream (the task whose solution *is* a T flip-flop) [@deletang2023chomsky]. **psMNIST-28** — permuted MNIST [@le2015simple] fed 28 pixels/step for 28 steps (chance 0.10). *This is the chunked variant, not the standard 784-step benchmark — published LSTM numbers (~0.90) are not comparable.* **delayed-MNIST** (P1, context only) — encode a digit, hold through up to 100 blank steps.

Configurations (full table in Appendix C): copy/distcopy at hidden 1024, psMNIST-28 at hidden 1000, parity at hidden 512; 20k iterations, batch 128, τ=30. The gated/clatch/combo/latch arms are **exactly gate-matched** (4,096 gates at hidden 1024; 4,000 at hidden 1000); the concat-recurrence control gets an equal-gates variant (hidden 2000/2048). Each mechanism runs at its task-tuned keep-bias (P1's finding: high to hold, low to absorb); where a comparison is keep-bias-matched we say so explicitly. ≥3 seeds on headline numbers (exceptions flagged, e.g. distcopy n=2). Hardware: 1–2× RTX 2080 Ti; ~25–40 GPU-minutes per run.

---

## 4. Rounding the state fails: the never-write collapse

**The obvious fix.** If the gap comes from continuous state drifting where the deployed circuit is binary, force the *training* state binary: after each update, apply a straight-through round `⌊·⌉` to the new state ("bistable restore" — each bit re-snaps to {0,1} like a settled latch, with identity backward so the hold Jacobian survives). We built three versions (§3.1): the SR-latch cell, the T-flip-flop cell, and `combo` (the P1 gated write with a rounded hold). On copy-8 (CPU smoke) the restore *worked* — an annealed SR cell reached discrete 1.000, gap 0. It does not survive scale.

**Result: every state-rounded variant is at chance on copy-50** (Table 1, middle block). SR latch: 0.118–0.126 across 3 seeds and all four ablations (soft state, no anneal, +entropy, keep-bias 6). `combo`: 0.122–0.252 across 3 seeds, no-anneal, and low-keep-bias rescues. Crucially, *soft* accuracy is also at chance (≈0.12) — these runs never learn even the relaxed task. No instability is involved (`n_skipped = 0` everywhere): the optimizer converges, calmly, to nothing.

**Gate-level forensics (the mechanism).** We log per-layer gate-usage distributions at the end of training. In the dead runs, the *write path* has collapsed to constant gates:

- SR cell: the `set`/`reset` networks' output layers select the constant-**FALSE** gate for 74–82% of neurons (keep-bias 6: **100%**) → S=R=0 always → the latch holds its initial zero forever.
- `combo`: the enable network collapses to constant-**TRUE** (84%) → the MUX always keeps → never writes.
- A *working* gated run, contrasted, is healthy: candidate dominated by pass-through/negation gates (writing), enable by keep-family gates.

`[FIG 1 — three-panel gate-usage histograms: latch set/reset → FALSE; combo enable → TRUE; healthy gated. Source: *_gate_distribution JSONs, 2026-07-03.]`

**Why rounding manufactures this.** With a hard-rounded state, a *partially-confident* write (candidate near 0.5) is rounded to a full write of a possibly-wrong bit — corrupting the memory scores *worse than chance*, while writing nothing scores exactly chance. There is therefore a **moat** around the never-write attractor: every gradient path from "hold zero" to "write correctly" passes through "write badly." Keep-bias — necessary against P1's cold-start — actively deepens the attractor (at keep-bias 6 the collapse is total), and the `(1−s)` factor starves the candidate network's gradient as `s→1`. The failure is a *forward loss-landscape* obstruction; BPTT gradients flow throughout (the hold-path Jacobian is exactly 1).

**Scope (this is not "hard state never trains").** The collapse is a property of the *long-hold regime*, not of the restore per se: the same `combo` cell trains normally on psMNIST-28 (discrete 0.624, healthy write gates, no skips) — an integration task has no "hold forever" attractor to fall into. And on copy-8 the restore closes the gap perfectly. The lesson is sharper than "rounding is bad": **rounding the *value* is what creates the moat** — which points directly at §6.2.

---

## 5. The reframe: deployment is already exact; the gap is a training-trajectory artifact

The state-rounding program of §4 was aimed at the wrong object.

**Observation (deployed state is binary by construction).** At inference the harness locks argmax gates and binarizes inputs. Then every `LogicMLP` maps binary inputs to binary outputs; the MUX of binary `(s, h, c)` is binary; and `h_0 = 0`. By induction the deployed recurrent state of the *plain gated cell* is exactly binary at every timestep — no rounding, no restore, nothing to add. The deployed model **is** already a clocked Boolean circuit: state bits in registers, learned combinational next-state logic, one timestep per tick. (Verified in the difflogic source and our eval path; this also holds for `rddlgn` and every other mechanism here.)

So exact-binary deployment was never the problem, and §4's rounding only forced the *soft training trajectory* to be binary — a constraint deployment never asked for, purchased at the price of the collapse.

**What the gap actually is.** Kim [-@kim2026align] decomposes the feedforward train/inference gap into a *selection* gap (soft mixture vs. argmax gate choice; removable by hard-selection training) and a *computation* gap (soft vs. hard evaluation of the *chosen* gate; method-independent, zero iff the propagated values are binary, up to 50–75% for values near 0.5). Three independent observations locate the recurrent gap in the computation half, compounding over time:

1. **Length-dependence.** The gated cell's copy gap is 0.000 at L=20 and +0.50 at L=50 (Table 1). A selection gap is length-independent for a shared recurrent cell (the argmaxed gate choice is the same at every unrolled step); only a values gap can grow with steps as the soft state drifts.
2. **A selection-side tool does not touch it.** Gate-entropy regularization (pushing gate distributions one-hot) on copy-50 committed the gates (entropy 1.6 → 0.03) and left discrete accuracy *unchanged* at 0.75, dragging soft accuracy down to meet it (1.00 → 0.886): the soft solution was using the mixture, and forcing selection just destroys it.
3. **The drift is real but mild.** An oracle histogram of the trained gated cell's soft state finds only 0.8% (t=0), 3.1% (mid), 0.8% (t=T) of values in the mushy band [0.4, 0.6] — the trajectory is *near*-binary, which is why the remaining gap is closable by training-side means at all.

**Interpretation.** In the recurrent setting, closing the *selection* gap (the solved problem [@yousefi2025mind; @kim2026align]) does not close the *computation-over-time* gap: the soft trajectory and the deployed binary trajectory are different dynamical systems, and their divergence grows with the horizon. The fix must either align the training trajectory with the deployed one (without triggering §4's moat), or make the training signal robust to the divergence. §6 does both.

---

## 6. Closing the gap

### 6.1 Deep per-timestep supervision (the method)

**Diagnosis.** Copy-style holds supervise only at the final step. The correct program — write at t=0, hold for L−1 steps — earns no credit until the very end, 49 steps after the write decision; meanwhile "never write" is a flat plateau at chance. This is a credit-assignment failure, and it, not the primitive, is what kept every §4 cell (and the plain `clatch`, §6.2) at chance.

**Method.** Add the label loss at every step:

```
L_total = CE(g(h_T), y) + λ · (1/T) Σ_t CE(g(h_t), y)          λ = 0.2
```

where `g` is the (shared) GroupSum head. This is valid whenever the target is time-invariant (copy, distcopy: the answer is defined from t=0). For parity the final-only loss is *deceptive* (every proper prefix-XOR is uncorrelated with the final label), so we supervise against the **running** target `y_t = XOR(x_1..x_t)` (λ = 0.3) — the dense-target repair of the same failure.

**Result (Table 1).** On copy-50, deep supervision takes the gated cell from disc 0.33 (gap +0.50) to **disc 1.000** (2/3 seeds exact; third seed 0.882 with zero gap), with no other change. It is the active ingredient: adding an activation-margin penalty on top changes nothing (3/3 at 1.000), while the margin penalty *alone* explodes (903 skipped steps, dead at chance) — the drift-targeting loss is dominated and unstable. A **length curriculum** (copy-8 → 20 → 35 → 50 by checkpoint warm-start, no deep supervision) reaches 1.000 at every rung on 3/3 seeds — an independent route that fixes the same credit-assignment problem by never letting the write signal be 49 steps from the loss. Capacity, for contrast, is not the fix: doubling hidden to 2048 lifted the old baseline only to 0.757 with the gap intact.

**Scope.** This is evidence on hold tasks with valid per-step targets, at a task (copy-50) that *saturates* once trainable. Deep supervision is not a universal gap eraser: on distcopy one gated seed keeps a +0.122 gap despite it (Table 4), and our psMNIST-28 runs do not use it. The claim is: *per-step supervision repairs long-horizon write/hold credit assignment, which is the actual barrier at 50 steps* — not "deep supervision closes all gaps."

### 6.2 The input-clocked latch: round the control, never the state

**Design, dictated by §4.** The moat came from rounding the *value*; the deployed value is binary anyway (§5). The only decision that must be hard for the training forward to have deployment semantics on a hold is the **write-enable**. So clock the enable:

```
c = C(z);   s = G(z);   ŝ = ⌊s⌉_STE   (identity backward; annealed)
h′ = ŝ · h + (1 − ŝ) · c
```

with the anneal `⌊s⌉_α = (1−α)s + α·round(s)`, α ramped 0→1 over the fraction [0.1, 0.6] of training (soft solution first, then commit; deterministic annealing [@rose1998deterministic]). Backward is identity for every α, so the hold-path carousel `∂h′/∂h = ŝ` is preserved exactly [@bengio2013estimating; @neftci2019surrogate; @zenke2021remarkable].

**Properties.** (i) **Hold is exact during training**: when `ŝ=1`, `h′ = h` bit-for-bit — zero drift over any horizon, by construction; the computation gap on the hold path is structurally 0 (Kim's condition — binary propagated values — is met on every held bit). (ii) **Write is never rounded**: an uncertain write stores the soft candidate, which argmaxes cleanly at eval — no corrupting round, no moat, and empirically no collapse (clatch trains wherever gated trains). (iii) **It is a register**: per bit, `clatch` is precisely a flip-flop with a clock-enable — the enable computed by learned logic, the stored value held exactly between writes. At α=0 it *is* the gated cell; the whole design is one STE placement away from P1.

**What it is not (deploy-equivalence, owned).** At inference, argmax binarizes the gated cell's enable too: `clatch` and `gated` deploy to the **same circuit class**, and a given trained checkpoint of either is an exact clocked circuit (§5). The contribution is *train-time*: `clatch`'s forward pass already executes the deployed hold semantics, so the trained trajectory cannot rely on fractional holds that deployment will snap away. Every deploy-consistency result in §7.4 should be read as a *consequence of this placement*, not as a property of a new inference-time object.

**Trainability (Table 1).** `clatch` alone at copy-50 is at chance (0.126/0.247/0.248 — soft too): the enable-round does not repair credit assignment, consistent with §6.1's diagnosis that the barrier was never drift. With deep supervision, `clatch` reaches **disc 1.000 on 3/3 seeds**. (One seed's *soft* accuracy is 0.754 against disc 1.000 — the discrete circuit is better than the relaxation that trained it; we flag it as the soft trajectory failing to track the hard one, not as a virtue.)

### 6.3 Remark: why not implicit/equilibrium gradients through a real latch loop

A physical SR latch is cross-coupled feedback; the tempting formalization is a within-step fixed point `q* = F(q*)` trained by implicit differentiation (DEQ [@bai2019deep]) or equilibrium propagation [@scellier2017equilibrium]. That road is closed for memory: a bistable element has **two** stable fixed points, violating the uniqueness/monotonicity assumptions of the implicit toolkit [@winston2020monotone]; the ideal (lossless) hold has a unit Jacobian eigenvalue, making `(I−J)` singular exactly at the marginal-hold operating point (a restoring latch is contractive and invertible [@bai2021stabilizing], but then it is the basin *selection* — a discontinuous, non-unique object — that carries the memory [@pineda1987generalization; @almeida1987learning]); EP additionally requires static inputs, excluding sequential state outright [@bal2023sequence], and bistable units have been observed to resist equilibrium training in physical implementations [@laydevant2024training]. (Exact RTRL is *not* blocked [@zucchet2023online]; the obstruction is specific to the fixed-point/implicit family.) We avoid the entire issue by construction: the characteristic next-state equation collapses the feedback loop into a closed-form per-step function — the recurrence across timesteps *is* the clock — and BPTT applies unchanged. We consider this one-paragraph obstruction a signpost, not a contribution of this paper.

---

## 7. Evaluation

### 7.1 Copy-50: what closes the gap (the method table)

**Table 1 — copy-50, alphabet 8 (chance 0.125), hidden 1024, 3 seeds unless noted.** disc / soft / gap; `skip` = optimizer steps skipped on non-finite gradient (of 20k).

| route | disc (per seed) | soft (per seed) | gap | skip |
|---|---|---|---|---|
| `rddlgn` control (also equal-gates h2048) | 0.259 (0.126–0.248) | 0.122 | ≈0 | 0 |
| `gated` (P1 baseline) | 0.380 / 0.241 / 0.378 | 0.879 / 0.743 / 0.877 | **+0.50** | 0 |
| `gated` + capacity (h2048, 30k) | 0.757 | 0.886 | +0.12 | 1477 |
| `gated` + entropy-reg (h2048) | 0.754 | 0.886→ | +0.13 | 0 |
| `gated` + margin only | 0.126 | 0.126 | 0 | **903 (dead)** |
| **`gated` + deep-sup** | **1.000 / 0.882 / 1.000** | 1.000 / 0.882 / 1.000 | **0.000** | 0 |
| `gated` + deep-sup + margin | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.879 | ≤0 | 0 |
| `latch` (SR, restore; all 4 ablations + kb6) | 0.118 – 0.126 | ≈0.12 | ≈0 (trivial) | 0 |
| `combo` (restore; incl. no-anneal, kb0/1) | 0.122 – 0.252 | ≈0.125 | ≈0 (trivial) | 0 |
| **`combo` + length curriculum** (8→20→35→50) | **1.000 / 1.000 / 1.000** | 1.000 | 0.000 | 0 |
| `clatch` alone | 0.126 / 0.247 / 0.248 | ≈0.12 | ≈0 (trivial) | 0 |
| **`clatch` + deep-sup** | **1.000 / 1.000 / 1.000** | 0.754 / 1.000 / 1.000 | ≤0 | 0 |

Readings: (1) **deep supervision is the gap-closer** — it rescues the gated cell with no primitive change; (2) the curriculum is an **independent second route** (same credit-assignment diagnosis); (3) the restore cells die *without* it (never-write collapse, §4) and `clatch` is the only hard-decision cell that becomes perfect *with* it; (4) copy-50 **saturates** — every trainable route lands on exactly 1.000, so this task certifies the method and cannot rank the primitives. `[FIG 2 — bar chart of disc/soft per route.]`

### 7.2 Accuracy across tasks: competitive, never superior (the honest map)

**Table 2 — where the primitive does and does not matter.** Discrete test accuracy (mean over seeds; per-seed in Appendix B).

| task (config) | `gated` | `clatch` | verdict |
|---|---|---|---|
| copy-50 + deep-sup (kb 3 / 1) | 1.000 (2/3; 0.882) | **1.000 (3/3)** | saturated — cannot rank |
| distcopy L50 D=8 + ds (kb3 both, n=2) | **0.936** | 0.877 | gated ahead |
| distcopy L50 D=20 + ds (kb3 both, n=2) | 0.874 | 0.874 | tie |
| psMNIST-28 (kb0 both, n=3) | 0.602 | 0.634 | tie (Welch p≈0.52; gated s2 outlier 0.519) |
| parity-32 + running-target ds (kb0, n=3) | 0.504 (chance) | 0.501 (chance) | both fail; see mechanism panel |
| selcopy L50/L100 (confounded) | 0.626 / 0.499 | 0.494 / 0.375 | **excluded** (see below) |

**The primitive-separator hunt failed, and we say so.** Two purpose-built separators — parity with dense running-XOR supervision, and distractor-copy with matched keep-bias — were designed to let a clean register beat a leaky soft MUX. Neither separated: distcopy ties at D=20 (0.8739 vs 0.8734) and *favors the gated cell* at D=8; on parity only the **T flip-flop** moves off chance (disc 0.572/0.582/0.586 — the toggle primitive on its home task, still short of solving), while clatch, gated and the control stay at 0.50. The parity panel is a *mechanism* result — the dense target unlocks exactly the primitive whose inductive bias matches the task — not an accuracy win for any cell we advocate. We found **zero tasks where `clatch` beats `gated` on accuracy**. `[FIG 5 (optional) — parity mechanism panel.]`

**Why selcopy is excluded from accuracy claims.** Our selective-copy runs carry three verified confounds, all favoring the gated arm (mismatched keep-bias 3-vs-1 on a hold task; deep supervision ill-posed for ~24% of timesteps because the symbol appears at a random position; and the K=1 variant is OR-solvable, never exercising hold-vs-overwrite). We keep the runs only in the stability census (§7.3), confounds stated.

**Distcopy gap footnote (scoped).** On distcopy the sign pattern is asymmetric — gated leaks upward (+0.122 / +0.012), clatch does not (−0.000 / −0.119) — but this held *only* on distcopy (clatch shows positive gaps elsewhere: +0.112 selcopy-L100, +0.041 psMNIST kb1), the clatch negative gaps trace to a worse soft optimum that rounding recovers, and n=2. We do not claim a gap-sign law. `[TODO: distcopy seed 2 backfill — 4 runs, ~40 min each.]`

### 7.3 Training stability: the register family never destabilized (a stratified census)

Our destabilization signal is `n_skipped` — optimizer steps skipped because the global gradient norm went non-finite (§3.2). Across the full 191-run archive:

- **Register family (clatch 23, latch 25, combo 27 = 75 runs): 0 skipped steps, ever.** This includes the *hard-trained* runs (copy-50 at 1.000, distcopy at 0.75–1.00, psMNIST-28 at 0.60–0.66), not only dead-at-chance runs.
- **The only 5 destabilized runs in the archive are all `gated`**: 16,695 and 13,339 skips/20k (early lr 0.01 runs), 1,477/30k (hidden 2048), 903/20k (margin-reg), and **2,082/20k at selcopy-L100** (lr 0.003 — the shared-config stratum).
- `rddlgn`, `lstm`, `gru_cell`: 0 skips (stability separates register-from-gated, not register-from-everything).

**Stratification (the honest version).** Four of the five gated explosions occurred at configurations the register family never ran (lr 0.01, hidden 2048, margin-reg). Within the shared stratum — lr 0.003, no margin, hidden ≤1024 — the census reads: **gated 1 destabilized run of ~50; register family 0 of 75.** The one shared-stratum pair (selcopy-L100 gated 2,082 skips vs. clatch 0) is *not* keep-bias-matched (3 vs. 1), and keep-bias is a documented explosion driver, so we present it as "each mechanism at its task-tuned setting," not as a controlled pair. `[TODO optional: kb-matched selcopy-L100 pair (2 runs) to upgrade this into a controlled comparison.]` The mechanism-level reading is nonetheless coherent: the gated cell's instability pathway (keep-bias pushing the recurrence Jacobian ≈1, long products creeping above 1 — P1 §5.4) runs through the *soft multiplicative hold*, which is exactly the path `clatch` replaces with a hard identity.

**Claim, scoped.** *The register family never destabilized across our sweep, including every configuration where it trains to high accuracy; the soft-gated cell is the only mechanism that ever did.* We do not claim "gated always explodes at length" (it is fine at L=101/112/128 elsewhere) nor that the census is a controlled experiment — it is an observational property of 191 logged runs, stratified above.

### 7.4 Deploy-consistency: the discretization gap at matched configuration

**Table 3 — psMNIST-28, keep-bias 0 for all arms, 3 seeds, gate-matched (4,000).** disc (per seed) and gap (per seed).

| mechanism | disc s0/s1/s2 (mean) | gap s0/s1/s2 (mean) |
|---|---|---|
| `gated` | .632 / .654 / .519 (.602) | +.077 / +.038 / **+.146** (+.087) |
| **`clatch`** | .657 / .647 / .598 (**.634**) | **+.023 / +.009 / +.027 (+.020)** |
| `combo` | .606 / .659 / .607 (.624) | +.021 / −.014 / −.025 (−.006) |
| `rddlgn` (own config†) | .620; equal-gates .655 | +.033; +.038 |

† control at keep-bias 3; no kb0-matched 3-seed control set exists — listed for orientation, not as a matched arm. Gated s0 is the 2026-06-18 run (older logging schema); s1/s2 are July runs. `[TODO optional: +2 seeds/arm to harden the non-overlap.]`

At statistically indistinguishable accuracy (clatch .634 vs gated .602; the difference is one gated outlier seed), the clatch gap is **bounded and non-overlapping with the gated gap across all seeds** (max clatch +0.027 < min gated +0.038; means 4.4× apart, ~2.9× if the gated outlier seed is excluded). The whole restore family shows the same cleanliness (combo gaps within ±0.025).

**Read this as a design property, not a discovered law.** `clatch` (and combo) *train with the hard decision already in the forward pass*, so their soft-vs-hard difference is largely the selection component by construction — that the deployed circuit inherits the trained behavior is precisely what "clock the enable" was designed to buy, and this table verifies the design delivers it **at zero cost in accuracy, trainability, gates (matched), or wall-clock (~25 min/run for both arms)**. It is not independent evidence of a universal "tighter gap" tendency, and on distcopy the clatch soft-side gap magnitude can be larger (§7.2). The falsifiable content is: *the enable-clocked forward achieves deploy-consistency without paying for it anywhere we measured.* `[FIG 3 — per-seed gap dot plot, non-overlap visible.]`

---

## 8. Deployment: registers, clocked circuits, and the verification surface

**What the trained artifact is.** Any argmax-deployed model in this paper is an exact, deterministic clocked Boolean circuit — a Mealy/Moore machine: H state bits in registers, learned two-level-of-`LogicLayer` combinational next-state logic, GroupSum as a popcount readout (§5). For `clatch` the per-bit structure is literally a **flip-flop with clock-enable**: enable = one output bit of the learned gate network, data = one output bit of the candidate network, H independent 1-bit registers sharing control logic. This maps 1:1 onto FPGA fabric (each state bit → one FF with CE; each learned gate → one LUT entry) and onto standard-cell registers. At our scale (4,096 gates + 1,024 state bits) the circuit fits the smallest commercial FPGAs with room to spare.

**Why train-time consistency matters here.** The verification appeal of LGNs — exact semantics, no floating point — has been demonstrated for feedforward networks [@kresse2025verification]; the sequential extension needs the *trained* behavior to be the *shipped* behavior, which is exactly the property §7.4 measures. Because the deployed object is a finite-state machine over explicit bits, sequence properties ("the register writes only on cue"; "the held value is unchanged through D distractors") are bounded-model-checking queries on the netlist — no RNN state-space abstraction or automaton extraction [@weiss2018extracting] is required, because the automaton is given.

**Status, stated plainly.** This paper contributes the *mapping and the training-side property*; we do not yet ship a netlist exporter, synthesis report, or model-checking run — that artifact (a sequential RTL/AIGER emitter over these checkpoints) is in progress and deliberately out of scope here. `[DECISION: either land the minimal exporter + one BMC smoke ("writes-only-on-cue" on a copy-50 clatch checkpoint) before submission — it is an afternoon of engineering and would let this section claim a demonstrated artifact — or keep this paragraph as an explicit descope. Do not leave the section between the two.]` No latency/energy numbers are claimed; published ns-scale figures are feedforward [@petersen2024conv; @gerlach2025cicada] and quantized-RNN FPGA flows sit at microseconds [@finnl2018; @finngl2025] — the sequential-LGN measurement is future work (see §10).

---

## 9. Limitations

1. **No accuracy win for the register — anywhere.** "Competitive, not superior" is literal: never better, occasionally worse (distcopy D=8). The contributions are the mechanism (§4), the method (§6.1), and the stability/consistency properties (§7.3–7.4). A reader who wants a primitive that wins on accuracy will not find it here, and our two purpose-built separators failed to produce one.
2. **Copy-50 saturates.** The method table certifies trainability, not ranking. Harder held-state benchmarks that do not saturate (e.g. n-bit memorization with long gaps [@martens2011learning]) are unexplored here.
3. **Seed counts.** distcopy is n=2 per cell; the psMNIST gated s0 run predates the others (older schema); several backfills are marked inline. Headline copy-50 and psMNIST numbers are n=3.
4. **The stability census is observational.** Stratified in §7.3; the flagship long-sequence pair is not keep-bias-matched. It is evidence of an asymmetry, not a controlled experiment.
5. **The gap edge is partly by construction.** §7.4's framing is deliberate: clatch's tight gap is the designed consequence of the hard-enable forward, not an independent empirical discovery; its soft-side gap can be loose on other tasks.
6. **No floating-point baseline.** A small float GRU would very likely dominate every accuracy number in this paper; the LGN value proposition is gates/energy/exactness, not raw accuracy. `[TODO: add a ~50k-param float GRU row on copy-50/distcopy/psMNIST-28 to price the trade explicitly — reviewers will ask.]`
7. **psMNIST-28 is the chunked variant** (28 steps × 28 pixels), not the standard 784-step benchmark; numbers are not comparable to published LSTM results, and we name the task accordingly everywhere.
8. **Deep supervision needs valid per-step targets** (time-invariant labels, or a running target as in parity). Where the per-step target is ill-posed it can actively mislead (the selcopy lesson, §7.2).
9. **No hardware measurement yet** (§8): the register→FF mapping is exact but unsynthesized in this draft; parity remains unsolved by every mechanism (the T-FF reaches 0.58, bar 0.9), so the "learned FSM you can verify" demo currently has no strong host task beyond copy/distcopy invariants.

---

## 10. Future work

**Learned per-bit primitive selection.** Here the memory type is a global architectural choice; the natural generalization — extend the per-neuron softmax to choose among {combinational, register/`clatch`, D-latch, T-FF, …} so each bit learns *which memory element it is*, argmax-deploying to a heterogeneous LUT+FF netlist — is deferred: the differentiable op-selection mechanism is occupied (DARTS [@liu2019darts]; per-dimension memory-op selection in MuFuRU [@weissenborn2016mufuru]), the accuracy tie removes the obvious carrier, and the payoff (a synthesis-measured FF/LUT Pareto at matched accuracy) needs the RTL toolchain first. **Hardware realization and measurement.** A sequential RTL emitter (clocked always-blocks, FFs with CE) over these checkpoints, synthesis/timing reports, and a latency/energy comparison against quantized-RNN FPGA flows [@finngl2025] — no public toolchain emits learned flip-flops today. **Sequential verification.** BMC/IC3 on exported netlists for hold/write invariants and, on tasks with known automata, equivalence checking against the reference machine — extending feedforward LGN verification [@kresse2025verification] to time. **Memory-required control.** A partially-observable control demo where a feedforward logic policy [@kresse2026dwc] provably fails and a register cell carries the belief state. We deliberately do not pursue parallel-scan training accelerations: associative-scan recurrences buy speed at the cost of the state expressivity this line exists for [@merrill2024illusion].

---

## 11. Conclusion

Recurrent logic gate networks already deploy as exact clocked circuits; what they lacked was a training procedure whose trajectory the deployed circuit inherits. We showed the intuitive fix (binarize the state in training) fails for a structural reason worth knowing — rounding the value makes never-writing the optimum — and that the real barriers were credit assignment over long holds, solved by deep per-timestep supervision, and train/deploy divergence on the hold path, solved by rounding the write-enable instead. The resulting write-enabled register trains wherever the soft gate trains, never destabilized in our sweep, matches its accuracy, and hands deployment a circuit whose training already spoke its semantics: registers with learned enables, ticking once per step. Making that circuit *measured* — synthesized, model-checked, and timed against arithmetic RNN flows — is the next paper.

---
---

# Appendices (draft stubs)

## A. Gate-usage forensics (never-write collapse)
Per-layer 16-gate usage histograms for the dead latch/combo runs vs. a healthy gated run; kb-6 total collapse (set/reset → FALSE 100%). Source JSONs: `copy_latch_cp50_latch_*_gate_distribution_*.json`, `copy_combo_cp50_*`, `copy_gated_cp50_gated_s*_gate_distribution_*.json`. `[FIG 1 assets]`

## B. Full per-seed tables
All 191 runs, grouped by task/mechanism/config, with disc/soft/gap/`n_skipped`/gates/train-minutes. Note the archive hygiene facts: 8 duplicate JSON pairs from a queue race (dedup before counting), 13 early-June runs predate the `n_skipped` counter, distractor count D lives in filenames (`d8`/`d20`) not JSON fields for the July runs. `[TODO: generate from collate.py with dedup.]`

## C. Hyperparameters
| family | task(s) | hidden | gates (gated/clatch arm) | kb | lr | iters | extras |
|---|---|---|---|---|---|---|---|
| copy | copy-50 (alphabet 8) | 1024 | 4,096 | gated 3 / clatch 1 | 0.003→3e-4 cosine | 20k | ds λ=0.2; clatch anneal 0.1–0.6; curriculum via `--init-from` 8→20→35→50 |
| distcopy | L50, D∈{8,20} | 1024 | 4,096 | 3 (both) | same | 20k | ds λ=0.2; clatch anneal 0.1–0.6 |
| psMNIST-28 | chunk 28 | 1000 | 4,000 | 0 (both) | same | 20k | no ds; clatch anneal 0.1–0.6 |
| parity | L32 (test L∈{32,128,256}) | 512 | 1,024 | 0 | same | 20k | ds λ=0.3 + running target; tff anneal 0.1–0.6 |
Batch 128, τ=30, grad-clip 1.0, skip-step on non-finite grad norm, Adam. ~25–40 GPU-min/run (RTX 2080 Ti).

## D. Task definitions
Formal definitions of copy / distcopy / selcopy / parity / psMNIST-28 / delayed-MNIST as generated by `seqlgn/data.py` (incl. the distcopy note that a distractor can coincide with the target symbol, effective overwrite pressure ≈ 7/8; and the selcopy K=1 OR-solvability argument).

## E. The SR/T-FF surrogates (for completeness)
Characteristic equations, multilinear relaxations, and Jacobians (`∂Q⁺/∂Q = (1−R)(1−S)`, `∂Q⁺/∂S = 1−Q+RQ`, `∂Q⁺/∂R = −Q(1−S)`; T-FF `∂Q⁺/∂Q = 1−2T`); corner-exactness check; the interaction-term gradient-starvation caution [@kim2026multilinear]. These cells are the paper's §4 evidence, not its recommendation.

---

# Evidence map (internal — strip before submission)

| claim / table | source |
|---|---|
| Table 1 copy-50 rows | `results/copy_gated_cp50_gated_s{0,1,2}_*.json`, `copy_gated_cpB_*` (dsonly/md/marginonly/oracle), `copy_gated_cp4_gated_ds_s{1,2}_*`, `copy_clatch_cpB_*`, `copy_clatch_cp4_*`, `copy_combo_cp50*`, `copy_combo_cp50A_curr_*`, `copy_combo_cp4_curr_*`, `copy_latch_cp50*`, `copy_rddlgn_*` |
| capacity/entropy rows | `copy_gated_L50cap_*`, `copy_gated_L50gap_*` (exp log 2026-06-10/11) |
| Table 2 accuracy map | `distcopy_*_dc_*_d{8,20}_s{0,1}_*`, `psmnist_*_psm_*_kb0_*`, `parity_*_pd_*`, `selcopy_*_bh_selcopy_*` |
| Table 3 psMNIST gaps | `psmnist_{gated,clatch,combo}_psm_*_kb0_*` + `psmnist_gated_psm28_gated_kb0_20260618-*` (s0) + `psmnist_rddlgn_psm28_rddlgn{,_eqgates}_*` |
| stability census | all 191 JSONs, `n_skipped` field; strata per `research/20_program_validation.md` §A2.2 (75 register runs: clatch 23 / latch 25 / combo 27; 5 gated skippers) |
| gate forensics / oracle mushiness | `*_gate_distribution_*.json`; `copy_gated_cpB_gated_oracle_*` (`state_mushy_by_t`) |
| never-write mechanism, reframe, Kim mapping | `research/04_experiment_log.md` (2026-07-02..08), `research/11_paper2_workmap.md` §A0/A0', `research/14_recurrent_lgn_2026_deepread.md` |
| claim scopings (census strata, by-construction gap, deploy-equivalence) | `research/20_program_validation.md` §A2 (all items addressed in-text) |

# Venue adaptation notes (internal — strip before submission)

- **ICML'27 (primary, ~8pp):** keep §1–§7 + §9 whole; compress §2 to ¾ page; §8 stays only if the exporter+BMC smoke lands (else fold its first paragraph into §6.2 and move the rest to future work); Appendix E to supplementary.
- **NeurIPS'27 (slip):** same shape; add the float-GRU row and distcopy seed backfills by then.
- **Workshop cut (4pp, defensive flag-plant):** §4 (collapse + forensics figure) + §5 (reframe) + Table 1 + §6.2 (clatch) + integrity statement. Drop §7.3–7.4, §8.
- **arXiv v1:** this document minus internal blocks, after: [ ] FIG 1–3 rendered, [ ] distcopy seed-2 (optional), [ ] exporter decision resolved, [ ] merge p2→main + tag commit `p2-draft1-numbers`.
- Known reviewer pressure points and their in-text answers: accuracy-tie (→ integrity statement §1, §7.2), census confound (→ stratification §7.3), by-construction gap (→ design-property framing §7.4), "clatch = gated at deploy" (→ owned in §6.2), no float baseline (→ Limitations 6 + TODO), chunked psMNIST naming (→ §3.3, Limitations 7).
