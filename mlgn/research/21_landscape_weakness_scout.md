# 21 — Scout: LGN-field weaknesses beyond P1–P4 (2026-07-08)

_27-agent adversarial scout (research-scout method): 1 axis-enumerator over the KB → 10 per-axis
web scouts (2025–26 window, all synonym families, threats read in full) → 10 independent
adversarial verifiers re-attacking every OPEN/GO call with fresh searches. Verdicts below are
POST-verification (scout → verified). Explicitly excluded territory: discretization gap/STE
(ETH lane), training speed (doc 09), recurrence/gating/latch (P1/P2), sequential FPGA (P3b),
sequential verification (P3a), primitive vocabulary (P4), RL/SNN/VSA (docs 15/13/08)._

## TL;DR ranking (what's actually open)

| # | Axis | Verdict (verified) | Cost | The gate |
|---|------|--------------------|------|----------|
| 1 | **UQ / calibration / OOD** | **GO ~65%** | post-hoc on existing ckpts | 1-afternoon posterior-collapse pilot |
| 2 | **Netlist plasticity (adaptation/fine-tune)** | **GO ~60–63%** | 2–4 GPU-wk MVP | 1-day warm-start falsifier |
| 3 | **Empirical robustness (attacks/corruptions/faults)** | COND ~55–60% (narrowed) | inference-heavy, cheap | OpenReview/ISTA-v2 freshness sweep |
| 4 | **ASIC gate-vocabulary (MAJ/AOI cells)** | COND ~55–60% | needs yosys/OpenLane (→P3b/P4 synergy) | tech-mapping control expt FIRST |
| 5 | **Netlist resynthesis audit** | COND ~55% (demoted from GO) | zero GPU | subsumed-by-ISTA check |
| 6 | **Regularization / why LGNs overfit** | COND ~65% | 300–500 GPU-h | read Kim DSLGN full text (LTU lib) |
| 7 | **Trainability theory** | COND ~55%, expressivity-only | zero GPU | Petersen thesis + Mommen full text |
| 8 | Logic-attention | killed as worded; thin paradigm-internal slice | post-P2 | BiHDTrans delta check |
| 9 | Input encoding | **NO-GO 95%** | — | drop (WARP + BitLogic) |
| 10 | Output heads / regression | **NO-GO 93%** | — | drop (LogicIR ECCV'26 + BEL/RLEL) |

**Cross-cutting finding:** the weakness space is being industrialized by exactly the two labs in
the PhD plan — **ETH (BitLogic, arXiv:2602.07400, TMLR 2026, v2 revised 2026-07-07** — a
five-axis {encoder, connectivity, fan-in, node, head} unified design-space study that killed two
of our fallback angles**)** and **ISTA (2507.02585 interconnect learning, which already does
SAT-equivalence pruning of trained nets)**. The surviving gaps are the **ML-culture axes**
(calibration, adaptation, robustness-empirical) that the hardware-culture labs skip. BitLogic v2
is now a must-read/must-cite for the whole program.

---

## 1. UQ / calibration / OOD — GO ~65% (scout 75→verified 65, repositioned)

**Gap (verified unoccupied after ~15 fresh attacks):** nobody has measured ECE/Brier/NLL,
conformal coverage, or OOD-AUROC on any *binary* LGN; nobody samples the trained per-neuron
16-gate categorical at TEST time. The gate distribution is a **free posterior over Boolean
circuits**: sampling K assignments = an implicit deep ensemble whose every member is a deployable
hard circuit — an efficiency story float ensembles can't match. Kim's Deep Stochastic LGN (IEEE
Access 2023) uses the same sampling **during training only**; Mind-the-Gap's Gumbel likewise.
**Paper:** "Are Logic Gate Networks Calibrated? Free Circuit Ensembles from the Gate Posterior"
(first-measurement + method).
**Verifier corrections (mandatory framing):** PST ternary LGNs (arXiv:2603.00302, UMD) already
publish abstention/selective-prediction — but *architectural* (Kleene UNKNOWN), not
*statistical*; hook = "calibration of the standard binary pipeline". Tsetlin UQ exists at
CIFAR-10 scale (arXiv:2507.04175) — caps "first discrete-logic UQ" claims; scope to LGNs.
CICADA (2511.01908) flags binarization-suppressed outlier scores = motivating exhibit; conformal
× L1-trigger confirmed empty.
**Fit:** best of all axes — post-hoc on checkpoints this repo already has; MNIST/F-MNIST/CIFAR/
SVHN + CMS open data on 2080Ti.
**GATE (run first, one afternoon):** sample 32 circuits from an existing trained checkpoint — if
gate distributions are near-one-hot (entropy→0), no ensemble diversity and the free-posterior
story dies (fallback: entropy-reg/Gumbel checkpoints, weakens "free").

## 2. Netlist plasticity — GO ~60–63% (scout 72)

**Gap (survived 8+ fresh angles incl. NAS warm-start, BNN re-binarization, CL vocab, full 2026
LGN corpus):** not one transfer / warm-start / fine-tune / continual / test-time-adaptation
result exists on gradient-trained gate netlists (Petersen line through LogicIR). "Netlist
plasticity": re-relax hardened one-hot logits, fine-tune under shift (MNIST-C/CIFAR-10-C,
rotated/permuted, task increments), + forward-only ZOA-style adaptation of the HARD netlist, +
adaptation cost measured in **gate-edits** (accuracy-recovered vs bits-flipped vs compute
frontier). Publishable both ways ("cheaply updatable" or "hardened LGNs lose plasticity —
deployment really is frozen").
**Verifier corrections:** the in-place-LUT-rewrite hardware punchline is occupied (Glette &
Kaufmann, AHS 2014 ICAP LUT rewriting; Dynamic Tsetlin 2504.19797) — recast as "fine-tune deltas
are LUT-mask-sized" (simulated), lead with method+forgetting analysis; cite WS-DARTS 2205.06355
(closest relative) and the EDA functional-ECO literature for the metric lineage.
**GATE (1 day):** hardened MNIST LGN → re-relax → fine-tune on rotated/corrupted vs from-scratch
vs forward-only; count gate edits. Dies if warm-start shows no compute advantage AND edits
aren't strikingly small.

## 3. Empirical robustness — CONDITIONAL ~55–60% (scout GO 78, narrowed)

**What survives:** first empirical attack/corruption/fault characterization of *Petersen-lineage
difflogic* nets — BPDA-via-the-relaxation, soft→hard transfer, black-box (Square/HopSkipJump),
CIFAR-10-C corruption curves, wire-level fault injection, + first AT-recipe-through-the-
relaxation with discretization-gap accounting. The "no inference gradient = safe" belief is
textbook gradient masking (thermometer front-end = Buckman/Athalye's broken defense) and remains
untested on difflogic.
**Verifier kills/corrections:** **TTnet (Benamira et al., IJCAI 2024)** — a differentiable
logic-gate-circuit CNN benchmarked against Petersen's net — already publishes complete-SAT
robust accuracy on MNIST/CIFAR at standard ε with PGD-AT-through-STE (and ISTA 2505.19932 cites
it); **Jia & Rinard (NeurIPS 2020)** already executed the verify-attack-tightness methodology on
BNNs. So: a *port with a finding*, not a first — scope claims to difflogic lineage + corruptions
+ faults (genuinely unmeasured), cite TTnet/JR20 as the method chain.
**Fit:** inference-dominated, cheap. **GATE:** OpenReview/arXiv 8-week freshness sweep for
in-flight LGN-robustness submissions (ISTA v2 bolting empirical attacks onto their verifier is
the obvious scoop).

## 4. ASIC gate-vocabulary — CONDITIONAL ~55–60% (scout 65)

**Gap (survived ~12 fresh angles):** differentiable gate selection over an actual **standard-cell
vocabulary** (MAJ3, AOI21/22, OAI21/22, 3–4-input NAND/NOR — combinational only, holding the P4
boundary) with liberty-derived area/delay/power in the relaxation, evaluated **through real
synthesis** against the control nobody ran: the same 16-gate net technology-mapped by ABC/Yosys
onto the full library. Claim = "training in the deployment basis changes what function is
learned and beats post-hoc mapping on the accuracy-vs-PPA Pareto", NOT "we can represent MAJ"
(WARP-LUTs kills that).
**Verifier corrections:** full-library mapping of trained DLGNs has been *run* publicly (LILogic
2511.12340 §4.3.2 SKY130/SG13G2/GF180 tapeouts; Zioma tt10-lgn-mnist Tiny Tapeout) — contribution
is the first *measured cross-basis comparison*, not the first run; cite HGQ/HGQ-LUT + PolyLUT
for cost-in-loss lineage; the misalignment observation is 2019 crypto prior art (NIST LWC "Does
gate count matter?"). Silicon Aware NN (2604.19334) read in full: 16-gate 1:1 mapping, area-only
loss, no tech mapping — the direct baseline to generalize.
**GATE (run before investing):** train vanilla 16-gate net → synthesize (a) 1:1 vs (b) full
Yosys/ABC mapping on SkyWater. If full mapping collapses the basis gap, drop the axis.
**Synergy:** shares the P3b emitter + gives P4's ONE-GATE its missing ASIC flow.

## 5. Netlist-resynthesis audit — CONDITIONAL ~55% (scout GO 78 → demoted)

**Verifier kills:** ISTA 2507.02585 already does SAT-equivalence merging + trivial/greedy/
similarity pruning on trained difflogic-family nets with before/after counts (vanilla-trained ≈
only 0.5% exact removable redundancy — bad news for big-win hopes) and a training-method axis;
**Miyasaka/Mishchenko FCCM 2024** already ran the multi-tool post-training synthesis audit on
LogicNets **with training-data don't-cares (20–75% area reduction)** — strictly subsuming the
thermometer-SDC lever as an upper bound.
**What survives:** (a) the **measurement-critique** — cross-paper gate-count comparisons are
provably inconsistent (Conv-DLGN App A.3 reports post-synthesis counts after an unreleased
in-house 4.6× synthesis shrink; Mind-the-Gap/LILogic report raw; CompactLogic after own pruning)
— still unclaimed and zero-GPU; (b) a lever-decomposed, tool-documented EDA audit specifically
of Petersen-lineage DLGNs (rewrite/refactor/dc2/fraig/espresso per-pass numbers). Keep as a
P3b-emitter by-product / short paper, not a standalone bet.

## 6. Regularization / overfitting mechanism — CONDITIONAL ~65% (scout 65–70)

**Survived:** nobody measures the LGN train/test gap systematically, characterizes WHY
categorical-gate nets overfit (memorization probes, gate-entropy dynamics, small-data scaling),
or demonstrates a generalization-*improving* LGN regularizer. Verifier found a third published
naive-dropout failure (RDDLGN Table 9: all dropout ≤ baseline) strengthening the premise, and
SAM-on-gate-logits is confirmed empty. Standing contradiction to resolve: Kim's DSLGN (IEEE
Access 2023) claims stochastic-gate-perturbation wins; Light DLGN App F says all regularizers
fail — the two literatures never cite each other.
**Corrections:** PST already ships an L1-of-Fourier regularizer (interpretability-framed, no
generalization claim) — reframe spectral component as "first shown to improve test accuracy".
**Cost is the problem:** 300–500 GPU-h grid + racing two active labs. Awkward during P2.
**GATE:** read DSLGN in full via LTU IEEE Xplore (resisted 5 open-web fetch routes).

## 7. Trainability theory — CONDITIONAL ~55%, expressivity slice only (scout 70)

**Verifier kill:** Kim 2605.08657 (read in full) proves exact gradient-cancellation at uniform
init + O(0.72^L) depth attenuation with a working fix — the gradient/trainability half is his.
**Survives:** "why (and when) random fixed fan-in-2 wiring works" — an embedding theorem (random
wired LGN contains any s-gate circuit whp at width/depth overhead X; matching/percolation proof,
pass-through gates = free routing) + Ω(n log n) input-coverage lower bounds, explaining
wide-and-shallow and IWP App-B.5's open flag. Zero-GPU, theorem-led; must position against
Functional Percolation 2512.09317 (owns the framing ring, no theorems, no training) and WiSARD
exact-VC (Neural Computation 2019). _Memory + this doc updated: earlier same-day memory entry
said ~70% — corrected to ~55%, expressivity-only._

## 8. Logic-attention — killed as worded; thin slice (scout COND 70)

**Verifier kill: BiHDTrans (arXiv:2509.24425, Sep 2025, read in full)** — trained, fully-binary-
at-inference, FPGA-deployed attention: learned binary Q/K/V binding, XNOR+popcount similarity,
hard-threshold masks (no softmax), majority-gate value bundling, benchmarked on 7 UEA datasets.
Also COBRA (2504.16269) replaces softmax with hard thresholds; HAD (2502.01770) trains Hamming
attention. Tsinghua's spec-only Boolean-attention paper (2511.17550) remains 0-citation/no
experiments. **Survives only:** attention *inside DLGNs proper* (Q/K/V as learned gate circuits
trained by gate relaxation, not STE) with discretization-gap analysis — paradigm-internal,
post-P2 at best, racing the spec-flag authors. Park.

## 9–10. DEAD: input encoding (NO-GO 95%) & output heads (NO-GO 93%)

- **Input encoding:** WARP-LNN (2602.03527) ships learnable thresholding with fixed-vs-learned
  ablations + bit-budget curves; Light DLGN App B.5 already falsified the naive ceiling story;
  and **BitLogic v2 killed the decomposition-analysis fallback** (five-axis study concludes
  fan-in, not encoder, explains accuracy differences). Only per-feature bit-allocation survives
  as a letter-sized sliver = Mecik&Kumm's declared future work. Drop.
- **Output heads:** LogicIR (2606.26609, ECCV 2026, 2026-06-25!) ships the differentiable
  bit-decoding layer for dense continuous outputs; GIC-DLC averages bits; CICADA regresses
  scores on FPGA; BEL/RLEL (ICLR'22/'23) own coded-regression-outputs generally; RTM owns
  Boolean regression. Many-class fix = ETH's named future work with BitLogic head ablations
  already published. Drop; fold a pure-logic actuator head into P3b as a section, citing these.

---

## New must-cite papers surfaced (add to 02_papers.md)

BitLogic 2602.07400 (TMLR'26, ETH — unified design space, v2 2026-07-07); ISTA interconnect
2507.02585 (SAT-pruning of trained nets); LogicIR 2606.26609 (ECCV'26); WARP-LNN 2602.03527;
WARP-LUTs 2510.15655; Kim 2605.08657 + DSLGN (IEEE Access 2023); CompactLogic 2602.05830;
Silicon Aware NN 2604.19334; TTnet (IJCAI 2024); Jia&Rinard NeurIPS 2020; PST 2603.00302;
Tsetlin-UQ 2507.04175; BiHDTrans 2509.24425; Miyasaka FCCM 2024; ReducedLUT 2412.18579;
Glette&Kaufmann AHS 2014; 2509.25933 (MNIST→ImageNet); Vieira 2509.22267 (bearing eval crisis,
for doc 22).
