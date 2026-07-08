# LGN Research Knowledge Base

This folder is the living knowledge base for my Logic Gate Network (LGN) research,
built on top of the `difflogic` fork. It is the result of a deep scout of the whole
field (done 2026-06-04) plus an ongoing experiment log.

## Files

| File | Purpose |
|------|---------|
| [01_landscape.md](01_landscape.md) | The field map: what LGNs are, the method primer, the research lineages, who the players are, and a timeline. Start here. |
| [02_papers.md](02_papers.md) | Annotated bibliography. One detailed entry per paper (contribution, method, numbers, limitations). |
| [03_open_problems.md](03_open_problems.md) | Gaps, open problems, and candidate research directions — with emphasis on the recurrent/sequential angle I'm pursuing. |
| [04_experiment_log.md](04_experiment_log.md) | Running log of my own experiments, results, and decisions. |
| [05_my_angles.md](05_my_angles.md) | My research angles (#1 gating, #2 latch, #3 Fourier, #4 CAM, #5 skip) with adversarial novelty-scout verdicts + a status board. |
| [06_paper_plan.md](06_paper_plan.md) | The 3-paper roadmap (P1 gating / **P2 latch anchor** / P3 FPGA), the 4-way comparison, and the shared infra. |
| [07_venues_timeline.md](07_venues_timeline.md) | Venue / deadline planning and honest paper-tier calls. |
| [08_paper1_checklist.md](08_paper1_checklist.md) | Paper 1 (gating) remaining-experiments → submission checklist (NeurIPS 2026 workshop). |
| [08_vsa_crosspollination_scout.md](08_vsa_crosspollination_scout.md) | Novelty scout on VSA/HDC × LGN (permutation-as-state → fold into P2; the rest parked/no-go). |
| [09_training_speed_scout.md](09_training_speed_scout.md) | Scout on "faster LGN training" — NO-GO/saturated; parallel-scan survives only as a P2 long-sequence enabler. |
| [10_fpga_scout.md](10_fpga_scout.md) | Scout on FPGA realization — feedforward = commodity; sequential/stateful = open, P2-coupled (D-FF demo → P2). |
| [11_paper2_workmap.md](11_paper2_workmap.md) | **Paper 2 work map (latch/flip-flop primitives): thesis, build, benchmarks, math, gate set, build order. The current active plan.** |
| [12_reading_sequential_memory.md](12_reading_sequential_memory.md) | Curated non-paper reading list on memory in sequential systems (digital logic, HDL, RNN memory, fixed-point/attractor, automata). |
| [13_snn_hebbian_scout.md](13_snn_hebbian_scout.md) | Scout on SNN / spiking / Hebbian ideas × LGN — **NO-GO as a direction** (all relabels/owned/W4-mismatch); real payoff = a **red-team hardening of P2's C2** (non-uniqueness > singularity, +4 cites) + 3 must-check 2026 papers (2603.14157, 2605.24649, 2605.08657). |
| [14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md) | **Full-text read** of the 3 papers from doc 13, claim-by-claim vs P1/P2. None scoops P2's core. **2603.14157** (gap = selection+computation) is a **gift** framing C3; **2605.24649 R-DTLGN** = closest neighbor, cite+distinguish on C3; **2605.08657** narrows claim-#2's novel surface to the characteristic-equation reduction. Action items for P1/P2. |
| [15_rl_lgn_scout.md](15_rl_lgn_scout.md) | Scout on **RL × LGN** — **NO-GO as a headline both ways** (train-LGN-by-RL killed by a pathwise-gradient theorem + Mind-the-Gap/Decoupled-STE; LGN-as-RL-policy occupied by **DWC 2512.01467**). Real payoff: a **memory-required POMDP "logic-DRQN" capstone demo for P2** (+ execute-a-model-check verification demo) + 3 must-cite papers (DWC, Petersen-2024a CoRL, CompactLogic 2602.05830). |
| [17_concepts_and_journey.md](17_concepts_and_journey.md) | **Plain-language explainer**: the two architecture levels (16-gate LogicLayer vs cell-level mechanism — latches are NOT in the gate pool), what "clocked" means (the recurrence IS the clock), how SR-latch instability is sidestepped (characteristic equation), and the reframe journey (bistable-restore collapse → gated-already-binary insight → the `clatch` input-clocked latch). Read to understand what we built. |
| [16_crosspollination_and_robotics.md](16_crosspollination_and_robotics.md) | **AI cross-pollination map** (no standalone trade — SSM-scan/CAM/GNN/diffusion/KAN/neurosymbolic all theorem- or occupancy-dead; only **sequential formal VERIFICATION** + robustness tooling flow in and harden P2) **+ Drones/Kyushu = CONDITIONAL-GO** (physical home of the P2 capstone; recurrent-memory + verified-safety, gated on latch training). **⚠ Strategic shift: ISTA (Kresse/Lampert/Henzinger) is now P2's #1 competitor** (owns DWC + feedforward LGN-verification 2505.19932 + connectivity). |
| [19_learnable_latch_scout.md](19_learnable_latch_scout.md) | **Novelty+paper-worthiness scout** on "learnable latches as trainable params" (per-neuron learned memory-type LGN). Verdict: **CONDITIONAL, park as P4** — gap real but MECHANISM occupied (DARTS + MuFuRU'16), accuracy ties (gated≈clatch), only carrier = a synthesis FF/LUT Pareto gated on P3b tooling (or a workshop interpretability paper). NOT a P2 section. |
| [18_sequential_benchmarks.md](18_sequential_benchmarks.md) | **Benchmark hunt (61 candidates, 6 families) for a task that SEPARATES clatch from the gated foil** — copy-50 saturates (all routes hit 1.000), so we need a discriminator. Recommended suite: **Selective Copying** (Mamba; headline-separator, ~20-line `_make_copy` extension) + **Parity** (Delétang ICLR'23; already coded, credibility + length-gen + T-FF verification capstone). Full ranked table + rejected list + the sweep design (kill the write-flag; discrete-acc-vs-gap-length figure). |
| [21_landscape_weakness_scout.md](21_landscape_weakness_scout.md) | **Field-weakness scout beyond P1–P4 (2026-07-08, 27-agent, adversarially verified).** Post-verification ranking: UQ/calibration GO ~65% (gate-posterior = free circuit ensembles; 1-afternoon gate), netlist-plasticity GO ~60%, robustness/gate-vocabulary/resynthesis/theory CONDITIONAL ~55–60%; input-encoding & output-heads NO-GO (WARP, BitLogic v2, LogicIR ECCV'26). Key find: ETH BitLogic + ISTA are industrializing the hardware axes — the ML-culture axes (calibration, adaptation, robustness) are what's left open. |
| [22_applications_scout.md](22_applications_scout.md) | **Application scout for the sequential-LGN stack (2026-07-08, verified).** CAN-bus IDS = GO ~72% flagship (stateful + ISO-21434 verification wedge, no logic-substrate incumbent); bearings/SKF CONDITIONAL ~65% (SKF–LTU UTC confirmed real; gate = bearing-wise-split spike); HFT CONDITIONAL ~65% academic / NO-GO commercial (kill-check clean; model is the 40–100× latency outlier vs 2µs VOLLO frontier; gate = FI-2010 run); NID = P2 motivation vignette. Feedforward flags already planted in KWS/ECG/trigger/HAR — the moat is {recurrent + registers + verified}. |
| [20_program_validation.md](20_program_validation.md) | **Full-program validation report (2026-07-08, 81-agent adversarially-verified audit).** All P1/P2 numbers reproduce exactly from the JSONs; BUT: the "matched" selcopy-L100 stability pair is kb-confounded, the 4.4× gap edge is partly by-construction, distcopy tie is n=2, beat-4 "verifiable" has zero artifacts, P3a/P3b share an unbuilt+unscheduled netlist/RTL exporter (dependency inversion), P4's accuracy-kill is an extrapolation. Race check: all novelty windows still open. Ranked fix list at the end. |
| [results_table.md](results_table.md) | Cross-paper accuracy / gate-count comparison tables for MNIST and CIFAR-10. |

## The one-paragraph state of the field (2026-06)

Logic Gate Networks learn a network of 2-input Boolean gates (AND, XOR, NAND, …)
directly by gradient descent, via a differentiable relaxation (Petersen et al.,
NeurIPS 2022). The discretized network runs as raw logic gates — extremely fast and
cheap on CPU/FPGA/ASIC (CIFAR-10 in <10 ns on FPGA). The field is small (~12 core
papers + an adjacent LUT/weightless cluster). The dominant active group is **ETH
Zürich (Wattenhofer lab)**, which is attacking the three core pain points —
**discretization gap**, **training cost / vanishing gradients**, and **fixed random
connectivity** — and has already published the **first recurrent LGN**
(translation). The original author **Felix Petersen** added the convolutional variant
(NeurIPS 2024 Oral). A separate **Princeton (Yue & Jha)** line pushes interpretable
LGNs for tabular/time-series, and an **adjacent UT-Austin "weightless / LUT" cluster**
(DWN) tackles the same hardware-efficiency goal with lookup tables instead of gates.

## My angle (why this fork exists)

I'm exploring **recurrent / sequential logic gate networks** (see
[../secuential.py](../secuential.py), a `LogicRNNCell` feeding MNIST rows over 28
timesteps). The closest prior art is the ETH **Recurrent DDLGN** paper (translation).
The gap I'm circling is in [03_open_problems.md](03_open_problems.md).

**Current work (2026-07):** Paper 1 (logic-native gating) experiments are essentially
done; **Paper 2 (latch / flip-flop primitives) is the active anchor** — full standing plan
in [11_paper2_workmap.md](11_paper2_workmap.md), with its non-paper reading list in
[12_reading_sequential_memory.md](12_reading_sequential_memory.md).
