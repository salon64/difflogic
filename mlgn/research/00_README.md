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
