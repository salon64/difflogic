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
