# Cross-paper results tables

Numbers gathered during the 2026-06-04 scout. Cross-paper comparisons are **rough** —
preprocessing (thermometer thresholds), gate counting, and "gates vs ops" conventions
differ. Treat as orientation, verify before quoting.

## MNIST (accuracy @ gate budget)

| Model | Acc | Gates | Source |
|---|---|---|---|
| Deep DLGN (small) | ~97.7% | ~48k | [1] / LILogic table |
| Deep DLGN (wide) | ~99% | large | [1] |
| LILogicNet-S | 97.96% | 4k | [7] |
| LILogicNet-M | 98.45% | 8k | [7] |
| LILogicNet-L | 98.95% | 32k | [7] |
| LogicTreeNet-M (conv) | 99.23% | 566k | [7] citing [2] |
| **My `mnist_test.py` (6×64k FC)** | **98.24% test** | — | this fork |
| **My `secuential.py` (LogicRNN)** | **98.04% val / 98.2%-ish** | — | this fork |

## CIFAR-10 (accuracy @ gate budget) — the field's main yardstick

| Model | Acc | Gates | Notes | Source |
|---|---|---|---|---|
| Deep DLGN (largest) | 62.14% | 5.12M | fully-connected | [2] |
| LILogicNet-L | 60.98% | 256k | learned Top-K conn. | [7] |
| LogicTreeNet-S | 60.38% | 400k | conv | [2]/[7] |
| LogicTreeNet-B | 80.17% | 16.0M | conv | [2] |
| LogicTreeNet-L | 84.99% | 28.9M | conv | [2] |
| **LogicTreeNet-G** | **86.29%** | **61M** | **conv, SOTA** | [2] |
| XNOR-Net (BNN baseline) | 86.28% | 1,780M | 29× more gates | [2] |
| FINN CNV (BNN baseline) | 80.10% | 901M | 56× more gates | [2] |
| LUTNet (baseline) | 84.95% | 1,290M | 44.6× more gates | [2] |

Takeaways:
- **Convolution (LogicTreeNet) is the only thing that cracks CIFAR-10 well** (>80%).
- **Learned connectivity (LILogic) wins the gate-efficiency-per-accuracy race** at small
  budgets (60% CIFAR-10 in 256k gates vs 5.12M for FC DLGN).
- Plain fully-connected DLGN tops out ~62% on CIFAR-10 regardless of size.

## Inference speed (headline hardware numbers)

| Model | Platform | Latency / throughput | Source |
|---|---|---|---|
| Deep DLGN (MNIST) | 1 CPU core | >1M images/s | [1] |
| LogicTreeNet-M | FPGA | 9 ns/image | [2] |
| LogicTreeNet-B | FPGA | 24 ns/image, ~41.6M FPS | [2] |
| DWC policy | FPGA | single clock cycle, nJ/action | [12] |

## Training-cost / quality knobs introduced by follow-ups

| Lever | Effect | Source |
|---|---|---|
| Residual init (favor pass-through "A") | fixes deep-net gradient decay | [2],[4] |
| Gumbel noise + STE | 4.5× faster, gap −98%, 0% unused gates | [3] |
| Input-Wise Parametrization (4-basis) | 4× smaller, 1.86× backward, 8.5× fewer steps | [4] |
| 4-basis projection gate eval | 3.4–4.0× training speedup | [7] |
| Top-K learnable connectivity | orders-of-magnitude fewer gates @ acc | [6],[7] |
| `grad_factor` (orig knob) | counters vanishing grads in deep nets | [1] |

Bracket numbers = entry IDs in [02_papers.md](02_papers.md).
