# seqlgn — Sequential / Recurrent Logic Gate Networks

Research infrastructure for the recurrent-LGN program (built on the `difflogic` fork).
Shared substrate for **Paper #1 (logic-native gating)** and **Paper #2 (latch primitives,
parked)**. See the research knowledge base in [`../research/`](../research/) for the
landscape, novelty scout, and the 3-paper plan.

## The core idea

A recurrent cell where the memory mechanism is **pluggable**, so every experiment differs
in exactly one variable — how state is carried across time:

| `--mechanism` | What it does | Role |
|---|---|---|
| `rddlgn` | concat-recurrence, state **recomputed** each step: `h' = LogicMLP([x;h])` | **control** (Bührer et al. 2025 / DiffLogic CA) |
| `gated`  | logic-native MUX gate: `h' = s*h + (1-s)*c` (keep vs write per bit) | **Paper #1** |
| `latch`  | bistable memory primitive (D-FF / latch) with custom STE | Paper #2 (parked → `NotImplementedError`) |

The `gated` "keep" branch `s*h` is a **constant-error carousel**: when a bit is kept
(`s≈1`), the gradient flows back through time un-attenuated. That's the mechanism this
whole paper is testing. See [docs/design.md](docs/design.md).

## Layout

```
seqlgn/
├── cells.py        LogicMLP, LogicRecurrentCell (the pluggable cell)
├── models.py       SequenceClassifier (unrolls the cell + GroupSum head)
├── data.py         get_task(): smnist, smnist-pixel, psmnist, parity, copy
├── train.py        CLI training/eval loop (discrete-locked eval, grad analysis)
├── utils.py        seeding, gate counting/distribution, grad-norm-through-time
├── _cpu_compat.py  lets difflogic import on a CPU-only machine (dev convenience)
├── results/        per-run JSON records (created on first run)
└── docs/           design.md, benchmarks.md, experiments.md, api.md
```

## Install / environment

Needs `torch` + the `difflogic` package at the repo root.

- **GPU machine (real runs):** install the `difflogic` CUDA extension (`difflogic_cuda`)
  per the repo root README. Use `--device cuda`.
- **CPU-only laptop (dev / debugging):** the CUDA extension is absent, so
  [`_cpu_compat.py`](_cpu_compat.py) injects a stub and you run the pure-Python
  implementation with `--device cpu`. This is **correct but slow** — keep `--hidden` and
  `--iters` small. Do real training on a GPU box/cluster.

> ⚠️ The Python/CPU path is ~50–100× slower than the CUDA kernels. CPU is for plumbing,
> debugging, and tiny sanity runs — not for the actual experiments.

## Quickstart

Run from the **repo root** (the folder containing `difflogic/`):

```bash
# CPU smoke test (a few seconds) — just checks the pipeline
python -m mlgn.seqlgn.train --task parity --seq-len 8 --mechanism gated \
    --hidden 20 --iters 20 --eval-freq 10 --batch-size 16 --device cpu

# The Paper #1 comparison (run on GPU): flip --mechanism, hold all else fixed
python -m mlgn.seqlgn.train --task psmnist --mechanism gated  --hidden 2000 --iters 50000
python -m mlgn.seqlgn.train --task psmnist --mechanism rddlgn --hidden 2000 --iters 50000

# Long-range memory + gradient-flow evidence
python -m mlgn.seqlgn.train --task parity --seq-len 128 --mechanism gated  --grad-analysis
python -m mlgn.seqlgn.train --task parity --seq-len 128 --mechanism rddlgn --grad-analysis
```

Key flags: `--task`, `--mechanism {rddlgn,gated}`, `--hidden`, `--cell-layers`, `--tau`,
`--grad-factor`, `--seq-len` (synthetic), `--iters`, `--grad-analysis`, `--show-gates`.
Full list: `python -m mlgn.seqlgn.train -h`.

## Constraints to remember (from difflogic)

- `hidden_dim >= input_dim` (difflogic needs `out_dim*2 >= in_dim`).
- `hidden_dim % num_classes == 0` (GroupSum partitions output bits into classes).
- Eval is always discrete: `model.eval()` (argmax gates) + inputs `.round()`.

## Status (2026-06-04)

- ✅ Pipeline built and smoke-tested on CPU (`rddlgn` + `gated`, grad-analysis, gates).
- ✅ Grad-analysis instrument works; control shows strong vanishing through time (expected).
- ⏳ Real comparison runs pending a GPU machine.
- ⏸ `latch` mechanism (Paper #2) stubbed — parked by decision.

See [docs/experiments.md](docs/experiments.md) for the experiment protocol and what to
report, and [../research/06_paper_plan.md](../research/06_paper_plan.md) for the plan.
