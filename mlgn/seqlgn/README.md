# seqlgn — Sequential / Recurrent Logic Gate Networks

Research infrastructure for the recurrent-LGN program (built on the `difflogic` fork).
Shared substrate for **Paper #1 (logic-native gating)** and **Paper #2 (latch primitives,
parked)**. See the research knowledge base in [`../research/`](../research/) for the
landscape, novelty scout, and the 3-paper plan.

## The core idea

A recurrent cell where the memory mechanism is **pluggable**, so every experiment differs
in exactly one variable — how state is carried across time:

| `--mechanism` | What it does                                                                                          | Role                                            |
| ------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `rddlgn`      | concat-recurrence, state **recomputed** each step: `h' = LogicMLP([x;h])`                             | **control** (Bührer et al. 2025 / DiffLogic CA) |
| `gated`       | logic-native MUX gate (GRU-style): `h' = s*h + (1-s)*c` (keep vs write per bit)                       | **Paper #1 primary** (2 LGNs, 1 state)          |
| `lstm`        | dedicated cell state `C` + forget/input/output: `C' = (C AND f) OR (i AND C̃)`, `h' = readout([o;C'])` | **Paper #1 richer arm** (5 LGNs, 2 states)      |
| `latch`       | bistable memory primitive (D-FF / latch) with custom STE                                              | Paper #2 (parked → `NotImplementedError`)       |

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

## Setting up a new machine (GPU build) — step by step

This is the full recipe for getting the CUDA path working from scratch, written from a
real Windows + RTX 2080 SUPER setup. The hard part is **compiling the `difflogic_cuda`
extension** — `pip install difflogic` does _not_ do it (it only installs the pure-Python
package; the fused kernels must be compiled locally).

> Known-good reference config: Windows 10, Python 3.13, `torch==2.6.0+cu124`,
> CUDA Toolkit 12.x, Visual Studio 2022 Build Tools, GPU compute capability 7.5 (sm_75).
> Linux is easier (gcc instead of MSVC) — skip the Visual Studio step and run the build
> from a normal shell.

### 1. Prerequisites

| Need                                   | What / why                                                                                                                                             | Check                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **NVIDIA GPU + driver**                | the actual hardware                                                                                                                                    | `nvidia-smi`                                                                    |
| **CUDA Toolkit (`nvcc`)**              | compiles the `.cu` kernels. Its **major** version must match the CUDA your torch was built with (minor mismatch is fine — just a warning on torch 2.x) | `nvcc --version`                                                                |
| **PyTorch (CUDA build)**               | e.g. `torch==2.6.0+cu124`. The `+cuXXX` tag must match the toolkit's major version                                                                     | `python -c "import torch;print(torch.version.cuda, torch.cuda.is_available())"` |
| **MSVC C++ compiler** _(Windows only)_ | `nvcc` needs `cl.exe` as host compiler. Install **Visual Studio Build Tools → "Desktop development with C++"**                                         | open _"x64 Native Tools Command Prompt for VS 2022"_                            |

### 2. Python env + difflogic package

```bash
python -m venv .venv
# install a torch wheel whose CUDA matches your toolkit's major version, e.g. cu124:
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -e .          # installs the difflogic package (editable, from repo root)
```

### 3. Compile the CUDA extension

On **Windows**, this must run from a **"x64 Native Tools Command Prompt for VS 2022"** —
that's the only shell where both `cl.exe` and `nvcc` are on `PATH`:

```bat
cd <repo-root>
.\.venv\Scripts\python.exe setup.py build_ext --inplace
```

Harmless warnings you can ignore: _ninja not found_, _CUDA 12.8 vs 12.4 minor mismatch_,
_"declared but never referenced"_. Success = a `difflogic_cuda.cp3XX-win_amd64.pyd`
appears at the repo root with no error at the end.

### 4. Verify (run from any shell — torch must be imported first!)

```bash
.venv/Scripts/python -c "import torch; import difflogic_cuda; print('real kernels:', difflogic_cuda.__file__)"
.venv/Scripts/python -m mlgn.seqlgn.train --task parity --seq-len 8 --mechanism gated \
    --hidden 32 --iters 50 --eval-freq 25 --batch-size 16
```

If the first line prints the `.pyd` path (not a stub) and the run finishes without a
`RuntimeError`, the GPU path is live and `--device cuda` (the default) works.

### Troubleshooting (everything that bit us)

- **`ModuleNotFoundError: No module named 'mlgn'`** — the package dir must be lowercase
  `mlgn/` and you must invoke it lowercase. Windows' filesystem is case-insensitive but
  Python imports are case-sensitive.
- **`pip install difflogic` says "already satisfied" but nothing changed** — that only
  confirms the _Python_ package. The compiled `difflogic_cuda` is separate; you must run
  the `build_ext` step above.
- **Build fails: _"Microsoft Visual C++ 14.0 or greater is required"_ / `cl.exe` not
  found** — you're not in the VS Native Tools prompt, or the C++ workload isn't installed.
- **Build fails with `nvcc` errors** about `.type()` → `c10::ScalarType` or a `{.member =}`
  designated initializer — stale upstream kernel vs newer PyTorch. Already patched in this
  fork ([`difflogic/cuda/difflogic_kernel.cu`](../../difflogic/cuda/difflogic_kernel.cu):
  `.type()`→`.scalar_type()`, and the union init rewritten for C++17).
- **Runtime: `RuntimeError: difflogic_cuda ... is not installed` even though the `.pyd`
  exists** — the real extension failed its DLL load and the CPU stub got injected. On
  Windows the `.pyd` needs torch's CUDA DLLs, which are only on the search path _after_
  `import torch`. [`_cpu_compat.py`](_cpu_compat.py) now imports torch before probing for
  the extension, which fixes this. If it recurs, confirm `import torch; import
difflogic_cuda` works standalone — that isolates a DLL/ABI problem from an import-order
  one.
- **Build artifacts** — `build/` and the `*.pyd` at the repo root are generated; add them
  to `.gitignore`.

## Quickstart

Run from the **repo root** (the folder containing `difflogic/`):

```bash
# CPU smoke test (a few seconds) — just checks the pipeline
python -m mlgn.seqlgn.train --task parity --seq-len 8 --mechanism gated \
    --hidden 20 --iters 20 --eval-freq 10 --batch-size 16 --device cpu
```

### Fast validation (single GPU, ~30 min/run) — DO THIS FIRST

Validate the idea cheaply on the `copy` task (chance = 12.5% for `--alphabet 8`). Short
delay = sanity (both learn it); long delay = the test (expect `gated` ≫ `rddlgn`):

```bash
# sanity (short delay): both should beat chance
python -m mlgn.seqlgn.train --task copy --seq-len 8  --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism rddlgn --tag sanity
python -m mlgn.seqlgn.train --task copy --seq-len 8  --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism gated  --tag sanity
# validation (long delay): gated should hold, rddlgn should collapse toward 12.5%
python -m mlgn.seqlgn.train --task copy --seq-len 50 --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism rddlgn --grad-analysis --tag val
python -m mlgn.seqlgn.train --task copy --seq-len 50 --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism gated  --grad-analysis --tag val
```

The first eval line prints elapsed time → you learn your real rate in ~2 min. See
[docs/experiments.md](docs/experiments.md) for success criteria and the full protocol.

### Cold-start fix re-run (run these after the above)

The first seq-50 run (2026-06-08) showed `gated` **cold-starting** — loss stuck flat at
`log(8)=2.08`, val at chance — because the gate wasn't keep-biased. `--keep-bias` (logic
forget-bias / residual init) turns the carousel ON at init. Re-run with it, vs a _fair_
control (`--grad-factor 2`, difflogic's anti-vanishing knob):

```bash
# gated WITH keep-bias — expect the seq-50 plateau to break (loss < 2.08, val > 12.5%)
python -m mlgn.seqlgn.train --task copy --seq-len 50 --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism gated  --keep-bias 3 --grad-analysis --tag keepbias
# fair control
python -m mlgn.seqlgn.train --task copy --seq-len 50 --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism rddlgn --grad-factor 2 --grad-analysis --tag fair
```

If `gated` still plateaus, sweep `--keep-bias 2` and `5` (too high saturates `s→1` and
kills the write path). Success = `gated` ≫ chance while `rddlgn` stays near 12.5%.

### follow up experiment

```bash
python -m mlgn.seqlgn.train --task copy --seq-len 20 --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 3 --grad-analysis --tag L20
python -m mlgn.seqlgn.train --task copy --seq-len 35 --hidden 1024 --iters 20000 --eval-freq 1000 --mechanism gated --keep-bias 3 --grad-analysis --tag L35
# is seq-50 just under-trained? give it more iters:
python -m mlgn.seqlgn.train --task copy --seq-len 50 --hidden 1024 --iters 50000 --eval-freq 2000 --mechanism gated --keep-bias 3 --tag L50long
```

### Full-scale runs (⚠️ GPU-HOURS — for the paper, not a casual check)

> Pixel-level sequential MNIST is **784 timesteps**; an RNN can't parallelise across time,
> so these take **~20–40 h** each. Don't run them to "try it out" — use the fast validation
> above first.

```bash
python -m mlgn.seqlgn.train --task psmnist --mechanism gated  --hidden 2000 --iters 50000   # ~40h
python -m mlgn.seqlgn.train --task psmnist --mechanism rddlgn --hidden 2000 --iters 50000   # ~20h
```

Key flags: `--task`, `--mechanism {rddlgn,gated}`, `--hidden`, `--cell-layers`, `--tau`,
`--grad-factor`, `--seq-len` (synthetic), `--iters`, `--grad-analysis`, `--show-gates`.
Full list: `python -m mlgn.seqlgn.train -h`.

## Constraints to remember (from difflogic)

- `hidden_dim >= input_dim` (difflogic needs `out_dim*2 >= in_dim`).
- `hidden_dim % num_classes == 0` (GroupSum partitions output bits into classes).
- Eval is always discrete: `model.eval()` (argmax gates) + inputs `.round()`.

## Status (2026-06-08)

- ✅ Pipeline built; `rddlgn` / `gated` / `lstm` all run (CPU dev + GPU).
- ✅ **First GPU validation (copy task, RTX 2080S):** at seq-8 `gated` hits 100% instantly
  vs `rddlgn`'s struggling 87%; gated gradient flow ~1e12× the control. At seq-50 **both
  failed** — `gated` cold-starts (gate not keep-biased). See
  [docs/experiments.md](docs/experiments.md) + `../research/04_experiment_log.md`.
- ✅ **Fix added:** `--keep-bias` (logic forget-bias / residual init). Re-run pending.
- ⏳ Re-run copy-50 `gated --keep-bias 3` (expect plateau to break); then equal-gates,
  ≥3 seeds; add `lstm` arm.
- ⏸ `latch` mechanism (Paper #2) stubbed — parked by decision.

See [docs/experiments.md](docs/experiments.md) for the experiment protocol and what to
report, and [../research/06_paper_plan.md](../research/06_paper_plan.md) for the plan.
