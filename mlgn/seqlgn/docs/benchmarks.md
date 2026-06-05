# Benchmarks

The sequential tasks in [`../data.py`](../data.py). All emit `[batch, seq_len, input_dim]`
in `[0,1]` + integer labels, so one training loop drives them. They are chosen to
**isolate memory**: how well does a cell carry information across time?

Get one with `get_task(name, batch_size=..., seq_len=..., alphabet=..., seed=...)`.

| name | seq_len | input_dim | classes | what it tests |
|---|---|---|---|---|
| `smnist` | 28 | 28 | 10 | warm-up; one MNIST **row** per step (as in `../secuential.py`) |
| `smnist-pixel` | 784 | 1 | 10 | long sequence; one **pixel** per step → gradient flow through time |
| `psmnist` | 784 | 1 | 10 | **permuted** pixel-MNIST (fixed perm) — the classic long-range memory benchmark |
| `parity` | `--seq-len` (def 64) | 1 | 2 | running 1-bit state: label = XOR of all bits |
| `copy` | `--seq-len` (def 64) | `alphabet+1` | `alphabet` | long-range recall: reproduce a symbol after a delay |

## Why these

- **psMNIST** is *the* standard yardstick for recurrent long-range memory. The fixed
  permutation destroys local pixel structure, so the model can't cheat with local
  features — it must integrate over all 784 steps. Strong discriminator between `rddlgn`
  and `gated`.
- **parity** is a pure 1-bit running-state task: maintain a toggling bit (XOR accumulator)
  over `L` steps. Two reasons it's ideal here: (1) difficulty scales cleanly with
  `--seq-len`, giving a controllable long-range axis; (2) **parity is exactly what a
  T flip-flop computes**, so it directly previews Paper #2 and tests "hold state" vs
  "recompute state." A net that can't keep its state will sit at chance (50%).
- **copy/recall** isolates *retention without computation*: the symbol is shown once at
  `t=0` (with a cue bit), then `L−1` blank steps; the answer is the original symbol.
  Accuracy as a function of delay `L` is a clean memory-decay curve.
- **smnist (row)** is the gentle warm-up that reproduces the `secuential.py` setting
  (~98% is reachable); not very memory-stressing, useful as a sanity baseline.

## Recommended sweep for Paper #1

- **Difficulty axis:** `parity`/`copy` with `--seq-len ∈ {16, 32, 64, 128, 256}`. Plot
  accuracy vs `L` for `rddlgn` vs `gated`. The expected story: both fine at small `L`;
  `rddlgn` collapses to chance earlier as `L` grows; `gated` holds.
- **Realistic task:** `psmnist` (and `smnist-pixel` as the unpermuted reference).
- **Gradient evidence:** add `--grad-analysis` on the long-`L` runs.

## Notes / gotchas

- Synthetic tasks are generated once with a fixed `--seed` (train/val/test from the same
  generator, disjoint draws). Default sizes: 50k/5k/10k.
- `copy` input is `one_hot(symbol) ++ cue_bit`; the cue bit is 1 only at `t=0`.
- `psmnist` uses a permutation seeded by a fixed constant (1234) so it's identical across
  runs/mechanisms — essential for a fair comparison.
- MNIST downloads to `../../data-mnist` (shared with the other mlgn scripts).

## TODO — adding problem (regression)

The classic "adding problem" (sum two marked values in a sequence) is a regression task
and needs a regression head instead of `GroupSum` + cross-entropy. Not implemented yet;
would require a small real-valued readout. Tracked as future work.
