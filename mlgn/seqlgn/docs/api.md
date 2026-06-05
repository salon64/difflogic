# difflogic API notes & gotchas

What I learned reading [`../../../difflogic/difflogic.py`](../../../difflogic/difflogic.py)
while building this. Reference for extending the cell.

## `LogicLayer(in_dim, out_dim, device='cuda', grad_factor=1., implementation=None, connections='random')`
- `weights`: `Parameter[out_dim, 16]` — the gate logits (one categorical per neuron over
  the 16 two-input Boolean gates).
- **train**: output = softmax-weighted mixture of the 16 gate surrogates, in `[0,1]`.
  **eval**: output = argmax gate applied hard (`one_hot(weights.argmax(-1))`).
- `implementation`: `'cuda'` (fast kernels, **cuda device only**) or `'python'`
  (pure torch, works on CPU/cuda, ~50–100× slower). Auto: cuda→cuda, cpu→python.
- **Input must be 2D** `[batch, in_dim]` for the cuda path (`forward_cuda` asserts
  `ndim==2`). We therefore step one timestep at a time.
- `grad_factor`: multiplies the gradient (a `torch.autograd.Function`); raise for deep/
  long nets to fight vanishing gradients.

### Connectivity constraint (bites the recurrent cell)
`get_connections` asserts **`out_dim * 2 >= in_dim`** — each output neuron picks 2 inputs,
so you need at least `in_dim/2` neurons to (potentially) use every input. For a cell whose
first layer maps `[x;h]` (`in = input_dim + hidden_dim`) to `hidden_dim`:
```
hidden_dim * 2 >= input_dim + hidden_dim   ⇒   hidden_dim >= input_dim
```
`LogicMLP` asserts this with a helpful message.

## `GroupSum(k, tau, device='cuda')`
- Partitions the last-layer output into `k` groups, sums each, divides by `tau`.
- Requires **`hidden_dim % k == 0`**. `SequenceClassifier` asserts it.
- Output range ≈ `[0, hidden_dim / k / tau]` → feed to cross-entropy.

## Inference helpers (NOT used here, CUDA-only)
- `PackBitsTensor` — fast packed-bit GPU inference; `import difflogic_cuda` at module top.
- `CompiledLogicNet` — compiles a trained net to C/`.so` for fast CPU inference.
Both are eval-time accelerators irrelevant to training; we don't use them in seqlgn.

## CPU-only environments (the `difflogic_cuda` import wall)
`difflogic.py` and `packbitstensor.py` both do `import difflogic_cuda` at module top, so
**the package won't import at all without the compiled CUDA extension** — even the Python
CPU path. [`../_cpu_compat.py`](../_cpu_compat.py) injects a stub `difflogic_cuda` into
`sys.modules` (only if the real one is missing) so you can develop on CPU. Its functions
raise a clear error if a CUDA-only path is actually hit. The seqlgn package `__init__`
calls it automatically before importing difflogic.

## Local fork change
We removed a dead debug block in `LogicLayer.forward_python`
([`difflogic.py` ~L97](../../../difflogic/difflogic.py#L94)) that printed index dtypes on
**every** forward call, flooding stdout on the CPU path. `indices` are already int64
(required for advanced indexing), so the block was a no-op besides the prints. Flagged
here so it's easy to find/revert if syncing with upstream.

## The 16 gate ids (from `functional.py`)
`0 FALSE, 1 AND, 2 A∧¬B, 3 A, 4 ¬A∧B, 5 B, 6 XOR, 7 OR, 8 NOR, 9 XNOR, 10 ¬B, 11 A∨¬B,
12 ¬A, 13 ¬A∨B, 14 NAND, 15 TRUE`. `utils.GATE_NAMES` mirrors this; `weights.argmax(-1)`
gives the chosen gate per neuron.
