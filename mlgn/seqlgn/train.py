"""
train.py — train & evaluate a recurrent Logic Gate Network on a sequential task.
================================================================================

Run from the project root (the folder that contains the ``difflogic/`` package):

    # gated cell (Paper #1) on permuted sequential MNIST
    python -m mlgn.seqlgn.train --task psmnist --mechanism gated --hidden 2000 --iters 50000

    # the RDDLGN-design CONTROL, same everything else
    python -m mlgn.seqlgn.train --task psmnist --mechanism rddlgn --hidden 2000 --iters 50000

    # a fast CPU smoke test (python implementation; small + short)
    python -m mlgn.seqlgn.train --task parity --seq-len 16 --mechanism gated \
        --hidden 100 --iters 1000 --eval-freq 200 --batch-size 64 --device cpu

The experiment that matters for Paper #1 is: hold everything fixed and flip
``--mechanism`` between ``rddlgn`` (control) and ``gated``, on a memory-stressing task
(``psmnist``, ``parity``/``copy`` with large ``--seq-len``). Add ``--grad-analysis`` to
measure the gradient norm reaching each timestep (the constant-error-carousel evidence).
See ``docs/experiments.md`` for the full protocol.

Eval is ALWAYS discrete: ``model.eval()`` locks each gate to its argmax and inputs are
binarised via ``.round()``, so the reported accuracy is the real logic-circuit accuracy,
not the soft relaxation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

# --- make 'difflogic' (project root) and the seqlgn package importable either way ------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if __package__ in (None, ""):  # run as a script: `python mlgn/seqlgn/train.py`
    from mlgn.seqlgn.models import SequenceClassifier
    from mlgn.seqlgn.data import get_task, AVAILABLE_TASKS
    from mlgn.seqlgn import utils
else:                          # run as a module: `python -m mlgn.seqlgn.train`
    from .models import SequenceClassifier
    from .data import get_task, AVAILABLE_TASKS
    from . import utils

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Discrete-locked accuracy: eval mode (argmax gates) + binarised inputs."""
    was_training = model.training
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device).round()
        y = y.to(device)
        preds = model(x).argmax(-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    model.train(was_training)
    return correct / max(total, 1)


def build_args():
    p = argparse.ArgumentParser(description="Train a recurrent Logic Gate Network.")
    p.add_argument("--task", default="psmnist", choices=AVAILABLE_TASKS)
    p.add_argument("--mechanism", default="gated", choices=("rddlgn", "gated"),
                   help="memory mechanism: 'rddlgn' control vs 'gated' (Paper #1). "
                        "'latch' (Paper #2) is parked.")
    p.add_argument("--hidden", type=int, default=2000, help="hidden_dim (must be >= input_dim and divisible by num_classes)")
    p.add_argument("--cell-layers", type=int, default=2, help="logic layers per candidate/gate/update network")
    p.add_argument("--tau", type=float, default=30.0, help="GroupSum temperature")
    p.add_argument("--grad-factor", type=float, default=1.0, help="difflogic grad_factor (raise for deep/long unrolls)")
    p.add_argument("--seq-len", type=int, default=None, help="sequence length for synthetic tasks (parity/copy)")
    p.add_argument("--alphabet", type=int, default=8, help="alphabet size for the copy task")

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=2_000)
    p.add_argument("--device", default=None, help="'cuda' / 'cpu' (default: auto)")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--grad-analysis", action="store_true", help="measure gradient norm through time at the end")
    p.add_argument("--show-gates", action="store_true", help="print the learned gate distribution at the end")
    p.add_argument("--tag", default="", help="optional tag added to the results filename")
    return p.parse_args()


def main():
    args = build_args()
    device = utils.get_device(args.device)
    utils.set_seed(args.seed)

    if device == "cpu":
        print("[warn] running on CPU with the slow 'python' difflogic implementation. "
              "Use a small --hidden and short --iters, or run on CUDA.")

    task = get_task(
        args.task, batch_size=args.batch_size, seq_len=args.seq_len,
        alphabet=args.alphabet, seed=args.seed,
    )
    print(f"task={task.name}  seq_len={task.seq_len}  input_dim={task.input_dim}  "
          f"num_classes={task.num_classes}")

    model = SequenceClassifier(
        input_dim=task.input_dim,
        hidden_dim=args.hidden,
        num_classes=task.num_classes,
        mechanism=args.mechanism,
        cell_layers=args.cell_layers,
        tau=args.tau,
        device=device,
        grad_factor=args.grad_factor,
    ).to(device)

    print(model)
    print(f"mechanism={args.mechanism}  logic gates={utils.count_gates(model):,}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state = 0.0, None
    t0 = time.time()
    model.train()
    for i, (x, y) in enumerate(cycle(task.train_loader)):
        if i >= args.iters:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.eval_freq == 0:
            val = evaluate(model, task.val_loader, device)
            print(f"[{i + 1:>7}/{args.iters}] loss={loss.item():.4f}  val={val:.4f}  "
                  f"({(time.time() - t0) / 60:.1f} min)")
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---- final test on the best checkpoint (discrete-locked) -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, task.test_loader, device)
    train_minutes = (time.time() - t0) / 60
    print("\n--- final (discrete locked gates) ---")
    print(f"best_val={best_val:.4f}  test={test_acc:.4f}  train_time={train_minutes:.1f} min")

    # ---- optional analyses -----------------------------------------------------------
    grad_profile = None
    if args.grad_analysis:
        xb, yb = next(iter(task.val_loader))
        grad_profile = utils.grad_norm_through_time(model, xb.to(device), yb.to(device), loss_fn)
        finite = [g for g in grad_profile if g == g]  # drop nan
        if finite:
            print("\n--- gradient norm through time (dL/dh_t) ---")
            print(f"  t=0 (earliest): {grad_profile[0]:.3e}   t=-1 (latest): {grad_profile[-1]:.3e}")
            ratio = grad_profile[0] / grad_profile[-1] if grad_profile[-1] else float("inf")
            print(f"  earliest/latest ratio: {ratio:.3e}   (closer to 1 = flatter = better flow)")

    if args.show_gates:
        print("\n--- learned gate distribution ---")
        print(utils.format_gate_distribution(utils.gate_distribution(model)))

    # ---- persist a results record ----------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(RESULTS_DIR, f"{task.name}_{args.mechanism}{tag}_{stamp}.json")
    record = {
        "task": task.name, "mechanism": args.mechanism, "seq_len": task.seq_len,
        "hidden": args.hidden, "cell_layers": args.cell_layers, "tau": args.tau,
        "grad_factor": args.grad_factor, "lr": args.lr, "batch_size": args.batch_size,
        "iters": args.iters, "seed": args.seed, "device": device,
        "logic_gates": utils.count_gates(model),
        "best_val": best_val, "test_acc": test_acc, "train_minutes": train_minutes,
        "grad_profile": grad_profile,
    }
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nresults -> {out}")
    print(f"LOG-LINE: task={task.name} mech={args.mechanism} hidden={args.hidden} "
          f"seq_len={task.seq_len} | val={best_val:.4f} test={test_acc:.4f} "
          f"gates={utils.count_gates(model):,} time={train_minutes:.1f}min")


if __name__ == "__main__":
    main()
