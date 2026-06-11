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
def evaluate(model, loader, device, discrete: bool = True) -> float:
    """Accuracy. discrete=True: eval mode (argmax gates) + binarised inputs — the REAL
    logic-circuit accuracy. discrete=False: train mode (softmax gates), inputs unrounded —
    the soft/relaxed model. The difference (soft - discrete) is the *discretization gap*."""
    was_training = model.training
    model.eval() if discrete else model.train()
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        if discrete:
            x = x.round()
        y = y.to(device)
        preds = model(x).argmax(-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    model.train(was_training)
    return correct / max(total, 1)


def build_args():
    p = argparse.ArgumentParser(description="Train a recurrent Logic Gate Network.")
    p.add_argument("--task", default="psmnist", choices=AVAILABLE_TASKS)
    p.add_argument("--mechanism", default="gated", choices=("rddlgn", "gated", "lstm"),
                   help="memory mechanism: 'rddlgn' control, 'gated' (GRU-style, Paper #1), "
                        "'lstm' (richer arm). 'latch' (Paper #2) is parked.")
    p.add_argument("--hidden", type=int, default=2000, help="hidden_dim (must be >= input_dim and divisible by num_classes)")
    p.add_argument("--cell-layers", type=int, default=2, help="logic layers per candidate/gate/update network")
    p.add_argument("--keep-bias", type=float, default=3.0,
                   help="keep-bias for the update/forget gate at init (logic forget-bias / "
                        "residual init). Turns the carousel ON at start to avoid cold-start. "
                        "0 disables it. Only affects 'gated'/'lstm'.")
    p.add_argument("--tau", type=float, default=30.0, help="GroupSum temperature")
    p.add_argument("--grad-factor", type=float, default=1.0, help="difflogic grad_factor (raise for deep/long unrolls)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="clip global grad norm to this value (RNN exploding-gradient fix). "
                        "0 disables. Keep-bias fixes vanishing but can over-correct into "
                        "exploding gradients on long sequences (NaN) — clipping prevents it.")
    p.add_argument("--seq-len", type=int, default=None, help="sequence length for synthetic tasks (parity/copy)")
    p.add_argument("--alphabet", type=int, default=8, help="alphabet size for the copy task")

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-min", type=float, default=-1.0,
                   help="if >=0 and < --lr, cosine-decay the LR from --lr to this over training "
                        "(absorbs the late-phase explosion once gates sharpen). -1 = constant LR.")
    p.add_argument("--entropy-reg", type=float, default=0.0,
                   help="coefficient on a gate-entropy penalty pushing gates toward one-hot, "
                        "to shrink the discretization gap. 0 = off.")
    p.add_argument("--entropy-ramp", type=float, default=1.0,
                   help="fraction of training over which --entropy-reg ramps 0→full (explore "
                        "early, commit late). 1.0 = ramp across the whole run.")
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
        keep_bias=args.keep_bias,
        tau=args.tau,
        device=device,
        grad_factor=args.grad_factor,
    ).to(device)

    print(model)
    print(f"mechanism={args.mechanism}  logic gates={utils.count_gates(model):,}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # TODO(future): make the LR schedule a first-class, pluggable choice — an explicit
    # `--lr-schedule {none,cosine,linear,step,...}` selected independently, rather than
    # inferring "cosine decay" from the presence of `--lr-min`. Coupling the schedule to
    # lr_min is a stopgap; a settable schedule (with its own params) is cleaner.
    scheduler = None
    if 0 <= args.lr_min < args.lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iters, eta_min=args.lr_min)

    best_val, best_state = 0.0, None
    grad_norm = float("nan")
    ent_val = float("nan")
    n_skipped = 0
    skips_at_last_eval = 0
    t0 = time.time()
    model.train()
    for i, (x, y) in enumerate(cycle(task.train_loader)):
        if i >= args.iters:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)          # task (CE) loss — reported as `loss`

        # Optional gate-entropy penalty (ramped): pushes gates one-hot to shrink the gap.
        total_loss = loss
        if args.entropy_reg > 0:
            ramp = min(1.0, (i + 1) / max(1, args.entropy_ramp * args.iters))
            ent = utils.gate_entropy(model)
            ent_val = ent.item()
            total_loss = loss + (args.entropy_reg * ramp) * ent

        optimizer.zero_grad()
        total_loss.backward()
        # Measure (and optionally clip) the global grad norm; SKIP the update when it is
        # non-finite. An exploding-gradient batch then never poisons the weights, and the
        # model is kept OUT of the NaN basin rather than pushed into it. Post-hoc clipping
        # alone can't do this: once a backward overflows to inf, clipping it yields nan.
        clip = args.grad_clip if args.grad_clip > 0 else float("inf")
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        grad_norm = total.item()
        if torch.isfinite(total):
            optimizer.step()
        else:
            n_skipped += 1
        if scheduler is not None:
            scheduler.step()

        if (i + 1) % args.eval_freq == 0:
            val = evaluate(model, task.val_loader, device, discrete=True)
            val_soft = evaluate(model, task.val_loader, device, discrete=False)
            window_skips = n_skipped - skips_at_last_eval
            skips_at_last_eval = n_skipped
            extra = ""
            if args.entropy_reg > 0:
                extra += f"  ent={ent_val:.3f}"
            if scheduler is not None:
                extra += f"  lr={optimizer.param_groups[0]['lr']:.4f}"
            print(f"[{i + 1:>7}/{args.iters}] loss={loss.item():.4f}  val={val:.4f}  "
                  f"soft={val_soft:.4f}  gap={val_soft - val:+.4f}  gnorm={grad_norm:.2f}  "
                  f"skip={n_skipped}{extra}  ({(time.time() - t0) / 60:.1f} min)")
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # Dead-weights early stop: if a whole eval window was skipped, the weights have
            # gone non-finite and will never recover — stop instead of spinning for 30 min.
            if window_skips >= args.eval_freq:
                print(f"[stop] all {args.eval_freq} steps this window skipped → weights are "
                      f"non-finite (dead). Best checkpoint kept. Prevent it with a lower "
                      f"--lr (e.g. 0.003) and/or --grad-factor 0.5.")
                break

    # ---- final test on the best checkpoint (discrete-locked) -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, task.test_loader, device, discrete=True)
    test_soft = evaluate(model, task.test_loader, device, discrete=False)
    train_minutes = (time.time() - t0) / 60
    print("\n--- final ---")
    print(f"best_val={best_val:.4f}  test={test_acc:.4f}  test_soft={test_soft:.4f}  "
          f"gap={test_soft - test_acc:+.4f}  skipped={n_skipped}/{args.iters}  "
          f"train_time={train_minutes:.1f} min")
    if n_skipped > args.iters * 0.2:
        print(f"[note] {100 * n_skipped / args.iters:.0f}% of steps skipped (exploding grads) "
              f"— consider a lower --lr (e.g. 0.003) and/or --grad-factor 0.5.")

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
        "hidden": args.hidden, "cell_layers": args.cell_layers, "keep_bias": args.keep_bias,
        "tau": args.tau,
        "grad_factor": args.grad_factor, "grad_clip": args.grad_clip,
        "lr": args.lr, "lr_min": args.lr_min, "entropy_reg": args.entropy_reg,
        "batch_size": args.batch_size,
        "iters": args.iters, "seed": args.seed, "device": device,
        "logic_gates": utils.count_gates(model),
        "best_val": best_val, "test_acc": test_acc, "test_soft": test_soft,
        "discretization_gap": test_soft - test_acc, "n_skipped": n_skipped,
        "train_minutes": train_minutes,
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
