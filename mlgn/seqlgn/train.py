"""
train.py - train & evaluate a recurrent Logic Gate Network on a sequential task.
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
    """Accuracy. discrete=True: eval mode (argmax gates) + binarised inputs - the REAL
    logic-circuit accuracy. discrete=False: train mode (softmax gates), inputs unrounded -
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
    p.add_argument("--mechanism", default="gated", choices=("rddlgn", "gated", "lstm", "gru_cell", "latch", "combo", "clatch"),
                   help="memory mechanism: 'rddlgn' control, 'gated' (GRU-style, Paper #1), "
                        "'lstm' (richer arm), 'gru_cell' (separate cell state + GRU MUX - the 2x2 "
                        "ablation), 'latch' (Paper #2 - bistable SR/T-FF primitive; see --latch-kind), "
                        "'combo' (gated write + bistable restore on the hold), "
                        "'clatch' (INPUT-CLOCKED LATCH - round the write-ENABLE not the value = a "
                        "learnable write-enabled register; exact hold, no drift, no collapse; "
                        "--anneal ramps the enable-hardening).")
    p.add_argument("--latch-kind", default="sr", choices=("sr", "tff"),
                   help="latch primitive when --mechanism latch: 'sr' (set/reset - hold/recall, e.g. "
                        "copy) or 'tff' (T flip-flop - toggle/integrate, e.g. parity).")
    p.add_argument("--soft-state", action="store_true",
                   help="latch: DISABLE the bistable restore (the v0 soft latch, for the ablation). "
                        "Default (off) = hard_state on = the v1 bistable latch.")
    p.add_argument("--hard-control", action="store_true",
                   help="latch: also round the S/R/T control lines (fully-hard workmap variant). "
                        "Off by default - at eval the gates are already argmax-binary.")
    p.add_argument("--anneal", default="",
                   help="latch: soft->hard restore anneal window 'START,END' (fractions of training), "
                        "e.g. '0.1,0.6' = soft until 10%%, ramp to fully-hard by 60%%. Empty = hard "
                        "from step 0 (un-annealed v1). Fixes the hard-from-scratch cold-start/plateau.")
    p.add_argument("--hidden", type=int, default=2000, help="hidden_dim (must be >= input_dim and divisible by num_classes)")
    p.add_argument("--cell-layers", type=int, default=2, help="logic layers per candidate/gate/update network")
    p.add_argument("--keep-bias", type=float, default=3.0,
                   help="keep-bias at init (logic forget-bias / residual init): turns the carousel ON "
                        "at start to avoid cold-start. 0 disables. Affects 'gated'/'lstm'/'gru_cell' "
                        "and 'latch' (biases S/R or toggle toward HOLD). TASK-DEPENDENT: HIGH for "
                        "hold/recall (copy), LOW for toggle/integrate (parity/psMNIST).")
    p.add_argument("--tau", type=float, default=30.0, help="GroupSum temperature")
    p.add_argument("--grad-factor", type=float, default=1.0, help="difflogic grad_factor (raise for deep/long unrolls)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="clip global grad norm to this value (RNN exploding-gradient fix). "
                        "0 disables. Keep-bias fixes vanishing but can over-correct into "
                        "exploding gradients on long sequences (NaN) - clipping prevents it.")
    p.add_argument("--seq-len", type=int, default=None, help="sequence length for synthetic tasks (parity/copy)")
    p.add_argument("--test-seq-len", type=int, default=None,
                   help="LENGTH-GENERALIZATION eval (synthetic tasks only): generate the TEST set at "
                        "this length while training at --seq-len. train-short/test-long is where an "
                        "exact register (clatch/tff) beats a soft-MUX (gated) that drifts. Model "
                        "selection stays on train-length val; test_acc is then the length-gen number.")
    p.add_argument("--alphabet", type=int, default=8, help="alphabet size for the copy/selcopy tasks")
    p.add_argument("--sel-flag", action="store_true",
                   help="selcopy ABLATION: re-add the cue bit marking the data step (default off = "
                        "content-based selection). Expect it to let gated re-saturate — the point of "
                        "the ablation is to show the gap only opens without the flag.")
    p.add_argument("--distractors", type=int, default=8,
                   help="distcopy: number of non-cued distractor symbol tokens scattered after the t=0 "
                        "target that the cell must HOLD its value through without overwriting (the "
                        "hold-vs-overwrite dial; more = more leak pressure on a soft-MUX).")
    p.add_argument("--chunk", type=int, default=1,
                   help="pixels per timestep for pixel-MNIST tasks (smnist-pixel/psmnist); "
                        "seq_len = 784//chunk. e.g. 14 -> 56 steps, 8 -> 98 steps.")
    p.add_argument("--delay", type=int, default=0,
                   help="append N blank steps after an MNIST image (RECALL test: hold the "
                        "digit through the delay, then classify). seq_len = image_steps + N.")

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-min", type=float, default=-1.0,
                   help="if >=0 and < --lr, cosine-decay the LR from --lr to this over training "
                        "(absorbs the late-phase explosion once gates sharpen). -1 = constant LR.")
    p.add_argument("--entropy-reg", type=float, default=0.0,
                   help="coefficient on a gate-entropy penalty pushing gates toward one-hot, "
                        "to shrink the discretization gap. 0 = off.")
    p.add_argument("--entropy-ramp", type=float, default=1.0,
                   help="fraction of training over which --entropy-reg ramps 0->full (explore "
                        "early, commit late). 1.0 = ramp across the whole run.")
    p.add_argument("--margin-reg", type=float, default=0.0,
                   help="ACTIVATION-margin penalty: coeff on mean h*(1-h) over the recurrent state, "
                        "pushing state VALUES toward {0,1} to close the COMPUTATION (drift) gap that "
                        "the plain gated cell leaves (its deployed state is already binary; this "
                        "aligns the soft training trajectory). 0 = off. Ramped by --entropy-ramp.")
    p.add_argument("--deep-sup", type=float, default=0.0,
                   help="DEEP per-timestep supervision: coeff on mean_t CE(head(state_t), y) over all "
                        "timesteps (valid when the target is time-invariant, e.g. copy). "
                        "Removes the flat never-write plateau + shortens delayed credit. 0 = off. "
                        "(Single-state mechanisms only: gated/clatch/latch/combo.)")
    p.add_argument("--running-target", action="store_true",
                   help="PARITY-ONLY: make --deep-sup supervise each state against the RUNNING target "
                        "(cumulative XOR of the bits seen so far, ytar[:,t]=(x[:,:1].cumsum%%2)) instead "
                        "of the final label. Parity's final-only loss is flat/deceptive (every proper "
                        "prefix XOR is uncorrelated with the final label) so nothing trains; the dense "
                        "running target gives a per-step gradient and makes the toggle learnable. "
                        "Requires --deep-sup > 0. Uses input channel 0 as the bit stream.")
    p.add_argument("--state-hist", action="store_true",
                   help="ORACLE: at the end, histogram the SOFT recurrent-state values per timestep "
                        "(fraction in the mushy band [0.4,0.6]) to decide DRIFT (grows with t = the "
                        "gap is escapable via --margin-reg) vs analog reliance. Saved to the record.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=2_000)
    p.add_argument("--device", default=None, help="'cuda' / 'cpu' (default: auto)")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--grad-analysis", action="store_true", help="measure gradient norm through time at the end")
    p.add_argument("--show-gates", action="store_true", help="print the learned gate distribution at the end")
    p.add_argument("--tag", default="", help="optional tag added to the results filename")
    p.add_argument("--init-from", default="",
                   help="warm-start: load a model state_dict from this .pt before training (length "
                        "curriculum — the params don't depend on seq_len, so a shorter-copy checkpoint "
                        "loads straight in). Missing path = fresh start (so a ladder is robust).")
    p.add_argument("--save-model", action="store_true",
                   help="save the best checkpoint's state_dict to results/ckpt_<tag>.pt (chain "
                        "curriculum stages with --init-from).")
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
        test_seq_len=args.test_seq_len,
        alphabet=args.alphabet, chunk=args.chunk, delay=args.delay,
        write_flag=args.sel_flag, n_distractors=args.distractors, seed=args.seed,
    )
    lg = f"  test_seq_len={task.test_seq_len} (LENGTH-GEN)" if task.test_seq_len != task.seq_len else ""
    print(f"task={task.name}  seq_len={task.seq_len}  input_dim={task.input_dim}  "
          f"num_classes={task.num_classes}{lg}")

    # latch bistable-restore anneal window (START,END fractions of training); None = hard from step 0.
    anneal_window = None
    if args.anneal:
        parts = [float(v) for v in args.anneal.split(",")]
        assert len(parts) == 2, "--anneal expects 'START,END', e.g. 0.1,0.6"
        anneal_window = (parts[0], parts[1])
    is_restore = args.mechanism in ("latch", "combo", "clatch")  # mechanisms that use the round anneal (state or enable)

    model = SequenceClassifier(
        input_dim=task.input_dim,
        hidden_dim=args.hidden,
        num_classes=task.num_classes,
        mechanism=args.mechanism,
        cell_layers=args.cell_layers,
        keep_bias=args.keep_bias,
        latch_kind=args.latch_kind,
        hard_state=not args.soft_state,
        hard_control=args.hard_control,
        tau=args.tau,
        device=device,
        grad_factor=args.grad_factor,
    ).to(device)

    print(model)
    print(f"mechanism={args.mechanism}  logic gates={utils.count_gates(model):,}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    if args.init_from:  # length-curriculum warm-start (same architecture across copy lengths)
        if os.path.exists(args.init_from):
            model.load_state_dict(torch.load(args.init_from, map_location=device))
            print(f"[init] warm-started from {args.init_from}")
        else:
            print(f"[init] --init-from {args.init_from} not found -> fresh start")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # TODO(future): make the LR schedule a first-class, pluggable choice - an explicit
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
        # latch bistable-restore anneal: ramp hard_alpha 0->1 over the window so the state
        # hardens gradually (soft solution first, then commit to {0,1}); avoids the
        # hard-from-scratch cold-start/plateau. No-op unless --mechanism latch --anneal set.
        if is_restore and anneal_window is not None:
            model.cell.hard_alpha = utils.hard_anneal_alpha(i / max(1, args.iters), *anneal_window)
        need_states = args.margin_reg > 0 or args.deep_sup > 0
        if need_states:
            logits, states = model(x, return_hidden=True)   # states = per-t carousel/hidden states
        else:
            logits = model(x)
        loss = loss_fn(logits, y)          # task (CE) loss - reported as `loss`
        total_loss = loss

        # Optional gate-entropy penalty (ramped): pushes gates one-hot (SELECTION gap).
        if args.entropy_reg > 0:
            ramp = min(1.0, (i + 1) / max(1, args.entropy_ramp * args.iters))
            ent = utils.gate_entropy(model)
            ent_val = ent.item()
            total_loss = total_loss + (args.entropy_reg * ramp) * ent
        # Activation-margin (ramped): push state VALUES to {0,1} (COMPUTATION/drift gap) — closes
        # the gap the plain gated cell leaves without ever destructively rounding a write.
        if args.margin_reg > 0:
            ramp = min(1.0, (i + 1) / max(1, args.entropy_ramp * args.iters))
            margin = torch.stack([(h * (1.0 - h)).mean() for h in states]).mean()
            total_loss = total_loss + (args.margin_reg * ramp) * margin
        # Deep per-timestep supervision: readout each state against the target.
        #  - default: the time-invariant final label y (valid for copy: symbol known from t=0).
        #  - --running-target (parity): the per-step RUNNING label ytar[:,t] = cumulative XOR of the
        #    bits so far. Parity's final-only loss is flat (every prefix XOR is uncorrelated with the
        #    final answer) so it never trains; the dense running target gives a real per-step gradient.
        if args.deep_sup > 0:
            if args.running_target:
                bits = x[:, :, 0]                                   # [B, L] the parity bit stream
                ytar = (bits.cumsum(dim=1) % 2).long()             # [B, L] running XOR, aligned to state_t
                ds = torch.stack([loss_fn(model.head(states[t]), ytar[:, t])
                                  for t in range(len(states))]).mean()
            else:
                ds = torch.stack([loss_fn(model.head(h), y) for h in states]).mean()
            total_loss = total_loss + args.deep_sup * ds

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
            # gone non-finite and will never recover - stop instead of spinning for 30 min.
            if window_skips >= args.eval_freq:
                print(f"[stop] all {args.eval_freq} steps this window skipped -> weights are "
                      f"non-finite (dead). Best checkpoint kept. Prevent it with a lower "
                      f"--lr (e.g. 0.003) and/or --grad-factor 0.5.")
                break

    # ---- final test on the best checkpoint (discrete-locked) -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    if is_restore:
        model.cell.hard_alpha = 1.0  # report the fully-hardened (deployed-consistent) numbers
    test_acc = evaluate(model, task.test_loader, device, discrete=True)
    test_soft = evaluate(model, task.test_loader, device, discrete=False)
    train_minutes = (time.time() - t0) / 60
    print("\n--- final ---")
    lg = f"  (train L={task.seq_len} -> test L={task.test_seq_len})" if task.test_seq_len != task.seq_len else ""
    print(f"best_val={best_val:.4f}  test={test_acc:.4f}  test_soft={test_soft:.4f}  "
          f"gap={test_soft - test_acc:+.4f}  skipped={n_skipped}/{args.iters}  "
          f"train_time={train_minutes:.1f} min{lg}")
    if n_skipped > args.iters * 0.2:
        print(f"[note] {100 * n_skipped / args.iters:.0f}% of steps skipped (exploding grads) "
              f"- consider a lower --lr (e.g. 0.003) and/or --grad-factor 0.5.")

    # ---- optional analyses -----------------------------------------------------------
    state_mushy = None
    if args.state_hist:
        # ORACLE: per-timestep fraction of SOFT state values in the mushy band [0.4,0.6]. If this
        # GROWS with t, the state DRIFTS off {0,1} over time -> the gap is a computation/drift gap,
        # closable by --margin-reg (escapable). If it stays low/flat, the eval-argmax state is clean
        # and any gap is gate-SELECTION. (Run on the trained soft model, one val batch.)
        model.train()
        xb, _ = next(iter(task.val_loader))
        logits_h, hstates = model(xb.to(device), return_hidden=True)  # grad on (no backward)
        state_mushy = [float(((h > 0.4) & (h < 0.6)).float().mean().item()) for h in hstates]
        if state_mushy:
            print("\n--- state-drift oracle: mushy-fraction (|h in [0.4,0.6]|) by timestep ---")
            print(f"  t=0: {state_mushy[0]:.3f}   t=mid: {state_mushy[len(state_mushy)//2]:.3f}   "
                  f"t=-1: {state_mushy[-1]:.3f}   (rising = DRIFT = margin-reg should help)")

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

    # gate usage (which of the 16 gates each neuron settled on at the best checkpoint) — SAVED
    # for every run so we can see which gates the latch/combo prefers (and diagnose failures,
    # e.g. a collapse to constant FALSE/TRUE = "dead" gates that can't compute anything).
    gate_dist = utils.gate_distribution(model)
    gate_totals = [int(c) for c in sum(gate_dist.values())] if gate_dist else [0] * 16
    if sum(gate_totals):
        _tot = sum(gate_totals)
        _top = sorted(range(16), key=lambda g: gate_totals[g], reverse=True)[:4]
        print("--- gate usage (top 4/16): "
              + ", ".join(f"{utils.GATE_NAMES[g]} {100 * gate_totals[g] / _tot:.0f}%" for g in _top))

    # ---- persist a results record ----------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.save_model:  # save best checkpoint for curriculum chaining (deterministic, tag-based path)
        ckpt = os.path.join(RESULTS_DIR, f"ckpt_{args.tag or args.mechanism}.pt")
        torch.save(best_state if best_state is not None else model.state_dict(), ckpt)
        print(f"checkpoint -> {ckpt}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(RESULTS_DIR, f"{task.name}_{args.mechanism}{tag}_{stamp}.json")
    record = {
        "task": task.name, "mechanism": args.mechanism, "seq_len": task.seq_len,
        "test_seq_len": task.test_seq_len,
        "latch_kind": args.latch_kind if args.mechanism == "latch" else None,
        "hard_state": (not args.soft_state) if is_restore else None,
        "hard_control": args.hard_control if args.mechanism == "latch" else None,
        "anneal": args.anneal if is_restore else None,
        "hidden": args.hidden, "cell_layers": args.cell_layers, "keep_bias": args.keep_bias,
        "tau": args.tau,
        "grad_factor": args.grad_factor, "grad_clip": args.grad_clip,
        "lr": args.lr, "lr_min": args.lr_min, "entropy_reg": args.entropy_reg,
        "margin_reg": args.margin_reg, "deep_sup": args.deep_sup,
        "running_target": args.running_target,
        "state_mushy_by_t": state_mushy,
        "batch_size": args.batch_size,
        "iters": args.iters, "seed": args.seed, "device": device,
        "logic_gates": utils.count_gates(model),
        "best_val": best_val, "test_acc": test_acc, "test_soft": test_soft,
        "discretization_gap": test_soft - test_acc, "n_skipped": n_skipped,
        "train_minutes": train_minutes,
        "grad_profile": grad_profile,
        # gate usage: per-LogicLayer histograms over the 16 gates + the aggregate 16-vector.
        # Map indices via utils.GATE_NAMES. Shows what the latch's set/reset (or combo's gate/
        # candidate) nets prefer, and flags "dead" collapses to FALSE(0)/TRUE(15).
        "gate_totals": gate_totals,
        "gate_distribution": {name: [int(c) for c in counts] for name, counts in gate_dist.items()},
    }
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nresults -> {out}")
    print(f"LOG-LINE: task={task.name} mech={args.mechanism} hidden={args.hidden} "
          f"seq_len={task.seq_len} | val={best_val:.4f} test={test_acc:.4f} "
          f"gates={utils.count_gates(model):,} time={train_minutes:.1f}min")


if __name__ == "__main__":
    main()
