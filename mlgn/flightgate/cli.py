"""
cli.py — run the D1 closed-loop distillation gate harness; JSON records per run.
================================================================================

Run FROM THE REPO ROOT (mlgn has no top-level __init__.py):

    # CPU smoke on the sim-independent mock (main env, torch 2.10+cpu):
    python -m mlgn.flightgate.cli --backend mock --arm gated --rounds 3 \
        --episodes 6 --eval-episodes 6 --iters 120 --hidden 40 --device cpu

    # clatch arm with enable-anneal:
    python -m mlgn.flightgate.cli --backend mock --arm clatch --anneal 0.1,0.6 ...

    # feedforward degradation control (same everything else):
    python -m mlgn.flightgate.cli --backend mock --arm ff ...

    # teacher-only collection + quantization gate (NO torch needed — this is the
    # mode the isolated sim venv can always run):
    python -m mlgn.flightgate.cli --backend hover --teacher-only --episodes 3

    # real sim (WSL .venv-flight or DUST; see README.md 'Install'):
    .venv-flight/bin/python -m mlgn.flightgate.cli --backend hover --arm gated ...

Seeding follows the seqlgn convention: utils.set_seed(seed) FIRST, then task/env
objects, then the model (keeps LogicLayer wiring replayable; new checkpoints are
self-contained via conn_a/conn_b buffers anyway). The JSON record mirrors
mlgn/seqlgn/train.py's shape plus flightgate-specific keys (frozen encoder
thresholds, bin edges, beta schedule, blackout config, sim versions/commit).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if __package__ in (None, ""):  # run as a script
    from mlgn.flightgate import encode, mock_env, trainer
    from mlgn.flightgate.env import SensorBlackout, hover_env_factory
    from mlgn.flightgate.teacher import make_teacher
else:
    from . import encode, mock_env, trainer
    from .env import SensorBlackout, hover_env_factory
    from .teacher import make_teacher

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_args(argv=None):
    p = argparse.ArgumentParser(description="D1 flight-gate closed-loop distillation.")
    p.add_argument("--backend", default="mock", choices=("mock", "hover"),
                   help="'mock' = deterministic 2D point-mass (sim-independent gate); "
                        "'hover' = gym-pybullet-drones HoverAviary (sim venv / DUST).")
    p.add_argument("--arm", default="gated", choices=("gated", "clatch", "ff"),
                   help="student arm (one per run, like train.py --mechanism).")
    p.add_argument("--teacher-only", action="store_true",
                   help="collect teacher episodes + run the quantization gate, no "
                        "student/torch (works in the torch-free sim venv).")
    p.add_argument("--teacher-masked", action="store_true",
                   help="a' CEILING run (deliverable 3): eval the PID teacher fed the "
                        "SAME masked obs the student sees (obs->state shim), no "
                        "student/torch. Meaningful under blackout; degenerates to the "
                        "raw teacher with --no-blackout. mechanism='teacher_masked'.")
    # sizes / encoding
    p.add_argument("--bits", type=int, default=None,
                   help="thermometer bits per obs dim (default: 4 mock, 16 hover).")
    p.add_argument("--n-bins", type=int, default=None,
                   help="action bins per motor channel (default: 5 mock, 9 hover).")
    p.add_argument("--hidden", type=int, default=None,
                   help="hidden width; must be >= input bits and divisible by "
                        "4*n_bins (default: 40 mock, 216 hover).")
    p.add_argument("--cell-layers", type=int, default=2)
    p.add_argument("--keep-bias", type=float, default=3.0)
    p.add_argument("--tau", type=float, default=None,
                   help="GroupSum temperature (default: head_group_width / 2, min 1 "
                        "— train.py's 30 assumes ~250-bit groups; the action head's "
                        "groups are hidden/(4*n_bins) bits, so 30 would flatten the "
                        "logits to near-zero gradient).")
    p.add_argument("--grad-factor", type=float, default=1.0)
    p.add_argument("--anneal", default="",
                   help="clatch enable-hardening window 'START,END' (fractions of "
                        "each round's iters), e.g. '0.1,0.6'. Empty = hard from 0.")
    # DAgger loop
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--beta", default="1.0,0.5,0.25",
                   help="comma beta schedule (prob. the TEACHER flies); round 0 is "
                        "always forced to 1.0; padded/truncated to --rounds.")
    p.add_argument("--episodes", type=int, default=8, help="collect episodes/round")
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--iters", type=int, default=200, help="train iters per round")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bptt", type=int, default=0,
                   help="BPTT window (episode-prefix) length; 0 = full episode.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # env / blackout
    p.add_argument("--max-steps", type=int, default=None,
                   help="episode cap (default: 40 mock; hover truncates itself at 8s).")
    p.add_argument("--no-blackout", action="store_true",
                   help="disable the SensorBlackout wrapper (the non-vacuity control "
                        "arm: feedforward must MATCH recurrent here).")
    p.add_argument("--blackout", default="0.05,3,6",
                   help="p_start,k_min,k_max for SensorBlackout (mock default; use "
                        "'0.02,6,12' for hover's 30 Hz per the recon spec).")
    p.add_argument("--settle-frac", type=float, default=0.25,
                   help="fraction of the episode excluded from RMS error (settling).")
    # bookkeeping
    p.add_argument("--device", default=None, help="'cuda' / 'cpu' (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", default=RESULTS_DIR)
    p.add_argument("--save-traj", action="store_true",
                   help="save every collection rollout as .npz next to the JSON.")
    return p.parse_args(argv)


def _backend_defaults(args):
    if args.backend == "mock":
        args.bits = 4 if args.bits is None else args.bits
        args.n_bins = 5 if args.n_bins is None else args.n_bins
        args.hidden = 40 if args.hidden is None else args.hidden
        args.max_steps = 40 if args.max_steps is None else args.max_steps
    else:
        args.bits = 16 if args.bits is None else args.bits
        args.n_bins = 9 if args.n_bins is None else args.n_bins
        args.hidden = 216 if args.hidden is None else args.hidden
        # HoverAviary self-truncates at 8 s * 30 Hz = 240 steps
        args.max_steps = 240 if args.max_steps is None else args.max_steps
    if args.tau is None:
        args.tau = max(1.0, args.hidden / (4 * args.n_bins) / 2.0)
    return args


def _make_env_factory(args, blackout_cfg, jitter_seed: int):
    """episode_index -> fresh env. Mock: one class, jitter via reset(seed). Hover:
    fresh jittered Aviary per episode (constructor-time jitter, recon spec §6).

    Jitter AND blackout are derived DETERMINISTICALLY from (run seed, episode
    index) — never from a shared consumed rng — so the eval envs (indices 0..n-1,
    used by every arm and every round) are frozen and bit-identical across arms;
    dagger() offsets collection indices by (round+1)*1_000_000 to decorrelate."""
    if args.backend == "mock":
        def factory(episode_index: int):
            env = mock_env.MockPointMass2D(max_steps=args.max_steps, jitter=True)
            if blackout_cfg is not None:
                b = dict(blackout_cfg)
                b["seed"] = int(b["seed"]) + episode_index
                env = SensorBlackout(env, **b)
            return env
        return factory

    def factory(episode_index: int):
        rng = np.random.default_rng([jitter_seed, episode_index])
        return hover_env_factory(rng, blackout_cfg, episode_index)
    return factory


def _sim_versions(backend: str) -> dict:
    import importlib.metadata as md
    out = {"python": sys.version, "platform": platform.platform(),
           "numpy": np.__version__}
    if backend == "hover":
        for pkg in ("gymnasium", "pybullet", "gym-pybullet-drones", "scipy"):
            try:
                out[pkg] = md.version(pkg)
            except Exception:
                out[pkg] = "NOT-INSTALLED"
        commit_file = os.path.join(_ROOT, ".venv-flight", "GPD_COMMIT")
        if os.path.exists(commit_file):
            with open(commit_file) as f:
                out["gym_pybullet_drones_commit"] = f.read().strip()
        else:
            out["gym_pybullet_drones_commit"] = "UNKNOWN (record it!)"
    return out


def main(argv=None):
    args = _backend_defaults(build_args(argv))

    no_student = args.teacher_only or args.teacher_masked
    mechanism = ("teacher_masked" if args.teacher_masked
                 else "teacher_only" if args.teacher_only else args.arm)

    # ---- seeding FIRST (seqlgn convention), torch only if a student trains ------
    if no_student:
        np.random.seed(args.seed)
        device = "cpu"
    else:
        from mlgn.seqlgn import utils
        device = utils.get_device(args.device)
        utils.set_seed(args.seed)
        if device == "cpu":
            print("[warn] CPU python-implementation difflogic: keep sizes tiny "
                  "(smoke); real runs go to DUST.")

    # ---- task objects ------------------------------------------------------------
    ranges = encode.MOCK_RANGES if args.backend == "mock" else encode.HOVER_RANGES
    encoder = encode.ThermometerEncoder(ranges, bits=args.bits)
    discretizer = encode.ActionDiscretizer(n_bins=args.n_bins)
    input_bits = encoder.n_bits + (0 if args.no_blackout else 1)

    blackout_cfg = None
    if not args.no_blackout:
        ps, kmin, kmax = args.blackout.split(",")
        blackout_cfg = {"p_start": float(ps), "k_min": int(kmin), "k_max": int(kmax),
                        "seed": args.seed * 7_919}
    env_factory = _make_env_factory(args, blackout_cfg, jitter_seed=args.seed + 1)
    teacher = make_teacher(args.backend)
    probe = env_factory(0)
    target = np.asarray(probe.target, dtype=np.float64)
    probe.close()
    if args.backend == "hover":
        print(f"[hover] versions: {_sim_versions('hover')}")
    print(f"backend={args.backend} arm={args.arm} input_bits={input_bits} "
          f"n_bins={args.n_bins} hidden={args.hidden} blackout={blackout_cfg}")

    # ---- teacher baselines: raw PID + PID-through-quantizer (the quantization
    # gate: replaying teacher actions through the bins must still solve the task,
    # else the student ceiling is dishonest) --------------------------------------
    t_raw = trainer.eval_actor(env_factory, trainer.TeacherActor(teacher), encoder,
                               discretizer, args.eval_episodes, target,
                               seed_base=args.seed * 100_000 + 900_000,
                               max_steps=args.max_steps, settle_frac=args.settle_frac)
    t_quant = trainer.eval_actor(env_factory,
                                 trainer.TeacherActor(teacher, discretizer), encoder,
                                 discretizer, args.eval_episodes, target,
                                 seed_base=args.seed * 100_000 + 900_000,
                                 max_steps=args.max_steps,
                                 settle_frac=args.settle_frac)
    print(f"teacher raw:       return={t_raw['mean_return']:.2f} "
          f"rms={t_raw['mean_rms_err']:.4f} exit={t_raw['envelope_exit_rate']:.2f}")
    print(f"teacher quantized: return={t_quant['mean_return']:.2f} "
          f"rms={t_quant['mean_rms_err']:.4f} exit={t_quant['envelope_exit_rate']:.2f}")
    quant_ratio = (t_quant["mean_return"] / t_raw["mean_return"]
                   if t_raw["mean_return"] else float("nan"))
    if not (quant_ratio > 0.9):
        print(f"[warn] QUANTIZATION GATE degraded: quantized/raw return ratio = "
              f"{quant_ratio:.3f} (< 0.9). Consider more --n-bins before training.")

    # ---- a' CEILING (deliverable 3): teacher on the MASKED obs the student sees.
    # SAME frozen eval seeds as t_raw/t_quant and the students -> matched jitter +
    # blackout mask sequences, so the ceiling is directly comparable per seed. -------
    t_masked = None
    if args.teacher_masked:
        from mlgn.flightgate.teacher import make_masked_teacher
        masked_teacher = make_masked_teacher(args.backend)
        t_masked = trainer.eval_actor(
            env_factory, trainer.MaskedTeacherActor(masked_teacher), encoder,
            discretizer, args.eval_episodes, target,
            seed_base=args.seed * 100_000 + 900_000,
            max_steps=args.max_steps, settle_frac=args.settle_frac)
        print(f"teacher MASKED(a'): return={t_masked['mean_return']:.2f} "
              f"rms={t_masked['mean_rms_err']:.4f} "
              f"rms_bk={t_masked['mean_rms_err_blackout']:.4f} "
              f"exit={t_masked['envelope_exit_rate']:.2f}  "
              f"(occlusion cost vs raw: "
              f"{t_raw['mean_return'] - t_masked['mean_return']:+.2f} return)")

    record = {
        "task": f"flightgate_{args.backend}",
        "mechanism": mechanism,
        "backend": args.backend,
        "hidden": args.hidden, "cell_layers": args.cell_layers,
        "keep_bias": args.keep_bias, "tau": args.tau,
        "grad_factor": args.grad_factor, "grad_clip": args.grad_clip,
        "lr": args.lr, "batch_size": args.batch_size, "bptt": args.bptt,
        "anneal": args.anneal or None,
        "iters_per_round": args.iters, "rounds": args.rounds,
        "episodes_per_round": args.episodes, "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps, "settle_frac": args.settle_frac,
        "seed": args.seed, "device": device,
        "input_bits": input_bits, "n_bins": args.n_bins,
        "encoder": encoder.to_config(),
        "discretizer": discretizer.to_config(),
        "blackout": blackout_cfg,
        "jitter_scheme": {"entropy": [args.seed + 1, "episode_index"],
                          "collect_offset": "(round+1)*1_000_000",
                          "eval_indices": "0..eval_episodes-1 (frozen across arms)"},
        "versions": _sim_versions(args.backend),
        "eval_teacher": t_raw,
        "eval_teacher_quantized": t_quant,
        "eval_teacher_masked": t_masked,   # a' ceiling (None unless --teacher-masked)
        "quantization_gate_return_ratio": quant_ratio,
    }
    if args.backend == "hover":
        from mlgn.flightgate.env import HOVER_JITTER
        record["hover_jitter"] = {k: list(v) for k, v in HOVER_JITTER.items()}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    base = f"flightgate_{args.backend}_{record['mechanism']}{tag}_{stamp}"
    traj_dir = os.path.join(args.out_dir, base + "_traj") if args.save_traj else None

    if args.teacher_masked:
        pass  # eval-only (a' ceiling already in record["eval_teacher_masked"])
    elif args.teacher_only:
        eps = trainer.collect_episodes(
            env_factory, teacher, encoder, discretizer, args.episodes,
            rng=np.random.default_rng(args.seed), seed_base=args.seed * 100_000,
            max_steps=args.max_steps)
        record["collected_steps"] = int(sum(ep["length"] for ep in eps))
        # velocity-range calibration snapshot (recon: HOVER velocity ranges are
        # PLACEHOLDERS until calibrated from >= 50 jittered teacher episodes)
        raw_obs = np.concatenate([ep["obs"][:, : encoder.obs_dim] for ep in eps])
        record["calibrated_ranges_preview"] = [
            list(r) for r in encode.calibrate_ranges(raw_obs)]
        if traj_dir is None:
            traj_dir = os.path.join(args.out_dir, base + "_traj")
        trainer.save_episodes(os.path.join(traj_dir, "teacher_collect.npz"), eps)
        record["traj_dir"] = traj_dir
    else:
        import torch  # noqa: F401  (via student import below)
        from mlgn.flightgate.student import StudentPolicy, build_student
        from mlgn.seqlgn import utils

        model = build_student(
            args.arm, input_bits=input_bits, hidden=args.hidden, n_act=probe.n_act,
            n_bins=args.n_bins, cell_layers=args.cell_layers,
            keep_bias=args.keep_bias, tau=args.tau, device=device,
            grad_factor=args.grad_factor)
        print(model)
        print(f"logic gates={utils.count_gates(model):,} "
              f"params={sum(p.numel() for p in model.parameters()):,}")
        record["logic_gates"] = utils.count_gates(model)

        anneal = None
        if args.anneal:
            a, b = (float(v) for v in args.anneal.split(","))
            anneal = (a, b)
        betas = [float(b) for b in args.beta.split(",")]
        betas = (betas + [betas[-1]] * args.rounds)[: args.rounds]

        dg = trainer.dagger(
            model, env_factory, teacher, encoder, discretizer, target,
            beta_schedule=betas, episodes_per_round=args.episodes,
            eval_episodes=args.eval_episodes, iters_per_round=args.iters,
            batch_size=args.batch_size, lr=args.lr, grad_clip=args.grad_clip,
            bptt=args.bptt, anneal=anneal, device=device, max_steps=args.max_steps,
            seed=args.seed, traj_dir=traj_dir, settle_frac=args.settle_frac)
        record["rounds_configured"] = args.rounds  # dg["rounds"] (per-round list) clobbers the config int below
        record.update(dg)
        record["gate_totals"] = [int(c) for c in (
            sum(utils.gate_distribution(model).values())
            if utils.gate_distribution(model) else [0] * 16)]
        record["gate_distribution"] = {
            name: [int(c) for c in counts]
            for name, counts in utils.gate_distribution(model).items()}
        last = dg["rounds"][-1]
        print(f"LOG-LINE: task={record['task']} arm={args.arm} hidden={args.hidden} "
              f"| teacher_ret={t_raw['mean_return']:.2f} "
              f"quant_ret={t_quant['mean_return']:.2f} "
              f"student_ret={last['eval_student_discrete']['mean_return']:.2f} "
              f"acc_d={last['train']['action_acc_discrete']:.4f} "
              f"gap={last['train']['discretization_gap']:+.4f} "
              f"skipped={last['train']['n_skipped']}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, base + ".json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2, default=float)
    print(f"results -> {out}")
    return record


if __name__ == "__main__":
    main()
