"""
trainer.py — closed-loop DAgger-style distillation: collect, relabel, train, iterate.
=====================================================================================

Layout (import discipline): everything ABOVE the "torch enters here" marker is
NUMPY-ONLY and importable in the isolated sim venv (rollout collection, actors,
metrics, trajectory persistence). Torch is imported lazily INSIDE the training
functions, so ``--teacher-only`` collection runs without torch.

The loop (recon spec §7, workmap D1 minimum):
  round 0:   beta=1 (teacher flies) -> dataset of (bits, teacher_bins, ...) per step
  round i>0: executed action = teacher w.p. beta_i else CURRENT student (per-step
             coin flip); the PRIVILEGED teacher labels EVERY visited state; data is
             AGGREGATED across rounds; the student is retrained on the aggregate.

Per-step CE on the teacher's (clipped, discretized) action = deep supervision — the
training method P2 proved necessary; here the per-step target is the PID action.

Training-loop guards mirrored from mlgn/seqlgn/train.py (house discipline):
clip_grad_norm_ returns the PRE-clip norm; the optimizer step is SKIPPED when that
norm is non-finite (post-hoc clipping of an inf backward yields nan); skipped steps
are counted; an all-skipped eval window early-stops the run. clatch's
``cell.hard_alpha`` is set per step from ``utils.hard_anneal_alpha`` and forced to
1.0 before any deployed eval.

BPTT window caveat (honest): recurrent students train on episode-aligned windows
starting at t=0 with a zero initial state (matching deployment, where the state
register resets per episode). ``bptt`` truncates the window LENGTH from the front
of each episode; random mid-episode offsets are deliberately NOT the default —
a window that starts inside a blackout has ill-posed targets w.r.t. the missing
pre-blackout context. Full-episode BPTT (bptt=0) is the DUST setting.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .encode import ActionDiscretizer, ThermometerEncoder


# ------------------------------------------------------------------------------------
# Actors (uniform closed-loop interface: reset(); action(env, bits) -> (n_act,))
# ------------------------------------------------------------------------------------

class TeacherActor:
    """Privileged PID arm. With a discretizer, actions are quantized through the
    student's bins — the QUANTIZATION GATE arm (teacher-through-bins must still
    solve the task, else the student ceiling is dishonest)."""

    def __init__(self, teacher, discretizer: ActionDiscretizer | None = None):
        self.teacher = teacher
        self.discretizer = discretizer

    def reset(self):
        self.teacher.reset()

    def action(self, env, bits):
        del bits  # privileged: reads env.true_state()
        a = self.teacher.act(env)
        if self.discretizer is not None:
            a = self.discretizer.quantize(a)
        return a


class MaskedTeacherActor:
    """a' CEILING actor (D1 deliverable 3): the PID teacher fed the SAME masked
    observation the student sees, reconstructed into the control law's state (see
    teacher.MaskedObsHoverTeacher / mock_env.MockMaskedObsTeacher). ``wants_obs`` makes
    ``_rollout_actor`` hand the masked obs to ``action`` each step — it does NOT read
    env.true_state(), which is what separates it from the privileged TeacherActor."""

    wants_obs = True

    def __init__(self, masked_teacher, discretizer: ActionDiscretizer | None = None):
        self.teacher = masked_teacher
        self.discretizer = discretizer

    def reset(self):
        self.teacher.reset()

    def action(self, env, bits, obs):
        del bits
        a = self.teacher.act_from_obs(obs, env)
        if self.discretizer is not None:
            a = self.discretizer.quantize(a)
        return a


# ------------------------------------------------------------------------------------
# Rollout collection (numpy-only)
# ------------------------------------------------------------------------------------

def rollout_episode(
    env,
    teacher,
    encoder: ThermometerEncoder,
    discretizer: ActionDiscretizer,
    student_actor=None,
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
    reset_seed: int | None = None,
    max_steps: int = 10_000,
) -> dict:
    """One closed-loop episode. The teacher labels EVERY step (from the true state);
    the executed action is teacher w.p. ``beta``, else the student's. The student's
    recurrent state is stepped on EVERY frame (it observes the actual trajectory
    regardless of who acts). Returns aligned per-step arrays.

    Observation convention: ``obs_t`` (masked view, includes the validity bit when
    the env is SensorBlackout-wrapped) is what the student sees BEFORE acting at t;
    the teacher label at t comes from the true state at t. The post-terminal
    observation is not stored (nothing acts on it).
    """
    rng = rng or np.random.default_rng(0)
    teacher.reset()
    if student_actor is not None:
        student_actor.reset()
    obs, _info = env.reset(seed=reset_seed)

    has_valid = obs.shape[-1] == encoder.obs_dim + 1
    enc = encoder.encode_with_valid if has_valid else encoder.encode

    rows: dict[str, list] = {k: [] for k in (
        "obs", "bits", "teacher_action", "teacher_bins", "executed_action",
        "teacher_flew", "reward", "position", "valid")}
    trunc, _t = False, -1
    for _t in range(max_steps):
        bits = enc(obs)
        a_teacher = teacher.act(env)                       # clipped normalized label
        t_bins = discretizer.to_bins(a_teacher)
        if student_actor is not None:
            a_student = student_actor.action(env, bits)    # steps its state every frame
        teacher_flies = student_actor is None or rng.random() < beta
        executed = a_teacher if teacher_flies else a_student
        state = env.true_state()
        rows["position"].append(env.position_from_state(state))
        rows["obs"].append(obs)
        rows["bits"].append(bits)
        rows["teacher_action"].append(a_teacher)
        rows["teacher_bins"].append(t_bins)
        rows["executed_action"].append(executed)
        rows["teacher_flew"].append(1.0 if teacher_flies else 0.0)
        rows["valid"].append(obs[-1] if has_valid else 1.0)
        obs, r, term, trunc, _info = env.step(executed)
        rows["reward"].append(r)
        if term or trunc:
            break
    ep = {k: np.asarray(v) for k, v in rows.items()}
    ep["truncated_by_envelope"] = bool(trunc and _t + 1 < max_steps)
    ep["length"] = len(ep["reward"])
    return ep


def collect_episodes(
    env_factory,
    teacher,
    encoder,
    discretizer,
    n_episodes: int,
    student_actor=None,
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
    seed_base: int = 0,
    max_steps: int = 10_000,
) -> list[dict]:
    """Fresh env per episode (jitter is a constructor argument in gym-pybullet-drones;
    each Aviary owns a pybullet client -> ALWAYS close()), deterministic reset seeds
    seed_base + i for replayability."""
    rng = rng or np.random.default_rng(0)
    episodes = []
    for i in range(n_episodes):
        env = env_factory(i)
        try:
            episodes.append(rollout_episode(
                env, teacher, encoder, discretizer, student_actor=student_actor,
                beta=beta, rng=rng, reset_seed=seed_base + i, max_steps=max_steps))
        finally:
            env.close()
    return episodes


# ------------------------------------------------------------------------------------
# Metrics (numpy-only)
# ------------------------------------------------------------------------------------

def episode_metrics(ep: dict, target: np.ndarray, settle_frac: float = 0.25,
                    blackout_tail: int = 5) -> dict:
    """Per-episode gate metrics: return, RMS position error after the settle window,
    envelope-exit flag, and the same RMS sliced to blackout(+tail) frames."""
    pos = ep["position"]
    err = np.linalg.norm(pos - np.asarray(target)[None, : pos.shape[1]], axis=1)
    t0 = int(settle_frac * len(err))
    masked = ep["valid"] < 0.5
    # blackout window = masked frames plus `blackout_tail` recovery frames after each
    in_window = masked.copy()
    for t in np.flatnonzero(masked):
        in_window[t: t + 1 + blackout_tail] = True
    out = {
        "return": float(ep["reward"].sum()),
        "rms_err": float(np.sqrt(np.mean(err[t0:] ** 2))) if len(err) > t0 else float("nan"),
        "envelope_exit": bool(ep["truncated_by_envelope"]),
        "length": int(ep["length"]),
        "n_masked": int(masked.sum()),
        "rms_err_blackout": (float(np.sqrt(np.mean(err[in_window] ** 2)))
                             if in_window.any() else float("nan")),
    }
    return out


def aggregate_metrics(eps_metrics: list[dict]) -> dict:
    keys = ("return", "rms_err", "rms_err_blackout", "length", "n_masked")

    def _nanmean(vals):  # all-NaN (e.g. rms_err_blackout with no masked frames) -> nan, no warning
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {f"mean_{k}": _nanmean([m[k] for m in eps_metrics]) for k in keys}
    agg["envelope_exit_rate"] = float(np.mean([m["envelope_exit"] for m in eps_metrics]))
    agg["n_episodes"] = len(eps_metrics)
    return agg


def eval_actor(env_factory, actor, encoder, discretizer, n_episodes: int,
               target, seed_base: int = 10_000, max_steps: int = 10_000,
               settle_frac: float = 0.25) -> dict:
    """Closed-loop evaluation of any actor (it alone flies) on a FROZEN eval seed
    list — the same seed_base across arms gives every arm the same jitter and the
    same blackout mask sequences (episode_index is threaded to the factory)."""
    del discretizer  # kept in the signature for symmetry with collect_episodes
    metrics = []
    for i in range(n_episodes):
        env = env_factory(i)
        try:
            ep = _rollout_actor(env, actor, encoder, seed_base + i, max_steps)
            metrics.append(episode_metrics(ep, target, settle_frac=settle_frac))
        finally:
            env.close()
    return aggregate_metrics(metrics)


def _rollout_actor(env, actor, encoder, reset_seed, max_steps) -> dict:
    """Roll an arbitrary actor closed-loop; teacher labels are not recorded
    (evaluation only). The caller owns (and closes) the env."""
    actor.reset()
    obs, _ = env.reset(seed=reset_seed)
    has_valid = obs.shape[-1] == encoder.obs_dim + 1
    enc = encoder.encode_with_valid if has_valid else encoder.encode
    rows = {k: [] for k in ("reward", "position", "valid")}
    trunc = False
    t = -1
    wants_obs = getattr(actor, "wants_obs", False)
    for t in range(max_steps):
        bits = enc(obs)
        a = actor.action(env, bits, obs) if wants_obs else actor.action(env, bits)
        rows["position"].append(env.position_from_state(env.true_state()))
        rows["valid"].append(obs[-1] if has_valid else 1.0)
        obs, r, term, trunc, _ = env.step(a)
        rows["reward"].append(r)
        if term or trunc:
            break
    ep = {k: np.asarray(v) for k, v in rows.items()}
    ep["truncated_by_envelope"] = bool(trunc and t + 1 < max_steps)
    ep["length"] = len(ep["reward"])
    return ep


# ------------------------------------------------------------------------------------
# Trajectory persistence (house replayability: every rollout is saved)
# ------------------------------------------------------------------------------------

_EP_FIELDS = ("obs", "bits", "teacher_action", "teacher_bins", "executed_action",
              "teacher_flew", "reward", "position", "valid")


def save_episodes(path: str, episodes: list[dict]) -> None:
    """Padded-stack .npz (no pickle): per-field [n_eps, T_max, ...] + lengths."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lengths = np.array([ep["length"] for ep in episodes], dtype=np.int64)
    t_max = int(lengths.max()) if len(lengths) else 0
    payload: dict[str, np.ndarray] = {"lengths": lengths, "truncated_by_envelope":
                                      np.array([ep["truncated_by_envelope"]
                                                for ep in episodes])}
    for f in _EP_FIELDS:
        arrs = []
        for ep in episodes:
            a = np.asarray(ep[f])
            pad = [(0, t_max - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
            arrs.append(np.pad(a, pad))
        payload[f] = np.stack(arrs)
    np.savez_compressed(path, **payload)


def load_episodes(path: str) -> list[dict]:
    z = np.load(path)
    lengths = z["lengths"]
    eps = []
    for i, ln in enumerate(lengths):
        ep = {f: z[f][i, :ln] for f in _EP_FIELDS}
        ep["length"] = int(ln)
        ep["truncated_by_envelope"] = bool(z["truncated_by_envelope"][i])
        eps.append(ep)
    return eps


# ====================================================================================
# torch enters here — training half (import kept INSIDE functions)
# ====================================================================================

def _episodes_to_tensors(episodes: list[dict], bptt: int, device: str):
    """[B, L, bits] float32, [B, L, n_act] int64 targets, [B, L] loss mask.
    Windows are episode PREFIXES (t=0-aligned, zero initial state — see module
    docstring); bptt=0 => full max episode length; shorter episodes are
    zero-padded and masked out of the loss."""
    import torch

    t_max = max(ep["length"] for ep in episodes)
    L = t_max if bptt <= 0 else min(bptt, t_max)
    B = len(episodes)
    bits_dim = episodes[0]["bits"].shape[-1]
    n_act = episodes[0]["teacher_bins"].shape[-1]
    x = torch.zeros(B, L, bits_dim, dtype=torch.float32, device=device)
    y = torch.zeros(B, L, n_act, dtype=torch.int64, device=device)
    m = torch.zeros(B, L, dtype=torch.float32, device=device)
    for i, ep in enumerate(episodes):
        n = min(ep["length"], L)
        x[i, :n] = torch.as_tensor(ep["bits"][:n], dtype=torch.float32)
        y[i, :n] = torch.as_tensor(ep["teacher_bins"][:n], dtype=torch.int64)
        m[i, :n] = 1.0
    return x, y, m


def _masked_ce(logits, y, m):
    """Per-step CE over n_act motor heads, averaged over unmasked steps.
    logits [B,L,n_act,n_bins], y [B,L,n_act], m [B,L]."""
    import torch.nn.functional as F

    b, l, n_act, n_bins = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, n_bins), y.reshape(-1),
                         reduction="none").view(b, l, n_act).mean(-1)  # [B, L]
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def train_student(
    model,
    episodes: list[dict],
    iters: int = 200,
    batch_size: int = 16,
    lr: float = 0.01,
    grad_clip: float = 1.0,
    bptt: int = 0,
    anneal: tuple[float, float] | None = None,
    device: str = "cpu",
    log_every: int = 50,
    rng_seed: int = 0,
) -> dict:
    """Distillation training pass with the seqlgn guard rails. Returns stats incl.
    the per-iteration loss curve, n_skipped, and discrete/soft action accuracy on
    the training aggregate (report BOTH + gap — house eval discipline)."""
    import torch

    from mlgn.seqlgn import utils

    x, y, m = _episodes_to_tensors(episodes, bptt, device)
    n = x.shape[0]
    g = torch.Generator().manual_seed(rng_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    is_restore = getattr(model, "mechanism", "") in ("latch", "combo", "clatch")

    losses: list[float] = []
    n_skipped = 0
    skips_at_last_log = 0
    model.train()
    t0 = time.time()
    for i in range(iters):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=g).to(device)
        if is_restore and anneal is not None:
            model.cell.hard_alpha = utils.hard_anneal_alpha(i / max(1, iters), *anneal)
        logits = model.forward_sequence(x[idx])
        loss = _masked_ce(logits, y[idx], m[idx])
        optimizer.zero_grad()
        loss.backward()
        clip = grad_clip if grad_clip > 0 else float("inf")
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if torch.isfinite(total):
            optimizer.step()
        else:
            n_skipped += 1
        losses.append(float(loss.item()))
        if (i + 1) % log_every == 0:
            window_skips = n_skipped - skips_at_last_log
            skips_at_last_log = n_skipped
            print(f"    [{i + 1:>5}/{iters}] loss={losses[-1]:.4f} "
                  f"gnorm={float(total):.2f} skip={n_skipped}")
            if window_skips >= log_every:
                print("    [stop] entire window skipped -> weights non-finite (dead); "
                      "lower --lr / --grad-factor.")
                break
    if is_restore:
        model.cell.hard_alpha = 1.0  # deployed-consistent from here on

    acc_d = action_accuracy(model, x, y, m, discrete=True)
    acc_s = action_accuracy(model, x, y, m, discrete=False)
    return {
        "losses": losses,
        "n_skipped": n_skipped,
        "train_minutes": (time.time() - t0) / 60.0,
        "action_acc_discrete": acc_d,
        "action_acc_soft": acc_s,
        "discretization_gap": acc_s - acc_d,
        "n_train_steps_available": int(m.sum().item()),
    }


def action_accuracy(model, x, y, m, discrete: bool = True) -> float:
    """Fraction of (step, motor) targets matched. discrete=True: model.eval()
    (argmax gates) = the deployed circuit; inputs are already {0,1} bits so no
    rounding is required. discrete=False: train mode (soft gate mixtures)."""
    import torch

    was_training = model.training
    model.eval() if discrete else model.train()
    old_alpha = None
    if hasattr(model, "cell"):
        old_alpha = model.cell.hard_alpha
        model.cell.hard_alpha = 1.0
    with torch.no_grad():
        preds = model.forward_sequence(x).argmax(-1)  # [B, L, n_act]
        match = (preds == y).float().mean(-1)  # [B, L]
        acc = float((match * m).sum().item() / max(m.sum().item(), 1.0))
    if old_alpha is not None:
        model.cell.hard_alpha = old_alpha
    model.train(was_training)
    return acc


def dagger(
    model,
    env_factory,
    teacher,
    encoder,
    discretizer,
    target,
    beta_schedule=(1.0, 0.5, 0.25),
    episodes_per_round: int = 8,
    eval_episodes: int = 8,
    iters_per_round: int = 200,
    batch_size: int = 16,
    lr: float = 0.01,
    grad_clip: float = 1.0,
    bptt: int = 0,
    anneal: tuple[float, float] | None = None,
    device: str = "cpu",
    max_steps: int = 10_000,
    seed: int = 0,
    traj_dir: str | None = None,
    settle_frac: float = 0.25,
) -> dict:
    """The closed-loop distillation loop. Returns the per-round record (training
    stats + closed-loop eval of the deployed/discrete student each round). All
    rollouts are saved to ``traj_dir`` when given (exact replayability gate)."""
    from .student import StudentPolicy

    rng = np.random.default_rng(seed)
    policy = StudentPolicy(model, discretizer, device=device, discrete=True)
    dataset: list[dict] = []
    rounds = []
    for r, beta in enumerate(beta_schedule):
        actor = None if r == 0 else policy  # round 0 = pure teacher (beta forced 1)
        eff_beta = 1.0 if r == 0 else float(beta)
        # collection env indices are offset per round so their jitter/blackout draws
        # never collide with the FROZEN eval envs (factory indices 0..n-1)
        off = (r + 1) * 1_000_000
        eps = collect_episodes(
            lambda i, _off=off: env_factory(_off + i), teacher, encoder,
            discretizer, episodes_per_round,
            student_actor=actor, beta=eff_beta, rng=rng,
            seed_base=seed * 100_000 + r * 1_000, max_steps=max_steps)
        dataset.extend(eps)
        if traj_dir:
            save_episodes(os.path.join(traj_dir, f"round{r}_collect.npz"), eps)
        stats = train_student(
            model, dataset, iters=iters_per_round, batch_size=batch_size, lr=lr,
            grad_clip=grad_clip, bptt=bptt, anneal=anneal, device=device,
            rng_seed=seed + r)
        ev = eval_actor(env_factory, policy, encoder, discretizer, eval_episodes,
                        target, seed_base=seed * 100_000 + 900_000,
                        max_steps=max_steps, settle_frac=settle_frac)
        print(f"  [round {r}] beta={eff_beta:.2f} dataset={len(dataset)}ep "
              f"loss {stats['losses'][0]:.4f}->{stats['losses'][-1]:.4f} "
              f"acc_d={stats['action_acc_discrete']:.4f} "
              f"gap={stats['discretization_gap']:+.4f} | eval return="
              f"{ev['mean_return']:.2f} rms={ev['mean_rms_err']:.4f}")
        rounds.append({"round": r, "beta": eff_beta, "dataset_episodes": len(dataset),
                       "train": stats, "eval_student_discrete": ev})
    return {"rounds": rounds, "beta_schedule": [float(b) for b in beta_schedule]}


def write_json(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=float)
