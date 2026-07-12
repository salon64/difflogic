"""
env.py — real-sim wrapper (gym-pybullet-drones HoverAviary) + observation-dropout.
==================================================================================

Two layers:

1. :class:`SensorBlackout` — the MEMORY-REQUIRED POMDP variant. Wraps ANY env whose
   observation is a flat float vector and masks it for ``k`` consecutive steps with a
   CONSTANT ZERO sentinel + a validity flag appended as a new last dimension. It is a
   plain duck-typed wrapper (NOT gymnasium.Wrapper) so it also wraps the numpy-only
   mock env in environments without gymnasium installed. HARD RULES (recon-verified):
   * masked frames are constant-zero + valid=0 — NEVER freeze-last-value (that would
     make the wrapper itself the memory and void the memory-required claim);
   * the wrapper owns its own seeded ``np.random.default_rng`` (recorded per run);
   * the per-episode mask sequence is exposed (``mask_history``) for trajectory logs.

2. :func:`make_hover_env` / :class:`HoverAdapter` — the real simulator. Imports
   gym_pybullet_drones LAZILY so this module stays importable (and the mock path
   fully functional) when the sim is not installed. The adapter:
   * flattens HoverAviary's (1, 72) KIN observation to the FIRST 12 dims only —
     dims 12:72 are a 60-dim buffer of the last 15 actions, a hidden memory channel
     that would let the "feedforward" baseline cheat the POMDP gate (recon pitfall);
   * takes flat (4,) normalized actions in [-1, 1] and reshapes to (1, 4);
   * exposes the privileged 20-dim true state (``true_state()``) for the teacher;
   * applies per-episode initial-state JITTER at construction (HoverAviary has NO
     built-in randomization — multi-seed eval is meaningless without this).
   Because jitter is a constructor argument in gym-pybullet-drones, closed-loop
   collection uses an ENV FACTORY (fresh env per episode, ``close()`` after — each
   Aviary owns a pybullet client and leaks it otherwise).

Local machine note: pybullet has no Windows/cp313 wheels — the real path runs only
inside the WSL venv (``.venv-flight/``) or on DUST. See README.md.
"""

from __future__ import annotations

import numpy as np


class SensorBlackout:
    """Mask the wrapped env's observation for k consecutive steps (constant-zero
    sentinel), appending a validity bit. Observation becomes [obs_dim + 1].

    p_start: per-step probability that a new blackout begins (when none is active).
    k_min, k_max: blackout length is drawn uniformly from [k_min, k_max] (inclusive).
    mask_dims: slice or index array of the dims to zero (default: all).
    """

    def __init__(self, env, p_start: float = 0.02, k_min: int = 6, k_max: int = 12,
                 mask_dims=None, seed: int = 0):
        assert 0.0 <= p_start <= 1.0 and 1 <= k_min <= k_max, (p_start, k_min, k_max)
        self.env = env
        self.p_start, self.k_min, self.k_max = float(p_start), int(k_min), int(k_max)
        self.mask_dims = mask_dims  # None = mask everything
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.blackout_left = 0
        self.mask_history: list[int] = []  # 1 = masked, per emitted observation

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim + 1  # + validity bit

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.blackout_left = 0
        self.mask_history = []
        return self._mask(np.asarray(obs, dtype=np.float64)), info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        if self.blackout_left == 0 and self.rng.random() < self.p_start:
            self.blackout_left = int(self.rng.integers(self.k_min, self.k_max + 1))
        return self._mask(np.asarray(obs, dtype=np.float64)), r, term, trunc, info

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        o = obs.copy()
        valid = 1.0
        if self.blackout_left > 0:
            if self.mask_dims is None:
                o[:] = 0.0
            else:
                o[self.mask_dims] = 0.0
            valid = 0.0
            self.blackout_left -= 1
        self.mask_history.append(0 if valid else 1)
        return np.concatenate([o, [valid]])

    def config(self) -> dict:
        return {"p_start": self.p_start, "k_min": self.k_min, "k_max": self.k_max,
                "seed": self.seed,
                "mask_dims": None if self.mask_dims is None else str(self.mask_dims)}

    def __getattr__(self, name):  # delegate true_state(), target, close(), ...
        return getattr(self.env, name)


# ------------------------------------------------------------------------------------
# Real simulator: gym-pybullet-drones HoverAviary (lazy imports)
# ------------------------------------------------------------------------------------

HOVER_TARGET = np.array([0.0, 0.0, 1.0])

# ---- per-episode initial-state jitter (HoverAviary has NO built-in randomization) ---
# Recorded in every run JSON (cli.py: record["hover_jitter"]) for replayability.
#
# HISTORY (D1 deliverable 4, 2026-07-12): the ORIGINAL recon-§6 jitter was too
# aggressive for the RAW DSLPIDControl teacher — with yaw U(-pi, pi) and 0.15 rad
# initial tilt the heading-recovery transient couples into roll/pitch and saturates
# the +/-5 % RPM band, driving the teacher PAST HoverAviary's |roll|,|pitch| > 0.4
# truncation envelope in ~1/2 of episodes (README caveat; ceiling arm a' broken).
# A broken teacher makes BOTH the quantization gate and the a'-ceiling meaningless,
# so the DEFAULT is tempered to keep the oracle in-envelope. Old values kept below
# for provenance / ablation; measured exit rates are in report notes.
#
#   dim     RECON-6 (old)          TEMPERED default (new)     why
#   x,y     U(-0.15, 0.15)         U(-0.10, 0.10)             smaller lateral offset
#   z       U(0.60,  1.40)         U(0.85,  1.15)             less climb/descent transient
#   roll    U(-0.15, 0.15)         U(-0.05, 0.05)             < envelope, small recovery
#   pitch   U(-0.15, 0.15)         U(-0.05, 0.05)             < envelope, small recovery
#   yaw     U(-pi,   pi)           U(-pi/4, pi/4)             heading->tilt coupling tamed
#
# The task stays non-trivial (off-target spawn + attitude error the teacher must null);
# it just no longer self-exits before the student even sees a blackout.
HOVER_JITTER_RECON6 = {  # ORIGINAL (kept for ablation; DO NOT use as default)
    "x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.6, 1.4),
    "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15), "yaw": (-np.pi, np.pi),
}
HOVER_JITTER = {  # tempered DEFAULT (deliverable 4) — keeps the raw PID in-envelope
    "x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (0.85, 1.15),
    "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-np.pi / 4, np.pi / 4),
}


class HoverAdapter:
    """Flatten HoverAviary to the flightgate contract: 12-dim obs, (4,) actions,
    privileged 20-dim true state. See module docstring for why dims 12:72 are cut."""

    obs_dim = 12
    n_act = 4

    def __init__(self, aviary):
        self.aviary = aviary
        self.target = np.array(getattr(aviary, "TARGET_POS", HOVER_TARGET),
                               dtype=np.float64).reshape(3)

    def reset(self, seed: int | None = None):
        obs, info = self.aviary.reset(seed=seed)
        return np.asarray(obs, dtype=np.float64).reshape(-1)[: self.obs_dim], info

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32).reshape(1, self.n_act), -1.0, 1.0)
        obs, r, term, trunc, info = self.aviary.step(a)
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)[: self.obs_dim]
        return obs, float(r), bool(term), bool(trunc), info

    def true_state(self) -> np.ndarray:
        """Privileged 20-dim state [pos 0:3, quat 3:7, rpy 7:10, vel 10:13,
        ang_vel 13:16, last_clipped_rpm_action 16:20] — teacher input ONLY."""
        return np.asarray(self.aviary._getDroneStateVector(0), dtype=np.float64)

    @staticmethod
    def position_from_state(state: np.ndarray) -> np.ndarray:
        return np.asarray(state)[..., 0:3]

    def close(self):
        self.aviary.close()

    def __getattr__(self, name):  # HOVER_RPM, CTRL_TIMESTEP, ... for the teacher
        return getattr(self.aviary, name)


def make_hover_env(jitter_rng: np.random.Generator | None = None, gui: bool = False,
                   jitter: dict | None = None):
    """Construct ONE fresh, optionally jittered HoverAviary episode env.

    jitter_rng: a run-level np.random.default_rng; when given, samples the initial
        position/attitude from ``jitter`` (default :data:`HOVER_JITTER`, the tempered
        deliverable-4 default; pass :data:`HOVER_JITTER_RECON6` to reproduce the old
        aggressive spec). Six uniform draws are consumed in the fixed order
        x, y, z, roll, pitch, yaw so a given ``jitter_rng`` seed stays replayable.
    Caller MUST env.close() after the episode (pybullet client leak).
    Raises ImportError with install pointers if gym-pybullet-drones is absent.
    """
    try:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ObservationType, ActionType
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "gym_pybullet_drones is not installed in this interpreter. Use the "
            "isolated sim venv (.venv-flight/ inside WSL) or install on DUST — see "
            "mlgn/flightgate/README.md 'Install'."
        ) from e

    j = HOVER_JITTER if jitter is None else jitter
    kwargs = dict(gui=gui, obs=ObservationType.KIN, act=ActionType.RPM)
    if jitter_rng is not None:
        xyz = np.array([[jitter_rng.uniform(*j["x"]),
                         jitter_rng.uniform(*j["y"]),
                         jitter_rng.uniform(*j["z"])]])
        rpy = np.array([[jitter_rng.uniform(*j["roll"]),
                         jitter_rng.uniform(*j["pitch"]),
                         jitter_rng.uniform(*j["yaw"])]])
        kwargs.update(initial_xyzs=xyz, initial_rpys=rpy)
    return HoverAdapter(HoverAviary(**kwargs))


def hover_env_factory(run_rng: np.random.Generator | None, blackout: dict | None,
                      episode_index: int = 0):
    """Factory used by the collector: fresh (jittered) HoverAdapter per episode,
    optionally wrapped in SensorBlackout. The blackout wrapper seed is derived
    deterministically from its base seed + episode index so every episode's mask
    sequence is replayable."""
    env = make_hover_env(jitter_rng=run_rng)
    if blackout is not None:
        b = dict(blackout)
        b["seed"] = int(b.get("seed", 0)) + episode_index
        env = SensorBlackout(env, **b)
    return env
