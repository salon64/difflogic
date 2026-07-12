"""
mock_env.py — sim-independent mock: deterministic 2D point-mass + scripted teacher.
===================================================================================

NUMPY-ONLY on purpose (no torch, no gymnasium, no pybullet): the mock exists so the
ENTIRE closed-loop DAgger harness can be exercised end-to-end on any machine — the
mandatory correctness gate when the real simulator install is unavailable
(``test_trainer_mock.py``). It duck-types the gymnasium API surface the harness uses
(``reset(seed=) -> (obs, info)``, ``step(a) -> (obs, r, terminated, truncated, info)``,
``close()``) plus the flightgate extensions (``true_state()``, ``target``,
``position_from_state()``, ``n_act``, ``obs_dim``).

Dynamics (explicit Euler, fully deterministic given the reset seed):
    pos' = pos + dt * vel
    vel' = vel + dt * (A_MAX * u - DRAG * vel),   u = mean of each action pair
The action is 4 channels in [-1, 1] (mirroring the drone's 4 motors so the student
head shape is identical): channels (0, 1) average to the x acceleration command and
(2, 3) to the y command. Reward mirrors HoverAviary: ``max(0, 2 - dist**2)``.
Termination: never (like Hover's 1e-4 ball, effectively). Truncation: |px| or |py|
exceeds the envelope (2.0 m) or t >= max_steps.

Scripted teacher: a saturating PID toward the target WITH an integral term — the
integral state is there deliberately so that a harness that forgets to call
``teacher.reset()`` between episodes produces detectably different actions (the mock
test uses this to prove episode-reset correctness).
"""

from __future__ import annotations

import numpy as np

A_MAX = 1.0     # [m/s^2] acceleration at |u| = 1
DRAG = 0.2      # [1/s]   linear drag
DT = 0.1        # [s]     control step
ENVELOPE = 2.0  # [m]     |px|,|py| truncation bound (matches encode.MOCK_RANGES)


class MockPointMass2D:
    """Deterministic 2D point-mass 'hover to target' task. Observation [px,py,vx,vy]."""

    obs_dim = 4
    n_act = 4  # paired: (a0+a1)/2 -> ax command, (a2+a3)/2 -> ay command

    def __init__(self, target=(0.0, 0.0), max_steps: int = 40, jitter: bool = True):
        self.target = np.asarray(target, dtype=np.float64)
        self.max_steps = int(max_steps)
        self.jitter = bool(jitter)
        self._rng = np.random.default_rng(0)
        self._pos = np.zeros(2)
        self._vel = np.zeros(2)
        self._t = 0

    # -- gym-style API ------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.jitter:
            self._pos = self._rng.uniform(-1.2, 1.2, size=2)
            self._vel = self._rng.uniform(-0.3, 0.3, size=2)
        else:
            self._pos = np.array([1.0, -1.0])
            self._vel = np.zeros(2)
        self._t = 0
        return self._obs(), {}

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64).reshape(self.n_act), -1.0, 1.0)
        u = np.array([(a[0] + a[1]) / 2.0, (a[2] + a[3]) / 2.0])
        self._vel = self._vel + DT * (A_MAX * u - DRAG * self._vel)
        self._pos = self._pos + DT * self._vel
        self._t += 1
        dist = float(np.linalg.norm(self.target - self._pos))
        reward = max(0.0, 2.0 - dist ** 2)
        terminated = False
        truncated = bool(
            self._t >= self.max_steps or np.any(np.abs(self._pos) > ENVELOPE)
        )
        return self._obs(), reward, terminated, truncated, {}

    def close(self):
        pass

    # -- flightgate extensions ----------------------------------------------------
    def true_state(self) -> np.ndarray:
        """Privileged full state (== the observation here; the base env is fully
        observed — partial observability is added by the dropout WRAPPER only)."""
        return self._obs()

    @staticmethod
    def position_from_state(state: np.ndarray) -> np.ndarray:
        """Extract position from a true_state vector (for RMS-error metrics)."""
        return np.asarray(state)[..., 0:2]

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._pos, self._vel]).astype(np.float64)


class MockPIDTeacher:
    """Scripted saturating PID toward ``env.target``, reading the PRIVILEGED true
    state (never the masked observation). Carries integral state across steps —
    ``reset()`` MUST be called at every episode start (mirrors DSLPIDControl)."""

    def __init__(self, kp: float = 2.0, kd: float = 1.8, ki: float = 0.15):
        self.kp, self.kd, self.ki = kp, kd, ki
        self._integral = np.zeros(2)

    def reset(self):
        self._integral = np.zeros(2)

    def act(self, env) -> np.ndarray:
        return self._control(env.true_state(), env.target)

    def _control(self, state, target) -> np.ndarray:
        pos, vel = np.asarray(state)[0:2], np.asarray(state)[2:4]
        err = np.asarray(target) - pos
        self._integral = np.clip(self._integral + DT * err, -1.0, 1.0)
        u = self.kp * err + self.kd * (-vel) + self.ki * self._integral
        u = np.clip(u / A_MAX, -1.0, 1.0)
        # expand paired command -> 4 channels (both members of a pair identical)
        return np.array([u[0], u[0], u[1], u[1]], dtype=np.float64)


class MockMaskedObsTeacher(MockPIDTeacher):
    """a' CEILING on the mock (deliverable 3): the identical saturating PID as
    MockPIDTeacher but reading the MASKED observation [px,py,vx,vy] the student sees,
    NOT env.true_state(). On a masked frame the wrapper feeds constant-zero, so the
    teacher acts on a zeroed belief — the occlusion cost. The mock is fully observed,
    so the shim is trivial (obs[:4] == the state); it exists to exercise the a'
    plumbing end-to-end WITHOUT the simulator (mirrors MaskedObsHoverTeacher's API)."""

    def act_from_obs(self, obs, env) -> np.ndarray:
        return self._control(np.asarray(obs, dtype=np.float64).reshape(-1)[:4],
                             env.target)
