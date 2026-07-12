"""
teacher.py — per-step PID teachers (the distillation / deep-supervision source).
================================================================================

NUMPY-ONLY (torch-free): importable inside the isolated sim venv.

Teacher contract (mirrored by :class:`mock_env.MockPIDTeacher`):

    teacher.reset()          # EVERY episode start — PID integral state carries over
                             # steps; forgetting this makes episode N depend on N-1
                             # and breaks replayability (recon pitfall).
    a = teacher.act(env)     # -> (n_act,) float64 normalized action in [-1, 1]

``act`` reads the PRIVILEGED true state via ``env.true_state()`` — never the masked
observation. The teacher is the label source for EVERY visited state during DAgger
collection, regardless of who flew.

HoverPIDTeacher (gym-pybullet-drones DSLPIDControl, Crazyflie-tuned cascaded PID):
DSLPIDControl outputs raw RPMs up to ~21.7k but HoverAviary's RPM action band is
only HOVER_RPM*(1 ± 0.05); the normalized action is the exact inverse of the env's
``_preprocessAction``:  a = (rpm / HOVER_RPM - 1) / 0.05, clipped to [-1, 1].
The CLIPPED action is the ONLY valid distillation target — it is what the env
executes; the raw pre-clip value is kept on ``last_rpm`` for the trajectory log.
"""

from __future__ import annotations

import numpy as np

# HoverAviary (BaseRLAviary._preprocessAction, ActionType.RPM):
#   rpm = HOVER_RPM * (1 + RPM_BAND * a).  0.05 is hardcoded upstream.
RPM_BAND = 0.05


class HoverPIDTeacher:
    """DSLPIDControl wrapped to the flightgate teacher contract.

    Lazy sim import so this module stays importable without gym-pybullet-drones.
    ``target``: hover target position (defaults to the env's own ``target`` at act
    time when None).
    """

    def __init__(self, target=None):
        try:
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.utils.enums import DroneModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "gym_pybullet_drones is not installed in this interpreter — "
                "HoverPIDTeacher needs the isolated sim venv (.venv-flight/ in WSL) "
                "or DUST. See mlgn/flightgate/README.md."
            ) from e
        self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.target = None if target is None else np.asarray(target, dtype=np.float64)
        self.last_rpm: np.ndarray | None = None      # raw pre-clip RPMs (log only)
        self.last_raw_action: np.ndarray | None = None  # pre-clip normalized (log only)

    def reset(self) -> None:
        self._ctrl.reset()  # clears the integral state
        self.last_rpm = None
        self.last_raw_action = None

    def act(self, env) -> np.ndarray:
        state = env.true_state()  # privileged 20-dim state (never the masked obs)
        target = self.target if self.target is not None else np.asarray(env.target)
        rpm, _pos_e, _yaw_e = self._ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target,
            target_rpy=np.zeros(3),
        )
        self.last_rpm = np.asarray(rpm, dtype=np.float64).reshape(-1)
        raw = (self.last_rpm / float(env.HOVER_RPM) - 1.0) / RPM_BAND
        self.last_raw_action = raw
        return np.clip(raw, -1.0, 1.0)


class MaskedObsHoverTeacher:
    """a' CEILING (D1 deliverable 3): DSLPIDControl fed the SAME masked observation
    the student sees, NOT the privileged true state. Answers "how much does occlusion
    cost the control law itself" — the honest upper bound on any student under blackout.

    obs -> state shim (VERIFIED exact against gym-pybullet-drones @ e712698a):
    the 12-dim KIN observation is ``[pos(0:3), rpy(3:6), vel(6:9), ang_vel(9:12)]`` ==
    ``hstack(state[0:3], state[7:10], state[10:16])``. DSLPIDControl.computeControlFrom
    State reads only pos=state[0:3], quat=state[3:7], vel=state[10:13], ang_vel=
    state[13:16]; state[7:10] (rpy) and state[16:20] (last action) are UNUSED by the
    control law. So the reconstruction is BIT-EXACT for unmasked frames:
      * pos, vel, ang_vel copied straight from the obs;
      * quat = getQuaternionFromEuler(rpy) — pybullet's own map, the exact inverse of
        the getEulerFromQuaternion that computeControl applies internally (round-trip
        verified allclose to state[3:7]); a numpy ZYX fallback is used if pybullet is
        unavailable (documented residual, only bites without pybullet — never on DUST).
    On a MASKED frame the wrapper zeroes the obs (sentinel) + sets valid=0, so the
    reconstructed state is origin-at-rest with identity attitude: the teacher flies
    "blind toward the last-known target from a zeroed belief" — that IS the definition
    of the occlusion cost this arm measures. RESIDUAL: none for unmasked frames; the
    masked-frame behaviour is the occlusion itself, not an approximation error.
    """

    def __init__(self, target=None):
        try:
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.utils.enums import DroneModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "gym_pybullet_drones is not installed in this interpreter — "
                "MaskedObsHoverTeacher (a' ceiling) needs the sim venv (.venv-flight/ "
                "in WSL) or DUST. See mlgn/flightgate/README.md."
            ) from e
        self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.target = None if target is None else np.asarray(target, dtype=np.float64)
        self.last_rpm: np.ndarray | None = None
        self.last_raw_action: np.ndarray | None = None
        try:
            import pybullet as _p  # noqa: F401
            self._quat_from_euler = lambda rpy: np.asarray(
                _p.getQuaternionFromEuler([float(rpy[0]), float(rpy[1]), float(rpy[2])]),
                dtype=np.float64)
        except Exception:  # pragma: no cover  (pybullet absent -> documented fallback)
            self._quat_from_euler = _euler_zyx_to_quat

    def reset(self) -> None:
        self._ctrl.reset()
        self.last_rpm = None
        self.last_raw_action = None

    def _reconstruct_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)[:12]
        state = np.zeros(20, dtype=np.float64)
        state[0:3] = obs[0:3]                       # pos
        state[3:7] = self._quat_from_euler(obs[3:6])  # quat from rpy (exact inverse)
        state[7:10] = obs[3:6]                      # rpy (unused by DSLPID; fill anyway)
        state[10:13] = obs[6:9]                     # vel
        state[13:16] = obs[9:12]                    # ang_vel
        return state                                # state[16:20] = last action (unused)

    def act_from_obs(self, obs, env) -> np.ndarray:
        """Return the (clipped, normalized) action from the MASKED obs. ``env`` supplies
        CTRL_TIMESTEP, HOVER_RPM and (fallback) target via the adapter delegation."""
        state = self._reconstruct_state(obs)
        target = self.target if self.target is not None else np.asarray(env.target)
        rpm, _pos_e, _yaw_e = self._ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target,
            target_rpy=np.zeros(3),
        )
        self.last_rpm = np.asarray(rpm, dtype=np.float64).reshape(-1)
        raw = (self.last_rpm / float(env.HOVER_RPM) - 1.0) / RPM_BAND
        self.last_raw_action = raw
        return np.clip(raw, -1.0, 1.0)


def _euler_zyx_to_quat(rpy) -> np.ndarray:
    """[x,y,z,w] quaternion from (roll,pitch,yaw), pybullet's Z*Y*X convention.
    Fallback ONLY when pybullet is unavailable (documented residual — on DUST the
    pybullet path is always taken, matching DSLPIDControl's internal inverse exactly)."""
    r, p, y = (float(v) for v in np.asarray(rpy).reshape(-1)[:3])
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    ], dtype=np.float64)


def make_teacher(backend: str):
    """Teacher factory: 'mock' -> MockPIDTeacher, 'hover' -> HoverPIDTeacher."""
    if backend == "mock":
        from .mock_env import MockPIDTeacher

        return MockPIDTeacher()
    if backend == "hover":
        return HoverPIDTeacher()
    raise ValueError(f"unknown backend {backend!r} (expected 'mock' or 'hover')")


def make_masked_teacher(backend: str):
    """a' CEILING factory: the teacher fed the MASKED observation (deliverable 3).
    'mock' -> MockMaskedObsTeacher, 'hover' -> MaskedObsHoverTeacher. Both expose the
    ``act_from_obs(obs, env)`` contract consumed by trainer.MaskedTeacherActor."""
    if backend == "mock":
        from .mock_env import MockMaskedObsTeacher

        return MockMaskedObsTeacher()
    if backend == "hover":
        return MaskedObsHoverTeacher()
    raise ValueError(f"unknown backend {backend!r} (expected 'mock' or 'hover')")
