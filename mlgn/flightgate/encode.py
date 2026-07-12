"""
encode.py — thermometer binarization of observation vectors + action discretization.
=====================================================================================

NUMPY-ONLY on purpose: this module must be importable inside the isolated sim venv
(``.venv-flight/``, which has NO torch) as well as in the main training environment.

Thermometer encoding
--------------------
Per observation dimension ``d`` with a fixed range ``[lo_d, hi_d]`` and ``T`` bits,
bit ``i`` fires iff ``x_d > lo_d + (i+1) * (hi_d - lo_d) / (T+1)`` — ``T`` monotone
threshold bits (a unary/ordinal code). Values are clipped to the range before
encoding. The encoder is exactly reproducible from its config dict
(:meth:`ThermometerEncoder.to_config`), which every run JSON records — the future
netlist exporter needs these thresholds bit-exactly.

Ranges (documented per the D1 recon spec)
-----------------------------------------
* ``HOVER_RANGES`` — gym-pybullet-drones ``HoverAviary`` 12-dim KIN observation
  ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]``. Position/attitude ranges
  are the env's own truncation bounds (|x|,|y|>1.5, z>2.0, |roll|,|pitch|>0.4,
  yaw in [-pi,pi]). Velocity ranges are PLACEHOLDER defaults (±1.5 m/s, ±3 rad/s per
  the recon estimate) — CALIBRATE from >=50 jittered teacher episodes with
  :func:`calibrate_ranges` (0.5/99.5 percentiles, padded x1.5) and freeze the result
  in the run JSON before any real training run.
* ``MOCK_RANGES`` — the 2D point-mass mock env observation ``[px, py, vx, vy]``
  (positions clipped at ±2 = the mock's envelope bound, velocities ±3).

Action discretization
---------------------
:class:`ActionDiscretizer` maps each continuous action channel in ``[-1, 1]`` to one
of ``n_bins`` uniform bins (per-step CE targets for distillation = deep supervision)
and back to bin centers (the executed student action).
"""

from __future__ import annotations

import numpy as np

# HoverAviary KIN observation, dims 0:12. See module docstring for provenance.
HOVER_RANGES: tuple[tuple[float, float], ...] = (
    (-1.5, 1.5),                    # x  [m]   (truncation bound)
    (-1.5, 1.5),                    # y  [m]   (truncation bound)
    (0.0, 2.0),                     # z  [m]   (truncation bound)
    (-0.4, 0.4),                    # roll  [rad] (truncation bound)
    (-0.4, 0.4),                    # pitch [rad] (truncation bound)
    (-np.pi, np.pi),                # yaw   [rad]
    (-1.5, 1.5),                    # vx [m/s]   PLACEHOLDER — calibrate before real runs
    (-1.5, 1.5),                    # vy [m/s]   PLACEHOLDER — calibrate
    (-1.5, 1.5),                    # vz [m/s]   PLACEHOLDER — calibrate
    (-3.0, 3.0),                    # wx [rad/s] PLACEHOLDER — calibrate
    (-3.0, 3.0),                    # wy [rad/s] PLACEHOLDER — calibrate
    (-3.0, 3.0),                    # wz [rad/s] PLACEHOLDER — calibrate
)

# Mock 2D point-mass observation [px, py, vx, vy] (see mock_env.py).
MOCK_RANGES: tuple[tuple[float, float], ...] = (
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-3.0, 3.0),
    (-3.0, 3.0),
)


class ThermometerEncoder:
    """Thermometer-binarize a float observation vector; optionally pass through a
    trailing validity bit (from the observation-dropout wrapper) unencoded.

    encode() input:  [obs_dim] floats            -> [obs_dim * bits] {0,1} float32
    encode_with_valid() input: [obs_dim + 1]     -> [obs_dim * bits + 1] {0,1} float32
        (last input element is the already-binary validity flag, appended as-is)
    Both also accept a leading batch/time axis ([..., obs_dim(+1)]).
    """

    def __init__(self, ranges, bits: int = 16):
        assert bits >= 1, bits
        self.ranges = tuple((float(lo), float(hi)) for lo, hi in ranges)
        for lo, hi in self.ranges:
            assert hi > lo, (lo, hi)
        self.bits = int(bits)
        self.obs_dim = len(self.ranges)
        lo = np.array([r[0] for r in self.ranges], dtype=np.float64)
        hi = np.array([r[1] for r in self.ranges], dtype=np.float64)
        # thresholds[d, i] = lo_d + (i+1)*(hi_d-lo_d)/(bits+1)  — strictly inside (lo, hi)
        steps = (np.arange(self.bits, dtype=np.float64) + 1.0) / (self.bits + 1.0)
        self._thresholds = lo[:, None] + (hi - lo)[:, None] * steps[None, :]
        self._lo, self._hi = lo, hi

    @property
    def n_bits(self) -> int:
        return self.obs_dim * self.bits

    @property
    def n_bits_with_valid(self) -> int:
        return self.n_bits + 1

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[-1] == self.obs_dim, (x.shape, self.obs_dim)
        xc = np.clip(x, self._lo, self._hi)
        bits = (xc[..., :, None] > self._thresholds).astype(np.float32)
        return bits.reshape(*x.shape[:-1], self.n_bits)

    def encode_with_valid(self, obs_v: np.ndarray) -> np.ndarray:
        obs_v = np.asarray(obs_v, dtype=np.float64)
        assert obs_v.shape[-1] == self.obs_dim + 1, (obs_v.shape, self.obs_dim)
        bits = self.encode(obs_v[..., : self.obs_dim])
        valid = obs_v[..., self.obs_dim :].astype(np.float32)
        assert np.all((valid == 0.0) | (valid == 1.0)), "validity flag must be binary"
        return np.concatenate([bits, valid], axis=-1)

    def to_config(self) -> dict:
        """Frozen, JSON-serializable config (record in every run JSON)."""
        return {
            "type": "thermometer",
            "bits": self.bits,
            "ranges": [list(r) for r in self.ranges],
            "thresholds": self._thresholds.tolist(),
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "ThermometerEncoder":
        enc = cls(cfg["ranges"], bits=cfg["bits"])
        got = np.asarray(cfg["thresholds"], dtype=np.float64)
        assert np.allclose(got, enc._thresholds), "config thresholds do not match"
        return enc


class ActionDiscretizer:
    """Uniform binning of continuous action channels in [lo, hi] (default [-1, 1]).

    to_bins(a)    : [..., n_act] floats -> [..., n_act] int64 bin indices (CE targets)
    to_values(b)  : [..., n_act] indices -> [..., n_act] float32 bin centers
    """

    def __init__(self, n_bins: int = 9, lo: float = -1.0, hi: float = 1.0):
        assert n_bins >= 2, n_bins
        self.n_bins = int(n_bins)
        self.lo, self.hi = float(lo), float(hi)
        edges = np.linspace(self.lo, self.hi, self.n_bins + 1)
        self._inner_edges = edges[1:-1]
        self.centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)

    def to_bins(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(a, dtype=np.float64), self.lo, self.hi)
        return np.digitize(a, self._inner_edges).astype(np.int64)

    def to_values(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.int64)
        assert np.all((b >= 0) & (b < self.n_bins)), "bin index out of range"
        return self.centers[b]

    def quantize(self, a: np.ndarray) -> np.ndarray:
        """Round-trip a continuous action through the bins (the quantization-gate check:
        replaying TEACHER actions through this must still solve the task)."""
        return self.to_values(self.to_bins(a))

    def to_config(self) -> dict:
        return {"type": "uniform", "n_bins": self.n_bins, "lo": self.lo, "hi": self.hi,
                "centers": self.centers.tolist()}


def calibrate_ranges(
    values: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
    pad: float = 1.5,
    fixed: dict[int, tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Percentile-calibrate per-dimension thermometer ranges from rollout data.

    values: [N, obs_dim] raw (UNMASKED) observations from teacher episodes.
    fixed:  {dim_index: (lo, hi)} entries that stay fixed (e.g. the truncation-bound
            position/attitude dims) — only the remaining dims are calibrated.
    Returns a list of (lo, hi) per dim: percentile band widened by ``pad`` about its
    midpoint. Freeze the result in the run JSON.
    """
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 2, values.shape
    out: list[tuple[float, float]] = []
    for d in range(values.shape[1]):
        if fixed is not None and d in fixed:
            out.append((float(fixed[d][0]), float(fixed[d][1])))
            continue
        lo = float(np.percentile(values[:, d], lo_pct))
        hi = float(np.percentile(values[:, d], hi_pct))
        mid, half = (hi + lo) / 2.0, max((hi - lo) / 2.0, 1e-6)
        out.append((mid - pad * half, mid + pad * half))
    return out
