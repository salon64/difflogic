"""
can_data.py — CAN-bus intrusion-detection tasks for the seqlgn harness (P3b gate C0.g).
========================================================================================

Turns raw CAN traffic (ROAD-style candump logs, HCRL-style CSVs, or a synthetic
generator) into the harness's standard batch contract: ``x: [batch, seq_len, input_dim]``
with values in {0,1} **by construction** (so ``evaluate()``'s ``x.round()`` is a no-op)
and either a final-message label ``y: [batch]`` or per-message labels ``y: [batch, T]``
(``per_step=True`` — feeds ``--deep-sup`` natively).

Tasks (registered in ``data.get_task``)
---------------------------------------
- ``can``     — real data from ``<repo>/data-can/<source>/`` (source: ``road`` | ``hcrl``).
- ``can-syn`` — zero-file synthetic fixture (CPU-fast smoke + non-vacuity control): periodic
  per-ID message schedules with masquerade-style (period change on a spoofed ID) and
  suspension-style (ID goes silent) TIMING attacks. A recurrent cell can carry per-ID
  timing state across the whole stream while the windowed-FF arm sees only its window —
  the synthetic separator should reproduce the expected direction; if it does not, the
  pipeline (not the hypothesis) is suspect.

Per-frame encoding (all bits {0,1} by construction)
---------------------------------------------------
``[ id bits | Δt(same-ID) thermometer | (Δt global thermometer) | (payload bits) ]``

- id bits: ``id_enc='onehot'`` = top-N train-frequency IDs one-hot + 1 'other' bucket
  (width ``min(N, |train vocab|) + 1``); ``'binary'`` = the raw 11-bit arbitration ID.
- Δt thermometer: ``bit[i] = (Δt >= edges[i])`` with ``dt_bins`` **log-spaced** edges
  between the 1st/99th percentile of the TRAIN-segment Δt distribution (edges are fit on
  the train split only, frozen for val/test, and persisted into the results JSON —
  encoder-calibration leakage guard). Δt = time since the previous frame of the SAME ID
  within the same chunk; the first frame of an ID in a chunk gets Δt = +inf = all bits 1
  ("gap longer than any bin" — which is also what a suspension looks like).
- payload: ``payload_bytes`` raw bytes -> ``8 * payload_bytes`` bits (default 0; ROAD
  masquerade attacks are frequency-preserving, so payload/content bits are what carries
  them — set ``--can-payload-bytes 8`` for the ROAD masquerade queue).

Splits (leakage discipline — NEVER copy the mnist/synthetic split patterns)
---------------------------------------------------------------------------
CAN frames repeat near-verbatim every few ms: any message- or window-level shuffle before
splitting puts near-duplicates in train AND test. Rules enforced here:
1. per attack-type capture group (captures sorted by name): n==1 -> contiguous-in-TIME
   split at the 70%/80% time points; n==2 -> [train, second-capture time-split into
   val|test]; n>=3 -> whole-capture holdout [train..., val, test].
2. Windows are built WITHIN a (capture, split) chunk only — they never straddle a split
   or capture boundary. Δt is likewise computed within-chunk (reset at boundaries).
3. Only train WINDOWS are shuffled (DataLoader shuffle=True), after the split.
4. The split is time/capture-based and seed-INDEPENDENT: all arms/seeds see identical data.

Labeling conventions
--------------------
ROAD raw logs carry no label column; labels are derived from the per-capture metadata
JSON (``injection_id``, ``injection_data_str`` with 'X' nibble wildcards,
``injection_interval`` = [start, end] in seconds elapsed since the capture's first frame):
- masquerade variant: EVERY frame of ``injection_id`` inside the interval is an attack
  frame (the legitimate ones were removed in post-processing).
- fabrication: attack iff in-interval AND ``id == injection_id`` AND the payload matches
  ``injection_data_str`` (X = wildcard nibble).
- no injection_id (fuzzing-style): every frame in the interval — coarse; fuzzing is NOT
  gate evidence (workmap §C0.g), supported only so the loader doesn't crash on it.
- ``attack_type == 'suspension'`` (synthetic extension; ROAD has none): every frame from
  ANY ID inside the interval is labeled 1 — "the bus is under attack". Per-message labels
  are ill-defined for an ABSENT ID (cf. SynCAN suppress), so this interval-level
  convention is the documented approximation.

RNG discipline (hard requirement — netlist wiring replay, netlist/README.md)
----------------------------------------------------------------------------
Builders here consume ZERO draws from the global torch CPU generator: all randomness
(synthetic generation only) goes through a dedicated ``torch.Generator`` seeded with the
FIXED ``_SYN_SEED`` (independent of ``--seed``, so seeds are paired across arms on
identical data and ``build_task()`` regeneration is bit-identical). Verified by
``test_can.py`` (rng-state fingerprint before == after).

Real data lives in ``<repo>/data-can/`` (mirrors ``data-mnist/``; NOT in git — see
``mlgn/seqlgn/data/can/README.md`` for download instructions). The checked-in fixture in
``mlgn/seqlgn/data/can/`` is written by ``python -m mlgn.seqlgn.can_data --write-fixture``
in the EXACT ROAD raw schema (candump ``(ts) iface ID#HEX`` lines + per-capture metadata
JSON) so the unit tests exercise the same parse->label->encode->split path as real data.

NOTE for netlist export (not C0.g-blocking): ``mlgn/netlist/extract.py`` RunSpec raises
for unknown tasks — a 'can' branch (reading ``input_dim`` from the results JSON, which
train.py now records) is needed before any CAN checkpoint is exported/verified.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os
import re
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from .data import TaskSpec

# real datasets (NOT in git; see mlgn/seqlgn/data/can/README.md)
CAN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data-can")
# tiny checked-in synthetic fixture (exact ROAD raw schema; <1 MB)
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "can")

# Fixed seed for the synthetic stream — INDEPENDENT of --seed by design: the data must be
# identical across arms/seeds (paired comparison) and bit-identically regenerable for the
# netlist accuracy gate. (Same pattern as data.py's fixed psmnist permutation seed.)
_SYN_SEED = 20260712

_SYN_IDS = (0x0B0, 0x110, 0x185, 0x224, 0x2A9, 0x333)
_SYN_JITTER = 0.10          # +-10% uniform jitter on each inter-arrival gap
_MASQ_IDX = 2               # 0x185 (period ~25 ms) — masquerade target
_SUSP_IDX = 1               # 0x110 (period ~16 ms) — suspension target


# --------------------------------------------------------------------------------------
# Capture container
# --------------------------------------------------------------------------------------
@dataclass
class Capture:
    """One contiguous CAN trace with per-message labels (already time-sorted)."""
    name: str                      # capture identity (file stem or synthetic name)
    group: str                     # split group: attack type ('ambient' for benign)
    ts: torch.Tensor               # [N] float64, seconds (non-decreasing)
    ids: torch.Tensor              # [N] int64 arbitration IDs
    dlc: torch.Tensor              # [N] int64
    payload: torch.Tensor          # [N, 8] uint8 (zero-padded)
    labels: torch.Tensor           # [N] int64, 0 normal / 1 attack
    meta: dict | None = None       # capture metadata (attack captures)
    extra: dict = field(default_factory=dict)   # generator bookkeeping (tests only)


# --------------------------------------------------------------------------------------
# Parsing — ROAD raw candump logs + metadata, HCRL CSVs
# --------------------------------------------------------------------------------------
def parse_candump(path: str):
    """Parse a can-utils candump log: ``(1364236952.554636) can0 0B4#00000813000013FC``.

    Returns ``(ts float64 [N], ids int64 [N], dlc int64 [N], payload uint8 [N, 8])``,
    time-sorted (stable). Malformed lines are skipped (defensive). Pure python line loop:
    ~1-2 us/line, so a multi-million-line ROAD ambient capture takes tens of seconds —
    a one-time cost per run."""
    ts_l: list[float] = []
    id_l: list[int] = []
    dlc_l: list[int] = []
    pl_chunks: list[bytes] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            try:
                tpart, rest = line.split(")", 1)
                t = float(tpart[1:])
                frame = rest.split()[-1]           # last token = "ID#HEXDATA"
                idpart, sep, datapart = frame.partition("#")
                if not sep:
                    continue
                cid = int(idpart, 16)
                datapart = datapart.strip()
                if datapart[:1].upper() == "R":    # remote frame: no data payload
                    data = b""
                else:
                    data = bytes.fromhex(datapart) if datapart else b""
            except (ValueError, IndexError):
                continue
            dlc = min(len(data), 8)
            ts_l.append(t)
            id_l.append(cid)
            dlc_l.append(dlc)
            pl_chunks.append(data[:8] + b"\x00" * (8 - dlc))
    n = len(ts_l)
    if n == 0:
        raise ValueError(f"no parseable candump frames in {path!r}")
    ts = torch.tensor(ts_l, dtype=torch.float64)
    ids = torch.tensor(id_l, dtype=torch.int64)
    dlc = torch.tensor(dlc_l, dtype=torch.int64)
    payload = torch.frombuffer(bytearray(b"".join(pl_chunks)), dtype=torch.uint8).reshape(n, 8).clone()
    if not bool((ts[1:] >= ts[:-1]).all()):
        perm = torch.argsort(ts, stable=True)      # defensive: enforce time order
        ts, ids, dlc, payload = ts[perm], ids[perm], dlc[perm], payload[perm]
    return ts, ids, dlc, payload


def find_metadata(log_path: str) -> dict | None:
    """Locate the per-capture metadata JSON for a log file (several ROAD layouts).

    Tries ``<log>.json`` / ``<stem>.json`` next to the log, then a shared
    ``capture_metadata.json`` in the log's directory or its parent (keyed by stem).

    VERIFIED against the real ROAD archive (road.zip, MD5 cab184cf…, 2026-07-12): the
    attack logs live in ``road/attacks/`` alongside a shared ``capture_metadata.json``
    keyed by bare capture stem (e.g. ``max_speedometer_attack_1_masquerade``) — the
    log's-directory branch below. Each entry carries ``injection_id`` as a ``0x``-hex
    string (``_parse_inj_id`` handles the prefix), ``injection_interval`` in elapsed
    seconds, ``injection_data_str`` (16 hex nibbles with ``X`` wildcards), and a
    ``modified`` bool that is ``true`` on the masquerade variant. There is NO ``masquerade``
    or ``attack_type`` key, so masquerade is recognised by the ``_masquerade`` filename
    suffix in :func:`label_capture` (every masquerade capture is named ``*_masquerade.log``).
    Ambient captures carry a metadata entry with null injection fields → all-benign labels."""
    d = os.path.dirname(log_path)
    stem = os.path.splitext(os.path.basename(log_path))[0]
    for cand in (log_path + ".json", os.path.join(d, stem + ".json")):
        if os.path.isfile(cand):
            with open(cand) as f:
                return json.load(f)
    for cand in (os.path.join(d, "capture_metadata.json"),
                 os.path.join(os.path.dirname(d), "capture_metadata.json")):
        if os.path.isfile(cand):
            with open(cand) as f:
                table = json.load(f)
            if isinstance(table, dict):
                for key in (stem, stem + ".log", os.path.basename(log_path)):
                    if key in table:
                        return table[key]
    return None


def _parse_inj_id(v) -> int | None:
    """injection_id field -> int arbitration ID, or None (fuzzing 'XXX' / absent)."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip().lower()
    if not s or "x" in s.replace("0x", ""):        # 'XXX' / 'X' = no single target ID
        return None
    try:
        return int(s, 16)
    except ValueError:
        return None


def _payload_matches(payload: torch.Tensor, data_str: str) -> torch.Tensor:
    """[N] bool: payload matches ``data_str`` (hex nibbles, 'X' = wildcard nibble)."""
    s = re.sub(r"[^0-9a-fA-FxX]", "", str(data_str))
    n = payload.shape[0]
    ok = torch.ones(n, dtype=torch.bool)
    for i, ch in enumerate(s[:16]):
        if ch in "xX":
            continue
        byte = payload[:, i // 2].to(torch.int64)
        nib = (byte >> 4) if i % 2 == 0 else (byte & 0xF)
        ok &= nib == int(ch, 16)
    return ok


def label_capture(ts, ids, payload, dlc, meta: dict | None, name: str = "") -> torch.Tensor:
    """Derive per-message labels from capture metadata (see module docstring rules)."""
    n = ts.numel()
    labels = torch.zeros(n, dtype=torch.int64)
    if not meta:
        return labels
    interval = meta.get("injection_interval") or meta.get("interval") or meta.get("attack_interval")
    if not interval:
        return labels
    s, e = float(interval[0]), float(interval[1])
    elapsed = ts - ts[0]
    in_int = (elapsed >= s) & (elapsed <= e)
    atk = str(meta.get("attack_type") or "").lower()
    inj = _parse_inj_id(meta.get("injection_id"))
    is_masq = bool(meta.get("masquerade")) or "masquerade" in atk or "masquerade" in name.lower()
    if atk == "suspension":
        labels[in_int] = 1                          # interval-level convention (docstring)
    elif inj is None:
        labels[in_int] = 1                          # fuzzing-style; NOT gate evidence
    elif is_masq:
        labels[in_int & (ids == inj)] = 1           # legit target frames were removed
    else:                                           # fabrication: match injected payload
        m = in_int & (ids == inj)
        ds = meta.get("injection_data_str")
        if ds:
            m &= _payload_matches(payload, ds)
        labels[m] = 1
    return labels


def parse_hcrl_csv(path: str):
    """HCRL Car-Hacking CSV: ``Timestamp,ID(hex),DLC,DATA0..DATA<dlc-1>,Flag(T/R)``.

    Returns ``(ts, ids, dlc, payload, labels)`` (labels from the Flag column). The
    differently-formatted attack-free .txt file is NOT supported (documented in the data
    README). Sanity-row dataset only — never gate evidence (research/22 + web recon)."""
    ts_l, id_l, dlc_l, lab_l = [], [], [], []
    pl_chunks: list[bytes] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                cid = int(parts[1], 16)
                dlc = int(parts[2])
            except ValueError:
                continue                            # header / malformed line
            data_hex = parts[3:3 + dlc]
            flag = parts[3 + dlc] if len(parts) > 3 + dlc else "R"
            try:
                data = bytes(int(b, 16) for b in data_hex)
            except ValueError:
                continue
            dlc = min(len(data), 8)
            ts_l.append(t)
            id_l.append(cid)
            dlc_l.append(dlc)
            lab_l.append(1 if flag.upper().startswith("T") else 0)
            pl_chunks.append(data[:8] + b"\x00" * (8 - dlc))
    n = len(ts_l)
    if n == 0:
        raise ValueError(f"no parseable HCRL rows in {path!r}")
    ts = torch.tensor(ts_l, dtype=torch.float64)
    ids = torch.tensor(id_l, dtype=torch.int64)
    dlc = torch.tensor(dlc_l, dtype=torch.int64)
    payload = torch.frombuffer(bytearray(b"".join(pl_chunks)), dtype=torch.uint8).reshape(n, 8).clone()
    labels = torch.tensor(lab_l, dtype=torch.int64)
    if not bool((ts[1:] >= ts[:-1]).all()):
        perm = torch.argsort(ts, stable=True)
        ts, ids, dlc, payload, labels = ts[perm], ids[perm], dlc[perm], payload[perm], labels[perm]
    return ts, ids, dlc, payload, labels


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------
def per_id_dt(ts: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[N] float64: time since the previous frame of the SAME ID; +inf at first
    occurrence (within the chunk — Δt never crosses a split/capture boundary)."""
    n = ts.numel()
    dt = torch.full((n,), float("inf"), dtype=torch.float64)
    if n < 2:
        return dt
    perm = torch.argsort(ids, stable=True)          # stable: time order kept within ID
    ts_s, ids_s = ts[perm], ids[perm]
    gaps = ts_s[1:] - ts_s[:-1]
    same = ids_s[1:] == ids_s[:-1]
    dt[perm[1:]] = torch.where(same, gaps, torch.full_like(gaps, float("inf")))
    return dt


def global_dt(ts: torch.Tensor) -> torch.Tensor:
    """[N] float64: time since the previous frame of ANY ID; +inf at the first frame."""
    n = ts.numel()
    dt = torch.full((n,), float("inf"), dtype=torch.float64)
    if n >= 2:
        dt[1:] = ts[1:] - ts[:-1]
    return dt


def fit_dt_edges(dts: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Log-spaced thermometer edges between the 1st/99th percentile of the finite,
    positive Δt values. MUST be fit on the TRAIN segment only (caller's contract)."""
    v = dts[torch.isfinite(dts) & (dts > 0)].to(torch.float64)
    if v.numel() == 0:                              # defensive fallback
        return torch.logspace(-4, 0, n_bins, dtype=torch.float64)
    if v.numel() > 4_000_000:                       # torch.quantile size limit; deterministic
        v = v[:: v.numel() // 4_000_000 + 1]
    lo = max(float(torch.quantile(v, 0.01)), 1e-6)
    hi = max(float(torch.quantile(v, 0.99)), lo * (1.0 + 1e-6))
    return torch.logspace(math.log10(lo), math.log10(hi), n_bins, dtype=torch.float64)


def thermometer(v: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """[N, n_bins] uint8: bit[i] = (v >= edges[i]). Monotone (1..10..0 pattern);
    v=+inf -> all ones."""
    return (v.unsqueeze(1) >= edges.unsqueeze(0)).to(torch.uint8)


def fit_id_vocab(ids: torch.Tensor, top_n: int) -> torch.Tensor:
    """Top-``top_n`` IDs by TRAIN-segment frequency (ties broken by ID value for
    determinism), returned SORTED ascending (one-hot slot order = sorted order)."""
    uniq, counts = torch.unique(ids, return_counts=True)
    order = sorted(range(uniq.numel()), key=lambda i: (-int(counts[i]), int(uniq[i])))
    keep = torch.tensor(sorted(int(uniq[i]) for i in order[:top_n]), dtype=torch.int64)
    return keep


def encode_frames(ts, ids, payload, *, id_enc: str, vocab: torch.Tensor | None,
                  dt_edges: torch.Tensor, dt_edges_global: torch.Tensor | None,
                  payload_bytes: int) -> torch.Tensor:
    """Encode one chunk -> uint8 [N, F], every entry in {0,1} by construction."""
    n = ts.numel()
    cols = []
    if id_enc == "onehot":
        assert vocab is not None
        pos = torch.searchsorted(vocab, ids).clamp(max=vocab.numel() - 1)
        hit = vocab[pos] == ids
        slot = torch.where(hit, pos, torch.full_like(pos, vocab.numel()))
        oh = torch.zeros(n, vocab.numel() + 1, dtype=torch.uint8)
        oh[torch.arange(n), slot] = 1
        cols.append(oh)
    elif id_enc == "binary":
        shifts = torch.arange(10, -1, -1, dtype=torch.int64)
        cols.append(((ids.unsqueeze(1) >> shifts) & 1).to(torch.uint8))   # 11-bit raw ID
    else:
        raise ValueError(f"id_enc must be 'onehot' or 'binary', got {id_enc!r}")
    cols.append(thermometer(per_id_dt(ts, ids), dt_edges))
    if dt_edges_global is not None:
        cols.append(thermometer(global_dt(ts), dt_edges_global))
    if payload_bytes > 0:
        b = payload[:, :payload_bytes].to(torch.int16)
        shifts = torch.arange(7, -1, -1, dtype=torch.int16)
        bits = ((b.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
        cols.append(bits.reshape(n, 8 * payload_bytes))
    x = torch.cat(cols, dim=1)
    assert int(x.max()) <= 1, "encoder must emit {0,1} bits only"
    return x


# --------------------------------------------------------------------------------------
# Splits — contiguous in time / whole-capture holdout (see module docstring)
# --------------------------------------------------------------------------------------
_SPLIT_FRACS = (0.7, 0.1, 0.2)
_SPLITS = ("train", "val", "test")


def _time_split(cap: Capture):
    """Cut ONE capture at the 70%/80% TIME points -> [(split, lo, hi)] index ranges."""
    t0, t1 = float(cap.ts[0]), float(cap.ts[-1])
    c1 = t0 + _SPLIT_FRACS[0] * (t1 - t0)
    c2 = t0 + (_SPLIT_FRACS[0] + _SPLIT_FRACS[1]) * (t1 - t0)
    i1 = int(torch.searchsorted(cap.ts, torch.tensor(c1, dtype=torch.float64)))
    i2 = int(torch.searchsorted(cap.ts, torch.tensor(c2, dtype=torch.float64)))
    n = cap.ts.numel()
    return [("train", 0, i1), ("val", i1, i2), ("test", i2, n)]


def assign_split_chunks(captures: list[Capture]):
    """-> (chunks, split_desc). chunks = list of (capture, split, lo, hi); windows are
    later built within one chunk only. Group rule per attack type (captures sorted by
    name): n==1 -> per-capture contiguous time split; n==2 -> [train, time-split(val|test)
    of capture 2]; n>=3 -> whole-capture holdout [train..., val, test]."""
    groups: dict[str, list[Capture]] = {}
    for c in captures:
        groups.setdefault(c.group, []).append(c)
    chunks = []
    split_desc: dict[str, str] = {}
    for gname in sorted(groups):
        caps = sorted(groups[gname], key=lambda c: c.name)
        if len(caps) == 1:
            cap = caps[0]
            for split, lo, hi in _time_split(cap):
                chunks.append((cap, split, lo, hi))
            split_desc[cap.name] = "time-split(0.7/0.1/0.2)"
        elif len(caps) == 2:
            chunks.append((caps[0], "train", 0, caps[0].ts.numel()))
            split_desc[caps[0].name] = "train"
            cap = caps[1]
            t0, t1 = float(cap.ts[0]), float(cap.ts[-1])
            cut = t0 + (_SPLIT_FRACS[1] / (_SPLIT_FRACS[1] + _SPLIT_FRACS[2])) * (t1 - t0)
            i1 = int(torch.searchsorted(cap.ts, torch.tensor(cut, dtype=torch.float64)))
            chunks.append((cap, "val", 0, i1))
            chunks.append((cap, "test", i1, cap.ts.numel()))
            split_desc[cap.name] = "time-split(val|test)"
        else:
            for cap in caps[:-2]:
                chunks.append((cap, "train", 0, cap.ts.numel()))
                split_desc[cap.name] = "train"
            chunks.append((caps[-2], "val", 0, caps[-2].ts.numel()))
            split_desc[caps[-2].name] = "val"
            chunks.append((caps[-1], "test", 0, caps[-1].ts.numel()))
            split_desc[caps[-1].name] = "test"
    return chunks, split_desc


# --------------------------------------------------------------------------------------
# Windows dataset
# --------------------------------------------------------------------------------------
class CANWindows(Dataset):
    """Fixed-length sliding windows over pre-encoded chunks.

    Frames are stored uint8 (4x memory saving on long captures) and converted to float32
    per window. ``per_step=True`` -> y is the full [T] per-message label vector (deep-sup
    path); else y = label of the LAST message (per-message detection with T-1 frames of
    context). ``flatten=True`` -> the window is reshaped to ONE step [1, T*F] (the
    information-matched windowed-FF control arm; psmnist-chunk precedent)."""

    def __init__(self, chunks, window: int, stride: int,
                 per_step: bool = False, flatten: bool = False):
        assert not (per_step and flatten), "per-step labels don't apply to a flattened window"
        self.chunks = chunks                        # list of (frames uint8 [n,F], labels [n])
        self.window = window
        self.per_step = per_step
        self.flatten = flatten
        self.index: list[tuple[int, int]] = []
        for ci, (frames, _labels) in enumerate(chunks):
            n = frames.shape[0]
            for s in range(0, n - window + 1, max(1, stride)):
                self.index.append((ci, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, s = self.index[i]
        frames, labels = self.chunks[ci]
        T = self.window
        w = frames[s:s + T].to(torch.float32)
        x = w.reshape(1, T * frames.shape[1]) if self.flatten else w
        y = labels[s:s + T] if self.per_step else labels[s + T - 1]
        return x, y


# --------------------------------------------------------------------------------------
# Synthetic generator (can-syn) + fixture writer
# --------------------------------------------------------------------------------------
def _schedule(t0: float, t1: float, period: float, gen: torch.Generator) -> torch.Tensor:
    """Jittered periodic arrivals in [t0, t1): gap_k = period * (1 +- jitter*U)."""
    n = int((t1 - t0) / (period * (1.0 - _SYN_JITTER))) + 3
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    gaps = period * (1.0 + _SYN_JITTER * (2.0 * u - 1.0))
    t = t0 + torch.cumsum(gaps, 0)
    return t[t < t1]


def _syn_periods() -> torch.Tensor:
    return torch.logspace(math.log10(0.01), math.log10(0.1), len(_SYN_IDS), dtype=torch.float64)


def _syn_capture(kind: str, idx: int, seconds: float, gen: torch.Generator) -> Capture:
    """One synthetic capture. kind: 'ambient' | 'masquerade' | 'suspension'.

    masquerade = period change on a spoofed ID: during the interval the target ID's
    legitimate frames are REMOVED and attacker frames are injected at HALF period —
    the ID appears with normal-looking content at double rate (ROAD-masquerade label
    semantics: every target-ID frame in the interval is an attack frame). Payload bits
    follow the SAME per-ID counter rule for legit and injected frames, so payload
    carries no attack signal — the attack is timing/sequence only, by construction.
    suspension = the target ID is silent for the interval; every frame (any ID) inside
    the interval is labeled 1 (interval-level convention, see module docstring)."""
    periods = _syn_periods()
    base = [_schedule(0.0, seconds, float(periods[k]), gen) for k in range(len(_SYN_IDS))]
    ts0 = min(float(t[0]) for t in base)            # capture start = first frame time
    a = b = None
    meta = None
    inj_sched = None
    if kind == "masquerade":
        a, b = 0.40 * seconds, 0.80 * seconds       # elapsed-time interval
        A0, A1 = a + ts0, b + ts0                   # absolute-time interval
        k = _MASQ_IDX
        base[k] = base[k][(base[k] < A0) | (base[k] >= A1)]
        inj_sched = _schedule(A0, A1, float(periods[k]) / 2.0, gen)
        meta = {"attack_type": "masquerade", "masquerade": True,
                "injection_id": f"{_SYN_IDS[k]:03X}", "injection_data_str": None,
                "injection_interval": [a, b], "synthetic": True}
    elif kind == "suspension":
        a, b = 0.40 * seconds, 0.70 * seconds
        A0, A1 = a + ts0, b + ts0
        k = _SUSP_IDX
        base[k] = base[k][(base[k] < A0) | (base[k] >= A1)]
        meta = {"attack_type": "suspension", "masquerade": False,
                "injection_id": f"{_SYN_IDS[k]:03X}", "injection_data_str": None,
                "injection_interval": [a, b], "synthetic": True}

    ts_parts, id_parts, inj_parts = [], [], []
    for k, cid in enumerate(_SYN_IDS):
        ts_parts.append(base[k])
        id_parts.append(torch.full((base[k].numel(),), cid, dtype=torch.int64))
        inj_parts.append(torch.zeros(base[k].numel(), dtype=torch.bool))
    if inj_sched is not None:
        ts_parts.append(inj_sched)
        id_parts.append(torch.full((inj_sched.numel(),), _SYN_IDS[_MASQ_IDX], dtype=torch.int64))
        inj_parts.append(torch.ones(inj_sched.numel(), dtype=torch.bool))
    ts = torch.cat(ts_parts)
    ids = torch.cat(id_parts)
    injected = torch.cat(inj_parts)

    ts = (ts * 1e6).round() / 1e6                   # candump text precision (round-trip exact)
    key = (ts * 1e6).round().to(torch.int64) * 4096 + ids   # time order, ID tie-break
    perm = torch.argsort(key)
    ts, ids, injected = ts[perm], ids[perm], injected[perm]
    n = ts.numel()

    # payload: per-ID frame counter — identical rule for legit and injected frames
    order = torch.argsort(ids, stable=True)
    ids_o = ids[order]
    starts = torch.cat([torch.tensor([True]), ids_o[1:] != ids_o[:-1]])
    gidx = torch.cumsum(starts.to(torch.int64), 0) - 1
    first_pos = torch.nonzero(starts).flatten()
    within = torch.arange(n, dtype=torch.int64) - first_pos[gidx]
    counter = torch.empty(n, dtype=torch.int64)
    counter[order] = within
    payload = torch.zeros(n, 8, dtype=torch.uint8)
    payload[:, 0] = (ids & 0xFF).to(torch.uint8)
    payload[:, 1] = (counter % 256).to(torch.uint8)
    payload[:, 2] = ((counter // 256) % 256).to(torch.uint8)
    dlc = torch.full((n,), 8, dtype=torch.int64)

    name = f"syn_{kind}_{idx}"
    labels = label_capture(ts, ids, payload, dlc, meta, name)   # single source of truth
    return Capture(name=name, group=f"syn_{kind}", ts=ts, ids=ids, dlc=dlc,
                   payload=payload, labels=labels, meta=meta,
                   extra={"injected": injected, "seconds": seconds})


def make_syn_captures(attack: str = "all", seconds: float = 20.0, n_each: int = 3,
                      gen: torch.Generator | None = None) -> list[Capture]:
    """The zero-file synthetic capture set: 1 ambient + n_each masquerade + n_each
    suspension captures (filtered by ``attack``). Deterministic: fixed _SYN_SEED,
    dedicated Generator — ZERO global-RNG draws."""
    gen = gen or torch.Generator().manual_seed(_SYN_SEED)
    caps = [_syn_capture("ambient", 1, seconds, gen)]
    for i in range(1, n_each + 1):                  # generation order fixed regardless of
        caps.append(_syn_capture("masquerade", i, seconds, gen))   # the attack filter, so
    for i in range(1, n_each + 1):                  # streams are identical across filters
        caps.append(_syn_capture("suspension", i, seconds, gen))
    if attack.lower() not in ("", "all"):
        caps = [c for c in caps if c.group == "syn_ambient" or _attack_match(c.group, attack)]
    return caps


def write_fixture(out_dir: str = FIXTURE_DIR, seconds: float = 8.0, n_each: int = 3) -> list[str]:
    """Serialize the synthetic captures in the EXACT ROAD raw schema: candump ``.log``
    (``(epoch.6f) can0 ID#HEX``) + per-capture ``<stem>.json`` metadata. Tiny (<1 MB
    total at 8 s/capture) — the checked-in parser/labeler test fixture."""
    os.makedirs(out_dir, exist_ok=True)
    offset = 1_700_000_000.0                        # mimic epoch timestamps
    written = []
    for cap in make_syn_captures("all", seconds=seconds, n_each=n_each):
        log_path = os.path.join(out_dir, cap.name + ".log")
        with open(log_path, "w", newline="\n") as f:
            for i in range(cap.ts.numel()):
                data = bytes(cap.payload[i, :int(cap.dlc[i])].tolist()).hex().upper()
                f.write(f"({float(cap.ts[i]) + offset:.6f}) can0 {int(cap.ids[i]):03X}#{data}\n")
        written.append(log_path)
        if cap.meta is not None:
            meta_path = os.path.join(out_dir, cap.name + ".json")
            with open(meta_path, "w", newline="\n") as f:
                json.dump(cap.meta, f, indent=2)
            written.append(meta_path)
    return written


# --------------------------------------------------------------------------------------
# Real-data loaders
# --------------------------------------------------------------------------------------
def _attack_match(stem: str, want: str) -> bool:
    """Capture filter: 'all' matches everything; otherwise '+'-separated substrings must
    ALL appear in the stem (e.g. 'max_speedometer+masquerade' matches ROAD's
    'max_speedometer_attack_1_masquerade'; plain 'masquerade' matches every masquerade
    capture)."""
    want = want.lower()
    if want in ("", "all"):
        return True
    return all(tok in stem.lower() for tok in want.split("+") if tok)


def _group_of(stem: str) -> str:
    """Split group from a capture file stem: 'ambient' captures form one group; attack
    captures group by name with counters stripped (correlated_signal_attack_1_masquerade
    -> correlated_signal_attack_masquerade)."""
    if "ambient" in stem.lower():
        return "ambient"
    return re.sub(r"_?\d+", "", stem)


def _discover(pattern: str) -> list[str]:
    """Sorted recursive glob that DROPS macOS AppleDouble sidecars ('._name.log') and
    anything under a '__MACOSX/' folder — defense-in-depth for the ROAD archive.

    ROAD's road.zip (verified 2026-07-12, MD5 cab184cf…) ships one 213-byte
    '._<capture>.log' resource fork next to EVERY real capture and collects them in a
    sibling '__MACOSX/' tree. Two things already make the standard path safe: (a) that
    sibling sits OUTSIDE the default data-can/road/ glob root, and (b) Python's glob
    skips leading-dot names, so '._foo.log' never matches '*.log' anyway. This guard is
    belt-and-suspenders for the cases those two don't cover — a broad user glob
    (--can-file 'data-can/**/*.log') combined with any tooling that strips the leading
    dot, or a future switch away from glob — where a 0-byte fork would otherwise reach
    parse_candump and raise 'no parseable candump frames' mid-run. No-op on a clean
    extraction and on the checked-in fixture (neither has such files)."""
    return sorted(p for p in _glob.glob(pattern, recursive=True)
                  if not os.path.basename(p).startswith("._")
                  and "__MACOSX" not in p.replace("\\", "/").split("/"))


def _load_candump_files(paths: list[str], attack: str) -> list[Capture]:
    caps = []
    for p in sorted(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        group = _group_of(stem)
        if group != "ambient" and not _attack_match(stem, attack):
            continue
        ts, ids, dlc, payload = parse_candump(p)
        meta = find_metadata(p)
        if group != "ambient" and meta is None:
            raise FileNotFoundError(
                f"attack capture {p!r} has no metadata JSON (looked for <stem>.json / "
                f"capture_metadata.json). ROAD labels are DERIVED from metadata "
                f"(injection_id / injection_data_str / injection_interval) — see "
                f"mlgn/seqlgn/data/can/README.md.")
        labels = label_capture(ts, ids, payload, dlc, meta, stem)
        caps.append(Capture(name=stem, group=group, ts=ts, ids=ids, dlc=dlc,
                            payload=payload, labels=labels, meta=meta))
    return caps


def _load_road(attack: str, file_glob: str, ambient: bool) -> list[Capture]:
    if file_glob:
        paths = _discover(file_glob)
        if not paths:
            raise FileNotFoundError(f"--can-file glob {file_glob!r} matched nothing")
        return _load_candump_files(paths, attack)
    root = os.path.join(CAN_ROOT, "road")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ROAD dataset not found at {os.path.abspath(root)}. Download road.zip "
            f"(556.7 MB, MD5 cab184cfc2fe12c0834bc46188c0f330, CC BY 4.0, no "
            f"registration) from https://zenodo.org/records/10462796 and unzip so raw "
            f"logs live under data-can/road/ (ambient/ + attacks/ subfolders). See "
            f"mlgn/seqlgn/data/can/README.md.")
    paths = _discover(os.path.join(root, "**", "*.log"))
    attack_paths = [p for p in paths if "ambient" not in p.lower()]
    ambient_paths = [p for p in paths if "ambient" in p.lower()]
    use = attack_paths + (ambient_paths if ambient else [])
    if not use:
        raise FileNotFoundError(f"no .log captures under {os.path.abspath(root)}")
    caps = _load_candump_files(use, attack)
    if not any(c.group != "ambient" for c in caps):
        raise FileNotFoundError(
            f"no ROAD attack capture matches --can-attack {attack!r} "
            f"(match = substring of the capture filename, e.g. 'masquerade', "
            f"'max_speedometer_masquerade').")
    return caps


def _load_hcrl(attack: str, file_glob: str) -> list[Capture]:
    if file_glob:
        paths = _discover(file_glob)
    else:
        root = os.path.join(CAN_ROOT, "hcrl")
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"HCRL Car-Hacking dataset not found at {os.path.abspath(root)}. See "
                f"mlgn/seqlgn/data/can/README.md (free Dropbox links at "
                f"https://ocslab.hksecurity.net/Datasets/car-hacking-dataset). NOTE: "
                f"sanity row only — never C0.g gate evidence.")
        paths = _discover(os.path.join(root, "**", "*.csv"))
    caps = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if not _attack_match(stem, attack):
            continue
        ts, ids, dlc, payload, labels = parse_hcrl_csv(p)
        caps.append(Capture(name=stem, group=_group_of(stem), ts=ts, ids=ids, dlc=dlc,
                            payload=payload, labels=labels, meta=None))
    if not caps:
        raise FileNotFoundError(f"no HCRL csv matches --can-attack {attack!r}")
    return caps


# --------------------------------------------------------------------------------------
# Task builder
# --------------------------------------------------------------------------------------
def build_can_dataset(*, window: int, stride: int = 1, eval_stride: int = 1,
                      id_enc: str = "onehot", top_ids: int = 20, dt_bins: int = 8,
                      dt_global: bool = False, payload_bytes: int = 0,
                      source: str = "syn", file: str = "", attack: str = "all",
                      ambient: bool = False):
    """Load/generate captures, split, fit the encoder on TRAIN ONLY, encode.

    Returns ``(encoded, meta)`` where ``encoded[split]`` is a list of
    ``(frames uint8 [n,F], labels [n])`` chunks. Exposed separately from
    :func:`can_task` so tests can verify splits/encoder state directly."""
    if source == "syn":
        captures = make_syn_captures(attack=attack)
        if file:
            raise ValueError("source 'syn' is zero-file; to run the fixture files use "
                             "--can-source road --can-file <fixture>/*.log")
    elif source == "road":
        captures = _load_road(attack, file, ambient)
    elif source == "hcrl":
        captures = _load_hcrl(attack, file)
    else:
        raise ValueError(f"unknown can source {source!r} (road / hcrl / syn)")

    for cap in captures:                            # contiguity guard (defensive)
        assert bool((cap.ts[1:] >= cap.ts[:-1]).all()), f"{cap.name}: not time-sorted"

    chunks, split_desc = assign_split_chunks(captures)
    for split in _SPLITS:
        if not any(s == split and hi > lo for _, s, lo, hi in chunks):
            raise ValueError(f"{split} split received no frames — check capture set "
                             f"({[c.name for c in captures]}) and the group rule in "
                             f"assign_split_chunks().")

    # --- fit encoder state on the TRAIN chunks only (calibration-leakage guard) -------
    train_ids = torch.cat([c.ids[lo:hi] for c, s, lo, hi in chunks if s == "train"])
    train_dts = torch.cat([per_id_dt(c.ts[lo:hi], c.ids[lo:hi])
                           for c, s, lo, hi in chunks if s == "train"])
    vocab = fit_id_vocab(train_ids, top_ids) if id_enc == "onehot" else None
    dt_edges = fit_dt_edges(train_dts, dt_bins)
    dt_edges_g = None
    if dt_global:
        tg = torch.cat([global_dt(c.ts[lo:hi]) for c, s, lo, hi in chunks if s == "train"])
        dt_edges_g = fit_dt_edges(tg, dt_bins)

    encoded: dict[str, list] = {s: [] for s in _SPLITS}
    n_frames = {s: 0 for s in _SPLITS}
    n_attack = {s: 0 for s in _SPLITS}
    boundaries: dict[str, list] = {}
    for cap, split, lo, hi in chunks:
        if hi - lo <= 0:
            continue
        frames = encode_frames(cap.ts[lo:hi], cap.ids[lo:hi], cap.payload[lo:hi],
                               id_enc=id_enc, vocab=vocab, dt_edges=dt_edges,
                               dt_edges_global=dt_edges_g, payload_bytes=payload_bytes)
        labels = cap.labels[lo:hi]
        encoded[split].append((frames, labels))
        n_frames[split] += hi - lo
        n_attack[split] += int(labels.sum())
        boundaries.setdefault(cap.name, []).append(
            [split, float(cap.ts[lo]), float(cap.ts[hi - 1])])
    input_dim = encoded["train"][0][0].shape[1] if encoded["train"] else 0

    meta = {
        "source": source, "file": file or None, "attack": attack, "ambient": ambient,
        "window": window, "stride": stride, "eval_stride": eval_stride,
        "id_enc": id_enc, "top_ids": top_ids, "dt_bins": dt_bins, "dt_global": dt_global,
        "payload_bytes": payload_bytes, "input_dim": input_dim,
        "id_vocab": [f"0x{int(v):03X}" for v in vocab] if vocab is not None else None,
        "dt_edges": [float(e) for e in dt_edges],
        "dt_edges_global": [float(e) for e in dt_edges_g] if dt_edges_g is not None else None,
        "splits": split_desc,
        "split_boundaries": boundaries,
        "frames": dict(n_frames),
        "attack_frac_frames": {s: (n_attack[s] / n_frames[s] if n_frames[s] else None)
                               for s in _SPLITS},
    }
    return encoded, meta


def can_task(name: str, batch_size: int, window: int, *, stride: int = 1,
             eval_stride: int = 1, id_enc: str = "onehot", top_ids: int = 20,
             dt_bins: int = 8, dt_global: bool = False, payload_bytes: int = 0,
             per_step: bool = False, flatten: bool = False, source: str = "syn",
             file: str = "", attack: str = "all", ambient: bool = False) -> TaskSpec:
    """Build the CAN TaskSpec. The returned spec carries ``task.can_meta`` (dict) with
    the fitted encoder state + split identity, which train.py persists into the results
    JSON (research/20 §C3 rule: every load-bearing arg must land in the record).

    NOTE: consumes ZERO draws from the global torch generator (netlist replay contract);
    the split is seed-independent, so all arms/seeds train on identical data."""
    encoded, meta = build_can_dataset(
        window=window, stride=stride, eval_stride=eval_stride, id_enc=id_enc,
        top_ids=top_ids, dt_bins=dt_bins, dt_global=dt_global,
        payload_bytes=payload_bytes, source=source, file=file, attack=attack,
        ambient=ambient)

    sets = {}
    for split in _SPLITS:
        st = stride if split == "train" else eval_stride
        sets[split] = CANWindows(encoded[split], window, st, per_step=per_step,
                                 flatten=flatten)
        if len(sets[split]) == 0:
            raise ValueError(
                f"{split} split has zero windows (window={window} vs chunk sizes "
                f"{[f.shape[0] for f, _ in encoded[split]]}) — shrink --seq-len or "
                f"check the capture/split assignment in task.can_meta['splits'].")
    if len(sets["train"]) < batch_size:
        raise ValueError(f"train split has only {len(sets['train'])} windows "
                         f"< batch_size={batch_size} (drop_last would starve the loop)")

    F = meta["input_dim"]
    meta["per_step"] = per_step
    meta["flatten"] = flatten
    meta["windows"] = {s: len(sets[s]) for s in _SPLITS}
    # window-level attack fraction under the final-label convention (test split):
    wtest = sets["test"]
    ylast = torch.cat([labels[torch.arange(0, f.shape[0] - window + 1, max(1, eval_stride))
                              + window - 1]
                       for f, labels in encoded["test"] if f.shape[0] >= window])
    meta["attack_frac_windows_test"] = float(ylast.float().mean()) if ylast.numel() else None
    assert len(wtest) == ylast.numel()

    spec = TaskSpec(
        name=name,
        input_dim=(window * F) if flatten else F,
        num_classes=2,
        seq_len=1 if flatten else window,
        train_loader=DataLoader(sets["train"], batch_size=batch_size, shuffle=True,
                                drop_last=True),
        val_loader=DataLoader(sets["val"], batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(sets["test"], batch_size=batch_size, shuffle=False),
        test_seq_len=1 if flatten else window,   # no length-gen eval for CAN
    )
    spec.can_meta = meta                            # plain dataclass: attribute rides along
    return spec


# --------------------------------------------------------------------------------------
# CLI: fixture writer / stream stats
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CAN data utilities (fixture writer).")
    p.add_argument("--write-fixture", action="store_true",
                   help=f"write the synthetic candump+metadata fixture to --out "
                        f"(default {os.path.relpath(FIXTURE_DIR)})")
    p.add_argument("--out", default=FIXTURE_DIR)
    p.add_argument("--seconds", type=float, default=8.0,
                   help="per-capture duration for the fixture (8 s keeps it < 1 MB)")
    args = p.parse_args()
    if args.write_fixture:
        files = write_fixture(args.out, seconds=args.seconds)
        total = sum(os.path.getsize(f) for f in files)
        print(f"wrote {len(files)} files, {total / 1e6:.2f} MB total -> {args.out}")
        for f in files:
            print("  ", os.path.basename(f), f"{os.path.getsize(f):,} B")
    else:
        caps = make_syn_captures("all")
        for c in caps:
            print(f"{c.name:22s} frames={c.ts.numel():6d} attack={int(c.labels.sum()):5d} "
                  f"({100 * float(c.labels.float().mean()):.1f}%)")


if __name__ == "__main__":
    main()
