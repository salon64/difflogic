"""
test_can.py — self-checks for the CAN-bus IDS task plumbing (can_data.py + the ff arm).
========================================================================================

Run from the repo root:

    python -m mlgn.seqlgn.test_can

Banks (all hard asserts; pytest-compatible test_* functions):

1.  RNG discipline + determinism — building can/can-syn tasks consumes ZERO draws from
    the global torch generator (netlist wiring-replay contract) and regenerates
    bit-identically.
2.  Parser round-trip — the synthetic fixture written in the EXACT ROAD raw schema
    (candump .log + metadata JSON) parses back to the in-memory stream: timestamps
    (elapsed), IDs, payloads, DLCs and DERIVED labels all match.
3.  Labeling rules — masquerade labels == generator injection bookkeeping; suspension ==
    interval convention; fabrication payload wildcard matching on engineered frames.
4.  Encoding — shapes/widths for onehot/binary/payload/global-dt variants, thermometer
    monotonicity, everything binary by construction (eval x.round() is a no-op).
5.  Splits — Δt edges fit on the TRAIN segment only; time-split contiguity (train before
    val before test within a capture); whole-capture holdout disjointness; windows never
    straddle a chunk boundary.
6.  Window/label alignment — final-label and per-step-label conventions against direct
    chunk indexing; flattened-window content equality.
7.  Batch contract + gate parity — loader shapes/dtypes; ff(hidden=2H) matches
    gated/clatch(hidden=H) gate counts exactly (the C0.g matched construction).
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

import torch

from .can_data import (CANWindows, _payload_matches, assign_split_chunks,
                       build_can_dataset, encode_frames, fit_dt_edges, fit_id_vocab,
                       label_capture, make_syn_captures, parse_candump, parse_hcrl_csv,
                       per_id_dt, write_fixture)
from .data import get_task
from .models import SequenceClassifier
from .utils import count_gates

_FIX_SECONDS = 8.0


def _rng_fingerprint():
    return torch.random.get_rng_state().clone()


# --------------------------------------------------------------------------------------
# 1. RNG discipline + determinism
# --------------------------------------------------------------------------------------
def test_rng_discipline_and_determinism():
    torch.manual_seed(1234)
    before = _rng_fingerprint()
    task1 = get_task("can-syn", batch_size=32, seq_len=16)
    after = _rng_fingerprint()
    assert torch.equal(before, after), "can-syn builder consumed global torch RNG draws!"

    task2 = get_task("can-syn", batch_size=32, seq_len=16)
    ds1, ds2 = task1.val_loader.dataset, task2.val_loader.dataset
    assert len(ds1) == len(ds2) and len(ds1) > 0
    for i in (0, len(ds1) // 2, len(ds1) - 1):
        x1, y1 = ds1[i]
        x2, y2 = ds2[i]
        assert torch.equal(x1, x2) and torch.equal(torch.as_tensor(y1), torch.as_tensor(y2))
    assert task1.can_meta["dt_edges"] == task2.can_meta["dt_edges"]
    print("[1] rng discipline + determinism        OK "
          f"(rng untouched; {len(ds1)} val windows bit-identical across builds)")


# --------------------------------------------------------------------------------------
# 2. Parser round-trip on the ROAD-schema fixture
# --------------------------------------------------------------------------------------
def test_fixture_roundtrip():
    mem = {c.name: c for c in make_syn_captures("all", seconds=_FIX_SECONDS)}
    with tempfile.TemporaryDirectory() as tmp:
        files = write_fixture(tmp, seconds=_FIX_SECONDS)
        total = sum(os.path.getsize(f) for f in files)
        assert total < 1_000_000, f"fixture must stay < 1 MB, got {total} B"
        logs = sorted(f for f in files if f.endswith(".log"))
        assert len(logs) == 7, logs
        n_checked = 0
        for lp in logs:
            stem = os.path.splitext(os.path.basename(lp))[0]
            cap = mem[stem]
            ts, ids, dlc, payload = parse_candump(lp)
            assert ts.numel() == cap.ts.numel(), stem
            assert torch.equal(ids, cap.ids), stem
            assert torch.equal(dlc, cap.dlc), stem
            assert torch.equal(payload, cap.payload), stem
            el_f, el_m = ts - ts[0], cap.ts - cap.ts[0]
            assert float((el_f - el_m).abs().max()) < 5e-7, stem
            # labels DERIVED from the written metadata JSON == generator labels
            from .can_data import find_metadata
            meta = find_metadata(lp)
            lab = label_capture(ts, ids, payload, dlc, meta, stem)
            assert torch.equal(lab, cap.labels), f"label mismatch on {stem}"
            n_checked += cap.ts.numel()
        print(f"[2] fixture parser round-trip           OK "
              f"({len(logs)} captures, {n_checked} frames, {total / 1e3:.0f} KB)")


# --------------------------------------------------------------------------------------
# 3. Labeling rules
# --------------------------------------------------------------------------------------
def test_labeling_rules():
    caps = make_syn_captures("all", seconds=_FIX_SECONDS)
    n_masq = n_susp = 0
    for cap in caps:
        if cap.group == "syn_masquerade":
            injected = cap.extra["injected"]
            assert torch.equal(cap.labels.bool(), injected), \
                f"{cap.name}: masquerade labels != injection bookkeeping"
            assert int(cap.labels.sum()) > 0
            # every labeled frame is the target ID inside the interval
            inj_id = int(cap.meta["injection_id"], 16)
            assert bool((cap.ids[cap.labels == 1] == inj_id).all())
            n_masq += 1
        if cap.group == "syn_suspension":
            s, e = cap.meta["injection_interval"]
            el = cap.ts - cap.ts[0]
            expect = ((el >= s) & (el <= e)).long()
            assert torch.equal(cap.labels, expect), \
                f"{cap.name}: suspension labels != interval convention"
            # the suppressed ID is genuinely absent inside the interval
            inj_id = int(cap.meta["injection_id"], 16)
            assert int(((cap.ids == inj_id) & (cap.labels == 1)).sum()) == 0
            n_susp += 1
        if cap.group == "syn_ambient":
            assert int(cap.labels.sum()) == 0
    assert n_masq == 3 and n_susp == 3

    # fabrication rule: payload pattern with X wildcards
    ts = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    ids = torch.tensor([0x111, 0x111, 0x111, 0x222])
    payload = torch.zeros(4, 8, dtype=torch.uint8)
    payload[1, 0] = 0xFF
    payload[2, 0] = 0xF0
    dlc = torch.full((4,), 8)
    meta = {"injection_id": "111", "injection_data_str": "FXX" + "X" * 13,
            "injection_interval": [0.5, 3.0]}
    lab = label_capture(ts, ids, payload, dlc, meta, "fab_test")
    assert lab.tolist() == [0, 1, 1, 0], lab.tolist()   # F? matches FF and F0; id 0x222 never
    m = _payload_matches(payload, "FF" + "X" * 14)
    assert m.tolist() == [False, True, False, False]
    print("[3] labeling rules                      OK "
          "(masq==bookkeeping x3, susp==interval x3, fabrication wildcards)")


# --------------------------------------------------------------------------------------
# 4. Encoding
# --------------------------------------------------------------------------------------
def test_encoding():
    caps = make_syn_captures("all", seconds=_FIX_SECONDS)
    cap = caps[0]
    vocab = fit_id_vocab(cap.ids, 20)
    assert vocab.numel() == 6 and bool((vocab[1:] > vocab[:-1]).all())
    edges = fit_dt_edges(per_id_dt(cap.ts, cap.ids), 8)
    assert edges.numel() == 8 and bool((edges[1:] > edges[:-1]).all()), "log-spaced, increasing"

    x = encode_frames(cap.ts, cap.ids, cap.payload, id_enc="onehot", vocab=vocab,
                      dt_edges=edges, dt_edges_global=None, payload_bytes=0)
    assert x.shape == (cap.ts.numel(), 7 + 8)           # (6 ids + other) + 8 dt bins
    assert x.dtype == torch.uint8 and int(x.max()) <= 1
    oh = x[:, :7]
    assert bool((oh.sum(1) == 1).all()), "one-hot rows must have exactly one bit"
    assert int(oh[:, -1].sum()) == 0, "no 'other' IDs in-vocab stream"
    th = x[:, 7:].to(torch.int16)
    assert bool((th[:, :-1] >= th[:, 1:]).all()), "thermometer bits must be monotone 1..10..0"
    # first frame of each ID -> dt=inf -> all thermometer bits set
    first_rows = [int((cap.ids == i).nonzero()[0]) for i in vocab.tolist()]
    assert bool((th[first_rows] == 1).all())

    xb = encode_frames(cap.ts, cap.ids, cap.payload, id_enc="binary", vocab=None,
                       dt_edges=edges, dt_edges_global=edges, payload_bytes=2)
    assert xb.shape == (cap.ts.numel(), 11 + 8 + 8 + 16)
    assert int(xb.max()) <= 1
    # 11-bit ID field decodes back to the arbitration ID
    shifts = torch.arange(10, -1, -1)
    dec = (xb[:, :11].to(torch.int64) << shifts).sum(1)
    assert torch.equal(dec, cap.ids)
    # payload bits decode back to the first 2 payload bytes
    pb = xb[:, 27:].reshape(-1, 2, 8).to(torch.int64)
    dec_b = (pb << torch.arange(7, -1, -1)).sum(-1)
    assert torch.equal(dec_b, cap.payload[:, :2].to(torch.int64))
    print("[4] encoding                            OK "
          "(onehot/binary/global-dt/payload widths, monotone thermometer, all bits {0,1})")


# --------------------------------------------------------------------------------------
# 5. Splits — train-only calibration, contiguity, holdout disjointness
# --------------------------------------------------------------------------------------
def test_splits_and_calibration():
    encoded, meta = build_can_dataset(window=16, source="syn", attack="all")
    # recompute the encoder fit INDEPENDENTLY from the train chunks
    caps = make_syn_captures("all")
    chunks, split_desc = assign_split_chunks(caps)
    train_dts = torch.cat([per_id_dt(c.ts[lo:hi], c.ids[lo:hi])
                           for c, s, lo, hi in chunks if s == "train"])
    edges_train = fit_dt_edges(train_dts, 8)
    assert meta["dt_edges"] == [float(e) for e in edges_train], \
        "dt edges must be fit on the train segment ONLY"
    all_dts = torch.cat([per_id_dt(c.ts, c.ids) for c in caps])
    edges_all = fit_dt_edges(all_dts, 8)
    assert meta["dt_edges"] != [float(e) for e in edges_all], \
        "train-fit edges coincide with all-data edges — calibration guard not exercised"

    # group rule: 3 masq captures -> whole-capture train/val/test; same for susp;
    # single ambient capture -> contiguous time split
    assert split_desc["syn_masquerade_1"] == "train"
    assert split_desc["syn_masquerade_2"] == "val"
    assert split_desc["syn_masquerade_3"] == "test"
    assert split_desc["syn_suspension_3"] == "test"
    assert "time-split" in split_desc["syn_ambient_1"]

    # time-split contiguity: within the ambient capture, train < val < test in TIME
    bounds = {s: (t0, t1) for s, t0, t1 in meta["split_boundaries"]["syn_ambient_1"]}
    assert bounds["train"][1] <= bounds["val"][0] <= bounds["val"][1] <= bounds["test"][0]

    # every split has frames + attack positives from both attack families
    for s in ("train", "val", "test"):
        assert meta["frames"][s] > 0
        assert meta["attack_frac_frames"][s] > 0
    print("[5] splits + train-only calibration     OK "
          f"(edges==train-fit, capture holdout, time contiguity; "
          f"attack_frac={ {k: round(v, 3) for k, v in meta['attack_frac_frames'].items()} })")


# --------------------------------------------------------------------------------------
# 6. Window/label alignment (+ no chunk straddling)
# --------------------------------------------------------------------------------------
def test_window_label_alignment():
    F, n1, n2, T = 3, 9, 7, 4
    f1 = torch.arange(n1 * F, dtype=torch.uint8).reshape(n1, F) % 2
    f2 = (torch.arange(n2 * F, dtype=torch.uint8).reshape(n2, F) + 1) % 2
    l1 = torch.arange(n1, dtype=torch.int64) % 2            # chunk-1 labels
    l2 = torch.full((n2,), 7, dtype=torch.int64)            # sentinel chunk-2 labels
    chunks = [(f1, l1), (f2, l2)]

    ds = CANWindows(chunks, window=T, stride=1)
    assert len(ds) == (n1 - T + 1) + (n2 - T + 1)
    for i in range(n1 - T + 1):                             # chunk 1: final-label rule
        x, y = ds[i]
        assert x.shape == (T, F) and x.dtype == torch.float32
        assert torch.equal(x, f1[i:i + T].float()), "window must be the contiguous slice"
        assert int(y) == int(l1[i + T - 1])
    for j in range(n2 - T + 1):                             # chunk 2: never straddles
        x, y = ds[(n1 - T + 1) + j]
        assert torch.equal(x, f2[j:j + T].float())
        assert int(y) == 7

    ds_ps = CANWindows(chunks, window=T, stride=2, per_step=True)
    x, y = ds_ps[1]                                          # start index 2 (stride 2)
    assert y.shape == (T,) and torch.equal(y, l1[2:2 + T])

    ds_fl = CANWindows(chunks, window=T, stride=1, flatten=True)
    x, y = ds_fl[0]
    assert x.shape == (1, T * F)
    assert torch.equal(x, f1[0:T].float().reshape(1, T * F))
    assert int(y) == int(l1[T - 1])

    try:
        CANWindows(chunks, window=T, stride=1, per_step=True, flatten=True)
        raise AssertionError("per_step+flatten must be rejected")
    except AssertionError as e:
        if "per-step" not in str(e):
            raise
    print("[6] window/label alignment              OK "
          "(final + per-step + flatten conventions, no chunk straddling)")


# --------------------------------------------------------------------------------------
# 7. Batch contract + ff gate parity
# --------------------------------------------------------------------------------------
def test_batch_contract_and_gate_parity():
    task = get_task("can-syn", batch_size=32, seq_len=16, can_per_step=True,
                    can_eval_stride=4)
    x, y = next(iter(task.train_loader))
    assert x.shape == (32, 16, task.input_dim) and x.dtype == torch.float32
    assert torch.equal(x, x.round()), "features must be binary — eval x.round() a no-op"
    assert y.shape == (32, 16) and y.dtype == torch.int64

    task_ff = get_task("can-syn", batch_size=32, seq_len=16, can_flatten=True,
                       can_eval_stride=4)
    assert task_ff.seq_len == 1 and task_ff.input_dim == 16 * task.input_dim
    xf, yf = next(iter(task_ff.val_loader))
    assert xf.shape == (32, 1, task_ff.input_dim) and yf.ndim == 1
    assert torch.equal(xf, xf.round())
    # same split -> same number of windows regardless of flattening
    assert task_ff.can_meta["windows"] == \
        get_task("can-syn", batch_size=32, seq_len=16, can_eval_stride=4).can_meta["windows"]

    H = 64
    kw = dict(num_classes=2, cell_layers=2, device="cpu")
    m_gated = SequenceClassifier(input_dim=task.input_dim, hidden_dim=H,
                                 mechanism="gated", **kw)
    m_clatch = SequenceClassifier(input_dim=task.input_dim, hidden_dim=H,
                                  mechanism="clatch", **kw)
    m_ff = SequenceClassifier(input_dim=task_ff.input_dim, hidden_dim=2 * H,
                              mechanism="ff", **kw)
    g_gated, g_clatch, g_ff = count_gates(m_gated), count_gates(m_clatch), count_gates(m_ff)
    assert g_gated == g_clatch == g_ff == 2 * 2 * H, (g_gated, g_clatch, g_ff)
    # ff really is stateless: output independent of the initial state
    xt = xf[:4, 0, :]
    s0 = m_ff.cell.init_state(4, device="cpu")
    out_a = m_ff.cell(xt, s0)
    out_b = m_ff.cell(xt, torch.ones_like(s0))
    assert torch.equal(out_a, out_b)
    print(f"[7] batch contract + gate parity        OK "
          f"(gated={g_gated} == clatch={g_clatch} == ff(2H)={g_ff}; ff stateless)")


# --------------------------------------------------------------------------------------
# 8. HCRL CSV adapter (synthetic file — schema check only)
# --------------------------------------------------------------------------------------
def test_hcrl_parser():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "spoof_rpm.csv")
        with open(p, "w", newline="\n") as f:
            f.write("1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n")
            f.write("1478198376.389636,018f,8,fe,5b,00,00,00,3c,00,00,T\n")
            f.write("1478198376.389864,0260,4,19,21,22,30,R\n")
            f.write("garbage,line,x\n")
        ts, ids, dlc, payload, labels = parse_hcrl_csv(p)
        assert ts.numel() == 3
        assert ids.tolist() == [0x316, 0x18F, 0x260]
        assert dlc.tolist() == [8, 8, 4]
        assert labels.tolist() == [0, 1, 0]
        assert payload[0, 0] == 0x05 and payload[2, 3] == 0x30 and payload[2, 4] == 0
    print("[8] hcrl csv adapter                    OK (schema, Flag labels, short DLC pad)")


# --------------------------------------------------------------------------------------
# 9. Full 'road' file path driven on the checked-in fixture (discovery/meta/filter/split)
# --------------------------------------------------------------------------------------
def test_road_path_on_fixture():
    fix = os.path.join(os.path.dirname(__file__), "data", "can", "*.log")
    assert glob.glob(fix), "checked-in fixture missing — run " \
        "`python -m mlgn.seqlgn.can_data --write-fixture`"
    before = _rng_fingerprint()
    task = get_task("can", batch_size=32, seq_len=16, can_source="road", can_file=fix,
                    can_eval_stride=4)
    assert torch.equal(before, _rng_fingerprint()), "road builder consumed global RNG!"
    assert task.num_classes == 2 and task.seq_len == 16
    x, y = next(iter(task.test_loader))
    assert x.shape[1:] == (16, task.input_dim) and torch.equal(x, x.round())
    for s in ("train", "val", "test"):
        assert task.can_meta["windows"][s] > 0
        assert task.can_meta["attack_frac_frames"][s] > 0
    # attack filter: only masquerade captures (+ ambient) survive
    task_m = get_task("can", batch_size=32, seq_len=16, can_source="road", can_file=fix,
                      can_attack="masquerade", can_eval_stride=4)
    names = set(task_m.can_meta["splits"])
    assert all(("masquerade" in n) or ("ambient" in n) for n in names), names
    assert task_m.can_meta["splits"]["syn_masquerade_3"] == "test"
    print(f"[9] road file path on fixture           OK "
          f"(windows={task.can_meta['windows']}, masquerade filter -> {sorted(names)})")


# --------------------------------------------------------------------------------------
# 10. Non-vacuity: the encoding must CARRY the masquerade signal (oracle separability)
# --------------------------------------------------------------------------------------
def test_masquerade_oracle_separability():
    """A 2-bit oracle over the ENCODED features — target-ID one-hot AND one Δt
    thermometer bit (Δt < ~18 ms, i.e. the spoofed half-period) — must pick out
    masquerade frames at precision 1.0. If this fails after an encoder change, the
    features no longer carry the timing attack and every training result is vacuous.
    (The residual FNs under 'all' labels are the suspension-interval frames, which a
    per-frame rule genuinely cannot resolve — that asymmetry IS the recurrence case.)"""
    encoded, meta = build_can_dataset(window=16, source="syn", attack="all")
    slot = meta["id_vocab"].index("0x185")          # masquerade target ID slot
    dt0 = len(meta["id_vocab"]) + 1                 # first Δt thermometer column
    assert meta["dt_edges"][2] < 0.020 < meta["dt_edges"][4], \
        "edge layout drifted — pick the bit that separates half-period (~12.5ms) from normal (~25ms)"
    tp = fp = 0
    for frames, labels in encoded["test"]:
        f = frames.to(torch.int64)
        pred = (f[:, slot] == 1) & (f[:, dt0 + 2] == 0)   # target ID AND dt < edges[2]
        tp += int((pred & (labels == 1)).sum())
        fp += int((pred & (labels == 0)).sum())
    assert fp == 0, f"oracle false positives: {fp} — encoding/labeling misaligned"
    assert tp >= 600, f"oracle only catches {tp} masquerade frames — timing signal lost"
    print(f"[10] masquerade oracle separability     OK "
          f"(2-bit rule: TP={tp}, FP=0 on the test split)")


def main():
    banks = [
        test_rng_discipline_and_determinism,
        test_fixture_roundtrip,
        test_labeling_rules,
        test_encoding,
        test_splits_and_calibration,
        test_window_label_alignment,
        test_batch_contract_and_gate_parity,
        test_hcrl_parser,
        test_road_path_on_fixture,
        test_masquerade_oracle_separability,
    ]
    for fn in banks:
        fn()
    print(f"\nALL {len(banks)} BANKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        print("\nFAILED", file=sys.stderr)
        raise
