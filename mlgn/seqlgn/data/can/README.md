# CAN-bus IDS data — fixtures + real-dataset download procedure

This folder holds the **checked-in synthetic fixture** (tiny, <1 MB) used by
`python -m mlgn.seqlgn.test_can` and the CPU smokes. **Real datasets do NOT live here** —
they go in `<repo-root>/data-can/<source>/` (the `data-mnist/` convention; large, not in
git — add `**/data-can/` to `.gitignore` if it isn't yet). On DUST only
`/home/jovyan/work` persists, so keep the repo (and `data-can/` inside it) there.

## Fixture (this folder)

`syn_*.log` + `syn_*.json` — 7 synthetic captures in the **exact ROAD raw schema**
(candump `(epoch.6f) can0 ID#HEXPAYLOAD` lines + per-capture metadata JSON with
`injection_id` / `injection_data_str` / `injection_interval`). 1 ambient + 3
masquerade-style (period change on a spoofed ID; ROAD-masquerade label semantics) + 3
suspension-style (ID silent; interval-level labels). Deterministic — regenerate with:

```bash
python -m mlgn.seqlgn.can_data --write-fixture
```

Run the full real-data code path on the fixture (parser, metadata labeling, capture
splits) without downloading anything:

```bash
python -m mlgn.seqlgn.train --task can --can-source road \
    --can-file "mlgn/seqlgn/data/can/*.log" --seq-len 16 --device cpu ...
```

## ROAD (ORNL) — PRIMARY dataset (gate C0.g evidence)

- **Download**: <https://zenodo.org/records/10462796> — single `road.zip`, 556.7 MB,
  **MD5 `cab184cfc2fe12c0834bc46188c0f330`** (verify after download!), no registration,
  license CC BY 4.0.
- **Citation (required by the record)**: Verma et al., *"A comprehensive guide to CAN IDS
  data and introduction of the ROAD dataset"*, PLOS ONE 19(1):e0296879, 2024
  (arXiv:2012.14600). Data DOI 10.5281/zenodo.10462796.
- **Layout the loader consumes**: unzip so the raw candump logs sit under

  ```
  data-can/road/ambient/*.log
  data-can/road/attacks/*.log
  ```

  (any nesting works — discovery is a recursive `**/*.log` glob; captures with `ambient`
  in the path are the benign group). **Labels are derived from metadata**: the loader
  looks for `<capture>.json` next to each log, or a shared `capture_metadata.json` in the
  log's folder / its parent keyed by capture stem. If the zip ships metadata in another
  shape, reshape it to one of those (keys used: `injection_id` hex, `injection_data_str`
  with `X` nibble wildcards, `injection_interval` `[start_s, end_s]` elapsed from the
  capture's first frame; masquerade variants are recognized by a `masquerade` flag or
  `masquerade` in the filename).
- **Use for the gate**: masquerade captures ONLY (`--can-attack masquerade`, or a specific
  suite e.g. `--can-attack max_speedometer_masquerade`). Fuzzing/fabrication are
  timing-transparent (any frequency window catches them) — not recurrence evidence.
  Masquerade is frequency-preserving, so include payload bits (`--can-payload-bytes 8`).
  `accelerator` captures have no per-frame labels — excluded from per-message metrics.

## SynCAN (ETAS) — SECONDARY (true suspension attack; synthetic)

`git clone https://github.com/etas/SynCAN` — `train_1..4.zip` (concatenate in order) +
6 `test_*.zip` in the repo. License: custom non-commercial/academic only (fine for the
paper; note it). Cite Hanselmann et al., *CANet*, IEEE Access 8:58194-58205, 2020.
**Not yet wired into the loader**: SynCAN is signal-valued (normalized floats per ID) —
it needs a quantization front-end (thermometer over signal values) that doesn't exist
here yet. `test_suppress` is the only public true suspension attack; score it at
interval/window level (per-message metrics are ill-defined for an absent ID).

## HCRL Car-Hacking — SANITY ROW ONLY (never gate evidence)

- <https://ocslab.hksecurity.net/Datasets/car-hacking-dataset> — free Dropbox links (a
  short requester form may appear). ~17M messages. Cite Song et al., Vehicular
  Communications 21 (2020) 100198.
- Layout: `data-can/hcrl/*.csv` (`Timestamp,ID(hex),DLC,DATA0..,Flag(T/R)` — the loader
  parses this schema; the differently-formatted attack-free `.txt` is NOT supported).
- All four attacks are high-rate injections with the vehicle stationary — ~0.99 F1 is
  table stakes and says nothing about recurrence (documented shortcut dataset). At most
  one comparability row.
- Related HCRL sets (instructions only): Survival-IDS
  (<https://ocslab.hksecurity.net/Datasets/survival-ids>, zip password `ai.spera!+`,
  cite Han/Kwak/Kim VehCom 14 (2018)); Challenge-2020 (IEEE DataPort DOI
  10.21227/qvr7-n418, needs a free IEEE account, cite Kang et al. NDSS AutoSec 2021).

## Split / evaluation discipline (enforced in `can_data.py`, don't undo it)

- NEVER random row/message splits (near-duplicate periodic frames leak into test — the
  documented anti-pattern of arXiv:2408.17235); splits are contiguous-in-time or
  whole-capture holdout, windows never straddle a boundary.
- Δt thermometer edges + the top-N ID vocabulary are fit on the TRAIN split only and
  persisted into the results JSON (`can_dt_edges`, `can_id_vocab`).
- Report message-level confusion (`test_fpr` / `test_recall` / `test_f1` in the JSON)
  against the always-normal floor; accuracy alone is dishonest at these base rates.
- Gate verdict protocol: recurrent vs the STRONGEST windowed-FF arm (sweep the window
  honestly), matched gate budget, masquerade/suspension-class attacks only.
