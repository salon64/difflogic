# 24 — Portfolio roadmap: the paper & product pool (2026-07-12)

_The grab-bag. Everything the 2026-07 scouts (docs [19](19_learnable_latch_scout.md),
[21](21_landscape_weakness_scout.md), [22](22_applications_scout.md)) left standing, turned
into ready-to-grab outlines: papers in `mlgn/paper/p4..p14/init.md`, products in
`mlgn/product/*/init.md`. **Deliberately no timeline for the papers** — pick by gate-cost and
mood when P1–P3b leave slack. The committed program (P1–P3b) is NOT re-planned here._

## The committed program (context only — plans live elsewhere)

| id | what | plan |
|---|---|---|
| P1 | logic-native gating | [08_paper1_checklist.md](08_paper1_checklist.md) — NeurIPS'26 workshop, submit ~Aug 29 |
| P2 | latch/clatch primitives (the anchor) | [11_paper2_workmap.md](11_paper2_workmap.md) — A\* main '27 (doc-23 calendar says ICML Jan '27; ICLR'27 was the earlier pencil) |
| P3a | verification / certificates | `../paper/p3a/p3a_skeleton.md` — arXiv-first |
| P3b | hardware + Kyushu + CAN carrier + tool paper | [23_p3b_workmap.md](23_p3b_workmap.md) — FPL'27 target |

## Paper pool (p4–p14) — numbered GO-first, then by nearness

| id | working title / topic | verdict (verified) | honest tier | the gate (cost) | depends on |
|---|---|---|---|---|---|
| [p4](../paper/p4/init.md) | Learnable memory primitives (latches as trainable params) | COND, soft-go (doc 19) | workshop/ALIFE floor; B-tier EDA (DATE/FPL/FCCM/TRETS) with synthesis; **not A\*** | FF/LUT Pareto vs register pruning, real synthesis (~30–40% odds) | P2 shipped + P3b flow |
| [p5](../paper/p5/init.md) | Are LGNs calibrated? Free circuit ensembles from the gate posterior | **GO ~65%** (doc 21 §1) | A\* main attempt; TMLR / UQ-workshop fallback | sample 32 circuits from a checkpoint — entropy check (**1 afternoon**) | existing checkpoints only |
| [p6](../paper/p6/init.md) | Netlist plasticity: warm-start / fine-tune / adapt trained netlists | **GO ~60–63%** (doc 21 §2) | main-track attempt; TMLR fallback | warm-start falsifier (**1 day**) | checkpoints + trainer |
| [p7](../paper/p7/init.md) | Recurrent logic-gate CAN IDS w/ model-checked safety (the app flagship write-up) | **GO ~72%** (doc 22 §3) | security venue: RAID/ACSAC/ESORICS/VehicleSec; USENIX/CCS stretch | C0.g — **already queued inside P3b** | P3b C0 complete |
| [p8](../paper/p8/init.md) | Verified in-sensor bearing condition monitor | COND ~65% (doc 22 §1) | domain: IEEE Sensors J. / TII / MSSP / PHM (not ML) | bearing-wise honest-split spike (**1–2 wks**) | P2 cell; P3b flow for hw numbers |
| [p9](../paper/p9/init.md) | Nanosecond LOB inference | COND ~65% academic (doc 22 §2) | ICAIF / FPL-FCCM app track; **not A\*** | FI-2010 run, existing code (**1–2 days**) | P2 + P3b emitter; quant co-author |
| [p10](../paper/p10/init.md) | Empirical robustness of difflogic (attacks/corruptions/faults) | COND ~55–60% (doc 21 §3) | SaTML; A\* only if masking finding is big | OpenReview/arXiv freshness sweep (**hours**) | checkpoints; attack tooling |
| [p11](../paper/p11/init.md) | Standard-cell-aware training (ASIC gate vocabulary) | COND ~55–60% (doc 21 §4) | DATE; ICCAD/DAC stretch; MLSys alt | tech-mapping control expt FIRST | yosys/OpenLane; P3b synergy; feeds p4 |
| [p12](../paper/p12/init.md) | Why LGNs overfit + what regularizes them | COND ~65% but **300–500 GPU-h + racing** (doc 21 §6) | A\* main IF a regularizer wins; else TMLR | read DSLGN full text (LTU Xplore, **0 GPU**) | DUST headroom post-P2 |
| [p13](../paper/p13/init.md) | Why random wiring works (embedding theorems; expressivity ONLY) | COND ~55% (doc 21 §7) | TMLR realistic; A\* stretch if bounds tight | read Petersen thesis + Mommen (**0 GPU**) | nothing (theory) |
| [p14](../paper/p14/init.md) | Gate counts aren't comparable (measurement critique + EDA audit) | COND ~55%, demoted (doc 21 §5) | IWLS / DATE-short / TRETS note — or a **P3b paper section** | subsumed-by-ISTA check (**hours**) | P3b emitter |

**Suggested first grabs when slack appears:** p5's afternoon gate → p6's 1-day falsifier →
p9's FI-2010 run. Three gates ≈ one week total, and they decide the next standalone paper
almost for free. p13/p12 gates are reading — commute material.

## Product pool

| dir | what | status | existence gate |
|---|---|---|---|
| [can-ids](../product/can-ids/init.md) | Certified CAN IDS core (ISO 21434 / UNECE R155 wedge) | candidate | P3b C0.g; then partner+data |
| [bearing-monitor](../product/bearing-monitor/init.md) | In-sensor µW verified condition monitor (SKF–LTU UTC route) | candidate | honest-split spike; front-end energy budget |
| [lgn-toolchain](../product/lgn-toolchain/init.md) | LGN→verified-RTL flow, open-core (P3b A4 ships the OSS) | strategic / OSS-first | external user completes quickstart |
| [hft-inference](../product/hft-inference/init.md) | ns learned inference for trading | **PARKED** (NO-GO near-term) | revive conditions in doc (p9 + partner + regulatory pull) |

Product logic in one line: the toolchain is the enabling asset (OSS moat), can-ids and
bearing-monitor are the two wedges with real buyers (regulation and battery life
respectively), HFT waits for its paper and a partner.

## Dependency map (what unlocks what)

- **Existing checkpoints only** → p5, p6, p10 (the "no excuses" tier — runnable today).
- **Reading only** → p12 gate, p13 gate, p14 gate.
- **P2 shipped** → p4 (also burns its workshop-tier novelty — accepted), p9 method, p7/p8
  sequential claims.
- **P3b emitter/synthesis flow** → p4 tier-(b), p7 hardware numbers, p8 hardware numbers,
  p9 latency table, p11 (mutual: p11 gives p4+products the ASIC leg), p14, all products.
- **P3a property machinery** → p7 certificates, p8 alarm certificates, can-ids +
  bearing-monitor evidence packs, hft property packs (if ever).
- **Cross-links worth remembering:** p6's gate-edit deltas = can-ids OTA-update feature;
  p10's fault injection = P3a/P3b synergy; p14 = P3b paper's reporting section by default.

## Parked / dead (so nobody re-scouts them)

| topic | verdict | where it's recorded |
|---|---|---|
| Line-rate NID grown into a paper | PARTIAL — **P2 motivation vignette ONLY**; certified evasion-robustness is the growth path if ever | doc 22, doc 23 §C2 |
| EEG seizure prediction (wearable) | PARTIAL — watch item; only if CAN & bearings stall; imec-delta check first | doc 22 |
| Logic-attention | killed as worded (BiHDTrans 2509.24425); paradigm-internal slice post-P2 at best | doc 21 §8 |
| Skip/Highway-LGN | parked post-P1/P2 future work | doc 05 |
| Smart-grid relays | strongest cert story but ms-scale = wrong-bottleneck (W2) trap | doc 22 |
| Space/rad-tolerant | unvalidatable solo | doc 22 |
| KWS / L1-trigger / HAR / RF | occupied (TsetlinKWS, CICADA, DWN nanoML, per-window) | doc 22 |
| Input encoding | **NO-GO 95%** (WARP-LNN + BitLogic v2) | doc 21 §9 |
| Output heads / regression | **NO-GO 93%** (LogicIR ECCV'26 + BEL/RLEL) — fold pure-logic actuator head into P3b as a section (⚠ not yet reflected in doc 23 — add when drafting the tool paper) | doc 21 §10 |
| RL×LGN, SNN/Hebbian, VSA headliners | NO-GO as directions; salvage already folded into P2 | docs 15, 13, 08 |

## Race watch (applies pool-wide)

- **ETH (BitLogic, 2602.07400)** industrializes the hardware-culture axes — freshness-sweep
  before committing to anything in docs 21 §3–§5 territory (p10, p11, p14). BitLogic v2 =
  must-cite everywhere.
- **ISTA (2507.02585, 2505.19932)** owns netlist surgery + feedforward verification — p6/p10/p14
  adjacency; also P2's #1 competitor generally (doc 16).
- **ETH-DISCO (RDDLGN)** named FF/latch FPGA work as future work → p4's scoop vector.
- The ML-culture axes (p5 calibration, p6 adaptation, p10 robustness-empirical) are the quiet
  lanes — that's exactly why they're top grabs. p12 is ML-culture too but NOT quiet
  (doc 21 §6: 300–500 GPU-h and racing two active labs).
- Standing defense everywhere: **speed-to-arXiv** once results exist.

## Bookkeeping

- Each `init.md` separates **must-cites (scout-verified)** from **background reading (added at
  outline time — sanity-check before citing)**.
- When a pool paper is picked up: promote it with a workmap doc here (the doc-11/23 pattern),
  and log the gate verdict in [04_experiment_log.md](04_experiment_log.md).
- When a gate kills something: mark the init.md header DEAD with one line of why, don't delete
  (the negative verdict is the record).
