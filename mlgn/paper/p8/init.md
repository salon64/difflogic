# P8 — The Verified Alarm: An In-Sensor Sequential Logic-Gate Network for Bearing Condition Monitoring

_Status: **pool — CONDITIONAL ~65%.** Source scout:
[research/22 §1](../../research/22_applications_scout.md) (2026-07-08, verified; scout 72 →
verified 65). Twin doc: [product/bearing-monitor](../../product/bearing-monitor/init.md) — same
work, industrial route. Depends on: P2 (cell) + P3b tooling (emitter/synthesis); gate is
runnable earlier with feedforward difflogic. Venue year per scout: 2027. No timeline here —
grab from [research/24](../../research/24_roadmap.md)._

## What it is

Always-on, sub-mW, in-sensor sequential LGN on a tiny FPGA/ASIC for vibration-based
bearing-fault monitoring: degradation state, operating-condition context, and alarm
persistence/debounce counters held in P2's exact-binary registers; P3a model-checks temporal
alarm properties — "alarm within k windows of persistent fault", "no alarm latch on transients
< m samples" — that are *meaningless for feedforward* models and *impossible for real-valued*
monitors. Claim: **the first fault monitor whose alarm logic is a formally verified sequential
circuit** (NOT "first LGN on machine health" — see scope).

## What it covers

- **Accuracy under the honest protocol:** adopt the Vieira et al. (MSSP 2026, arXiv:2509.22267)
  leakage-free **bearing-wise splits** — the field is in a documented evaluation crisis (99.9%
  CWRU numbers collapse under honest splits); adopting their protocol is an instant rigor
  differentiator. Target: parity with the (collapsed) realistic DL baselines.
- **Energy:** 2–4 orders less than MCU-TinyML (mJ/inference commodity tier), ~10–100× under
  BNN/LSTM-FPGA (Vitolo 138 µW/MHz 65nm autoencoder; 17 mW iCE40 LSTM). The sub-100 µW digital
  always-on **multi-class** tier is unclaimed.
- **The sequential story:** IMS / XJTU-SY run-to-failure data — degradation tracking and alarm
  persistence, not just per-window classification. This is what separates it from every
  feedforward logic-substrate app.
- **Verified alarm certificates:** the P3a decode-type machinery on debounce/persistence
  properties.
- **Datasets:** CWRU, Paderborn-KAt, XJTU-SY, IMS, MFPT. **Hardware:** $100–300 iCE40/Artix
  board + MEMS accelerometer; reuses the P3b emitter + P3a flow 1:1.

## Scope — claim discipline (verifier-mandated)

- **NOT "first LGN/weightless on machine health":** Abid & Afshan (Rev. Sci. Instrum. 2024)
  touch motor-fault diagnosis with weightless NNs; Yue & Jha DLNs run FordA/FordB/Wafer
  time-series (arXiv:2508.17512); imec published the biosignal LGN application shape (ECG,
  arXiv:2601.11433, feedforward). **The claim is the sequential + verified system.**
- Temporal-verification method layer is occupied generically (Hosseini & Lomuscio, AAMAS 2025 —
  LTL verification of memoryful agents): cite as lineage; the *deployed-netlist
  condition-monitoring instance* is the open slice.
- Cite the commercial trendline (ST ISM330IS ML-core-in-IMU; neuromorphic streaming PdM,
  PHM-Europe 2026) to show the in-sensor tier is real, then differentiate on
  multi-class + verified + µW.
- (Added here, not a scout item:) honest energy accounting must include the **feature
  front-end** (envelope spectrum needs rectify/filter/FFT before the LGN) — see the product
  doc; reviewers at sensors venues will ask.

## Venue & tier (honest call)

- **Not an ML-venue paper.** Domain venues where this is a strong fit:
  - **IEEE Sensors Journal / IEEE TII** (top domain journals for in-sensor ML systems),
  - **MSSP** (the top mechanical-signals journal — where the eval-crisis paper lives; adopting
    their protocol makes it a natural home),
  - **PHM Society / IEEE PHM** conferences (community + industrial visibility).
- Tier translation for the CV: strong domain journal ≈ solid non-A* main-track; the PhD-package
  value is the industrial-collab story (SKF–LTU UTC) more than venue prestige.

## The gate (1–2 weeks, schedulable around P2; also P3b §C2's fallback-carrier trigger)

Feedforward difflogic on thermometer-encoded envelope-spectrum features, Paderborn + CWRU,
**bearing-wise leakage-free splits** — must land within a few points of the (collapsed)
realistic DL baselines. **If LGN accuracy craters under cross-bearing domain shift, the pitch
dies regardless of energy/verification.**

## Read up on

Must-cites (verified in the scout):
- **Vieira et al.** — MSSP 2026, arXiv:2509.22267 (the evaluation crisis + honest protocol).
- **Vitolo et al.** — IEEE Sensors J. 2022 (partially-binarized AE, 65nm, 138 µW/MHz — the low-power frontier).
- **Abid & Afshan** — Rev. Sci. Instrum. 2024 (weightless NN motor-fault; scoping cite).
- **Yue & Jha** — arXiv:2508.17512 (DLNs on FordA/B/Wafer; scoping cite).
- **imec ECG** — arXiv:2601.11433 (feedforward biosignal LGN, Artix-7, 50–70 pJ/inf; the substrate-application shape).
- **Hosseini & Lomuscio** — AAMAS 2025 (LTL verification of memoryful agents; method lineage).
- Commercial exhibits: SKF Insight / IMx-1 (energy = binding constraint), Aspinity AML100 (<20 µA analog wake-up; µW tier sells), ST ISM330IS.

Background (added here, not from the scout — sanity-check before citing):
- Randall & Antoni, MSSP 2011 — *Rolling element bearing diagnostics — a tutorial* (envelope analysis canon; the feature front-end justification).
- Smith & Randall, MSSP 2015 — the CWRU benchmark study (dataset caveats).
- Lessmeier et al., PHM Europe 2016 — Paderborn dataset paper.

## Risks / kill conditions

- Gate fails on cross-bearing shift → dead regardless of energy story.
- The front-end energy dominates and erases the µW claim → scope to the compute stage with
  explicit accounting, or move feature extraction into logic too (research question in itself).
- imec-style scoop (they hold the LGN-biosignal pipeline; bearings are one lateral step) —
  freshness-sweep before committing.
