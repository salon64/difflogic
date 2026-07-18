# 25 — UAV-LGN detector scout (2026-07-18): drone thesis, reframed from control → detection

_Adversarial novelty scout (research-scout method, 8 kill-angle agents + synthesis, threats read
in full). Trigger: the flight-CONTROL gate failed (P3b findings §1.3 — LGNs don't converge on
hover), and Malcolm's read that "LGNs fit fast/simple supervised sub-components, not the loop." So:
what NON-RL drone task fits? Scouted the reframe = **recurrent LGN as a fast, on-board, VERIFIABLE
UAV fault/attack detector** (the flight twin of the CAN-IDS carrier), two variants._

## TL;DR verdict

| variant | verdict | why |
|---|---|---|
| **(a) UAV FAULT detection** (ALFA / UAV-SEAD) | **CONDITIONAL ~60%** | genuine data headroom + no FPGA/verified prior work on these sets; but the whole novelty rests on the ONE surviving pillar (verification of a learned circuit) |
| **(b) GPS-spoofing / attack IDS** | **NO-GO ~70% blocked** | R2U2 already does verified on-board GPS-spoof detection in FPGA; incumbents ~0.998 F1 @ 15 ms (ns latency unneeded); UiA/Granmo Tsetlin race |

**The reframe that survives:** an application+verification paper on **variant (a)**, where the moat is
**a machine-checked TEMPORAL detection theorem (bounded detection latency / no-false-trip over a
horizon) proved on a gradient-LEARNED gate-level circuit** — NOT recurrence, NOT FPGA-speed (both
pre-claimed). Recurrence is demoted to a hardware/verifiability ablation; **default to a
WINDOWED-feedforward LGN** for the detector itself.

## What got killed (every moat pillar except one is pre-claimed)

- **Recurrent-LGN mechanism → OWNED by ETH.** RDDLGN "Recurrent Deep Differentiable Logic Gate
  Networks" (Bührer/Plesner/Aczel/Wattenhofer, EdgeFM'25, **arXiv:2508.06097**). The memory-flagged
  ETH race risk is now **published**. Translation only; no FPGA, no verification, no UAV. → We can
  APPLY+deploy+verify recurrent LGN, never claim to invent it.
- **"LGN = fast/cheap/zero-DSP on-FPGA anomaly detector" → OWNED by Princeton.** CLGN "Rapid
  Inference of LGNs for Anomaly Detection in HEP" (**arXiv:2511.01908**, 2025): conv-LGN anomaly
  detector synthesized to Virtex-7, **18.75 ns / 3 cycles / 0 DSP**, ≥ quantized NN, explicitly
  "…and beyond." Feedforward, unverified, non-UAV. → The pitch **cannot rest on "LGN is fast on
  FPGA"** — that flag is planted.
- **Verified on-board UAV temporal monitor → largely OWNED by R2U2** (Rozier/Schumann/Moosbrugger,
  FMSD'17, v3.0 2023): on-board FPGA MLTL/MTL runtime observers + Bayesian diagnosis over
  GPS/GCS/sensor/actuator streams, **including a GPS-spoofing case study**, flight-certifiable.
  **THE single most dangerous flag-owner.** Differentiation survives ONLY because R2U2 specs are
  **hand-authored temporal logic**, not a learned classifier whose exact circuit is the verified
  object. This is the crux the whole idea lives or dies on.
- **Recurrence-earns-its-keep on UAV telemetry → REFUTED.** AeroTSBoost (**arXiv:2605.25639**, 2026):
  windowed statistical features + gradient boosting (memoryless) beats recurrent baselines on BOTH
  ALFA (0.926 AUPRC) and UAV-SEAD (0.752, +5.79 over the strongest incl. recurrent), states
  "recurrence not necessary"; spoofing detectors deliberately drop LSTM; 1D-CNN detects faster than
  GRU/LSTM. → The recurrence-earns-its-keep gate that CAN **passed** would **fail** on UAV telemetry.
  (CAN's timing/sequence structure is genuinely stateful; UAV telemetry is windowable. Clean, honest
  distinction — keep the CAN recurrence claim, don't port it to UAV.)
- Also planted: LGN static-verification-friendliness (Kresse SAIV'25 **2505.19932**, already in our
  KB); exact model-checking of learned binary nets (QNN-SMT 2106.05997, certified-BNN SAT'25);
  ns-latency FPGA IDS (PolyLUT-Add 2406.04910 ~92%@8ns; QNN-CAN 2401.12240; BIDS).

## The surviving gap (narrow, composition-novelty, ~55–60%)

A gradient-**learned** detector whose exact gate-level (state-holding) circuit is **itself** the
machine-checked object, carrying a **non-trivial temporal detection theorem** (bounded latency k /
no-false-trip over horizon H), on **real UAV fault data** (UAV-SEAD is baseline-free; ALFA has
headroom, SOTA ~0.89 macro-F1). Every ingredient is individually pre-claimed; the **composition** —
"temporal detection property proved on a learned circuit at deployable latch scale" — is not. It is
an **application** contribution (+ an analysis sub-contribution reusing the P3a sim+temporal-
induction recipe), NOT a mechanism claim. A reviewer WILL say "composition, not mechanism" — so the
temporal theorem must be substantive and hard for a hand-written R2U2 spec to replicate.

## The decisive GATE (run first, before any commitment)

Take ONE trained LGN fault detector (**windowed-feedforward is fine**) on UAV-SEAD (or ALFA) and
verify end-to-end that a single useful temporal theorem — bounded detection latency k, or
no-false-trip over horizon H — is: **(i)** machine-checkable with the existing P3a stack at the real
detector's latch count (naive IC3 died at ~1k latches → sim + temporal-induction recipe); **(ii)**
genuinely hard to replicate as a hand-written R2U2 MLTL spec; **(iii)** the detector still hits
competitive AUPRC (≥~0.89 macro-F1 ALFA / beats AeroTSBoost's 0.752 on UAV-SEAD). **If you can't
produce a scalable, non-trivial temporal theorem that R2U2's hand-spec can't trivially match, the
idea collapses into R2U2 — drop it.**

## Cross-program implications (matter beyond the drone thread)

1. **RDDLGN is published** (2508.06097) — the ETH recurrent-LGN race risk is realized. Confirms P2
   must cite+distinguish it and get arXiv'd; it does NOT scoop P2's register/verification/deployment
   angle (translation only).
2. **R2U2 is a must-cite competitor for the ENTIRE P3a verification story**, not just drones — it
   owns "verified on-board runtime monitor." Our P3a differentiator is the same everywhere: the
   **learned circuit is the verified object**, vs R2U2's hand-authored specs. Add to P3a related work.
3. **CLGN (Princeton) already planted "LGN-anomaly-detector-on-FPGA, and beyond"** — the tool-paper /
   any LGN-detector framing must cite it and lead with verification, not FPGA-speed.

## How to apply
Don't re-scout this. If the drone thread is un-parked: pursue **variant (a) fault detection**, run
the GATE above FIRST, frame on **verification-of-a-learned-circuit** (recurrence = ablation, FPGA-
speed = table not headline), and position vs **R2U2** explicitly. Kyushu/Vargas fit is neutral-to-
slightly-(a): the verification leg extends Kresse-style bounded-bit-flip robustness into *temporal*
robustness of a fault detector — a clean formal hook for the adversarial-robustness thesis, without
(b)'s crowded security turf. Variant (b) GPS-spoofing = dead, don't pursue.
