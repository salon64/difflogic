# 22 — Scout: applications for the sequential-LGN stack (2026-07-08)

_Adversarial application scout (research-scout method, same 27-agent workflow as doc 21): three
scouts — bearings/SKF, high-frequency trading, broad domain sweep — each with all-synonym
2025–26 kill-searches, threats read in full, and an independent adversarial verifier re-attacking
every OPEN call. What the stack offers: {streaming/recurrent inference, exact-binary state in
hardware registers (P2 clatch), ~ns latency + µW on FPGA/ASIC (P3b), formally verifiable learned
circuit (P3a)}. Drones = already planned (P3b, doc 16)._

## TL;DR

| Application | Verdict (verified) | Route | The gate |
|---|---|---|---|
| **Automotive CAN-bus IDS** | **GO ~72%** — best new flagship | post-P3b app paper (security/systems venue) | recurrence-earns-its-keep ablation |
| **Bearings / SKF (kullager)** | **CONDITIONAL ~65%** | PHM/sensors venue 2027 + SKF-LTU industrial route | 1–2 wk honest-split feasibility spike |
| **HFT / low-latency trading** | **CONDITIONAL ~65% academic; NO-GO commercial (near-term)** | ICAIF / FPL-FCCM app track, 2027 | 1–2 day FI-2010 feasibility run |
| Line-rate NID (per-flow state) | PARTIAL — use as **P2 motivation vignette** | inside P2 / short paper | stateful-vs-windowed ablation |
| Wearable EEG seizure *prediction* | PARTIAL — certification-forward alternative | post-P3b | imec-delta check (they hold ECG feedforward) |
| KWS, physics trigger, HAR, RF, grid relays, space | occupied / weak-fit | cite-only | — |

**Unifying result:** the *feedforward* logic substrate has already colonized the obvious
flagships (KWS = TsetlinKWS 16.6µW; ECG = imec 2601.11433; L1-trigger = CICADA; NID = LogicNets;
HAR = DWN nanoML 2502.12173 — a verifier kill on the sweep's one factual error). But the
**{recurrent + exact-binary registers + verified temporal properties}** combination is applied
NOWHERE. That triple is exactly P2+P3a+P3b — the moat is the combination, not the substrate.

---

## 1. Bearings / SKF condition monitoring — CONDITIONAL ~65% (scout 72 → verified ~65)

**The pitch:** always-on, sub-mW, in-sensor sequential LGN on a tiny FPGA/ASIC for
vibration-based bearing-fault monitoring — degradation state, operating-condition context, and
alarm persistence/debounce counters held in P2's exact-binary registers; P3a model-checks
temporal alarm properties ("alarm within k windows of persistent fault", "no alarm latch on
transients < m samples") that are meaningless for feedforward and impossible for real-valued
monitors.

**Landscape (threats read in full):**
- DL bearing diagnosis: mega-saturated (99.9% CWRU) AND in a documented **evaluation crisis** —
  Vieira et al. (MSSP 2026, arXiv:2509.22267): accuracy collapses under leakage-free
  bearing-wise splits. Adopt their protocol = instant rigor differentiator.
- Low-power frontier: Vitolo (IEEE Sensors J. 2022) partially-binarized autoencoder, 65nm
  synthesis 138 µW/MHz; 17 mW iCE40 LSTM; MCU TinyML tier is mJ/inference commodity. All
  arithmetic NNs. The sub-100µW digital always-on **multi-class** tier is unclaimed.
- **Verifier kills on "first" claims (scope carefully):** weightless NNs already touch motor
  fault diagnosis (Abid & Afshan, Rev. Sci. Instrum. 2024); Yue&Jha DLNs run on FordA/FordB/
  Wafer fault time-series (2508.17512); imec published the LGN/LUTN biosignal application shape
  (ECG, 2601.11433, Artix-7, 50–70 pJ/inf, feedforward). Also: a streaming/stateful/online
  PdM engine exists on neuromorphic (PHM-Europe 2026) and ST's ISM330IS puts an ML core inside
  an industrial IMU — cite as the commercial trendline. **So: NOT "first LGN on machine
  health" — the claim is the sequential+verified system.**
- Temporal-verification method layer is occupied generically (Hosseini/Lomuscio AAMAS 2025, LTL
  verification of memoryful agents) — cite as method lineage; the *deployed-netlist
  condition-monitoring* instance stays open.

**The local edge (verifier-CONFIRMED and sharpened):** the **SKF–LTU University Technology
Centre exists** (est. Dec 2011, SKF's 5th UTC worldwide, scope = condition monitoring & asset
management, involving LTU Machine Elements / Operation & Maintenance / EISLAB; cooperation
publicly renewed), and **SKF Condition Monitoring Center (Luleå) AB builds the IMx monitoring
hardware in Luleå**. SKF Insight (rotation-harvested in-bearing sensors) and IMx-1 (battery
mesh) prove **energy is the binding commercial constraint** — the LGN µW wedge; SKF Enlight AI
does inference off-sensor today. Aspinity AML100 (<20 µA analog wake-up) proves the µW tier
sells but can't do multi-class or verification.

**Contribution shape:** application/system paper — "first logic-gate-network condition-monitoring
system; the first fault monitor whose alarm logic is a formally verified sequential circuit" —
accuracy parity under the honest bearing-wise protocol at 2–4 orders less energy than MCU-TinyML
and ~10–100× under BNN/LSTM-FPGA, + model-checked alarm certificates. Venues: IEEE Sensors J.,
TII, MSSP, PHM Society — not ML venues. Datasets: CWRU, Paderborn-KAt, XJTU-SY + IMS
(run-to-failure for the sequential story), MFPT. Hardware: $100–300 iCE40/Artix board + MEMS
accelerometer; reuses P3b emitter + P3a flow 1:1.

**GATE (1–2 weeks, schedulable around P2):** feedforward difflogic on thermometer-encoded
envelope-spectrum features, Paderborn+CWRU, **bearing-wise leakage-free splits** — must land
within a few points of the (collapsed) realistic DL baselines. If LGN accuracy craters under
cross-bearing domain shift, the pitch dies regardless of energy/verification.

## 2. High-frequency trading — CONDITIONAL ~65% academic / NO-GO near-term commercial

**The pitch:** each market-data event clocks a learned circuit once — clatch registers hold
LOB-derived state, no window recompute — emitted as clocked RTL at ns-scale reported latency.

**Landscape (threats read in full):**
- **Direct kill-check CLEAN and verifier-confirmed** (~20 differently-worded searches, 6
  full-text reads): no logic-native substrate (LGN/LUT-net/DWN/BNN/Tsetlin/weightless) has ever
  been applied to market data.
- Closest prior art: LOBIN (HPSR'23) + Hong et al. (ECAI'24) — trees/tables squeezed into Tofino
  switches at **µs** latency with 2–16% accuracy loss, explicitly ruling DL out of the data
  plane; **FINN-GL (FPL'25, 2506.20810** — verifier find, must-cite**)**: FINN/BNN-lineage
  ConvLSTM on FI-2010, W8A6, **4.3 ms** — three orders off the hot path. Audited commercial
  frontier: Myrtle VOLLO ~**2 µs** p99 LSTM (STAC-ML record, Apr 2026), Xelera Silva 1.6 µs GBT.
- **The feared counterargument INVERTED:** feature-building on FPGA is ns-scale (parse 20–25ns,
  book update 30–40ns/msg; top firms quote sub-100ns tick-to-trade) while the learned model is
  the 40–100× outlier at 1.6–2µs — so ns-scale *inference* is exactly the unserved niche. The
  ECAI'24 paper's own profitability simulation (+4.08% from latency vs −2.17% from accuracy
  loss) prices the trade.
- Publishability: precedent says public-data backtest + synthesized/measured latency clears the
  bar (LOBIN@HPSR with one day of free ITCH; FINN-GL@FPL) — venues: ICAIF, FPL/FCCM app tracks,
  ECAI-class. Data: FI-2010 free, ITCH samples free, LOBSTER academic ~$100s, crypto L2 free.
- Commercial: **occupied at µs by audited incumbents; no LGN vendor exists.** Prop shops
  hand-code RTL; a learned circuit is a compliance question — but the **MiFID II RTS 6 /
  ESMA 2026** explainability + kill-switch regime makes "formally checkable learned circuit"
  (P3a) the actual commercial wedge, unclaimed. Verdict: paper first; the paper IS the demo.

**Contribution shape:** "Nanosecond limit-order-book inference with sequential logic gate
networks" — first logic-native learned model in the trading hot path; headline artifact = the
accuracy-vs-wire-latency frontier table (trees-in-switch µs / VOLLO-Xelera 1.6–2µs / feedforward
LGN ~10–25ns / clatch-LGN O(1)-per-event). Method-gated by P2, tool-gated by P3b emitter →
lands 2027.

**GATE (1–2 days, existing code):** feedforward difflogic on FI-2010 mid-price movement
(3-class, k=100). If discretized F1 falls far below the ~40–46% tree/DeepLOB band on this noisy
regime, the application dies regardless of latency.

## 3. Broad sweep — GO ~72%: CAN-bus IDS is the flagship

**Winner: automotive CAN-bus intrusion detection.** Survived 6+6 fresh attack angles across two
agents: **no learned-logic-circuit CAN IDS exists** (incumbents are all quantized/BNN: BIDS
<0.17ms, SecCAN, FPGA zero-day IDS 2401.10724; "can-logic" is hand-written temporal logic, not
learned). CAN message-ID timing/sequence = genuine hidden state (P2 load-bearing);
**ISO-21434/26262 makes model-checked no-false-trip / bounded-detection-latency properties a
paying feature** (P3a load-bearing); ns/µW on an ECU-adjacent FPGA (P3b load-bearing). Open
datasets: Car-Hacking/HCRL, ROAD, SynCAN. ~4–8 week add-on once P2+P3b tooling exists.
**Contribution:** "A recurrent logic-gate intrusion detector: exact-binary per-message state on
FPGA with model-checked safety" — application+system+measurement, security venue.
**GATE:** the recurrence-earns-its-keep ablation — recurrent clatch-LGN must beat a windowed
feedforward LGN given the same input window on CAN timing attacks; if stateless matches, the
domain collapses to "another feedforward LGN app".

**Runner-up / P2 vignette: line-rate NID with per-flow recurrent state.** NID is already the
canonical feedforward logic-substrate benchmark (LogicNets 91.3%/10.5ns UNSW-NB15 — instant
credibility); stateful-beats-stateless is proven by NON-logic hybrids (Brain-on-Switch
binarized-GRU +0.13–0.31 F1, NSDI'24; FENIX/Quark follow-ups piling on) — all unverified,
none logic-native. Use as P2's motivation example (zero domain onboarding); certified
evasion-robustness is the differentiator if grown into a paper.

**Third: wearable EEG seizure *prediction*** (long-memory, FDA-certification wedge, open
CHB-MIT/TUH data) — real but carries scoop risk: imec planted the feedforward biosignal-LGN flag
(ECG, 2601.11433) and holds the hardware pipeline. Only attractive if CAN/bearings stall.

**Ruled out (with reasons):** KWS (TsetlinKWS 16.58µW 65nm + BNN commodity; no cert wedge);
particle L1 trigger (CICADA owns it; per-bunch-crossing = weak recurrence need); **HAR (verifier
kill: DWN nanoML 2502.12173 — LUT substrate on HAR, 5ns/sample, FPGA — occupied)**; RF/AMC
(per-window, no cert wedge); smart-grid relays (strongest cert story but ms-scale = W2
wrong-bottleneck trap; park as future work); space/rad-tolerant (differentiated on paper,
unvalidatable solo).

---

## Sequencing recommendation

1. **Now (around P2 writing):** nothing new starts. Optionally the two micro-gates that de-risk
   2027 choices: FI-2010 run (1–2 days) and the bearing-wise-split spike (1–2 wks, only if slack).
2. **P2 paper:** use the NID streaming vignette as the motivation example (zero cost, benchmark
   credibility); cite doc-21/22 landscape finds (BitLogic v2, LogicIR, FINN-GL) where relevant.
3. **Post-P3b (with emitter + verification flow in hand):** CAN-bus IDS flagship application
   paper; bearings/SKF as the industrial-collab track through the SKF-LTU UTC (thesis-adjacent,
   Swedish funding logic: Vinnova/SSF industrial-PhD patterns); HFT paper if the FI-2010 gate
   passed and a quant-finance co-author appears.
4. The Kyushu drone thesis (P3b) remains the physical capstone; these applications don't
   compete with it — they reuse its tooling.
