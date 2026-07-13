# 23 — P3b work map: hardware realization + Kyushu thesis (kicked off 2026-07-11)

_P3b = the deployment phase: from P3a's verified netlists to measured silicon (Fmax/ns/nJ,
on-board), a real-domain benchmark carrier, and the Kyushu drone thesis. Kickoff decisions
recorded here. Supersedes the P3b portions of research/20 §D2 (written 07-08, before the
P3a sprint banked the emitter, synthesis, and verification demo) AND research/22's
sequencing of CAN-IDS as post-P3b (pulled forward, see Decision 2). Operationalizes
research/10 (FPGA scout), research/16 §C (drones/Kyushu), research/22 (applications)._

> **⏸ PARKED 2026-07-13** to focus on the P1/P2 papers. **The self-contained findings +
> resume point is `../paper/p3b/findings.md`** — read that first to pick this back up.
> Status at park: T1/T2 hardware done (367.9 MHz measured), CAN carrier PASSES (recurrence
> > ff; register > recompute clean on rlon), flight D1 recalibrated-and-queued (round 1
> undertrained → `run_queue_d1.sh` tags `d1v2_*` ready to re-run). This is a head start —
> a 3rd paper for the Dec'26→Japan gap, or a PhD-start publication runway.

## THE DECISIONS (both 2026-07-11)

### 1. Kyushu pitch leads; tool paper follows on the same trunk

Both legs share their first ~4–6 weeks of build (the trunk, §B); the ordering governs what
runs alongside it and which deliverable claims the first finish slot. **Malcolm chose
Kyushu-first.** Rationale, ranked:

1. **The Kyushu window is externally fixed and earliest** — pitch out ~Sep–Oct '26,
   lead-in Oct '26–Jan '27, in-country Apr–Sep '27. The thesis is one of the four
   load-bearing elements of the PhD package; Apr '27 can't slip.
2. **The closed-loop gate needs maximum lead time** — new code + GPU runs, mandated by
   research/16+20 BEFORE the thesis topic is committed.
3. **The hardware timestamp no longer hinges on FCCM.** P3a's imminent arXiv contains the
   yosys synthesis of a *verified* sequential LGN; the tool paper arXivs independently the
   day measurements exist (standing race defense = speed-to-arXiv).
4. **January '27 is already congested** (ICML = P2, possibly CAV = P3a). FPL '27
   (~late Mar) de-conflicts the third deadline inside an ETH race window (~Jun '27) that
   will already contain public timestamps.

**Tool-paper venue policy:** arXiv the moment measured numbers exist; **FPL '27 = planning
target**; FCCM '27 (~early Jan, verify exact date ~Sep '26) = stretch; NeurIPS '27 demo
track = fallback (per research/20 §D2).

### 2. Benchmark carrier = CAN-bus IDS; flight sim demoted to thesis-gate-only

The flight sim was never "the benchmark" — it is the closed-loop trainability **gate**
research/16+20 mandate before the thesis commitment. As a benchmark, flight control fails
on every axis: no dataset/leaderboard, ISTA's DWC owns feedforward LGN-control, the
unoccupied slice ({recurrent + real drone + verified}) needs Kyushu's hardware to claim,
and control framing must avoid latency (research/16's W2 trap). The benchmark role goes to
research/22's own verified flagship, **CAN-bus intrusion detection (GO ~72%)** — pulled
forward INTO P3b (doc 22 had sequenced it post-P3b; the full application paper still lands
after, at a security venue). **Extension arrows from one trunk: drone = the research
extension** (embodied, Kyushu, rungs 2–4); **automotive = the commercial extension**
(certified IDS; ISO-21434 compliance-artifact hook). Secondary benchmarks: gated
portfolio, §C2.

## A. Inventory — what P3a already banked for P3b (2026-07-10/11)

| in hand | where |
|---|---|
| Verilog-2001 emitter + BLIF, iverilog golden/random equivalence, post-synth re-sim, per-gate truth-table cross-check | `mlgn/netlist/verilog.py`, `synth/` |
| yosys `synth_xilinx`: 901 LUT + 885 FDRE ≈ 8.7%/4.3% of XC7A15T, 0 BRAM/DSP | `synth/report.md` |
| In-netlist GroupSum/argmax head (11.7k gates, exact tie-break) — built, NOT yet synthesized | `netlist/head.py` |
| Machine-checked theorems incl. d20 full-correctness ("any write + ANY distractor stream ⇒ correct readout forever") | `netlist/out/`, p3a skeleton |
| 12 verified circuits across mechanisms/regimes + self-contained checkpoints | `netlist/out/*/report.json` |

**Missing (= the P3b work):** any P&R (no Vivado/nextpnr anywhere) → no Fmax/ns/nJ; any
board/bitstream; a CAN data loader (nothing in repo); the closed-loop POMDP gate (zero
code); recompute-vs-register RTL control; pitch documents.

## B. Shared trunk (start immediately; feeds BOTH legs)

- **T1 — head-in-fabric deployable top. DONE (see §H').** `lgn_top` = FSM + popcount head:
  `x[…]` in, `class_out[2:0]` out; extends `synth/export_fsm.py`; golden vectors incl.
  decode. Kills the 1,035-port I/O blocker; doubles as P3a §R's open "head-included
  synthesis". (A consumer-visible `valid`/settled bit is descoped to T2 — the K≥16 settle
  envelope is documented, not yet surfaced as a port.)
- **T2 — P&R + timing. OPEN-FLOW DONE (see §H'); Vivado sign-off pending.** **FIRST REAL
  Fmax (openXC7, `top.v`/`lgn_top`, xc7a35tcsg324-1, 2026-07-12): 367.9 MHz best-of-4
  (spread 316–368, ~14% seed variance) → 2.72 ns/step, 43.5 ns/decision (K16), 51.6 ns
  (K19).** Util 2,651 LUT / 885 FDRE / 0 CARRY4, 12.7% LUT of xc7a35t; A7-100T 326.7 MHz /
  4.2% LUT. **Critical path is INSIDE the FSM register recurrence (2 LUT levels,
  routing-dominated 2.2 ns) — the head's 0-CARRY4 popcount tree is OFF the critical clock
  path** (answers the §7.6 open question, resolves the §G "T2 timing unknown" risk). Raw
  `fsm.v` (1024-bit q) is I/O-unplaceable — machine-confirms the I/O-bound caveat; the
  deployable `top.v` places at 14 IOB. Caveat: open-flow nextpnr timing is a delay-model
  ESTIMATE, not sign-off. **Vivado 2026.1 free "Basic" tier (all 7-series, but now
  Windows-only; 2025.2 = last free-on-Linux) → authoritative Fmax + the T3 power numbers;
  the ONE remaining T2/T3 user action.** Details: report.md §8, `synth/run_pnr.sh`.
- **T3 — energy.** Vivado `report_power` with SAIF activity from the testbenches →
  **nJ/step, nJ/decision**; on-board current measurement later as the honest upgrade.
  (Verify SAIF/report_power availability in the free Basic tier at install — reported,
  unconfirmed.)
- **T4 — board bring-up + demo.** Order an **Arty A7-100T now** (~$299, DigiKey / RS
  Sweden se.rs-online.com / Mouser). **NOTE (verified 2026-07-12): the Arty A7-35T is
  retired/EOL** — residual stock only, unpredictable pricing; the A7-100T is the
  in-production sibling (same board, bigger XC7A100T, open-flow-supported). Two stages:
  **bring-up demo** = the copy register (circuits exist today; UART/LED "write →
  distractor storm → readout never wrong" — the d20 theorem made physical); **headline
  demo** = CAN-IDS once §C0 lands: replay a recorded CAN trace over UART, the board flags
  injections at line rate — "live intrusion detection on a ~$300 Artix-7 board, with a
  machine-checked no-false-alarm envelope." Video = pitch + paper collateral.

## C. Benchmark carrier + portfolio

### C0 — CAN-bus IDS (primary carrier; research/22 flagship, GO ~72%)

Why it fits P3b exactly:
- **Genuine hidden state** (message-ID timing/sequence — P2 load-bearing); incumbents are
  all quantized/BNN (BIDS <0.17 ms, SecCAN, FPGA zero-day IDS 2401.10724) — **no
  learned-logic-circuit CAN IDS exists** (verified, doc 22).
- **The P3a property machinery ports ~1:1**: a legal-traffic input automaton plays the
  blank/distractor automaton's role; **no-false-trip = a decode-type invariant over legal
  traffic** (`distractor_decode`'s analog); **bounded detection latency = shadow-armed
  `protocol_decode`** (shadow arms on the attack pattern; bad = not-alarmed within K).
  The dose-response result becomes *"train with injected attacks → machine-checked
  no-false-alarm envelope"* — the certified-IDS story ISO-21434/26262 pays for.
- **Open datasets, no rig**: ROAD (primary — realistic masquerade/targeted-ID attacks),
  Car-Hacking/HCRL, SynCAN (suppress/masquerade). ns/µW on an ECU-adjacent FPGA = the
  trunk's T2/T3 numbers with a buyer.

**GATE C0.g — run FIRST (1–2 wks, DUST): recurrence-earns-its-keep.** Recurrent
clatch/gated vs a **windowed feedforward LGN given the SAME input window**, on
timing/sequence attacks specifically (masquerade/suspension-style; NOT flooding — any
window solves flooding). Multi-seed. Stateless matches ⇒ the flagship collapses to
occupied feedforward territory (doc 22's own kill condition) ⇒ fallback chain in §C2.

**Encoding sketch:** per-frame token = ID bits (one-hot over top-N IDs or 11-bit binary)
+ thermometer-coded inter-arrival Δt bins (+ payload bytes only if payload attacks in
scope); ~20–40 PIs. MC note: engines are symbolic so the wider input space is fine in
principle, but BFS-closure enumeration won't scale — lean on MC + the P3a round-3 finding
that trained closures are tiny. Free-input proofs at this alphabet may land as **bounded
certificates — an acceptable deliverable; do not promise unbounded theorems in the pitch.**

**Deliverables:** trained + verified CAN circuit → T4 headline demo + Leg-A real-domain
row; full application paper ("a recurrent logic-gate intrusion detector: exact-binary
per-message state on FPGA with model-checked safety", security venue) post-P3b.

### C2 — secondary portfolio (from research/22; gates only, nothing builds until triggered)

| track | verdict (doc 22) | gate (cost) | trigger / role |
|---|---|---|---|
| **Bearings / SKF** | COND ~65% | bearing-wise leakage-free split spike, Paderborn+CWRU, thermometer envelope-spectrum features (1–2 wks) | if slack, or when the SKF–LTU UTC route activates; industrial-collab track (Vinnova/SSF logic), PHM/sensors venue '27; **same Arty board + a MEMS accelerometer**; IMS run-to-failure = the sequential story |
| **HFT / FI-2010** | COND ~65% academic, NO-GO commercial | FI-2010 mid-price run, existing code (1–2 days) | cheap — slot whenever; paper only if the gate passes AND a quant-finance co-author appears; ICAIF / FPL-FCCM app track '27 |
| **Line-rate NID** | PARTIAL | none | **P2 motivation vignette ONLY** — no P3b work |
| **EEG seizure prediction** | PARTIAL | imec-delta check | watch item, post-P3b; certification-forward alternative (imec holds feedforward ECG) |

Fallback chain: **C0.g fails → bearings becomes the carrier candidate** (its gate then
runs); both fail → the demo stays on the synthetic register circuits (trunk + tool paper
unaffected; only the application flagship dies).

## D. Leg B — Kyushu (LEADS)

- **D1 — flight-sim gate (rescoped: thesis-gate ONLY, minimal).** Closed-loop POMDP
  trainability: PyFlyt or gym-pybullet-drones (Crazyflie model), teacher = stock PID,
  student = seqlgn cell (gated + clatch arms) on thermometer-encoded obs; **closed-loop**
  (student flies during training rollouts — DAgger-style at minimum), memory-required
  variant (IMU dropout/flicker, gust history, or T-maze cue). Multi-seed, DUST.
  Bar: memory cell ≳ teacher under occlusion where a feedforward student degrades.
  **Recorded (discharges research/20 F12):** distillation-from-PID supplies per-step
  action targets **≡ deep supervision — the training method P2 proved necessary**.
  **Pivot rule (pre-committed):** gate FAILS → thesis topic pivots to "verified logic
  flight controller" (feedforward + P3a verification + board); memory claims get scoped.
- **D2 — pitch package.** Narrative (updated per Decision 2): arrive with a **working,
  verified streaming stack** — P3a theorems + the T4 board demo (CAN-IDS if C0 has
  landed) — and pitch the drone as its **embodied next body**, not as promised future
  work. Three differentiators unchanged: (a) first real-drone weightless controller
  **with memory** (DWC is sim-only, feedforward); (b) belief-state in clocked FPGA
  registers; (c) machine-checked temporal safety of the deployed circuit. **New
  handshake:** Vargas's adversarial-robustness/security lineage (one-pixel) meets the
  intrusion-detection flagship directly — certified detector robustness is a natural
  joint topic. 2-page proposal + deck, rungs 0–4 ladder + division of labor from
  research/16 §C. Frame on **energy/area/determinism/verifiability — never latency**.
- **D3 — outreach.** Email Vargas ~Sep (after P1 submission, with T4 video + first gate
  results); loop in the co-driving PhD student.

## E. Leg A — tool paper (FOLLOWS, on trunk artifacts)

- **A1** breadth sweep: all 12 verified circuits through emit→synth→P&R (area/Fmax/energy
  × mechanism/width/regime) **+ the CAN circuit as the real-domain row** — likely the
  paper's strongest table line ("a CAN intrusion detector at X ns/frame, Y nJ/frame, with
  a machine-checked no-false-alarm envelope").
- **A2** the method-carrying ablation: register FSM vs **RDDLGN-style concat-recompute
  recurrence in RTL**, same task — latency/area/energy head-to-head (the research/10
  headline nobody can publish without reproducing P2).
- **A3** comparison table: FINN-GL (µs quantized LSTM), VOLLO (~2 µs audited), published
  feedforward-LGN ns — noting ETH's sub-20 ns is synthesis-report timing, ours on-board.
- **A4** OSS toolchain release (**first public LGN→RTL flow of any kind**) + the angle:
  **certified bitstreams** — checkpoint → bit-exact netlist → machine-checked theorems →
  measured hardware, one gated chain.

## F. Calendar

| when | what |
|---|---|
| Jul–Aug '26 | P1 workshop pick → submit ~Aug 29; P3a arXiv draft; **trunk T1–T4; C0.g CAN gate + D1 flight gate on DUST**; (opt.) FI-2010 micro-gate |
| Sep–Oct '26 | **D2 pitch + D3 outreach out** (with gate verdicts in hand); CAN circuit trained + verified → T4 headline demo; Leg A starts on trunk artifacts |
| Nov–Dec '26 | Leg A measurements + draft → **arXiv**; (stretch: FCCM '27 abstract ~early Jan) |
| ~late Mar '27 | **FPL '27 submission (planning target)** |
| Apr–Sep '27 | Kyushu in-country (rungs 2–4); CAN application paper drafted (security venue) |

## G. Risks / verify-items

- **C0.g fails** (stateless matches on timing attacks) → fallback chain §C2; the
  application flagship dies, P3b trunk/paper unaffected.
- **Free-input MC at 20–40 PIs** may not prove unbounded no-false-trip → bounded
  certificates + the dose-response training story remain the deliverable; pitch and paper
  wording must not promise unbounded theorems for the CAN case.
- **D1 gate failure** → pivot rule in D1 (pre-committed).
- **Deck-FPGA identity — RESOLVED 2026-07-12.** The "Artix-7 XC7A15T deck" in 2403.18703
  is the **paper authors' custom research prototype**, not a product; Bitcraze's stock
  Lighthouse deck is a **Lattice iCE40UP5K**, and **no commercial Artix-7 Crazyflie deck
  exists**. Consequence for the pitch: promise EITHER (a) the stock iCE40UP5K deck (5,280
  LUT4 — needs a real nextpnr-ice40 fit check; note the FSM+head is 2,963 LUTs on Artix,
  so an iCE40 fit is genuinely tight and MUST be checked before promising it), (b) a
  bench-top Arty A7-100T demo (safe; carries the video regardless), or (c) custom deck
  PCB design — **never an off-the-shelf Artix-7 deck.** Best pitch framing: bench-top Arty
  now, iCE40-deck fit as a co-scoped rung with the Kyushu student.
- **ETH race** → arXiv-first on P3a and the tool paper; their published pattern (GIC-DLC:
  "no access to an FPGA implementation") suggests no on-board capability — measured
  on-board is the leg they can't fake quickly.
- **Vivado grind** (install size, timing closure) → openXC7 open flow in parallel (§T2);
  neither blocks T1. **Solo January** → de-conflicted by the FPL-target policy.
- **T2 timing unknown — RESOLVED 2026-07-12 (openXC7 P&R).** The head's 0-CARRY4 popcount
  tree is NOT the critical path; the clock is limited by the FSM register recurrence
  (routing-dominated). Fmax 367.9 MHz (open-flow estimate). The remaining timing question
  is only sign-off-grade closure (Vivado / a pblock floorplan of the register file).

## H'. Execution status (workflow `p3b-kickoff-execute`, 2026-07-11/12)

A 13-agent recon→build→adversarial-verify workflow executed the in-lane parts of §H. Net:
- **T1 (lgn_top) — DONE, verified.** FSM+head composed (18,897 gates), ports **1,035 → 14**;
  RTL golden 8/8, 3 mutation controls (FSM/head observable FAIL, masked PASS), 4 yosys
  flows (flow A **2,963 LUT / 885 FDRE**, 0 CARRY4/BRAM/DSP), post-synth re-sim 8/8. Head
  costs ~2,062 LUT over the FSM-only 901. Artifacts: `netlist/synth/top.v`, `export_top.py`,
  `run_top.sh`, report.md §7.
- **C0.g (CAN) — DONE, verified incl. REAL DATA.** `seqlgn/can_data.py` + `test_can.py`
  (10 banks) + `run_queue_c0g.sh` (26-job DUST queue). **ROAD verified against the real
  557 MB archive (2026-07-12):** the loader already matched the real schema (injection_id
  0x-hex, `modified:true` masquerade flag, elapsed-sec intervals) — **no labeling/path fix
  was needed** (only an additive AppleDouble glob guard). Label sanity PASSED on all 9
  masquerade captures (attack frames 0 outside interval, ~4.2% inside; inj_id present
  outside as legit = frequency-preserving masquerade confirmed) + independently re-derived
  on a 4th family (reverse_light_off). Leakage-safe whole-capture holdout, input_dim=93.
  Real-data train.py smoke ran end-to-end (exit 0). The highest-risk silent-failure item is
  discharged. **Remaining = the gate itself (DUST GPU, hidden 1024–2048/20k iters).**
- **D1 (flight) — RAN on DUST 2026-07-13; gate FAILS but INCONCLUSIVE (mis-calibrated,
  do NOT fire the pivot).** `mlgn/flightgate/` closed-loop DAgger (gated/clatch/ff @ hidden
  432, matched 1,728 gates); `run_queue_d1.sh` 22 jobs; `gate_eval.py` table. Result: verdict
  FAIL (blackout return gated 20.8 / clatch 16.5 < ff 34.3), BUT **no arm actually flew —
  every student exits the envelope 100% in both conditions, returns 15–51 vs teacher 440, and
  the a′ oracle-on-masked-obs itself exits 97%.** So ff is only "least-catastrophic," and the
  memory-required task as configured is near-impossible for anything → the gate can't measure
  whether memory helps (this is flight's "CAN round-1"). Discretization gap ≈ 0 (models fine).
  **Fix before re-run:** soften the blackout/jitter so the a′ ceiling can hold the envelope
  (then students have separation room) and/or scale training (returns far below teacher =
  undertrained). Pivot rule (→ verified feedforward controller) stays UNTRIGGERED pending a
  calibrated re-run. Ran on base CUDA torch + conda-forge pybullet (off the 8 GB quota).
- **T2 open-flow P&R — DONE (headline above): 367.9 MHz / 43.5 ns-decision on xc7a35t.**
  `netlist/synth/run_pnr.sh` + `build_chipdb.sh`, report.md §8; openXC7 snaps in WSL.
- **extract-can — DONE, verified CONFIRMED.** `netlist/extract.py` `can`/`can-syn` RunSpec:
  clatch → MUX-register FSM, ff → combinational latch-free netlist, both bit-exact vs torch
  on fresh random batches; `test_can_extract.py`; existing suites unbroken. CAN checkpoints
  are now exportable/verifiable the day the gate finishes.
- **Facts — all 6 discharged**; folded into §B/§G above.
- **Cleanups (2026-07-12):** `.gitignore += **/data-can/`; stray `bash.exe.stackdump`
  removed; T1 verify-loop re-run (clears stale `top_mut_*.v`); flightgate cli/trainer
  cosmetics (rounds record, empty-slice warning).

**C0.g round 1 (2026-07-12, DUST) — flagship gate INCONCLUSIVE, fix in hand.** On the two
HARD masquerade attacks (max_speedometer, reverse_light_on) **every arm collapsed to the
always-normal predictor** (recall ≈ 0, the gate's "must beat the floor" precondition failed
for all) — classic class-imbalance collapse at ROAD's ~2–3% attack-frame rate under
unweighted CE. BUT on the one attack where anything trained (correlated_signal, easy),
recurrence won decisively: **gated F1 0.74 / clatch 0.67 vs ff 0.12** (n=1) — the
recurrence-earns-its-keep signal, where the task is learnable. Fix: **`--can-pos-weight`**
(train.py; `auto` = neg/pos on the train frame rate) — validated on the fixture, recall
0.00→0.31 / F1 0.00→0.47 at FPR 0.002 (a too-gentle weight of 6 still collapsed → use
`auto`). run_queue_c0g.sh re-tagged `c0g_*→c0gpw_*` (round 1 kept as the unweighted-collapse
baseline for the paper).

**C0.g round 2 (2026-07-13, pos-weighted, gated speedo now n=3) — GATE PASSES its core
question.** Recall (means, low FPR everywhere): correlated_signal gated/clatch **0.94** vs
ff 0.57; max_speedometer gated **0.49**(n=3, .16–.75)/clatch 0.51(n=3, .18–.75) vs ff 0.11,
rddlgn 0.49(n=1); reverse_light_on gated/clatch **0.23**(n=3) vs ff 0.04, rddlgn 0.03(n=1).
**Solid headline — recurrence earns its keep:** recurrent LGNs beat the gate-matched
windowed-feedforward on BOTH hard attacks (~4–6×); bearings fallback NOT triggered.
**Register-specific claim HOLDS (firm-up seeds, 2026-07-13, rddlgn now n=3):** the register
arms beat recompute-recurrence (rddlgn) on both hard attacks — max_speedometer gated 0.49/
clatch 0.51 > rddlgn **0.36** > ff 0.11; reverse_light_on gated 0.21/clatch 0.19 >> rddlgn
**0.03** ≈ ff 0.04. On reverse_light_on it's a CLEAN, non-overlapping separation (gated
0.12–0.32 vs controls ~0.02–0.05) = register-specific, not generic recurrence. On speedo the
ordering register>recompute>ff is real in the means but register [.16–.75] and rddlgn [.21–.49]
ranges OVERLAP (high variance) → suggestive there, clean on rlon. (NB: an earlier single-seed
rddlgn=0.49 on speedo made this look attack-dependent; n=3 corrected it.) **NOT paper-tight:**
register arms high-variance (speedo .16–.75); a 4-layer `ff_deep` reaches 0.70 on speedo only
(budget-mismatched); rlon absolute recall modest (~0.20). Round-1 (c0g_*, unweighted collapse)
kept as the baseline.

**All four board-independent tracks are complete + adversarially verified.** Remaining is
**USER actions only:** (1) launch the CAN C0.g DUST queue (the flagship gate verdict);
(2) launch the flight D1 DUST queue (thesis gate); (3) install Vivado 2026.1 for sign-off
Fmax + T3 power/energy; (4) order the Arty A7-100T (T4 board demo, weeks out). Optional
agent work: FI-2010 micro-gate; head reg-to-out timing constraint; A7-100T seed sweep.

## H. Immediate next actions

1. Order the Arty A7-35T (shipping lead time).
2. T1: extend `export_fsm.py` to emit `lgn_top` (FSM+head) + golden decode vectors →
   yosys → deployment-realistic counts.
3. **C0.g: CAN data loader + gate queue** — ROAD/HCRL ingest → ID+Δt tokenizer →
   windowed-feedforward vs recurrent arms, multi-seed → DUST.
4. D1: minimal harness skeleton (PyFlyt vs gym-pybullet-drones pick), PID teacher,
   thermometer front-end; queue first closed-loop runs on DUST.
5. Install Vivado; bring up nextpnr-xilinx on `fsm.v` in parallel (T2).
6. Verify the 2403.18703 platform + exact FCCM '27 dates when drafting D2/venue notes.
7. (Optional, when convenient) FI-2010 micro-gate (1–2 days, existing code).
