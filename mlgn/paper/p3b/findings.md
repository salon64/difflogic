# P3b — Findings & Resume Point

**Status: PARKED 2026-07-13** to focus on the P1 and P2 papers. This document is the
self-contained handoff: what P3b established, what the numbers are (with honest caveats),
and the exact steps to pick each thread back up. Written so future-me (or a PhD-start) can
resume cold. Running plan + decisions: `../../research/23_p3b_workmap.md`. All artifacts are
committed to `main` (github salon64/difflogic); every number below traces to a committed
result JSON / log.

## What P3b is (one paragraph)

P3b is the **deployment phase** of the sequential-LGN program: take P3a's verified
gate-level netlists to *measured* silicon, and stand up a real-domain **benchmark carrier**
plus the **Kyushu drone thesis** gate. Decisions (research/23): the **Kyushu pitch leads**,
the **tool paper** follows on a shared hardware trunk (arXiv-first; FPL'27 target / FCCM'27
stretch); the benchmark carrier is **CAN-bus intrusion detection** (masquerade attacks); the
flight sim is the **thesis-gate**, not a benchmark. The strategic frame: this is a head start
— a third paper if there's time between Dec'26 and the Japan thesis, or an easy publication
runway at the start of a PhD.

## TL;DR status

| component | status | headline |
|---|---|---|
| **T1** head-in-fabric top (`lgn_top`) | ✅ done, verified | FSM+argmax head, ports **1,035→14**, 2,963 LUT / 885 FDRE |
| **T2** open-flow place-and-route | ✅ done (estimate) | **367.9 MHz on xc7a35t → 2.72 ns/step, 43.5 ns/decision** |
| **T2/T3** Vivado sign-off + energy | ⏳ user action | needs Vivado 2026.1 (Win); gives sign-off Fmax + nJ/decision |
| **T4** board demo | ⏳ blocked | needs an Arty A7-100T (order; weeks out) |
| **C0.g** CAN gate (carrier) | ✅ PASSES core | recurrence > feedforward; register > recompute (clean on rlon) |
| **extract** CAN→netlist bridge | ✅ done, verified | `extract.py` can/can-syn, bit-exact |
| **D1** flight gate (thesis) | 🔁 inconclusive → recalibrated | round 1 undertrained (nothing flew); v2 queue ready |
| **Leg A** tool paper | ▫ not started | numbers exist; it's writing |
| **Leg B** Kyushu pitch | ▫ ~Sep–Oct'26 | stronger with CAN verdict + Fmax in hand |

## 1. Results

### 1.1 Hardware trunk (T1 → T2) — the measured-silicon story

- **`lgn_top`** (`mlgn/netlist/synth/`, report.md §7): the eval-time copy-FSM composed with the
  in-netlist GroupSum/argmax head into one deployable module — `x[8:0]` in, `class_out[2:0]`
  out. Ports drop **1,035 → 14**, killing the I/O-bound blocker. 18,897 gates; verified
  bit-exact vs torch, RTL golden 8/8, mutation-controlled, post-synth re-sim 8/8. yosys
  `synth_xilinx`: **2,963 LUT / 885 FDRE, 0 CARRY4/BRAM/DSP**.
- **First real Fmax** (open flow: openXC7 nextpnr-xilinx + prjxray, no Vivado, no board;
  report.md §8): `lgn_top` on **xc7a35tcsg324-1 = 367.9 MHz** best-of-4 (range 316–368,
  ~14% seed spread) → **2.72 ns/step, 43.5 ns/decision** (K=16 settle), 51.6 ns (K=19).
  2,651 LUT / 885 FDRE, **12.7% of the chip**. A7-100T: 326.7 MHz / 4.2%.
- **Load-bearing finding:** the reg-to-reg critical path is **inside the FSM register
  recurrence** (routing-dominated) — the head's 0-CARRY4 popcount tree is *off* the critical
  clock path (answers the T1 timing question). Raw `fsm.v` (1024-bit q) is I/O-unplaceable,
  machine-confirming the I/O-bound caveat.
- **Caveat:** open-flow timing is a delay-model *estimate*, not sign-off. Vivado gives the
  authoritative Fmax **and** the only path to energy (nJ/decision, T3).

### 1.2 CAN-bus IDS gate (C0.g) — the benchmark carrier, PASSES

The gate: does a **recurrent** logic detector beat a **gate-matched windowed-feedforward**
one on ROAD *masquerade* attacks (frequency-preserving — the ones a rate window can't catch)?
Arms: gated / clatch (register) vs ff (stateless) vs rddlgn (recompute-recurrence control).
Loader verified against the real 557 MB ROAD archive; splits leakage-safe; low FPR throughout.

**Round 1 (unweighted CE) — collapsed:** all arms → the always-normal predictor (recall 0) on
the hard attacks. Cause: ~2–3% attack-frame imbalance. **This is a committed baseline** (tags
`c0g_*`) and part of the paper's honest arc.

**Fix:** `--can-pos-weight auto` (= neg/pos) in the CE that drives the final + per-step
deep-sup loss. Validated: fixture recall 0.00→0.31.

**Round 2 (pos-weighted, tags `c0gpw_`) — recall means (low FPR):**

| attack | gated | clatch | rddlgn (recompute) | ff (stateless) |
|---|---|---|---|---|
| correlated_signal (easy) | 0.94 | 0.94 | — | 0.57 |
| max_speedometer (hard) | 0.49 [.16–.75] | 0.51 [.18–.75] | 0.36 [.21–.49] | 0.11 |
| reverse_light_on (hard) | 0.21 [.12–.32] | 0.19 [.06–.32] | 0.03 [.02–.04] | 0.04 |

Two claims:
1. **Recurrence earns its keep** — recurrent LGNs beat matched feedforward on both hard
   attacks (~4–6×). Solid. (The workmap's bearings fallback is NOT triggered.)
2. **The register specifically** beats recompute-recurrence — **clean/non-overlapping on
   reverse_light_on** (gated 0.12–0.32 vs rddlgn/ff ~0.02–0.05), directional on
   max_speedometer (means order 0.50 > 0.36 > 0.11 but register/recompute ranges overlap).
   This is the sharp claim that lands on the P2 register thesis.

**Honest caveats before it's paper-tight:** high register-arm variance (speedo .16–.75); a
4-layer `ff_deep` reaches 0.70 on speedo only (budget-mismatched); rlon absolute recall is
modest (~0.20). *(Note for the record: I mis-stated claim 2 as "attack-dependent" at rddlgn
n=1 — the extra seeds corrected it; rddlgn n=3 confirms register > recompute on both.)*

**CAN→verification bridge:** `extract.py` now rebuilds CAN checkpoints to bit-exact netlists
(clatch→FSM, ff→combinational), so a trained detector becomes a *verifiable* circuit — the
P3a property machinery (`distractor_decode` = no-false-trip; shadow-armed `protocol_decode` =
bounded detection latency) ports to the CAN input automaton. Not yet exercised on a CAN
checkpoint; that's the paper's "certified IDS" payoff.

### 1.3 Flight gate (D1) — DIAGNOSED: hover distillation doesn't converge to a flyable policy

The thesis gate: does a memory cell hold belief-state through sensor blackout where a
stateless net degrades? **Two rounds, same outcome:** every student arm (gated/clatch/ff)
exits the envelope **100% in both conditions** (returns 15–70 vs teacher 440). Round 1
(`d1_*`) looked like undertraining; **round 2 (`d1v2_*`, 4× the training + softer blackout)
came back identical → it's NOT compute, it's the distillation.** Diagnosed from the per-round
logs:

- **The training loss is unstable, not plateaued-low.** Starts at 2.20 (= ln 9, random over
  9 bins/motor), dips to a best of **1.2–1.5**, then **drifts back up to ~1.6 by the end** —
  non-monotone — and **4/18 runs hit NaN (all on the recurrent arms; zero on ff)**. So the
  optimizer partially learns, then degrades/diverges; the final policy is worse than its own
  mid-training best. lr 0.01 + BPTT-through-discrete-recurrence is the prime suspect.
- **The target is achievable and the discretization is fine:** teacher-through-9-bins hovers
  perfectly (return 464, 4% exit, quant-gate ratio 1.02). So 9 bins fly; the student just
  can't hit them (best action-match ~50%, insufficient for a knife-edge-unstable hover).
- **The method works on easy dynamics:** the mock 2D point-mass env trains fine (mock gate
  5/5, loss decreases, students control it). So the logic+DAgger stack is sound — **it's
  quadrotor hover specifically that the discrete logic policy can't track.**
- **Recurrence HURT here:** the recurrent arms take all the NaNs and fly *worse* than the
  stateless ff (control return ff 70 vs gated/clatch ~20) — the opposite of the memory
  hypothesis. But moot, since nothing flies.

**What this means:** the memory question is unanswerable (no arm flies the control), AND the
pre-committed pivot ("verified *feedforward* flight controller") **also fails** — ff doesn't
fly hover either. The drone-*hover* framing, as built, does not yield a flyable logic
controller. This is a real finding, not a tuning bug; scale and blackout-softening both ruled
out. **Implication for the Kyushu pitch:** don't promise "recurrent logic flight brain beats
feedforward under occlusion" on hover — lead the pitch with the CAN/verification results that
*work*, and treat the drone as a forgiving-task or exploratory rung.

**Resume options (ranked), for when flight is un-parked:**
1. **Move the memory demo to a task logic policies CAN fly** (most promising): the mock 2D
   point-mass already trains; a rate-stabilized hover, waypoint-following, or slower dynamics
   where the student reaches teacher-level — then add the occlusion to test memory. This puts
   the memory comparison on solvable ground.
2. **Optimization fixes** (cheap first try, likely insufficient alone): lr 0.01→0.001–0.003,
   tighter grad-clip / verify the NaN step-skip guard fires, shorter BPTT window on the
   recurrent arms. Might get below the 1.2 loss floor — but ff (no BPTT) plateaus too, so fit
   precision is also a wall.
3. **Reframe** the drone thesis away from unstable-hover control toward a task where logic
   policies achieve flyable fidelity, or toward verified-control-on-a-simpler-platform.

Round-1 `d1_*` and round-2 `d1v2_*` both committed as the record.

## 2. Where this becomes papers

- **Tool paper (Leg A) — the nearest, and it's writing not compute.** Checkpoint → bit-exact
  netlist → machine-checked theorems (P3a) → *measured* hardware (Fmax/area, and the CAN
  detector as the real-domain row). Angle: **certified bitstreams** — one gated chain from
  trained weights to silicon. Venue: FPL'27 (~Mar) / FCCM'27 (~Jan) stretch, arXiv-first. All
  numbers exist; needs the Vivado sign-off row + the head-included synthesis to be complete.
- **CAN application paper — the flagship, post-tooling.** "A recurrent logic-gate intrusion
  detector: exact-binary per-message state on FPGA with model-checked no-false-trip." Security
  venue. Needs: the register result firmed up (seeds) + the certified-IDS demo (run the P3a
  properties on a CAN checkpoint via the new extract branch).
- **P2 reinforcement:** the register-beats-recompute result (clean on rlon) is direct evidence
  for the deployable-register thesis — fold a sentence/figure into P2 if useful.
- **Kyushu thesis (Leg B):** gated on the flight gate landing (or a clean pivot). Pitch ~Sep–Oct.

## 3. Resume checklist (parked — pick up in priority order)

Each item is standalone; none blocks the P1/P2 writing.

1. **Flight D1 v2 re-run** (DUST, GPU, no new code). `git pull`, run `run_queue_d1.sh`, read
   `gate_eval --glob 'flightgate_hover_*_d1v2_*.json'`. **Watch first:** do the *control*
   students' `envelope_exit` drop well below 100% (they can hover)? If yes, the blackout
   comparison is finally meaningful. **If control still ~100% exit at this scale, the
   bottleneck is architectural** (action discretization / DAgger / head) — diagnose before
   adding seeds. Env: base CUDA torch + conda-forge pybullet (off the 8 GB quota; recipe in
   research/23 §H' and the dust memory).
2. **CAN paper-tightening** (DUST, cheap). More register seeds (variance is the weak point);
   the register-vs-recompute claim is already seed-backed (rddlgn n=3). Then run the P3a
   properties on a CAN checkpoint (the `extract.py` branch exists) for the certified-IDS row.
3. **Vivado 2026.1** (user, Windows). Sign-off Fmax + `report_power` → nJ/step, nJ/decision
   (T3). The open-flow 367.9 MHz is an estimate; this is the authoritative number and the only
   energy path. Also the head-included synthesis for a deployment-realistic LUT count.
4. **Arty A7-100T** (~$299) → the T4 board demo (bring-up on the copy register; headline =
   live CAN detection). The one genuinely hardware-gated item.
5. **Tool-paper draft** — writing; the numbers are in.
6. **Kyushu pitch** (~Sep–Oct) — with the CAN verdict + Fmax as collateral.

**Open infra note:** DUST `~/work` is an **8 GB** quota — too small for a CUDA torch (~6 GB)
+ ROAD (~3.5 GB) together. Reuse base's CUDA torch (it's in `/opt/conda`, off-quota); a quota
bump to ~30–50 GB was requested and would remove the friction. Details in the dust memory +
research/23 §G.

## 4. Provenance

- **Code/artifacts:** `mlgn/netlist/synth/` (T1/T2: `export_top.py`, `run_pnr.sh`,
  `build_chipdb.sh`, report.md §7–§8), `mlgn/seqlgn/` (CAN: `can_data.py`, `train.py`
  `--can-pos-weight`, `run_queue_c0g.sh`, `test_can.py`), `mlgn/netlist/extract.py` +
  `test_can_extract.py` (CAN→netlist), `mlgn/flightgate/` (D1 harness + `run_queue_d1.sh` +
  `gate_eval.py`).
- **Results:** `mlgn/seqlgn/results/can_*_c0g_*` (round 1) + `*_c0gpw_*` (round 2);
  `mlgn/flightgate/results/*_d1_*` (round 1) + `*_d1v2_*` (round 2, pending).
- **Real ROAD data** lives at repo-root `data-can/` (gitignored; re-download from zenodo
  10462796, MD5 cab184cfc2fe12c0834bc46188c0f330).
- **Commits:** the P3b arc is on `main` (branch `p3b-kickoff` merged); grep `git log --oneline
  | grep p3b`.
