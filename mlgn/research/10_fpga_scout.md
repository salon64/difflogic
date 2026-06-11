# Scout #10 — FPGA realization of LGNs: who has ACTUALLY built it? (P3 viability)

_Adversarial scout run 2026-06-11. Question from Malcolm: "papers leave FPGA to future
work — has anyone done it? Can we pick it up as (a) a product, (b) another A* paper?"_

**TL;DR: Feedforward-LGN-on-FPGA is now COMMODITY (5 independent implementations,
2 in the last 7 months). Sequential/stateful-LGN-on-FPGA is UNTOUCHED — and it is
method-gated by P2's latch primitives, which is exactly our moat. GO (~80%) as the
P2-coupled story; NO-GO as a standalone "we ported it to FPGA" paper. Product avenue:
real, validated by Myrtle VOLLO's µs-scale finance records, but it's a company-build,
not a side-sale.**

---

## Q1 — Toolchain: trained LGN → RTL/bitstream, does an open flow exist?

| Flow | Status | Detail |
|---|---|---|
| `Felix-Petersen/difflogic` | **no RTL** | exports C + CUDA only (README verified 2026-06-11) |
| Conv-DLGN NeurIPS'24 FPGA (9–24 ns) | **closed one-off** | numbers published, flow never released; nobody cites a released flow |
| **torchlogix** (`ligerlac/torchlogix`, Princeton/CICADA) | **open, partial** | Python → C → **Vitis HLS** → HDL; only the *simple feedforward* LGN synthesized; full CLGN synthesis "ongoing — quite complicated" (arXiv:2511.01908) |
| UF characterization (arXiv:2605.04109, May'26) | study, no tool | Vitis HLS flow, Alveo U200, depth/width/power tradeoffs, feedforward-only ("combinatorial logic"), no code found |
| **BitLogic** (arXiv:2602.07400, **ETH Wattenhofer**, Feb'26) | **automated RTL pipeline, NOT released** | PyTorch → synthesizable HDL, LUT-node nets, CIFAR-10 72.3% @ <0.3M gates, **sub-20 ns** (likely post-synthesis timing, not on-board — sister paper GIC-DLC admits "no access to an FPGA implementation" Jan'26); code "public once **licensing** has been resolved" (full text, 2026-06-11) |

**Verdict Q1: feedforward toolchain = OCCUPIED/PARTIAL.** BitLogic claims the automated
pipeline; CICADA has an open partial one; HLS-from-C-export is the de-facto workaround.
A "we release a feedforward LGN→Verilog tool" paper is **no longer a contribution**.
A **sequential** toolchain (emitting clocked `always @(posedge)` netlists with learned
FFs) exists nowhere.

## Q2 — Sequential/recurrent LGN on FPGA: has ANYONE done it?

**No. Zero hits across all framings searched** (recurrent/stateful/streaming LGN FPGA;
learned flip-flop/latch primitives; sequential circuit learning + synthesis; recurrent
LUT-NN/weightless variants). Every implementation found in Q1 is combinational.

Verified from RDDLGN full text (arXiv:2508.06097v1, also publ. MobiCom'25 EdgeFM wksp):
recurrence = `k_t = [h_t; k_{t-1}]` concat-recompute, **no state primitive**; flip-flops/
latches appear ONLY as a motivational hardware analogy; FPGA synthesis explicitly named
future work. DiffLogic CA likewise (per scout 2026-06-04).

**Must-cite adjacent (the "well actually" reviewers will raise):**
- **FINN-L** (arXiv:1807.04093, FPL'18) — first binarized LSTM/BiLSTM on FPGA; and
  **FINN-GL** (arXiv:2506.20810, Jun'25) — generalized mixed-precision LSTM/ConvLSTM
  FPGA flow. These are **quantized-arithmetic RNNs** (MACs + weight memories); registers
  hold state as an *implementation artifact*. NOT learned gate-level circuits, NOT
  latches in a learnable gate vocabulary. The hardware-cost gap vs LGN is the story.
- Unclocked recurrent Boolean circuits in FPGAs via **evolutionary** search
  (arXiv:2403.13105) — not gradient-trained, unclocked, no learned latch.
- Recurrent CircuitSAT sampling (arXiv:2502.21226) — FFs as hidden state for SAT on a
  *fixed* netlist, nothing learned.
- DWC (arXiv:2512.01467) — combinational policy in a closed control loop (environment
  provides the recurrence, circuit is stateless).

**⚠ The adversarial nuance — the weak headline is scoopable by ENGINEERING:** anyone
who implements RDDLGN's concat-recurrence in hardware gets "first recurrent LGN on
FPGA" (state lands in BRAM/FFs as an implementation choice). And it is worse than "one
lab has both halves": **RDDLGN and BitLogic share the same first author — Simon Bührer
(ETH/Wattenhofer)** (RDDLGN = Bührer et al. 2025 per GIC-DLC's reference list), and
BitLogic's future work **declares the merge in writing**: "sequence modeling will likely
require stronger support for recurrent and encoder–decoder style components" + improve
"handling of sequential logic (e.g., explicit pipelining, **stateful modules**, and
streaming interfaces)". His obvious next paper is recurrent-LGN-on-FPGA. **Window
revised: 3–12 months** (was 6–18).
**The defensible headline is method-gated:** "learned latch primitives map 1:1 to
hardware FFs; measured ns-latency/energy vs recompute-recurrence (RDDLGN-style) and vs
quantized-LSTM flows (FINN-GL)" — nobody can publish that without first reproducing P2.

## Q3 — Commercial signals

- **Myrtle.ai VOLLO** — THE incumbent product: STAC-ML audited FPGA inference for
  capital markets, **LSTM model zoo**, Intel Agilex 7; 5.1 µs LSTM (2023) → **~2 µs
  99th-pct record (Apr 2026)**. The audited commercial frontier for *recurrent* models
  is microseconds — a sequential LGN at tens-of-ns is **~100× below it**. That is the
  quantified product wedge.
- AMD Alveo FinTech cards ship with FINN; Algo-Logic, Exegy (Nexus, Q4'25) sell
  FPGA trading infra; NVIDIA pushes single-digit-µs GPU inference for markets.
- **InftyLabs Research** = Petersen's outfit (affiliation on Conv-DLGN with J. Welzel)
  — founders clearly see commercial value; no public product yet.
- Nobody sells LGN-based inference today.

## Angle status

| Sub-angle | Status | Evidence |
|---|---|---|
| Feedforward LGN on FPGA (numbers) | **OCCUPIED** | Conv-DLGN 9–24 ns; CICADA 18.75 ns/3 cyc, 0 DSP; UF study; BitLogic <20 ns |
| Feedforward LGN→RTL toolchain | **OCCUPIED/PARTIAL** | BitLogic pipeline claim; torchlogix open partial |
| FPGA resource/power characterization (FF LGNs) | **OCCUPIED** | arXiv:2605.04109 |
| "First recurrent LGN on FPGA" (weak headline) | **OPEN but race-prone** | scoopable by RDDLGN+BitLogic engineering; no method moat |
| **Learned sequential circuit (latch primitives) + hardware realization + ns/energy vs recompute & vs FINN-GL** | **OPEN, method-gated by P2** | zero prior art; deferred by RDDLGN & DiffLogic CA (verified) |
| Sequential LGN→clocked-RTL toolchain artifact | **OPEN** | nothing emits learned FFs anywhere |
| LGN inference as a product | OPEN (no LGN vendor) | incumbents: VOLLO ~2 µs (recurrent), FINN ecosystem |

## Verdict

**P3 standalone ("port LGNs to FPGA"): NO-GO** — that train left (5 implementations).
**P3 as the P2-coupled story: GO (~80%)** — "Sequential Logic Gate Networks: learning
clocked circuits with latch primitives, realized on FPGA at X ns vs µs-scale quantized
RNN flows." Contribution type: method (P2) + hardware realization & measurement (P3) +
tool artifact (sequential RTL emitter). The FPGA section upgrades P2 from "nice idea"
to the killer table: **LGN-sequential ~tens-of-ns vs FINN-GL / VOLLO ~2–5 µs ⇒ ~100×.**

- **Venue logic:** ML core (NeurIPS/ICLR) — P2 method is the claim, FPGA = wow-demo
  (Conv-DLGN's oral proves the formula). Hardware (FCCM/FPL/DAC/DATE) — the toolchain +
  measurements stand alone IF ML reviewers reject. Two shots on goal from one artifact.
- **Plan impact:** P3 stops being "deferred indefinitely" → becomes **P2's demo
  section (minimal: D-FF variant on a small board) + a full P3 tool/eval paper for
  FCCM/FPL'27**. Urgency ↑ because of ETH's BitLogic.
- **Resource fit:** executable solo. Cheap board first (Artix-7/DE10-class — UF used a
  DE10-Lite); direct RTL emission is *easier* for LGNs than HLS (each gate = one
  `assign`, each latch = one FF; no HLS scheduling). Alveo-class only for headline
  throughput. No cluster needed for synthesis; P2 training remains the compute item.
- **Risks:** (a) ETH wires RDDLGN→BitLogic first → lose weak headline, keep method
  story (mitigate: speed + framing); (b) P2 latch-training stability (unchanged);
  (c) ML reviewers discount HW numbers → keep method primary; (d) HW reviewers demand
  FINN-GL/VOLLO comparison → that IS our table, embrace it.

**GATE CLOSED (2026-06-11, full-text reads):**
- **BitLogic:** future work EXPLICITLY names recurrent components for sequence modeling
  AND "sequential logic / stateful modules / streaming interfaces" in HDL export — the
  race is **declared**, not inferred. BUT: their framing is export *plumbing*
  (pipelining registers, stateful modules as implementation), **NOT learned latch
  primitives in the gate vocabulary** — P2's method moat holds. No RNN/LSTM/flip-flop/
  latch mentions in main text. Code **unreleased** ("once licensing has been resolved" —
  ETH commercialization signal #2 after InftyLabs) → **a released open-source
  sequential-RTL emitter would be the FIRST public LGN→RTL toolchain of any kind**
  (torchlogix is partial/HLS). Toolchain-artifact credit is back on the table for us.
- **GIC-DLC:** purely combinational, two-layer LUT nets, **estimation-only** ("we do not
  have access to an FPGA implementation") — no threat; also reveals the BitLogic/GIC-DLC
  group may not have on-board measurement capability yet (their ns numbers are likely
  synthesis-report timing).
- **Net effect:** verdict unchanged (GO ~80% P2-coupled), urgency ↑ (window 3–12 mo),
  and the minimal D-FF-on-board demo inside P2 gains value: it redefines "first" as
  first *learned* sequential circuit (method) before Bührer ships first *implemented*
  recurrent LGN (system). Standing tripwire: Bührer / Wattenhofer-lab arXiv feed.

## Product avenue (Malcolm's question 2a) — honest read

Validated demand, real wedge, wrong weight class for a side-sale: VOLLO's STAC records
prove buyers pay for every µs; an LGN product would undercut by ~100× on latency, but
incumbents own distribution (STAC audits, exchange colo, model-zoo UX), and difflogic's
accuracy ceiling caps which models port. Realistic sequence: **paper + open tool + ns
demo first** (it doubles as the product demo), then license/consult/join/raise from a
position of being the only person who has built it. Do not start with sales calls.

## New papers to fold into 02_papers.md / 01_landscape.md
- **BitLogic** arXiv:2602.07400 (ETH, Feb'26) — gradient LUT-nets + automated RTL export, sub-20 ns CIFAR-10. [toolchain competitor, feedforward]
- **GIC-DLC** arXiv:2601.14130 (ETH, Jan'26) — differentiable logic circuits for grayscale image compression. [application, combinational]
- **UF FPGA characterization** arXiv:2605.04109 (Wormald/Kravatsky/Woodard/Forte, May'26) — feedforward LGN FPGA tradeoffs, Alveo U200 / DE10-Lite. [measurement]
- **CICADA LGN** arXiv:2511.01908 (Princeton: Gerlach/Kauffman/Våge/Ojalvo, Nov'25) — LGN for CMS L1 trigger, Virtex-7, 3 cyc/18.75 ns, 0 DSP; `ligerlac/torchlogix`. [deployment + partial open flow]
- **FINN-GL** arXiv:2506.20810 (Jun'25) — generalized quantized LSTM/ConvLSTM on FPGA. [must-cite baseline for P3]
- Evolutionary unclocked recurrent Boolean circuits arXiv:2403.13105. [cite-differentiate]
- Myrtle **VOLLO**: STAC-ML audited, 5.1 µs LSTM'23 → ~2 µs Apr'26, Agilex 7. [commercial anchor for P3 motivation]
