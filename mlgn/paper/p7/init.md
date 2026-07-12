# P7 — A Recurrent Logic-Gate Intrusion Detector: Exact-Binary Per-Message State on FPGA with Model-Checked Safety

_Status: **pool — GO ~72% (the application flagship), post-P3b.** Source scouts:
[research/22 §3](../../research/22_applications_scout.md) (2026-07-08, verified) +
[research/23 §C0](../../research/23_p3b_workmap.md) (2026-07-11: the CAN **carrier work** —
gate C0.g, trained+verified circuit, board demo — was pulled INTO P3b; **this paper is the
post-P3b security-venue write-up** that consumes those artifacts). Depends on: P3b C0 complete
(gate passed, circuit verified, T2/T3 numbers, T4 demo). No timeline — grab from
[research/24](../../research/24_roadmap.md) when P3b lands._

## What it is

The first learned-logic-circuit CAN-bus intrusion detector (verified vacant: incumbents are all
quantized/BNN — BIDS <0.17 ms, SecCAN, FPGA zero-day IDS arXiv:2401.10724; "can-logic" is
hand-written temporal logic, not learned). Application + system + measurement paper: a recurrent
clatch-LGN holding CAN message-ID timing/sequence state in exact-binary registers, deployed on
an ECU-adjacent FPGA at ns/frame and nJ/frame, **with machine-checked temporal safety
properties** — the certified-IDS story that ISO 21434 / ISO 26262 compliance actually pays for.

## What it covers

- **The recurrence-earns-its-keep result** (P3b gate C0.g): recurrent clatch/gated vs a windowed
  feedforward LGN *given the same input window*, on timing/sequence attacks specifically
  (masquerade/suspension — NOT flooding; any window solves flooding). Multi-seed. This ablation
  IS the paper's legitimacy; without it the domain collapses to occupied feedforward territory.
- **The verification wedge** (P3a machinery ports ~1:1): legal-traffic input automaton plays the
  distractor automaton's role; **no-false-trip** = a decode-type invariant over legal traffic;
  **bounded detection latency** = shadow-armed protocol_decode (bad = not-alarmed within K).
  Dose-response reframed: *train with injected attacks → machine-checked no-false-alarm
  envelope*.
- **Measured hardware:** ns/frame + nJ/frame from P3b T2/T3 (open-flow Fmax already 367.9 MHz on
  xc7a35t; Vivado sign-off pending), the $300 Arty board demo (replayed CAN trace, line-rate
  flagging), vs the BIDS/SecCAN/2401.10724 quantized-NN frontier.
- **Datasets:** ROAD (primary — realistic masquerade/targeted-ID; loader already built and
  verified against the real archive), Car-Hacking/HCRL, SynCAN (suppress/masquerade).
  Leakage-safe whole-capture holdout, input_dim=93 (built in P3b).

## Scope — claim discipline

- **Do not promise unbounded theorems.** At 20–40 PIs, free-input proofs may land as **bounded
  certificates — an acceptable deliverable** (P3b work map pre-commits this wording).
- The P3b **tool paper** carries the flow/toolchain claims; this paper carries the security
  story. Don't double-claim the emitter.
- Detection scope = what the datasets support (injection/masquerade/suspension on classic CAN);
  CAN-FD / automotive-Ethernet is future-work, say so.
- False-positive framing matters more than detection rate in this domain: a false trip on a
  safety bus is itself a hazard — lead with the no-false-alarm envelope.

## Venue & tier (honest call)

- Scouts say security/systems venue. Honest ladder:
  - **Realistic main targets: RAID / ACSAC / ESORICS** (solid security venues; systems+measurement
    papers with real hardware do well) or **VehicleSec** (community bullseye; a USENIX symposium
    co-located with USENIX Security since '25 — smaller but exactly the right reviewers).
  - **Stretch: USENIX Security / CCS (A\*)** — plausibly reachable *only* with something beyond
    public-dataset replay (e.g. live-bus/vehicle validation or a strong adversarial-evasion
    analysis). Budget honestly before aiming here.
  - Embedded/systems alternative: an FPL/FCCM app-track version exists, but that lane is what
    the P3b tool paper occupies — keep this one security-native.

## The gate

**C0.g — already queued inside P3b** (26-job DUST queue built and verified 2026-07-12; awaiting
launch). This paper only exists if C0.g passes. If stateless matches on timing attacks →
fallback chain is P3b §C2 (bearings becomes carrier candidate) and this dir goes dormant.

## Read up on

Must-cites (verified in the scouts):
- **BIDS** (<0.17 ms BNN IDS), **SecCAN**, **FPGA zero-day IDS** arXiv:2401.10724 — the quantized-NN incumbent frontier.
- **ROAD** dataset (realistic attacks incl. frequency-preserving masquerade), **Car-Hacking/HCRL**, **SynCAN** — data + their papers' eval conventions.
- "can-logic" (hand-written temporal-logic CAN monitoring) — the non-learned verified incumbent.
- **Brain-on-Switch** (NSDI'24) + FENIX/Quark — stateful-beats-stateless at line rate, non-logic, unverified (motivation lineage from the NID vignette).
- **LogicNets** (91.3% / 10.5 ns UNSW-NB15) — feedforward logic-substrate credibility anchor.
- P2/P3a/P3b self-cites: the cell, the certificate machinery, the toolchain.

Background (added here, not from the scouts — sanity-check before citing):
- Checkoway et al., USENIX Sec 2011 + Miller & Valasek 2015 (Jeep) — the canonical CAN-attack motivation.
- A recent CAN-IDS survey (several 2022–2025 exist) for the related-work skeleton.
- ISO/SAE 21434 and UNECE R155 primers — the compliance wedge needs precise wording.

## Risks / kill conditions

- **C0.g fails** → dormant (pre-committed, P3b §G).
- Bounded-certificate wording slips into "proved for all traffic" → reviewer kill; keep the
  envelope language from P3a.
- Public-dataset-only evaluation caps the venue at the realistic tier — decide early whether
  live-bus validation is worth chasing (it is what unlocks the A\* stretch).
