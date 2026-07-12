# Certified CAN-bus intrusion-detection core

_Status: **candidate — the strongest commercial wedge in the portfolio.** Gated on P3b's C0.g
(recurrence-earns-its-keep, DUST queue built, awaiting launch). Sources:
[research/22 §3](../../research/22_applications_scout.md) (GO ~72%, verified),
[research/23 §C0](../../research/23_p3b_workmap.md) ("automotive = the commercial extension").
Paper twin: [paper/p7](../../paper/p7/init.md)._

## What it is

An **IP core + compliance-artifact package** for in-vehicle network intrusion detection:

- A trained recurrent LGN that holds CAN message-ID timing/sequence state in exact-binary
  registers, clocked once per frame — deployable as a soft core on an ECU-adjacent small FPGA,
  or as a synthesizable RTL netlist for integration into a gateway ECU / zonal controller.
- **The differentiator ships with it:** machine-checked certificates — a *no-false-trip
  invariant over a legal-traffic automaton* and a *bounded-detection-latency* property — as
  audit-ready artifacts (bounded certificates; never promise unbounded theorems).
- ns/frame, µW-class marginal power, ~thousands of LUTs (P3b: 2,963 LUT + 885 FF class
  circuits, 367.9 MHz open-flow on xc7a35t).

## The wedge (why anyone pays)

- **Regulation created the budget line:** UNECE R155 makes a cybersecurity management system
  mandatory for new vehicle type approval; **ISO/SAE 21434** is how OEMs/Tier-1s demonstrate
  it. A detector whose safety properties are *model-checked on the deployed netlist* is a
  compliance artifact no DL-based IDS can produce.
- **False positives are the incumbent pain:** a false trip on a safety bus is itself a hazard
  (ISO 26262 lens). The no-false-alarm envelope is the sales pitch, not the detection rate.
- Learned (adapts to per-platform traffic) where today's shipped IDS rules are hand-written;
  verified where learned competitors are black boxes. The moat is the combination
  {recurrent + exact registers + certificates} — verified vacant (no learned-logic-circuit
  CAN IDS exists; incumbents are quantized/BNN: BIDS, SecCAN, arXiv:2401.10724).

## Potential customers

- **Tier-1 suppliers** shipping gateways/zonal ECUs: Bosch/ETAS (ESCRYPT lineage), Continental
  (whose Elektrobit subsidiary houses PlaxidityX, formerly Argus), Aptiv, ZF, Vector
  Informatik (CAN tooling monopoly — also a channel).
- **Automotive cybersecurity vendors** needing a hardware-tier detector: PlaxidityX (formerly
  Argus), Upstream (cloud — complementary), C2A Security.
- **Silicon vendors** with CAN controller IP: NXP, Renesas, Infineon (an IDS block next to the
  CAN controller is a natural SKU extension).
- **Certification/testing bodies** (TÜV, exida) — not buyers, but the channel that makes the
  certificate artifact legible; early engagement is marketing.
- Realistic first customer: **none of the above directly** — a pilot via an academic-industrial
  project (see below) that a Tier-1 co-signs.

## What it needs to work — technical

1. **C0.g passes** (recurrence beats windowed-feedforward on timing attacks) — the existence
   gate for the whole product; already queued in P3b.
2. Cross-dataset generalization: ROAD → HCRL/SynCAN → an *unseen vehicle platform*; public
   datasets ≠ product data — per-platform retraining story needed (this is where
   [paper/p6](../../paper/p6/init.md) netlist-plasticity connects: **LUT-mask-sized OTA update
   deltas** would be a genuine product feature).
3. **Live-bus validation**, not replay: a bench CAN setup (two nodes + injector, ~$100s), then
   a real vehicle harness. Fleet-scale false-positive-rate estimation.
4. **CAN-FD support** (classic CAN is legacy; new platforms are CAN-FD/automotive-Ethernet
   mixes) — tokenizer + timing model rework.
5. Certificate scope engineering: the legal-traffic automaton must be derived from a real DBC /
   platform spec, not a synthetic protocol — bounded certificates with stated envelopes.
6. Integration form factors: Vivado-signed soft core, ASIC-synthesizable RTL (P11's
   standard-cell flow helps), MISRA-friendly integration shim, documentation.

## What it needs beyond the tech

- **A partner with vehicles and data.** Solo is not viable past the demo stage. Routes:
  automotive-security research projects (EU Horizon, Vinnova FFI in Sweden), a Tier-1 research
  group, or the Kyushu security connection (Vargas's lineage meets IDS — doc 23 D2 names this
  handshake).
- **Process credibility:** ISO 21434 expects a development process (ASPICE-flavored), not just
  an artifact. Realistic path = license the IP to someone who owns the process, don't become an
  automotive supplier.
- **IP strategy:** the toolchain is heading OSS (P3b A4); the *trained+certified per-platform
  circuit* and the property/automaton packs are the licensable assets. Decide before the OSS
  release what stays proprietary.
- Funding shapes that fit: Vinnova automotive calls, EU Horizon cybersecurity topics,
  industrial-PhD constructions.

## Competition & prior art

- Shipped today: rule-based IDS inside gateways (Tier-1 in-house), software IDS suites
  (PlaxidityX née Argus, ESCRYPT), cloud fleet analytics (Upstream). Academic hardware tier: BIDS (<0.17 ms,
  BNN), SecCAN, FPGA zero-day IDS (arXiv:2401.10724) — none learned-logic, none verified.
- Hand-written temporal-logic monitors ("can-logic") = the verified-but-not-learned incumbent;
  we are the learned-AND-verified quadrant.

## Signals to go / kill

- **GO:** C0.g passes cleanly (multi-seed); p7 paper accepted; one industrial conversation
  turns into data access.
- **KILL:** C0.g fails (stateless matches) → the whole quadrant collapses to occupied
  feedforward territory; or CAN-FD rework proves the state advantage doesn't transfer.

## First three concrete steps (when activated)

1. Launch C0.g (already the P3b next-action) and hold until verdict.
2. Bench CAN-FD literature + dataset scan (what ROAD-equivalent exists for FD?).
3. One-page teaser for Vinnova/Horizon partner search, written from p7's results section.
