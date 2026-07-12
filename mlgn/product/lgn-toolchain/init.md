# LGN → verified RTL toolchain ("certified bitstreams")

_Status: **strategic / OSS-first — not a revenue product today, the enabling asset for
everything else.** The release itself is already committed work: P3b §A4 ships the OSS flow
(**first public LGN→RTL flow of any kind**). This doc is about what sits *around* that release.
Sources: [research/23 §E](../../research/23_p3b_workmap.md),
[research/21](../../research/21_landscape_weakness_scout.md) (ETH/ISTA industrialization
context). Paper twin: the P3b tool paper itself (+ [paper/p14](../../paper/p14/init.md) as its
reporting/audit section)._

## What it is

The one gated chain nobody else has, packaged: **checkpoint → bit-exact netlist → machine-checked
temporal theorems → measured hardware.**

- Core (OSS, from P3b): exporter/emitter (`mlgn/netlist/`), bit-exactness gates, property
  harnesses, ABC recipes (`tempor→scorr→pdr`), synthesis/P&R scripts, board demo collateral.
- Around it (the "product" layer, if ever): certification evidence packs (per-domain property
  libraries — CAN legal-traffic automata, alarm/debounce specs), a hosted verify-and-report
  service, engineering support/consulting, training.

## The wedge (why anyone would pay)

- For safety-critical ML deployment (DO-178C/DO-254 avionics, IEC 61508 industrial, ISO 26262 /
  ISO 21434 automotive), the assurance gap for learned components is THE blocker. An LGN's
  deployed object *is* a circuit — the toolchain turns "trust my test set" into "check my
  theorem", which is language certification bodies already speak.
- Nobody can fake the chain quickly: ETH's published pattern suggests no on-board capability
  (GIC-DLC: "no access to an FPGA implementation"); ISTA verifies feedforward only. The
  sequential+deployed+verified chain is ours (P2+P3a+P3b).

## Honest market assessment (read this before dreaming)

- **The addressable market today is ≈ zero.** The LGN field is ~a dozen core papers; nobody
  deploys LGNs commercially yet. A toolchain for a substrate without industrial users is a
  bet on the substrate, not a business.
- Therefore: **OSS-first is the strategy, not a compromise.** Comparables: YosysHQ (OSS EDA +
  support contracts), hls4ml (community → CERN-adjacent adoption), cocotb (OSS → consulting).
  The flow being *the* public reference implementation is worth more to the PhD/consulting
  trajectory than a closed tool with zero users.
- Revenue, if any, arrives as: (a) consulting on the two application products
  ([can-ids](../can-ids/init.md), [bearing-monitor](../bearing-monitor/init.md)), (b) funded
  projects that need the chain (certification research calls), (c) much later, licensing
  property packs / hosted verification if LGN deployment materializes.

## Potential users → potential customers (in adoption order)

1. **Researchers** (ETH/ISTA/UT-Austin lineages, weightless-NN community) — free tier;
   adoption = citations = the moat.
2. **Edge-FPGA ML teams** (hls4ml-adjacent physics/aerospace/industrial groups) — first
   external users; they surface the packaging bugs.
3. **Safety-critical integrators** (avionics/automotive/industrial suppliers with FPGA teams) —
   the first plausible payers, via evidence packs + support.
4. **EDA vendors** — not customers; the acqui-hire/collab endgame if the substrate takes off.

## What it needs to work — technical

1. P3b A1–A4 complete (breadth sweep, RTL ablation, comparison table, release) — already the
   committed plan; this doc adds only packaging intent.
2. Release hygiene that makes it *usable*, not just public: docs + a 15-minute quickstart
   (checkpoint → theorem → bitstream on an Arty), pinned deps (yosys/ABC/nextpnr versions),
   CI on the golden circuits, a worked CAN example once C0 lands.
3. License choice **decided before the release** (permissive Apache-2 maximizes adoption;
   keep trained circuits + domain property packs out of the repo — that's the IP line, see
   can-ids).
4. A second engine (rIC3/Pono/nuXmv) behind the ABC recipe — P3a §R already wants this;
   for users it's the "not an ABC quirk" guarantee.
5. One external user completing the quickstart without help = the real release gate.

## What it needs beyond the tech

- A name, a repo home (separate from the research fork), a paper DOI to cite (the P3b tool
  paper), and a demo video (P3b T4 already plans it).
- Maintenance budget honesty: OSS with zero maintenance is negative marketing; scope the
  public surface to what one person can keep green.
- Community seeding: IWLS/FPL/FCCM hallway track, the weightless-NN Slack/mailing lists,
  a HN/r/FPGA post timed with the arXiv drop.

## Competition & prior art

- No public LGN→RTL flow exists (verified, doc 10/23) — difflogic repos stop at simulation;
  ETH publishes synthesis-report numbers without a flow.
- Adjacent flows: hls4ml (NN→HLS), FINN (BNN→FPGA dataflow), LogicNets' NN→LUT flow (public,
  Xilinx) — none for the difflogic gate substrate + none with verification in the loop.
- ISTA's verifier (2505.19932) is the verification-side neighbor: feedforward, SAT-based,
  no RTL/hardware leg. Complementary; consider interop rather than competition.

## Signals to go / kill

- **GO (for the paid layer):** an external team asks for support/evidence-pack work
  unprompted; or a certification-research call funds the chain.
- **KILL (for the paid layer, never for the OSS):** two years post-release with no external
  users → keep it as research infrastructure, stop polishing.

## First three concrete steps (when A4 approaches)

1. Pick license + split the repo boundary (flow public / circuits+packs private).
2. Write the quickstart against a clean machine; time it.
3. Line up the release train: arXiv (tool paper) + repo + video + one seeded example, same week.
