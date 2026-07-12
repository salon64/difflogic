# P11 — Training in the Deployment Basis: Standard-Cell-Aware Differentiable Logic

_Status: **pool — CONDITIONAL ~55–60%.** Source scout:
[research/21 §4](../../research/21_landscape_weakness_scout.md) (2026-07-08, adversarially
verified). Needs: yosys/OpenLane + liberty files (→ direct P3b synergy; gives
[P4](../p4/init.md) its missing ASIC flow). No timeline — grab from
[research/24](../../research/24_roadmap.md)._

## What it is

Differentiable gate selection over an actual **standard-cell vocabulary** — MAJ3, AOI21/22,
OAI21/22, 3–4-input NAND/NOR (**combinational only**; stateful primitives stay in P4) — with
**liberty-derived area/delay/power in the relaxation**, evaluated **through real synthesis**
against the control nobody has run: the same 16-gate net technology-mapped by ABC/Yosys onto
the full library. Claim: **"training in the deployment basis changes what function is learned
and beats post-hoc mapping on the accuracy-vs-PPA Pareto"** — NOT "we can represent MAJ"
(WARP-LUTs kills that framing).

## What it covers

- The extended relaxation (bigger softmax, 3–4-input cells, fan-in handling) with per-cell
  area/delay/power costs pulled from real liberty files in the loss.
- **The cross-basis experiment matrix:** {16-gate trained, cell-trained} × {1:1 emission,
  full ABC/Yosys tech-mapping} × {SkyWater 130, one more PDK} → accuracy-vs-PPA Pareto through
  actual synthesis reports, not gate counts.
- The "what function is learned" analysis: does basis-aware training find different circuits,
  or the same circuit expressed differently? (Function-level diff via the P3a equivalence
  tooling — free synergy.)
- Cost-in-loss lineage acknowledged: HGQ/HGQ-LUT, PolyLUT.

## Scope — claim discipline (verifier-mandated)

- **Full-library mapping of trained DLGNs has already been *run* publicly** — LILogic
  (arXiv:2511.12340 §4.3.2; SKY130/SG13G2/GF180 tapeouts) and Zioma's tt10-lgn-mnist Tiny
  Tapeout. **The contribution is the first *measured cross-basis comparison*, not the first
  mapping run.**
- **Silicon Aware NN** (arXiv:2604.19334, read in full by verifier): 16-gate 1:1 mapping,
  area-only loss, **no tech mapping** — the direct baseline to generalize.
- The basis-misalignment observation itself is 2019 crypto prior art (NIST LWC "Does gate count
  matter?") — cite, don't rediscover.
- Hold the **P4 boundary**: combinational vocabulary only here.

## Venue & tier (honest call)

- **EDA main track: DATE** (A/B, realistic) or **ICCAD / DAC** (A/A\*-EDA, stretch — needs the
  Pareto to be decisive and the PDK story clean).
- **MLSys** alternative framing (training-for-deployment-cost) if the ML-side result is the
  stronger half.
- Floor: IWLS (the logic-synthesis workshop — friendly expert audience) or an ML-for-systems
  workshop.

## The gate (run FIRST, before any training investment)

**The tech-mapping control experiment:** train a vanilla 16-gate net → synthesize (a) 1:1
emission vs (b) full Yosys/ABC mapping onto SkyWater. **If full mapping already collapses the
basis gap (post-hoc mapping recovers the PPA), drop the axis** — the paper's premise is that it
doesn't.

## Read up on

Must-cites (verified in the scout):
- **Silicon Aware NN** — arXiv:2604.19334 (the direct baseline: area-only loss, 1:1, no mapping).
- **LILogic** — arXiv:2511.12340 (full-library mapping runs + tapeouts; the "already run" cite).
- **WARP-LUTs** — arXiv:2510.15655 (kills representation-based framing).
- **HGQ / HGQ-LUT**, **PolyLUT** — cost-in-loss lineage.
- NIST LWC 2019 — "Does gate count matter?" (basis-misalignment prior art).
- Zioma tt10-lgn-mnist (Tiny Tapeout) — public mapping exhibit.
- **BitLogic v2** — arXiv:2602.07400 (the five-axis design-space study; program-wide must-cite).

Background (added here, not from the scout — sanity-check before citing):
- ABC documentation + Brayton & Mishchenko 2010 (ABC system paper); Yosys/OpenLane flow docs.
- Liberty format / static-timing basics (any synthesis textbook; De Micheli for depth).
- SkyWater SKY130 PDK docs (the open PDK this will actually run on).

## Risks / kill conditions

- Gate says post-hoc mapping closes the gap → drop (pre-committed).
- Softmax over a bigger, multi-fan-in vocabulary may train badly — budget a relaxation-
  engineering phase; BitLogic's fan-in findings say fan-in is the accuracy lever, which cuts
  both ways (opportunity + confound: separate basis effect from fan-in effect explicitly, or a
  reviewer will).
- Racing the hardware-culture labs on their home turf (ETH/ISTA industrialize these axes) —
  freshness-sweep before committing.
