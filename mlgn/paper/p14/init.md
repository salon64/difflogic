# P14 — Gate Counts Are Not Comparable: A Measurement Critique and Synthesis Audit of Trained Logic Networks

_Status: **pool — CONDITIONAL ~55% (demoted from GO by verifier); short paper / P3b by-product,
NOT a standalone bet.** Source scout:
[research/21 §5](../../research/21_landscape_weakness_scout.md) (2026-07-08, adversarially
verified). Cost: **zero GPU** — synthesis tooling only (P3b emitter feeds it directly). No
timeline — grab from [research/24](../../research/24_roadmap.md)._

## What it is

Two surviving slices after the verifier took the big-win hopes away:

1. **The measurement critique (unclaimed, zero-GPU):** cross-paper gate-count comparisons in
   the LGN literature are **provably inconsistent** — Conv-DLGN App A.3 reports post-synthesis
   counts after an *unreleased in-house 4.6× synthesis shrink*; Mind-the-Gap and LILogic report
   raw counts; CompactLogic reports after its own pruning. The field's headline efficiency
   numbers are not on a common scale. Document it, quantify it, propose the reporting standard.
2. **A lever-decomposed, tool-documented EDA audit** of Petersen-lineage DLGNs specifically:
   per-pass numbers for `rewrite` / `refactor` / `dc2` / `fraig` / `espresso` etc., so the
   community knows what post-training synthesis actually buys and where it comes from.

## Scope — claim discipline (verifier-mandated — this is why it was demoted)

- **ISTA** (arXiv:2507.02585) already does SAT-equivalence merging + trivial/greedy/similarity
  pruning on trained difflogic-family nets with before/after counts — and found vanilla-trained
  nets have only **~0.5% exactly-removable redundancy** (bad news for big-win narratives). Run
  the **subsumed-by-ISTA check** before writing a word.
- **Miyasaka & Mishchenko** (FCCM 2024) already ran the multi-tool post-training synthesis
  audit on LogicNets **with training-data don't-cares (20–75% area reduction)** — strictly
  subsuming the thermometer-SDC lever as an upper bound. The audit here must be scoped to the
  Petersen lineage + the per-lever decomposition they didn't publish.
- (Editorial, not from the verifier:) this is a **service/critique paper** — keep the tone
  constructive (reporting checklist + reference flow), or it reads as an attack note.

## Venue & tier (honest call)

- **IWLS** (Int'l Workshop on Logic & Synthesis) — the bullseye audience, workshop tier by
  design.
- **DATE short paper / TRETS note** — if the audit numbers are strong.
- Best realistic use: **a section/appendix of the P3b tool paper** ("our emitter reports raw,
  post-ABC, and post-mapping counts; here's why the literature's numbers aren't comparable") —
  the critique lands with the toolchain release and costs nothing extra.
- Not a standalone bet; only break it out if the P3b paper is over-full.

## The gate (hours)

**The subsumed-by-ISTA check:** re-read arXiv:2507.02585 against the two slices above. If
their v2 (or a follow-up) added a per-pass decomposition or a reporting critique, fold whatever
survives into P3b and close this dir.

## Read up on

Must-cites (verified in the scout):
- **ISTA interconnect** — arXiv:2507.02585 (SAT-pruning; 0.5% redundancy result; the subsumption threat).
- **Miyasaka & Mishchenko** — FCCM 2024 (multi-tool audit with training-data don't-cares, 20–75%).
- **Conv-DLGN** App A.3 (the unreleased 4.6× shrink exhibit), **Mind-the-Gap**, **LILogic** arXiv:2511.12340, **CompactLogic** arXiv:2602.05830 — the inconsistent-reporting exhibits.
- **ReducedLUT** — arXiv:2412.18579 (adjacent LUT-compression lineage).

Background (added here, not from the scout — sanity-check before citing):
- ABC pass documentation (rewrite/refactor/dc2/fraig semantics) + Brayton & Mishchenko 2010.
- Espresso two-level minimization (classic reference) for the espresso lever.
- MLPerf-style benchmarking-methodology papers as the reporting-standard template.

## Risks / kill conditions

- ISTA check fails → fold residue into P3b, close.
- The critique without the audit is an opinion piece; the audit without the critique is a lab
  note — only the pair carries even a short paper.
- Low ceiling by design: budget accordingly (this exists to be cheap, citable, and to make the
  P3b tool paper's reporting section authoritative).
