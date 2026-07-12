# P13 — Why Random Wiring Works: Embedding Theorems for Fixed Fan-In Logic Gate Networks

_Status: **pool — CONDITIONAL ~55%, expressivity slice ONLY.** Source scout:
[research/21 §7](../../research/21_landscape_weakness_scout.md) (2026-07-08, adversarially
verified; earlier same-day memory said ~70% — corrected to ~55%). Cost: **~zero GPU** —
theorem-led, toy-scale sanity panels only. No timeline — grab from [research/24](../../research/24_roadmap.md)._

## What it is

A theory paper answering "**why (and when) does random fixed fan-in-2 wiring work?**" — the
question every LGN paper hand-waves. Two candidate results:

1. **An embedding theorem:** a randomly wired LGN of width w / depth d contains any s-gate
   circuit with high probability at overhead X — proof route via matching/percolation
   arguments, with pass-through gates (the identity/buffer entries of the 16-gate vocabulary)
   acting as **free routing**.
2. **Lower bounds:** Ω(n log n)-type input-coverage bounds — how wide must a random-wiring net
   be before every input bit even *reaches* the output cone — explaining the empirical
   wide-and-shallow preference and the open flag in IWP App B.5.

## What it covers

- The theorem(s) + proof; explicit constants/overheads, not just asymptotics, where possible
  (the field is small enough that concrete numbers get cited).
- Empirical sanity panels (cheap, toy-scale): predicted vs observed *representable*-width
  thresholds on toy targets — any training is only a search proxy, no trainability claim —
  theory papers in this space land better with one honest plot.
- Positioning: this explains *representability* under random wiring; it deliberately does NOT
  touch trainability (see scope).

## Scope — claim discipline (verifier-mandated)

- **The gradient/trainability half is TAKEN:** Kim (arXiv:2605.08657) proves exact
  gradient-cancellation at uniform init + O(0.72^L) depth attenuation, with a working fix.
  **Expressivity only.** Any drift into "and therefore it trains" walks into his theorem.
- **Functional Percolation** (arXiv:2512.09317) owns the framing ring (percolation language, no
  theorems, no training) — must position against it explicitly.
- **WiSARD exact-VC** (Neural Computation 2019) — the adjacent substrate's capacity result;
  cite to show the lane's conventions.
- Random-graph embedding /-routing machinery is classical — the novelty is the LGN
  instantiation (fan-in-2, 16-gate vocabulary with pass-throughs, layered DAG), not the tools.

## Venue & tier (honest call)

- Honest read: **expressivity-only theory is a hard A\* main-track sell** unless the theorem is
  surprisingly strong (tight threshold, matching upper/lower bounds).
- Realistic homes: **TMLR** (theory-friendly, no novelty-vs-impact fights), **NeurIPS/ICLR
  main as a stretch** if bounds are tight, **ALT/theory-workshop** floor.
- Strategic value beyond venue: this becomes the citation that P2/P4/P11 lean on for "why
  random wiring suffices" — internal value even at modest venue tier.

## The gate (reading, zero cost)

Full-text read of the **Petersen thesis** + **Mommen** (the two documents most likely to
contain a buried version of the embedding argument). If either states the theorem, the lane
closes; if they only conjecture, they become the motivating cites.

## Read up on

Must-cites (verified in the scout):
- **Kim** — arXiv:2605.08657 (gradient-cancellation + depth attenuation; the boundary of our claim).
- **Functional Percolation** — arXiv:2512.09317 (framing incumbent, no theorems).
- **WiSARD exact-VC** — Neural Computation 2019 (adjacent capacity result).
- IWP App B.5 (the open wide-vs-deep flag) + Petersen thesis + Mommen (gate reads).

Background (added here, not from the scout — sanity-check before citing):
- Bollobás, *Random Graphs* — matching/threshold machinery.
- Hoory, Linial, Wigderson 2006 — expander survey (routing/embedding vocabulary).
- Valiant 1976 — superconcentrators (the classical "small graphs route everything" result the
  embedding argument will resemble).
- Coupon-collector / occupancy bounds (any probability text) for the Ω(n log n) input-coverage side.

## Risks / kill conditions

- Gate read finds the theorem already stated → dead.
- The theorem may be true but loose (polylog overheads with huge constants) — decide a
  minimum-strength bar *before* investing proof effort: if the bound can't explain the
  empirical wide-and-shallow numbers within ~an order of magnitude, it won't carry a paper.
- Solo-theory risk: no co-author with random-graph fluency = slow; consider recruiting one
  (this is the pool's most collaboration-friendly item).
