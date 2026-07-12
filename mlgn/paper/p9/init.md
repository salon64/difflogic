# P9 — Nanosecond Limit-Order-Book Inference with Sequential Logic Gate Networks

_Status: **pool — CONDITIONAL ~65% academic; commercial = NO-GO near-term** (see
[product/hft-inference](../../product/hft-inference/init.md) for the parked commercial view).
Source scout: [research/22 §2](../../research/22_applications_scout.md) (2026-07-08, verified;
direct kill-check CLEAN — no logic-native substrate has ever touched market data). Method-gated
by P2, tool-gated by P3b emitter → realistically lands 2027. Scout condition: paper only if the
gate passes **and a quant-finance co-author appears.** No timeline — grab from
[research/24](../../research/24_roadmap.md)._

## What it is

First logic-native learned model in the trading hot path: each market-data event clocks a
learned circuit **once** — clatch registers hold LOB-derived state, no window recompute —
emitted as clocked RTL at ns-scale reported latency. The niche is real and inverted from the
obvious objection: feature-building on FPGA is *already* ns-scale (parse 20–25 ns, book update
30–40 ns/msg; top firms quote sub-100 ns tick-to-trade) while the **learned model** is the
40–100× outlier (VOLLO ~2 µs p99, Xelera 1.6 µs) — ns-scale *inference* is exactly the unserved
slot.

## What it covers

- **Headline artifact: the accuracy-vs-wire-latency frontier table** — trees-in-switch (µs,
  2–16% accuracy loss) / VOLLO–Xelera (1.6–2 µs audited) / feedforward LGN (~10–25 ns) /
  clatch-LGN (O(1) per event, no window recompute). Synthesized + measured latency on our own
  hardware (P3b flow), accuracy on public benchmarks.
- **Benchmark:** FI-2010 mid-price movement (3-class, k=100) as the community anchor; ITCH
  sample days and/or crypto L2 feeds for the streaming/event-clocked story; LOBSTER (academic
  ~$100s) if depth matters.
- **The event-clocked recurrence claim:** per-event O(1) state update vs windowed recompute —
  the same recurrence-earns-its-keep discipline as P7, on LOB features.
- Optionally: the ECAI'24-style profitability framing (their own simulation: +4.08% from
  latency vs −2.17% from accuracy loss) to price the accuracy/latency trade — with a
  quant co-author.

## Scope — claim discipline

- Precedent says **public-data backtest + synthesized/measured latency clears the publication
  bar** (LOBIN @ HPSR with one day of free ITCH; FINN-GL @ FPL). Do not promise live-market
  results.
- Commercial claims stay out: the µs tier is occupied by audited incumbents; we have no STAC
  audit. Reported latency = synthesis/P&R + on-board measurement of *our* pipeline stage, with
  the measurement boundary stated precisely (this is where HFT reviewers kill papers).
- FI-2010 is noisy and stale (2010 Nordic data) — treat as comparability anchor, not as the
  claim.
- The compliance angle (MiFID II RTS 6 / ESMA 2026: explainability + kill-switch regime →
  "formally checkable learned circuit" via P3a) is a **discussion-section wedge**, not a
  validated claim.

## Venue & tier (honest call)

- **ICAIF** (ACM Int'l Conf. on AI in Finance) — the domain main track; systems papers with
  hardware numbers are rare there = differentiation.
- **FPL / FCCM application track** — B+/A− hardware venues; the FINN-GL precedent is exactly
  this shape.
- ECAI-class general-AI venue possible but weaker fit.
- **Not A\*.** Value to the program: the frontier table is citable forever, and it exercises
  the P2+P3b stack on a third domain.

## The gate (1–2 days, existing code — cheapest gate in the pool)

Feedforward difflogic on FI-2010 mid-price movement (3-class, k=100). **If discretized F1 falls
far below the ~40–46% tree/DeepLOB band on this noisy regime, the application dies regardless
of latency.** (P3b §C2 lists this as slot-whenever.)

## Read up on

Must-cites (verified in the scout):
- **LOBIN** — HPSR 2023 (trees in Tofino, µs, the data-plane precedent) + **Hong et al.** — ECAI 2024 (tables-in-switch + the profitability simulation).
- **FINN-GL** — FPL 2025, arXiv:2506.20810 (BNN-lineage ConvLSTM on FI-2010, W8A6, 4.3 ms — the must-cite that defines the gap).
- **Myrtle VOLLO** (~2 µs p99 LSTM, STAC-ML record Apr 2026) + **Xelera Silva** (1.6 µs GBT) — the audited commercial frontier.
- FI-2010 dataset paper + **DeepLOB** — the accuracy band (scout names only the dataset/band;
  attributions added here: Ntakaris et al. 2018; Zhang, Zohren & Roberts, IEEE TSP 2019).
- MiFID II RTS 6 / ESMA 2026 algo-governance texts — for the compliance wedge paragraph.

Background (added here, not from the scout — sanity-check before citing):
- Gould et al. 2013 — limit-order-book survey (LOB mechanics for the uninitiated reader).
- STAC-ML benchmark methodology docs — to phrase latency claims in the vocabulary the field audits.

## Risks / kill conditions

- FI-2010 gate fails → dead (pre-committed).
- No quant-finance co-author materializes → scout says don't write it solo; park.
- Latency-measurement-boundary sloppiness = the reviewer kill in this domain; define
  tick-to-signal scope early.
- Note W2 discipline (research/16): frame on the *model* being the bottleneck outlier — never
  claim end-to-end tick-to-trade wins.
