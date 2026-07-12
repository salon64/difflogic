# ns-class learned inference for trading

_Status: **PARKED — NO-GO commercial near-term (scout verdict, verified).** The academic paper
([paper/p9](../../paper/p9/init.md)) comes first in every scenario: *the paper IS the demo.*
This doc exists so the commercial option is written down with its revive conditions, not
re-derived from scratch later. Source: [research/22 §2](../../research/22_applications_scout.md)
(2026-07-08, verified)._

## What it would be

- **An inference IP core for the trading hot path:** learned feature→signal circuit at
  ns-scale — feedforward LGN ~10–25 ns class, or the clatch variant clocking O(1) per
  market-data event with LOB state held in registers (no window recompute).
- Delivered as RTL for the customer's own NIC/FPGA infrastructure (prop shops run their own
  boards; nobody buys appliances for the hot path).
- **The differentiated add-on: the compliance artifact.** MiFID II RTS 6 + the ESMA 2026
  algo-governance regime demand explainability + kill-switch discipline for algorithmic
  trading; a learned strategy component that is a *formally checkable circuit* (P3a: bounded
  properties like "never emits an order when X", checked on the deployed netlist) is a
  genuinely unclaimed wedge — today's answer is documentation, not proof.

## Why it's parked (the scout's case, kept honest)

- **The µs tier is occupied by audited incumbents:** Myrtle VOLLO ~2 µs p99 (STAC-ML record,
  Apr 2026), Xelera Silva 1.6 µs GBT. They have STAC audits (and, presumably, the sales
  channels and reference customers that follow — our inference); we'd have a benchmark table.
- **The ns tier's incumbent is "no model at all":** prop shops hand-code RTL
  (scout-verified; feature-building is already ns-scale); convincing them to insert a
  *learned* circuit is a trust problem before it's a latency problem.
- A learned circuit in the execution path is a compliance question with no precedent — the
  wedge above is real but unproven, and nobody pays first.
- **No LGN vendor exists** — which also means no market validation of the substrate anywhere
  in finance.

## Revive conditions (ALL of these, roughly in order)

1. **p9 published** with the frontier table (accuracy vs wire latency) holding up: LGN ns-class
   numbers at accuracy within the tree/DeepLOB band, measured (not synthesis-estimated).
2. **A quant partner** — prop shop, market-making desk, or trading-infra vendor — engages off
   the paper (the scout's own condition for even the paper is a quant co-author; for the
   product it's a design partner with real feeds and a P&L view).
3. **Regulatory pull materializes:** ESMA/FCA enforcement makes formal artifacts for algo
   components valuable in practice (watch RTS 6 review outcomes), OR a customer's compliance
   team asks for exactly this.
4. Budget for a **STAC-ML audit** (the tier's credibility currency) — realistically only with
   a partner's money.

## Potential customers (if revived)

- Proprietary trading firms / market makers with FPGA teams (Optiver, IMC, Jump, HRT class) —
  as design partners, not license buyers, initially.
- Trading-infrastructure vendors (Exegy/Enyx, LDA Technologies, Xelera itself) — the
  ns-inference block as a portfolio addition; most realistic licensing route.
- Exchanges/venues (surveillance-side, softer latency needs) — a lateral, unexplored.

## What it would need technically (beyond p9)

- Real feed handlers (ITCH/OUCH, MDP3) feeding the tokenizer at line rate — partner territory.
- Retraining/adaptation cadence story (markets drift daily; links to
  [paper/p6](../../paper/p6/init.md) netlist plasticity — LUT-mask deltas between sessions).
- Property packs for the compliance wedge: order-rate limits, price-band never-cross,
  kill-switch reachability — P3a machinery on trading specs.
- Determinism/jitter characterization (p99 story, not just typical-path).

## Kill (permanent)

- p9's FI-2010 gate fails (accuracy floor) — kills paper and product.
- VOLLO/Xelera-class incumbents publish sub-100 ns learned inference — the niche closes.
- Two years post-p9 with zero inbound interest — the wedge was academic; leave it as a paper.
