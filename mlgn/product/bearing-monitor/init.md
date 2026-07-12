# In-sensor verified condition-monitoring core (bearings)

_Status: **candidate — the portfolio's best local-edge play.** Gated on the bearing-wise
honest-split spike (1–2 wks, P3b §C2 lists it as the fallback-carrier trigger). Sources:
[research/22 §1](../../research/22_applications_scout.md) (CONDITIONAL ~65%, verified;
SKF–LTU UTC confirmed real). Paper twin: [paper/p8](../../paper/p8/init.md)._

## What it is

An **always-on, sub-100 µW, multi-class bearing-fault monitor core** for in-sensor / in-bearing
deployment:

- Sequential LGN on a tiny FPGA (iCE40-class) or ASIC block, fed by a MEMS accelerometer;
  degradation state, operating-condition context, and **alarm persistence/debounce counters in
  exact-binary registers**.
- Alarm logic ships with machine-checked certificates: "alarm within k windows of persistent
  fault", "no alarm latch on transients < m samples" — verified debounce, fewer false truck
  rolls.
- Product forms: (a) IP block for sensor-chip vendors, (b) reference design (accel + iCE40 +
  firmware) for CM-system vendors, (c) the SKF route: a drop-in inference tier for
  battery/harvester-powered sensor nodes.

## The wedge (why anyone pays)

- **Energy is the binding commercial constraint — proven by the buyers' own products**
  (scout-verified): SKF Insight (rotation-harvested in-bearing sensors) and IMx-1 (battery
  mesh). The *why* — wireless-vs-wiring cost, battery life in years as the purchasing
  criterion — is our added commercial reading, not scout-checked. SKF Enlight AI does
  inference *off-sensor* today.
- The sub-100 µW **digital always-on multi-class** tier is unclaimed: Aspinity AML100 (<20 µA)
  proves the µW tier sells but is an analog wake-up (binary, no multi-class, no verification);
  MCU TinyML is mJ/inference; BNN/LSTM-FPGA is 10–100× above.
- **The local edge is real and specific:** the **SKF–LTU University Technology Centre**
  (est. 2011, SKF's 5th UTC worldwide, scope = condition monitoring & asset management,
  cooperation publicly renewed) and **SKF Condition Monitoring Center (Luleå) AB builds the IMx
  hardware in Luleå.** This is a walk-across-town industrial route, thesis-adjacent, with
  Swedish funding logic (Vinnova/SSF industrial-PhD patterns).

## Potential customers

- **SKF** — the UTC route (first and obvious; also the reason to do bearings and not generic
  vibration).
- Other CM/bearing vendors: Schaeffler (OPTIME battery mesh = the direct analog), Emerson,
  ifm, Banner — all ship wireless vibration nodes with battery-life pain.
- **Sensor-chip vendors:** ST (ISM330IS already puts an ML core in an industrial IMU — the
  trendline exhibit), Analog Devices (CbM line), TDK/InvenSense — an LGN block is a
  differentiating hard-macro for them.
- End-user verticals via CM service providers: wind (gearbox/generator bearings), rail axle
  boxes, pulp & paper (LTU's backyard industry).

## What it needs to work — technical

1. **The honest-split spike passes:** feedforward difflogic, thermometer envelope-spectrum
   features, Paderborn + CWRU, bearing-wise leakage-free splits, within a few points of the
   (collapsed) realistic DL baselines. If accuracy craters under cross-bearing shift, kill.
2. **Honest front-end energy accounting** — the open technical risk the scout under-weighted:
   envelope spectrum needs rectification/filtering/FFT *before* the LGN. If the DSP front-end
   dominates the budget, the µW claim dies. Options: logic-friendly features (zero-crossing/
   threshold statistics), a minimal fixed-point envelope stage costed explicitly, or moving
   feature extraction into trained logic (research question).
3. The sequential story on run-to-failure data (IMS, XJTU-SY): degradation tracking +
   persistence, not just snapshot classification.
4. Operating-condition robustness (variable speed/load — order tracking or condition inputs).
5. Platform reality: Artix-7 static power (~100 mW class) breaks the story — target
   iCE40UP5K-class (µW static) or ASIC; the P3b flow needs an ice40/ASIC leg
   ([paper/p11](../../paper/p11/init.md) supplies the standard-cell flow).
6. P3a property machinery ported to alarm/debounce specs.

## What it needs beyond the tech

- **The UTC conversation** — timing matters: arrive with the spike verdict + a board demo, not
  a promise (mirrors the P3b pitch discipline). Route via LTU EISLAB / Operation & Maintenance
  contacts inside the UTC scope.
- Funding: Vinnova production/industry calls, SSF, industrial-PhD or postdoc-with-SKF
  constructions — this product is *designed* to be an academic-industrial collaboration, not a
  startup.
- Certification later (machinery directive / SIL levels if alarms become safety functions) —
  the verified-alarm artifact is again the differentiator, but don't front-load this.
- IP: same open-core split as [lgn-toolchain](../lgn-toolchain/init.md) — flow open, trained
  per-application circuits + property packs licensable.

## Competition & prior art

- Incumbent practice: envelope analysis + thresholds computed off-sensor (SKF @ptitude/Enlight,
  Emerson AMS) — the products this augments rather than replaces.
- In-sensor tier: Aspinity (analog wake-up), ST ISM330IS (programmable ISPU ML core; the
  decision-tree MLC lives in siblings like ISM330DHCX), neuromorphic
  streaming PdM (PHM-Europe 2026) — none multi-class + verified + digital µW.
- Academic: Vitolo 2022 (138 µW/MHz partially-binarized AE), 17 mW iCE40 LSTM, mJ-tier MCU TinyML;
  weightless-NN motor diagnosis exists (Abid & Afshan 2024) — feedforward, unverified.

## Signals to go / kill

- **GO:** spike passes; a UTC/SKF conversation yields data or a pilot node; front-end energy
  budget closes under ~100 µW total.
- **KILL:** cross-bearing accuracy collapse; front-end dominates energy with no logic-friendly
  alternative; SKF route stays cold after two attempts (without it, this is a generic TinyML
  play against thousands).

## First three concrete steps (when activated)

1. Run the 1–2 wk honest-split spike (also P3b's fallback-carrier gate — double duty).
2. Cost the envelope front-end honestly (fixed-point envelope stage on iCE40; µW estimate).
3. Map the UTC: who at LTU sits in it now, what projects are running, where a µW inference
   tier fits — then one exploratory conversation with the spike numbers in hand.
