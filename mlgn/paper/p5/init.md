# P5 — Are Logic Gate Networks Calibrated? Free Circuit Ensembles from the Gate Posterior

_Status: **pool — GO ~65%** (best near-term standalone; the scout's "best fit of all axes" —
post-hoc on existing checkpoints). Source
scout: [research/21 §1](../../research/21_landscape_weakness_scout.md) (2026-07-08,
adversarially verified). Depends on: **nothing new** — post-hoc on checkpoints this repo
already has; MNIST/F-MNIST/CIFAR/SVHN (+ CMS open data) on a 2080 Ti. No timeline — grab from
[research/24](../../research/24_roadmap.md) when ready._

## What it is

First-measurement + method paper. Nobody has measured **ECE / Brier / NLL, conformal coverage,
or OOD-AUROC on any *binary* LGN**, and nobody samples the trained per-neuron 16-gate
categorical **at test time**. The trained gate distribution is a **free posterior over Boolean
circuits**: sampling K assignments gives an implicit deep ensemble whose *every member is a
deployable hard circuit* — an efficiency story float ensembles structurally can't match (K
circuits ≈ K× gates, not K× float networks; on FPGA potentially K parallel copies at LUT cost).

## What it covers

- **Measurement half:** calibration (ECE/Brier/NLL, reliability diagrams), conformal prediction
  coverage/set-size, OOD detection (AUROC on standard near/far-OOD splits), across the standard
  difflogic pipeline (thermometer front-end, GroupSum head) on MNIST/F-MNIST/CIFAR-10/SVHN;
  optionally CMS open data as the "binarization suppresses outlier scores" exhibit (CICADA
  motivates exactly this).
- **Method half:** test-time sampling of the gate categorical → circuit ensembles. Diversity,
  disagreement-as-uncertainty, ensemble calibration vs single-argmax circuit, cost accounting
  (the "free" claim quantified). Compare against temperature scaling, deep ensembles of
  independently trained LGNs, and MC-sampling baselines.
- Deliverable framing: "the first UQ report card for logic gate networks, plus the one UQ method
  their parametrization gives away for free."

## Scope — claim discipline (verifier-mandated)

- **PST ternary LGNs** (arXiv:2603.00302, UMD) already publish abstention/selective-prediction —
  but *architectural* (Kleene UNKNOWN), not *statistical*. Hook = "calibration of the standard
  binary pipeline"; cite and position, don't ignore.
- **Tsetlin UQ exists** at CIFAR-10 scale (arXiv:2507.04175) — caps any "first discrete-logic
  UQ" claim. **Scope claims to LGNs.**
- Kim's **DSLGN** (IEEE Access 2023) samples the gate distribution **during training only**;
  Mind-the-Gap's Gumbel likewise — the test-time posterior use is the vacant slice.
- Conformal × L1-trigger confirmed empty; CICADA (arXiv:2511.01908) flags
  binarization-suppressed outlier scores = the motivating exhibit, not a competitor.

## Venue & tier (honest call)

- **A\* main-track attempt is justified** (NeurIPS/ICLR) *if* the gate shows real ensemble
  diversity and one headline lands crisp (e.g. "free ensemble halves ECE at zero float cost" or
  a clean OOD win). First-measurement + usable method is a main-track shape.
- Realistic fallbacks: **TMLR** (measurement papers land well there) or an **UQ/reliable-ML
  workshop** at NeurIPS/ICLR (safe floor).
- If the posterior is degenerate (see gate) it becomes a workshop-sized negative-result note —
  still publishable, much smaller.

## The gate (ONE afternoon — run first, before anything else in the pool)

Sample 32 circuits from an existing trained checkpoint. **If gate distributions are
near-one-hot (entropy → 0), there is no ensemble diversity and the free-posterior story dies.**
Fallback: entropy-regularized / Gumbel checkpoints — works, but weakens "free" to "cheap".

## Read up on

Must-cites (verified in the scout):
- **DSLGN** — Kim, IEEE Access 2023 (training-time sampling; full text via LTU IEEE Xplore — resisted open-web fetch).
- **PST** ternary LGNs — arXiv:2603.00302 (architectural abstention; the framing constraint).
- **Tsetlin UQ** — arXiv:2507.04175 (CIFAR-10-scale discrete-machine UQ; the claim cap).
- **CICADA** — arXiv:2511.01908 (binarization-suppressed outlier scores; motivating exhibit).
- **BitLogic v2** — arXiv:2602.07400 (TMLR 2026, ETH) — must-cite for the whole program.
- Mind-the-Gap (Gumbel relaxation) — training-time stochastic relative.

Background (added here, not from the scout — sanity-check before citing):
- Guo et al., ICML 2017 — *On Calibration of Modern Neural Networks* (ECE canon).
- Lakshminarayanan et al., NeurIPS 2017 — deep ensembles (the baseline to beat on efficiency).
- Ovadia et al., NeurIPS 2019 — *Can You Trust Your Model's Uncertainty?* (shift protocol).
- Angelopoulos & Bates, arXiv:2107.07511 — conformal prediction tutorial (coverage protocol).
- Hendrycks & Gimpel 2017 (MSP baseline) + standard OOD-eval suites for the AUROC half.

## Risks / kill conditions

- **Gate fails** (one-hot posteriors) → downgrade to negative-result note or park.
- Diversity exists but ensembles don't improve calibration/OOD → the measurement half still
  stands alone (weaker, TMLR/workshop tier).
- Race: ML-culture axes are exactly what ETH/ISTA *skip* (doc-21 cross-cutting finding), so the
  lane is quiet — but BitLogic's design-space machinery could bolt UQ on quickly; re-run a
  freshness sweep before committing.
