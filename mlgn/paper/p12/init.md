# P12 — Why Logic Gate Networks Overfit (and What Actually Regularizes Them)

_Status: **pool — CONDITIONAL ~65%, expensive + racing.** Source scout:
[research/21 §6](../../research/21_landscape_weakness_scout.md) (2026-07-08, adversarially
verified). Cost: **300–500 GPU-h grid** (with p6, the pool's most expensive tier) + racing two active
labs — scout's own note: awkward during P2. No timeline — grab from
[research/24](../../research/24_roadmap.md)._

## What it is

Nobody measures the LGN train/test gap systematically, characterizes **why**
categorical-gate networks overfit (memorization probes, gate-entropy dynamics, small-data
scaling), or demonstrates a generalization-**improving** LGN regularizer. The premise is
strong and got stronger under verification: **three independent published naive-dropout
failures** (incl. RDDLGN Table 9: all dropout variants ≤ baseline), and SAM-on-gate-logits is
confirmed empty. There is also a **standing contradiction to resolve**: Kim's DSLGN (IEEE
Access 2023) claims stochastic-gate-perturbation *wins*; Light DLGN App F says *all*
regularizers fail — the two literatures never cite each other.

## What it covers

- **The mechanism study:** train/test gap vs width/depth/data-size; memorization probes
  (label-noise fitting); gate-entropy dynamics over training (when do neurons commit, and does
  early commitment predict overfitting?); small-data scaling curves.
- **The regularizer bake-off, done right:** dropout variants (documenting the failure mode,
  not just the failure), stochastic gate perturbation (the DSLGN claim, reproduced), entropy
  penalties/annealing, spectral/L1-of-Fourier (PST ships one, interpretability-framed — reframe
  as "first shown to improve test accuracy" *if it does*), SAM-on-logits, data augmentation as
  the control.
- **Resolving DSLGN vs Light DLGN App F** head-to-head under one protocol — a service to the
  field regardless of which wins.

## Scope — claim discipline

- PST's L1-of-Fourier regularizer exists (interpretability-framed, no generalization claim) —
  the reframe must credit it.
- If no regularizer wins, the mechanism study + contradiction resolution still stand, but the
  tier drops (negative-result + measurement paper).
- Keep P5's calibration axis out of scope (separate paper; cross-cite).

## Venue & tier (honest call)

- **A\* main (NeurIPS/ICLR) if a regularizer genuinely improves test accuracy** — "first
  working LGN regularizer + the mechanism for why others fail" is a main-track story.
- Mechanism-study-only outcome: **TMLR** (fits claims-based review) or a strong workshop.
- The GPU cost only pays at the upper tier — do not start this for a workshop-tier outcome.

## The gate (zero-GPU, do first)

**Read Kim's DSLGN in full** via LTU IEEE Xplore access (it resisted 5 open-web fetch routes).
If DSLGN's win is real and general, the "first working regularizer" slot is already taken and
this collapses to the mechanism study; if it's narrow/flawed, the slot is open and the
contradiction is the hook.

## Read up on

Must-cites (verified in the scout):
- **DSLGN** — Kim, IEEE Access 2023 (the claimed win; THE gate read).
- **Light DLGN** App F (all-regularizers-fail) + **RDDLGN** Table 9 (dropout ≤ baseline) — the failure exhibits.
- **PST** — arXiv:2603.00302 (L1-of-Fourier regularizer, interpretability-framed).
- **BitLogic v2** — arXiv:2602.07400 (design-space context; also the race exhibit).
- Kim — arXiv:2605.08657 (gradient-cancellation at init; the trainability side that interacts with entropy dynamics).

Background (added here, not from the scout — sanity-check before citing):
- Zhang et al., ICLR 2017 — *Understanding deep learning requires rethinking generalization* (memorization-probe protocol).
- Foret et al., ICLR 2021 — SAM (for the on-logits arm).
- Arpit et al., ICML 2017 — memorization in DNNs (probe design).

## Risks / kill conditions

- **The race**: ETH (BitLogic machinery) and ISTA both have the tooling to run this grid faster;
  a freshness sweep is mandatory, and any sign of an in-flight submission kills it.
- 300–500 GPU-h on shared 2×2080 Ti is weeks of wall-clock — needs DUST headroom P2 isn't using
  (realistically: after P2 experiments freeze).
- DSLGN read reveals the slot is taken → collapse to mechanism study, re-tier before investing.
