# P10 — Attacking the Relaxation: An Empirical Robustness Study of Differentiable Logic Gate Networks

_Status: **pool — CONDITIONAL ~55–60%** (narrowed from scout GO 78 by verifier). Source scout:
[research/21 §3](../../research/21_landscape_weakness_scout.md) (2026-07-08, adversarially
verified). Cost: inference-dominated, cheap (existing checkpoints + attack tooling). No
timeline — grab from [research/24](../../research/24_roadmap.md)._

## What it is

First empirical attack / corruption / fault characterization of **Petersen-lineage difflogic**
networks. The field's folk belief — "no inference gradient = safe" — is textbook **gradient
masking** (a thermometer front-end is literally Buckman et al.'s broken defense, per
Athalye et al.), and it has never been tested on difflogic. This is a **port with a finding,
not a first** (see scope): the finding is whatever the masking test reveals.

## What it covers

- **Adaptive attacks through the relaxation:** BPDA using the soft net as the differentiable
  surrogate; soft→hard transfer rates; the gradient-masking checklist run honestly.
- **Black-box attacks** that bypass gradients entirely: Square, HopSkipJump — the control that
  decides whether apparent robustness is real or masked.
- **Corruption curves:** CIFAR-10-C / MNIST-C across severities (genuinely unmeasured on
  difflogic).
- **Wire-level fault injection:** bit-flips on wires/gates of the hard netlist — the
  hardware-facing robustness axis nobody has for this lineage (and a P3a/P3b synergy: we can
  *verify* fault-tolerance properties on small circuits).
- **First AT-recipe-through-the-relaxation** with honest discretization-gap accounting
  (adversarially train soft, measure hard).

## Scope — claim discipline (verifier-mandated)

- **TTnet** (Benamira et al., IJCAI 2024) — a differentiable logic-gate-circuit CNN benchmarked
  against Petersen's net — already publishes **complete-SAT robust accuracy** on MNIST/CIFAR at
  standard ε with PGD-AT-through-STE; **Jia & Rinard** (NeurIPS 2020) already executed the
  verify-attack-tightness methodology on BNNs. **Scope claims to: difflogic lineage +
  corruptions + faults**; cite TTnet/JR20 as the method chain, don't compete with them.
- ISTA's feedforward-LGN verifier (arXiv:2505.19932) is the formal complement — and the scoop
  vector (them bolting empirical attacks onto the verifier is the obvious v2).
- Follow the Carlini-style evaluation checklist religiously — a robustness paper that gets its
  own eval wrong is worse than no paper.

## Venue & tier (honest call)

- **SaTML** (IEEE Conf. on Secure & Trustworthy ML) — main-track bullseye for
  characterization-with-adaptive-attacks papers.
- **A\* main (NeurIPS/ICLR) only if** the masking finding is spectacular (e.g. "reported LGN
  robustness collapses 40 points under adaptive attack" or the inverse "discreteness confers
  real black-box robustness at ε=X").
- Floor: AdvML/TrustML workshop at a main venue.
- TMLR viable for the systematic-characterization framing.

## The gate (hours, run before anything)

**OpenReview/arXiv 8-week freshness sweep** for in-flight LGN-robustness submissions —
specifically anything from ISTA attaching attacks to 2505.19932, and TTnet follow-ups touching
Petersen-lineage nets. The lane was clear 2026-07-08; it is the pool's most scoopable slot.

## Read up on

Must-cites (verified in the scout):
- **TTnet** — Benamira et al., IJCAI 2024 (SAT-complete robust accuracy on a logic-gate CNN; the ceiling on "first").
- **Jia & Rinard** — NeurIPS 2020 (verify-attack-tightness on BNNs; the method template).
- **ISTA verifier** — arXiv:2505.19932 (feedforward LGN verification; complement + scoop vector).
- **BitLogic v2** — arXiv:2602.07400 (program-wide must-cite).

Background (added here, not from the scout — sanity-check before citing):
- Athalye, Carlini, Wagner, ICML 2018 — obfuscated gradients / BPDA (the framework).
- Buckman et al., ICLR 2018 — thermometer encoding as a (broken) defense (the exhibit).
- Carlini et al. 2019, arXiv:1902.06705 — *On Evaluating Adversarial Robustness* (the checklist).
- Andriushchenko et al., ECCV 2020 — Square attack; Chen et al., IEEE S&P 2020 — HopSkipJump.
- Hendrycks & Dietterich, ICLR 2019 — corruption benchmarks.
- Fault-injection lineage: bit-flip attacks on quantized nets (e.g. Rakin et al., ICCV 2019 BFA) for the wire-fault axis.

## Risks / kill conditions

- Freshness sweep finds an in-flight difflogic-robustness paper → park immediately (thin moat).
- If adaptive attacks land exactly at BNN-known levels with no difflogic-specific structure →
  contribution shrinks to corruptions+faults only (workshop tier).
- AT-through-relaxation may destabilize discretization — report the gap honestly; that
  interaction is itself a finding (links to P12's overfitting mechanics).
