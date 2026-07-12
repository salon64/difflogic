# P6 — Netlist Plasticity: Warm-Starting, Fine-Tuning, and Adapting Trained Logic Gate Networks

_Status: **pool — GO ~60–63%.** Source scout:
[research/21 §2](../../research/21_landscape_weakness_scout.md) (2026-07-08, adversarially
verified). Cost: ~2–4 GPU-weeks MVP (DUST). Depends on: existing checkpoints + training stack
only. No timeline — grab from [research/24](../../research/24_roadmap.md) when ready._

## What it is

Not one transfer / warm-start / fine-tune / continual-learning / test-time-adaptation result
exists on gradient-trained gate netlists (the whole Petersen line through LogicIR). "**Netlist
plasticity**": what happens when you re-relax a hardened one-hot netlist and train it again —
and what does adaptation *cost in gate edits*? Publishable in both directions: "hardened LGNs
are cheaply updatable" **or** "hardened LGNs lose plasticity — deployment really is frozen."
Either answer is a first.

## What it covers

- **Warm-start / fine-tune under shift:** re-relax hardened one-hot logits → fine-tune on
  MNIST-C / CIFAR-10-C, rotated/permuted variants, task increments — vs from-scratch training
  at matched compute.
- **Forward-only adaptation of the HARD netlist:** ZOA-style (zeroth-order) updates without
  re-relaxation — the deployment-realistic regime (a fielded circuit that can't backprop).
- **The metric that carries the paper: adaptation cost in gate-edits** — accuracy-recovered vs
  bits-flipped vs compute, as a frontier. Metric lineage from EDA functional-ECO (engineering
  change orders = minimal netlist patches).
- **Forgetting analysis:** what do the flipped gates do to the old task (continual-learning
  face of the same coin).

## Scope — claim discipline (verifier-mandated)

- **The in-place-LUT-rewrite hardware punchline is occupied** (Glette & Kaufmann, AHS 2014 —
  ICAP LUT rewriting; Dynamic Tsetlin arXiv:2504.19797). **Recast as "fine-tune deltas are
  LUT-mask-sized" (simulated), lead with method + forgetting analysis** — do not promise a
  live-reconfiguration demo.
- Cite **WS-DARTS** (arXiv:2205.06355) as the closest relative (warm-starting differentiable
  architecture search) and the functional-ECO literature for the gate-edit metric.
- ISTA's interconnect paper (arXiv:2507.02585) already SAT-prunes trained nets — adjacent
  netlist-surgery lineage, cite and distinguish (they compress, we adapt).

## Venue & tier (honest call)

- **Main-track ML attempt (ICLR / NeurIPS)** — adaptation/CL/TTA is an ML-culture axis the
  hardware-culture labs skip, and a clean gate-edit frontier figure is main-track-legible.
- **TMLR** is a natural home if results are solid but not flashy (either-direction findings
  fit TMLR's claims-based review).
- Floor: efficient-ML / on-device workshop.
- Product hook (not for the paper, for us): LUT-mask-sized deltas = OTA update story for
  [product/can-ids](../../product/can-ids/init.md).

## The gate (1 day — falsifier first)

Hardened MNIST LGN → re-relax → fine-tune on rotated/corrupted vs (a) from-scratch and (b)
forward-only; count gate edits. **Dies if warm-start shows no compute advantage AND edits
aren't strikingly small.** (Either one alive keeps the paper alive.)

## Read up on

Must-cites (verified in the scout):
- **WS-DARTS** — arXiv:2205.06355 (warm-starting DARTS; closest relative).
- **Glette & Kaufmann** — AHS 2014 (ICAP in-place LUT rewriting; the occupied punchline).
- **Dynamic Tsetlin Machines** — arXiv:2504.19797 (adaptive discrete-logic learner; occupied-punchline #2).
- **ISTA interconnect** — arXiv:2507.02585 (SAT-equivalence pruning of trained nets).
- **BitLogic v2** — arXiv:2602.07400 (program-wide must-cite).
- EDA **functional ECO** literature (minimal netlist patch under spec change) — the gate-edit metric's ancestry.

Background (added here, not from the scout — sanity-check before citing):
- Hendrycks & Dietterich, ICLR 2019 — CIFAR-10-C corruption benchmark; Mu & Gilmer,
  arXiv:1906.02337 (2019) — MNIST-C (separate paper, commonly mis-merged).
- Kirkpatrick et al., PNAS 2017 (EWC) + a continual-learning survey for forgetting protocol.
- Wang et al., ICLR 2021 (TENT) — test-time adaptation baseline framing.
- Zeroth-order optimization for the forward-only arm (e.g. SPSA / ZO-SGD surveys).

## Risks / kill conditions

- Gate falsifier fails both arms → park (the negative result alone is thin for main track;
  becomes a workshop note).
- Re-relaxation may destroy the discretization the hardening bought — measure the soft→hard gap
  after fine-tune explicitly (this is a finding either way, budget for it).
- Scoop check: nothing in-flight found 2026-07-08; re-sweep OpenReview/arXiv before starting.
