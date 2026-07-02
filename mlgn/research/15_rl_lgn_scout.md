# 15 — Reinforcement Learning × LGN: Novelty Scout

**Scouted:** 2026-07-01 (research-scout skill + multi-agent: 6 sub-angle readers that read the
threat papers *in full* → adversarial kill-pass per opening → synthesis). **Trigger:** "I'm
new to RL — is there anything we can get FROM it or contribute TO it with LGNs?" Two directions:
**(a)** RL/policy-gradient/REINFORCE/bandit/evolution to **TRAIN** LGNs (gate choice = a discrete
action); **(b)** LGNs as ultra-cheap/fast/interpretable/FPGA **POLICIES or value functions FOR RL**.

Builds on [09_training_speed_scout.md](09_training_speed_scout.md), [11_paper2_workmap.md](11_paper2_workmap.md),
[13_snn_hebbian_scout.md](13_snn_hebbian_scout.md), [14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md),
`lgn-recurrent-scout-verdicts`, `lgn-landscape-key-facts`.

### RL in one paragraph (for orientation)
Reinforcement learning trains an **agent** to pick **actions** in an **environment** to maximize
cumulative **reward**; the two learned objects are a **policy** (state → action) and often a
**value function** (expected future reward). The gradient trick RL uses when the action is
discrete/non-differentiable is the **score-function (REINFORCE) estimator** — an *unbiased but
high-variance* gradient. That's the hinge for both directions here: LGN gate-choice is discrete,
so RL *looks* applicable to training; and LGNs are cheap/fast, so they *look* attractive as
policies. Both intuitions collide with prior art below.

---

## Verdict (headline)

**RL is a NO-GO as a thesis/headline for this program, both directions (confidence ~88%).** Keep
the P1-gating → P2-latch plan. **But the scout is not empty:** it (1) confirms one genuinely
strong, low-risk use — RL/POMDP as **P2's capstone application demo** (the single most compelling
argument for why a learnable sequential logic primitive must exist), and (2) surfaces **three
must-cite 2026 papers you didn't have**, one of which (**DWC**) is the paper that *closes* the
obvious "LGN-as-cheap-RL-policy" idea, and one (**Petersen 2024a**) shows Petersen himself already
dipped LGNs into RL.

**Why RL fails as a headline, in one line each:**
- **(a) RL-trains-LGN** → a **theorem** kills it: LGN training is a *supervised, fully-relaxable*
  objective, so a low-variance **pathwise** gradient exists *by construction*; a score-function
  estimator reintroduces variance for zero structural gain (Mohamed et al., JMLR 2020). Both RL
  motivations are already delivered *pathwise* on the exact substrate (Mind-the-Gap, Decoupled STE).
- **(b) LGN-as-RL-policy** → **DWC (arXiv:2512.01467)** is exactly the feared paper and demonstrates
  every load-bearing claim (weightless-logic policy, SAC/DDPG/PPO, FPGA, ns/nJ, interpretability).
- **Interpretable-logic-RL, evolution-of-logic-circuits, RL-for-structure/synthesis** → all
  land in mature, owned fields (programmatic/DDT RL; Evolvable Hardware/CGP; RL-for-logic-synthesis).

---

## Angle status

| # | Sub-angle | Status | Disp. | Conf. | Killed by |
|---|---|---|---|---|---|
| 1 | **Recurrent-LGN as POMDP memory ("logic-DRQN")** | PARTIAL (empty intersection) | **FOLD → P2 (capstone demo)** | 0.85 open / 0.3 standalone | its novelty *is* P2's latch |
| 2 | Formal **verifiability** of a clocked-sequential LGN controller | PARTIAL | FOLD → P2 (safety framing) | 0.7 | R-DTLGN + DWC named it as future work |
| 3 | (b) generic LGN-as-cheap-FPGA-RL-policy | **OCCUPIED** | CITE-ONLY | 0.9 | **DWC 2512.01467** + Petersen 2024a |
| 4 | (b) LGN policy = *interpretable* RL contribution | **OCCUPIED** | NO-GO | 0.9 | DDT/INTERPRETER/prog-RL; DWC |
| 5 | (a) RL/REINFORCE/bandit **trains** gate choice | **OCCUPIED** | NO-GO | 0.9 | Mind-the-Gap + Decoupled STE; Mohamed'20 theorem |
| 6 | (a) RL/NAS/search for LGN **structure/synthesis** | **OCCUPIED** | NO-GO | 0.88 | CompactLogic 2602.05830; RL-for-synthesis; CGP |
| 7 | **Evolution/CGP** to build a logic-gate policy | **OCCUPIED** | NO-GO (defensive ablation only) | 0.88 | difflogic's founding thesis; EHW/CGP |

*(Walls: W1 linearity-vs-expressivity, W2 wrong-bottleneck, W3 lane-saturation, W4 substrate-mismatch.)*

---

## The one survivor — angle 1: recurrent-LGN as POMDP memory ("logic-DRQN")

**The empty intersection is real.** {2-input difflogic gates} × {learned cross-timestep latch
state} × {RL / POMDP} × {clocked FPGA deployment} is verifiably **unclaimed as of mid-2026**:
- **DWC (2512.01467)** — read in full — is a logic RL policy but **feedforward/memoryless**
  (its flip-flops are *pipeline* registers, not belief state; all 5 MuJoCo tasks are fully-observable
  MDPs; it never mentions recurrence/POMDP).
- The recurrent-LGN papers (**RDDLGN** 2508.06097, **R-DTLGN** 2605.24649, **DiffLogic CA** 2506.04912)
  are all **supervised / non-RL**.
- The recurrent-RL-memory field (**DRQN** 1507.06527, R2D2, recurrent-PPO) and finite-memory POMDP
  controllers (**FSCs** 2602.08734, **QBN** state-extraction 1811.12530) are all **real-valued**,
  not gate-level FPGA-native.

**But it is not a separable paper — its entire novelty is P2's latch primitive.** Subtract every
owned framing (logic-RL-policy = DWC; finite-memory POMDP controller = FSC/Mealy machine; recurrent
hidden state solves POMDP = DRQN) and the only residue is the *bistable latch synthesized 1:1 to
FPGA registers*. "logic-DRQN minus P2" = "DWC with a hidden-state vector" — the obvious low-hanging
follow-up for ISTA (DWC) or ETH (RDDLGN + BitLogic already names "stateful modules").

**→ Use it as P2's flagship capstone demo, not a standalone paper.** A *tiny, memory-required*
POMDP (T-maze / memory-length / flickering-observation — **NOT** MuJoCo continuous control, which is
DWC's turf and invites a maturity comparison DWC wins) where **a feedforward logic controller
(DWC-style) provably fails and P2's stateful latch provably fixes it**, synthesized to a clocked
sequential circuit holding belief-state in FPGA registers at ns latency / nJ per action. This is
simultaneously (i) the single most compelling argument for *why a learnable sequential logic
primitive should exist*, and (ii) the natural **P2 → P3 FPGA-deployment bridge** — while adding zero
novelty risk because the contribution stays anchored on the *latch*, not on "RL policy" or "FPGA
deployment" (both owned).

**Secondary survivor (angle 2), fold into P2 as safety framing, not interpretability:** actually
**execute a formal model-check** (a safety/liveness property) on the deployed clocked LGN netlist +
verify the exhaustive state-transition table against the synthesized RTL. This is the one thing
**R-DTLGN** (which verbatim names "DFA extraction for formal model checking of the learned circuit"
as future work) and **DWC** (names DBN formal verification as future work) *named but did not do*.
Do not pitch "interpretable logic policy" — that lane is saturated and a raw gate-netlist is *less*
human-simulatable than the oblique decision trees that already win user studies.

---

## Why each other angle is dead (grounded in full reads)

- **Angle 5 — RL to train gate choice.** A *bias-variance theorem* (Mohamed et al., *Monte Carlo
  Gradient Estimation*, JMLR 2020): when a differentiable path exists — and it does, by difflogic's
  construction — the **pathwise** estimator provably dominates the score-function one. Both RL
  motivations are already delivered pathwise on the exact substrate: **Mind-the-Gap** (2506.07500)
  puts the *hard Gumbel-argmax gate = the deployed circuit* in the forward pass + STE backward
  (closes ~98% of the discretization gap, so "optimize the hard circuit, no relaxation proxy" is
  done without RL); **Decoupled STE** (2410.13331) tunes a *forward* temperature for exploration
  over the 16 gates and **explicitly rejects REINFORCE** as high-variance/costly, *demonstrated on
  DLGN*. The "REINFORCE is unbiased" escape is closed by **REBAR/RELAX** (the optimal unbiased
  discrete estimator uses the relaxation itself as a control variate). The hardware-in-the-loop /
  device-non-ideality sliver is owned by **Evolvable Hardware** (Thompson 1996), **CGP** (an LGN is
  a constrained CGP genome), and the **Physical-NN-training** taxonomy (2406.03372; in-situ PPO on
  optical hardware 2507.05583) — and LGN device robustness is already handled pathwise via Gumbel
  noise-injection.
- **Angle 6 — RL/NAS/search for LGN structure.** **CompactLogic** (2602.05830, ETH-SRI/Vechev, read
  in full) makes gate-function + wiring a **single parameter-free softmax** over (gate, input, input)
  triples and *already ships "adaptive resampling"* — the deterministic, low-variance discrete-
  structure explorer RL/bandit would claim — and shows it dominates brute-force candidate expansion.
  Macro-topology is 2–3 cheaply-swept hyperparameters (W2); synthesis-aware training is
  CompactLogic's *own named future work* and the RL-for-logic-synthesis lane is saturated (DRiLLS
  1911.04021, 2205.07614) + CGP multi-objective area/delay.
- **Angle 7 — evolution/CGP.** "Pathwise gradient beats evolution for logic gates" is **difflogic's
  founding thesis** (Petersen 2022, verbatim: evolutionary training "becomes infeasible for larger"
  models). Evolving gate-level controllers is a 25+ year owned subfield (Evolvable Hardware; Miller's
  CGP; M-CGP evolves *sequential* circuits with flip-flops). Going gradient-free throws away the
  pathwise gradient that *defines* difflogic and doesn't compose with P2's BPTT-unrolled latch
  training. Only legitimate use: a **defensive ablation footnote** in P2 (Petersen's "evolution
  infeasible at scale" was shown for *feedforward* nets; a small ES/CEM baseline on the *recurrent-
  latch* task pre-empts the "the circuit is tiny — why not evolve it?" reviewer and shows STE+BPTT
  still wins with a degraded gradient).
- **Angles 3 & 4 — LGN-as-cheap/interpretable-RL-policy = DWC.** See below.

---

## Bonus prior art you didn't have (all read in full unless flagged)

1. **DWC — "Differentiable Weightless Controllers: Learning Logic Circuits for Continuous Control"**
   (Kresse & Lampert, **ISTA**, arXiv:**2512.01467**, ICML 2026) — **read in full.** The
   direction-(b) occupier. DWN-style **k-input LUTs** (default k=6; **ablates k=2 = Petersen gates**),
   real obs z-normalized + thermometer-encoded to bits; **only the actor** is weightless (critic/σ
   stay FP), trained by **SAC** (DDPG/PPO in appendix) with EFD surrogate gradients + learnable
   interconnect. **5 MuJoCo** tasks (matches FP on 4/5). **FPGA** (Artix-7 XC7A15T @100 MHz): **1–3
   cycle latency, ~2 nJ/action, 1e8 actions/s, zero BRAM/DSP.** **Feedforward/memoryless** (FFs =
   pipelining), interpretability = sparse-connectivity heatmaps only; formal verification named as
   future work. **This neutralizes the "2-input gates vs LUTs" escape via its own k=2 ablation.**
2. **Petersen, Borgelt & Ermon — "Efficient RL Agents with Differentiable Logic Gate Networks"**
   (**CoRL 2024 DiffOpt workshop**, "Petersen 2024a") — *flag: not read in full; surfaced via DWC's
   related work.* Earlier **2-input LGN RL agents via behavioral cloning** (discrete actions,
   preliminary). **Petersen himself already touched LGN+RL** — verify its exact scope before any
   comparative claim (reader believes feedforward/BC/discrete-action).
3. **CompactLogic — "Learning Compact Boolean Networks"** (Wang/Mao/Zhang/**Vechev**, **ETH-SRI**,
   arXiv:**2602.05830**, repo eth-sri/CompactLogic) — **read in full.** Parameter-free joint
   gate+wiring learning + "adaptive resampling" + adaptive progressive discretization; beats
   DiffLogicNet (+2.73% CIFAR-10) and TreeLogicNet (47× fewer BOPs, 99.38% MNIST @6.48 ns FPGA) and
   undercuts weight-matrix connectivity (DWN, LILogic). **A second ETH group (Vechev/SRI, distinct
   from Wattenhofer) is now in LGN, and it beats the connectivity SOTA** — a strong new baseline and
   reconfirmation that the connectivity/structure lane is owned.

*Minor landscape:* DLGN scalability boundaries (2509.25933), FPGA resource utilization (2605.04109),
Decoupled STE (2410.13331) as training infra. FSC-for-POMDP (2602.08734) + QBN state-extraction
(1811.12530) are the "logic-DRQN concept isn't novel by itself" anchors.

---

## Contribution shape / resource fit / remaining gate

- **Contribution shape:** *none as an RL paper.* The only carry-forward is an **application/eval
  section inside P2** (a memory-required POMDP demo) + a **verification demo** (execute a model-check
  on the deployed clocked netlist). Both are *empirical/demo* contributions anchored on P2's latch.
- **Resource fit:** excellent — a tiny discrete POMDP (T-maze / memory-length) is *cheaper* than
  psMNIST-784 and runs on the 2080S / DUST cluster; no new infra beyond the P2 recurrent cell + a
  minimal RL loop (recurrent policy-gradient or DRQN). Off-the-shelf model-checkers (ABC/nuXmv/
  SymbiYosys) for the verification demo.
- **REMAINING GATE (before committing P2's headline to it):** **trainability** — high-variance
  policy-gradient / Q-learning credit assignment stacked on a **BPTT-unrolled, discrete-relaxed
  recurrent latch** (custom-STE-through-feedback) may be unstable; this *compounds* P2's own feedback-
  training risk (§C2 obstruction) and could silently sink the demo. **Prove it on a tiny POMDP before
  making it P2's flagship.** Secondary gate: **scoop race** — DWC/ISTA or Wattenhofer/ETH bolting a
  hidden state onto their RL/recurrent-logic stack is the obvious low-hanging follow-up; monitor those
  feeds and verify Petersen 2024a stays feedforward/BC.

---

## Bottom line / steer

**Do NOT pursue RL as a research direction or paper headline** — direction (a) is killed by a
bias-variance theorem + ETH's pathwise gap/exploration methods; direction (b) is occupied by DWC;
the interpretability, evolution, and structure-search framings are all in mature owned fields. **Do
harvest one thing:** a small **memory-required POMDP "logic-DRQN" demo** as **P2's capstone** — the
cleanest possible argument that a *learnable sequential logic primitive* is necessary (feedforward
logic controllers provably can't hold belief-state; P2's latch can, synthesized to FPGA registers)
— plus the **execute-a-model-check** verification demo that R-DTLGN and DWC only *named*. Both fold
into P2, gated on the trainability check. And **add DWC (2512.01467), Petersen 2024a (CoRL DiffOpt),
and CompactLogic (2602.05830) to P2's citations** regardless.

## How to apply
Don't re-scout RL. When P2 reaches the benchmark/demo stage, decide whether to include the
POMDP-capstone + verification demo (recommended, but run the tiny-POMDP trainability check first).
Cite DWC as the direct predecessor of any LGN-RL-policy claim and explicitly distinguish
*feedforward-memoryless DWC* from *P2's clocked-sequential latch*. Keep evolution/CGP only as a
one-table defensive ablation. Watch the ISTA (Kresse/Lampert) and ETH (Wattenhofer/DISCO;
Vechev/SRI) arXiv feeds — three separate groups are now within one move of recurrent-logic-RL.
