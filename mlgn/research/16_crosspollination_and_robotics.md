# 16 — AI cross-pollination map + Robotics/Drones application (Kyushu)

**Scouted:** 2026-07-01 (multi-agent: 5 area-cluster finders + 2 drone deep-readers → 12
adversarial kills → synthesis; threat papers read in full where noted). **Triggers:** (1) "what
OTHER AI areas can trade knowledge to/from LGNs?"; (2) an invitation to join **Kyushu University's
robotics group** (advisor Danilo) to work with a PhD student on **drones**, possibly real ones —
"how to apply LGNs there?"

Builds on all prior scouts, especially [15_rl_lgn_scout.md](15_rl_lgn_scout.md) (the RL verdict:
recurrent-LGN-as-POMDP-memory = a P2 capstone) and [11_paper2_workmap.md](11_paper2_workmap.md).

---

## Headline

**Cross-pollination: no clean standalone new-method trade exists (~conf 0.8).** LGNs' home
clusters are ETH-saturated (W3) or substrate-mismatched (W4), and every "obvious" import is killed
by a theorem/substrate wall or freshly occupied (SSM parallel-scan, CAM, KAN, GNN, diffusion,
neurosymbolic, BNN tricks). **The only knowledge worth importing flows ONE way and HARDENS P2, it
doesn't become its own paper:** (1) **hardware formal-verification tooling** (model checkers +
sequential equivalence checking) — the load-bearing fold-in; (2) **adversarial-robustness tooling**
(the low-friction entry into the Kyushu lab); (3) **distillation** (the enabling method for the
drone demo).

**Drones: CONDITIONAL-GO** — join the collaboration; it is the *physical home of the P2 capstone*
(a robotics/systems demo), **not** a standalone method paper and **not** "DWC on a robot." 100%
gated on P2's latch training working.

**⚠ The most important strategic finding — a new #1 competitor: ISTA.** The lab of **Kresse,
Lampert & Henzinger (ISTA)** now owns three pieces of *exactly P2's frontier*: **DWC** (weightless
control, arXiv:2512.01467), **"Logic Gate Neural Networks are Good for Verification"** (feedforward
LGN → SAT verification, arXiv:2505.19932, **NeuS 2025 Disruptive Idea Award**), and **LGN
connectivity** (arXiv:2507.02585). Their stated future work is "larger nets / more formal tools."
**The sequential + verification + control extension P2 is aiming at is ISTA's obvious v2.** The
race is no longer just ETH-Wattenhofer — **ISTA is arguably ahead on the control+verification axis.**
See §B.

---

## A. Cross-pollination landscape (ranked, grouped by disposition)

**PURSUE (1) — the one import worth building, but only as a P2/P3 section:**

| Area | Dir | Verdict / one-line |
|---|---|---|
| **Sequential/temporal formal verification** of the clocked (P2) LGN — BMC / IC3-PDR / LTL-CTL on the AIGER netlist via ABC/nuXmv | TO | **PURSUE (0.7).** P2's clocked latch-LGN is an **AIGER-native FSM** → drops zero-gap into stock hardware model checkers for **exact** LTL/CTL safety+liveness certificates — *impossible* for real-valued controllers, *meaningless* for the feedforward LGNs ISTA already verified. This is P2's differentiator + the drone safety hook + the Vargas handshake in one. Push-button MC alone is thin → pair with counterexample-guided repair or the drone cert. **ISTA owns the feedforward flag → move with urgency.** |

**WATCH (2) — fold-ins / vehicles, not papers:**

| Area | Dir | Verdict / one-line |
|---|---|---|
| **Adversarial robustness / attack surface** of recurrent LGNs (the Vargas hook) | BOTH | **WATCH (0.65).** The "no inference gradient = new threat" premise is **gradient-masking** (BPDA/transfer on the training relaxation defeats it, arXiv:1802.00420); bit-flip is owned by the DWN team (2603.22770). But Vargas's DE/one-pixel toolkit drops onto an LGN **near-free** → the **entry vehicle** into his lab, as a P2/P3 robustness section. Empty sliver: *targeted 16-way gate-choice corruption on a recurrent netlist.* |
| **Distillation** teacher(PID/MPC/RL)→recurrent-LGN | TO | **WATCH (0.68).** The reliable, RL-trap-avoiding **method** that makes the drone capstone actually train. High utility as a vehicle, ~0 as a contribution (DWC + Petersen-2024a cover imitation-into-logic). |
| **tinyML / MCU deployment** (bit-packed sequential-LGN on the flight controller) | FROM | **WATCH (0.6).** Not a trade — LGN *is* tinyML; the value is the **deployment path** + borrowable thermometer/unary sensor front-ends (shared with DWN) to get real sensor bits into gates. |

**CITE-ONLY (fold a sentence / cite; don't build):**

| Area | one-line |
|---|---|
| **Automata / DFA / FSM extraction** | Exact extraction is **FREE** (discrete state = read a truth table + classical Hopcroft/ABC minimization) and **owned** (Koul QBN→Moore 2019 arXiv:1811.12530; Wang & Niepert state-regularized RNN; R-DTLGN names it). Value = *execute the model-check* (→ verification), not write the extraction paper. **Swap Tomita for parity/mod-counting/Dyck in P2 eval** (Tomita is a memoryless benchmark). |
| **Sequential EDA / SAT-equivalence checking of "the discretization gap"** | Category error — the hard net = argmax-of-the-soft-model **by construction**, nothing to certify; the real gap is a continuous accuracy delta, not a Boolean-equivalence property. Sequential equivalence checking of the clocked circuit folds into verification; drop the "certified gap" framing. |
| **Associative / CAM / Hopfield** (Angle #4) | Match = XNOR+popcount+threshold is a **fixed optimal textbook circuit** (nothing to learn); "stored patterns" = the P2 latch bank (storage); learnable cleanup = **DiffLogic-CA's** damage recovery (2506.04912); DWN/WiSARD/LogicWiSARD already synthesize trained associative memory to raw gates. Survives only as a P2 "content-addressable" latch read-port. |
| **Analog / in-memory / memristor stateful logic** | Multi-step **write-bound** (IMPLY-NAND = 3 writes) serializes a deep LGN into millions of write cycles — the inverse of single-pass <10 ns eval; endurance/variability make it a *worse* latch than an FPGA register; netlist→crossbar mapping is owned EDA (SIMPLER-MAGIC, STAR). One future-work sentence in P3. |
| **Boolean-function analysis (Fourier/Walsh)** (Angle #3) | Method layer double-occupied (Sinkhorn 2601.13953 + WARP-LUTs 2510.15655). Companion **analytical lens** for accuracy ceilings only. |
| **Tsetlin machines / LUT-NN family (DWN/PolyLUT/NeuraLUT)** | Competing edge-logic **substrates**, not donors (clause-feedback / EFD-gradient don't booleanize onto softmax-over-16-gates; W4/W3). Cite as bake-off baselines. Genuine adjacent cite: **logic/LUT nets are far more bit-flip resilient than arithmetic NNs** (2603.22770) — for the drone robustness story. |

**NO-GO (traps & dead lanes — do not pursue):**

| Area | why dead |
|---|---|
| **SSM / Mamba / linear-attention "carry-lane"** (the seductive trap) | **Killed by a theorem:** parallel-scan needs linear/commutative recurrence, which "Illusion of State" (arXiv:2404.08819, ICML'24) + 2603.01959 prove is stuck in **TC⁰ and cannot track state** (non-negative diagonal SSMs can't even do parity). Over GF(2) the scannable lane degenerates to an **LFSR/rule-90 CA** DiffLogic-CA/R-DTLGN already have; HiPPO decay is magnitude-less-bit-incoherent (W4). **RDDLGN has publicly staked "associative blocks → O(log n) training."** Only a hand-designed shift-register lane survives as a minor P2 embellishment. |
| **GNN — LGN as message/aggregation** | **W1 structural kill:** permutation-invariant variable-degree aggregation = a symmetric Boolean fn needing O(n·log n)-depth popcount trees per node → destroys the latency/sparsity win. DiffLogic-CA works only on a fixed-degree lattice. Binarized-GNN/FPGA accelerators own the cheap-message axis. |
| **Discrete/masked diffusion — LGN as denoiser** | **W4:** reverse steps are real-valued categorical kernels you must sample; good masked decoding needs a real-valued confidence head → reintroduces exactly what LGNs avoid. Strip it → collapses to a recurrent LGN = P1/P2 + DiffLogic-CA. |
| **LGN as mechinterp ground-truth-circuit benchmark** | Occupied (Tracr, InterpBench, MIB, Formal-MI) and they use **transformers on purpose** because the methods fight superposition — an LGN's "no superposition" *deletes* the pathology → ~0 external validity; wrong consumer. |
| **Neurosymbolic / ∂ILP / DeepProbLog / NLM / LNN** | **W4:** LGN = fixed-arity *propositional* Boolean fns; these need *first-order* relations/quantifiers/probabilistic facts. |
| **Efficient-ML tail** (BNN/XNOR-QAT, pruning/LTH, **KAN**, soft/oblique trees, causal abstraction, concept-bottleneck, VQ-VAE, HDC/VSA, federated/meta/SSL, continual, symbolic-regression, SAT-as-a-model) | Consolidated dead lane: STE/pruning in ETH's lane (W3); **KAN→FPGA-LUT is done** (KANELÉ, FPGA'26 Best Paper); soft-tree distillation owned by Yue&Jha + structurally redundant (an LGN is already a Boolean tree); the rest W4-mismatched or occupied-by-analogy. |

---

## B. ⚠ Strategic bombshell — ISTA is now P2's #1 competitor

The scout's most consequential finding is a **landscape shift**, not a new idea. Three papers,
one lab (**Kresse, Lampert, Henzinger, ISTA**):
- **DWC** (arXiv:2512.01467) — weightless-logic RL control policies, FPGA, ~2 nJ/action. *Owns
  "LGN as controller."* Its stated limitation: "training only feasible in simulation, not on-device."
- **"Logic Gate Neural Networks are Good for Verification"** (arXiv:2505.19932, **NeuS 2025
  Disruptive Idea Award**) — LGN discreteness → a SAT encoding ~10³× faster than α,β-CROWN for
  robustness/fairness. *Owns "LGN is verification-friendly."* **Feedforward only** — stated future
  work: extend to larger nets / more formal tools.
- **LGN connectivity** (arXiv:2507.02585) — *also* in the owned wiring lane.

**Implication for P2.** The two things this scout says are P2's *only* worthwhile imports —
**(i) sequential/temporal verification** and **(ii) recurrent control** — are the *exact*
extensions ISTA is one step from. They own the feedforward version of both and the award flag.
**This raises P2's scoop risk from "ETH might extend RDDLGN" to "ISTA is actively building the
control+verification stack and the sequential case is their obvious v2."** Two takeaways:
1. **The recurrent/clocked-sequential angle is P2's moat** — it's the one thing feedforward-DWC and
   feedforward-2505.19932 structurally *cannot* claim. Lean into it hard (latch = state = the object
   that makes temporal verification meaningful and POMDP memory possible).
2. **Add a tripwire** on the ISTA/Kresse–Lampert–Henzinger arXiv feed alongside the existing
   Bührer/Wattenhofer one. Time-to-publish on the sequential-verification + recurrent-control
   result now matters.

---

## C. Drones / Kyushu — the actionable part

**Advisor context (verify — agent-surfaced):** "Danilo" is almost certainly **Danilo Vasconcellos
Vargas, Laboratory of Intelligent Systems (LIS), Kyushu University** — evolutionary computation,
RL, and **adversarial robustness** (the *one-pixel attack*, arXiv:1710.08864). That makes the
**robustness + formal-verification** angle a *direct handshake* with his research agenda, and it
explains why robotics/drones is the offered vehicle. (Keep the plan valid regardless of the exact
identity.)

### Verdict: CONDITIONAL-GO — the physical home of the P2 capstone, not a standalone paper

**Fit is textbook.** LGN's nJ/action, native FPGA/LUT mapping, binary I/O, deterministic timing,
and formal-verifiability match **nano-UAV control**, where the binding constraint is
**energy/mass/area and safety**, not accuracy. Crucially, **the real-hardware path already exists
and flies**: the Crazyflie **Lighthouse-FPGA Artix-7 XC7A15T** deck (arXiv:2403.18703) runs a
learned NN flight controller **on the same silicon DWC synthesizes for** → sim-to-real is de-risked
to "swap the bitstream." The one ingredient the whole LGN-control line lacks — a **real drone +
flight arena + a co-driving PhD student** — is exactly what the group supplies.

**The defensible novel angle is NOT the substrate swap.** It is the empty intersection **{stateful
difflogic latch (P2)} × {real drone with GENUINE partial observability} × {belief-state in clocked
FPGA registers}**: a *recurrent* logic controller that holds belief-state under **wind-gust history,
dropped/delayed IMU, target occlusion, or single-motor degradation** — where feedforward DWC
structurally **cannot** hold state and provably degrades. Bolted to that: **exhaustive formal/
temporal verification** of the deployed clocked netlist (model-check a geofence/attitude/liveness
envelope with ABC/nuXmv) — the thing R-DTLGN and DWC only *named* as future work, that SNN/DNN
controllers structurally cannot offer, and that maps 1:1 onto Vargas's robustness agenda. **Frame
on energy/area/determinism/verifiability — never "faster than a PID"** (a 200–1000 Hz control loop
is not latency-bound; 100 MHz throughput is a W2 trap).

### Entry plan — sim→real ladder (distill from a teacher throughout; NEVER RL-on-a-real-drone)

| Rung | What | Who / when | Novelty |
|---|---|---|---|
| **0** | Feedforward LGN attitude/rate stabilization in `gym-pybullet-drones`/PyFlyt (Crazyflie model), **imitation-distilled from stock PID**; validates the thermometer sensor-encoding front-end + booleanized controller | solo, **now** on 2080S/DUST | ~0 (de-risk) |
| **1** | Synthesize via P2's sequential-RTL emitter → FPGA/MCU; **measure** latency/energy/determinism vs the FPGA-DNN thrust controller (2403.18703) + PID | solo | engineering/measurement (DWC territory) |
| **2** | **THE publishable rung:** introduce the **recurrent latch-LGN** on a genuine-hidden-state task (gust/motor-fault adaptation or sensor dropout) where feedforward provably fails and the latch holds belief-state; sim, multi-seed | solo→student | **the P2 capstone** (robotics venue IROS/ICRA/RA-L, or P2's capstone §) |
| **3** | **Model-check** safety/liveness of the deployed netlist + adversarially certify the input→action map | solo + Vargas | the Vargas differentiator |
| **4** | **Fly it** on the Crazyflie Artix-7 deck with a PID safety-supervisor fallback | group's drone + PhD | the wow-demo/validation |

*Fallback:* hardware-in-the-loop on the FPGA deck driven by sim is a strong result; the
memory-vs-feedforward POMDP **sim** result (Rung 2) **stands alone even without flight**. Run
Rungs 0–1 immediately to de-risk *while P2's latch matures*.

### Risks
1. **Scoop race** — ISTA (DWC + verification) is one step from a real-drone / recurrent / sequential-
   verification follow-up; ETH (RDDLGN/BitLogic "stateful modules") and the **SNN-on-real-nano-drone**
   camp (2411.13945; TU Delft *Science Robotics* 2024) also crowd it. **Time-to-flight matters.**
2. **Sim-to-real brittleness** of hard thermometer binarization + argmax gates + latched feedback on
   a safety-critical loop — only stress-tested vs synthetic Gaussian noise so far; vibration, IMU
   bias/latency are a different regime.
3. **Compounded training risk** — BPTT-through-discrete-latch + sim-to-real + closed-loop stability,
   three hard things at once.
4. **Wrong-bottleneck (W2)** — if pitched as "speed," reviewers kill it. Pitch **energy/area/
   verifiability**.
5. **Lab-fit** — Vargas's LIS leans RL/evolutionary/robustness, **not FPGA**; the P3 hardware+
   verification half may need scaffolding, and the lab's easy default (a sim-RL policy) is precisely
   the low-novelty DWC-like one. **Keep evolution/novelty-search as a one-table ablation only** (the
   evolution-of-logic-circuits lane is already ruled dead — [15](15_rl_lgn_scout.md)).

### Remaining gate (the single blocker)
**P2's learnable latch/flip-flop must actually TRAIN.** Pass a **tiny-sim-POMDP trainability check**
(BPTT-through-discrete-feedback + closed-loop stability, e.g. a T-maze / memory-recall toy) **BEFORE
any drone commitment.** Both novel angles (recurrent memory, verified safety) are 100% gated on the
latch working — if P2's feedback-training (its own §C2 obstruction) stalls, there is no drone paper.
Rungs 0–1 are safe to run in parallel; do not let the group's real drone carry Rungs 2–4 until the
gate clears.

### How to pitch it to the group ("a recurrent logic-gate flight brain")
An ultra-low-energy, FPGA-native, **interpretable-and-verifiable-as-a-circuit** nano-drone
controller — positioned as the **deterministic, synthesizable, formally-verifiable alternative to
the neuromorphic-SNN controllers** the field is currently excited about (SNN's *digital twin* in the
same energy niche, but **model-checkable on a commodity FPGA — no Loihi needed**). Lead with the
three differentiators the incumbents lack **together**: (a) **first real-drone weightless controller
with MEMORY** (DWC is sim-only, feedforward); (b) **belief-state via recurrent LGN** for partial
observability (all LUT-NN/DWN are memoryless); (c) **exhaustive formal verification / adversarial
certification** of the deployed circuit — extending Danilo's robustness/one-pixel lineage from
*perception* to the *control substrate*. **Division of labor:** the group brings the real drone,
flight arena/mocap, sim-to-real pipeline, and the co-driving PhD; you bring the LGN + recurrent
training stack (P1 done, P2 latch), the sequential-RTL emitter, FPGA-synthesis, and the verification
demo. Offer Rungs 0–1 as immediate no-risk solo deliverables; co-scope Rungs 2–4 with the student.

---

## D. Bonus must-cite prior art (new this scout; verify 2026 IDs before citing)

- **arXiv:2505.19932** — "Logic Gate Neural Networks are Good for Verification" (Kresse, Yu, Lampert,
  **Henzinger**, ISTA, NeuS'25 award). *The* feedforward LGN-verification paper → the base P2's
  sequential verification extends, and the scoop threat. **Highest-priority new citation.**
- **arXiv:2603.22770** — DWN team (UT Austin/UFRJ): logic/LUT nets are far more **bit-flip resilient**
  than arithmetic NNs (drone/edge robustness; names "structured fault models" as its own future work).
- **arXiv:2403.18703** — FPGA-Based Neural Thrust Controller for UAVs (Crazyflie + Artix-7 XC7A15T,
  **real flight**). The hardware platform/template that de-risks the drone capstone's sim-to-real.
- **arXiv:2411.13945** / *Science Robotics* 2024 — TU Delft fully-**neuromorphic** drone (Loihi,
  real flight). The competitor to position *against* ("SNN's verifiable digital twin").
- **arXiv:2404.08819** — "Illusion of State in SSMs" (ICML'24). The theorem that kills the
  SSM-parallel-scan trap — cite it in P2 to explain why the associative-scan sirens (incl. RDDLGN's
  own future-work direction) don't buy structured memory.
- Also: **Koul QBN 2019 (1811.12530)** + **R-DTLGN (2605.24649)** for the "extraction is free/owned"
  point; **Petersen 2024a** (already in workmap) for imitation-into-logic.

---

## Bottom line / steer

**Don't open a new cross-pollination lane** — the honest landscape verdict is that LGNs' home
clusters are saturated (W3) or substrate-mismatched (W4), and every tempting import (SSM scan, CAM,
GNN, diffusion, KAN, neurosymbolic) is theorem- or occupancy-dead. **Pour effort into P2's latch
trainability and its VERIFICATION story** — the one import that hardens P2, is P2's moat vs
feedforward-ISTA, and is the drone safety hook. **Take the Kyushu offer** as the physical home of the
P2 capstone (recurrent memory + verified safety on a real nano-drone), framed on energy/
verifiability, distilled from a teacher, and **gated on a tiny-POMDP latch-trainability check first**
— with Rungs 0–1 as safe immediate solo work. **Treat robustness (Vargas's tooling) as the low-
friction lab entry, distillation as the delivery method, everything else as cite/skip.** And
**update the competitive picture: ISTA (Kresse/Lampert/Henzinger) is now the front-runner on
control+verification — the recurrent/sequential case is the ground you must take first.**

## How to apply
Don't re-scout these areas. When P2's latch trains, (1) build the sequential-verification demo
(cite 2505.19932, distinguish feedforward-vs-clocked), (2) start the drone Rungs 0–1 now to de-risk,
(3) hold Rungs 2–4 for after the trainability gate + in collaboration with the Kyushu student.
Add the tripwire on the ISTA arXiv feed. Keep evolution as a one-table ablation, never the method.
