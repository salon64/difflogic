# 13 — SNN / Hebbian ideas × LGN: Novelty Scout

**Scouted:** 2026-07-01 (multi-agent: 6 blind literature finders → consolidate → adversarial
kill-pass per surviving idea; arXiv IDs marked *(verified)* were checked live this session).
**Trigger:** "I've been learning about spiking neural networks (SNNs), event/temporal coding,
and Hebbian learning ('wire together, fire together') — brain-like nets. Could those ideas
apply to LGNs?" Exploratory scout: does any transfer *work* and is it *novel*, or is it a
dead-end / cite-only framing / genuine future work? Must **not** divert P1/P2 unless it
strengthens them.

Builds on [09_training_speed_scout.md](09_training_speed_scout.md) (training-lane = ETH-
saturated), [11_paper2_workmap.md](11_paper2_workmap.md) (P2 latch anchor + C2 obstruction),
`lgn-recurrent-scout-verdicts`, `lgn-landscape-key-facts`.

---

## Verdict (headline)

**NO-GO as a new research direction (confidence ~85%).** Every SNN/Hebbian transfer either
(a) **relabels** the sequential/recurrent LGN you already build (a bit going high at step *t*
*is* an event; BPTT-over-timesteps already owns the temporal axis), (b) lands in a lane
already **owned** (ETH training/reparam **W3**; connectivity-learning **W3**; the "spiking
logic gate / spiking latch" vocabulary is taken by biocomputation, in the reverse direction),
or (c) hits the **substrate wall W4** — SNN/Hebbian mechanisms (membrane potentials, spike
timing, eligibility traces, adjustable synaptic weights) live on *real-valued* state that
does not exist in a gate-choice unit whose only parameter is a categorical over 16 gates. No
spiking/Hebbian idea earns a paper slot, and none should pull effort off P1/P2.

**But the scout paid off twice — precisely because it forced a systematic sweep of the whole
non-BPTT trainer family (EqProp, e-prop, RTRL, predictive coding, target-prop), which is the
exact family C2 makes claims about:**

1. **★ It red-teamed your own C2 (the P2 "moat" theorem) and found a real scoping/tension bug
   + 4 missing citations.** Highest-value output. It hardens C2 and pre-empts the single
   sharpest reviewer objection available against P2. See **§ The real payoff**.
2. **3 bonus prior-art finds newer than the last scout (2026-06-11), all touching P1/P2:** a
   *verified* gap-closing paper (2603.14157), a **third** recurrent-LGN entrant (R-DTLGN
   2605.24649) whose "stability of recurrent LGN" framing brushes C2's turf, and a
   multilinear-polynomial-fitting LGN paper (2605.08657) adjacent to C2's reduction. See
   **§ Bonus prior art**.

**One SNN idea is already, correctly, in P2** — the *surrogate-gradient parallel*. SNNs are
non-differentiable (spike = step) and trained with surrogate gradients; LGNs are non-diff
(discrete gate) and trained with a relaxation/STE — the same problem shape. The workmap
already imports this (§Tier-3 justification (ii), citing Neftci/Mostafa/Zenke 2019 and Zenke &
Vogels 2021). The scout confirms that is the **right scoping**: surrogate-gradient transfer is
**cite/companion infra, not a contribution** (the "close the gap via training method" lane is
ETH's — W3). Nothing to add there beyond what's already in the workmap.

---

## Status board

| Idea | Disposition | Conf. killed | Wall | What to do with it |
|---|---|---|---|---|
| Spiking / event-driven LGN | **NO-GO (relabel)** | 0.90 | W2/W4/W1 | 1 cite-only ¶ in P2 related-work |
| Hebbian *trains* the LGN (gate-select **or** wiring) | **NO-GO** | 0.85 | W3 + owned + credit-assignment | cite Oliveira'93; MI-attribution = minor P2 interpretability ablation at most |
| e-prop / RTRL / STDP local temporal credit → recurrent LGN | **FOLD → C2 / FUTURE** | 0.90 | W2 (+W4, W3) | its *failure feeds C2*; park SnAp-on-sparse-wiring as future work gated on a **measured** BPTT wall |
| Target-Prop-Through-Time (backprop-free trainer) | **NO-GO** | 0.88 | W4 (non-invertible gates) + W2 | cite-only *foil* in C2 |
| Neuromorphic energy / async LGN | **NO-GO / cite** | 0.85 | W2/W4 | 2-story energy ¶ in P3 motivation; park "async LGN" as a 1-line future note |
| Learnable-delay / temporal-code primitive | **FOLD-INTO-P2** | 0.90 | W4/W1 + subsumed | 1 sentence (delay = D-FF chain), cite DCLS |
| **★ C2 red-team (moat hardening)** | **ACT NOW** | — | — | **re-anchor C2 on non-uniqueness; scope the singularity; +4 cites — see below** |

*(Walls: W1 linearity-vs-expressivity; W2 wrong-bottleneck; W3 lane-saturation/ETH; W4
substrate-mismatch — real-valued-neuron mechanisms don't booleanize onto gate-choice units.)*

---

## Per-idea detail

### 1. Spiking / event-driven LGN — NO-GO (relabel), conf 0.90
The adversarial question answers itself: an LGN is **already 1-bit, already gate/event-native,
already cheap**, so the three things "spiking" normally buys all evaporate. **(a) Energy
sparsity:** a logic gate/register already burns dynamic power only when it toggles; clock-/
signal-gating is native CMOS, not a new algorithm (W2). **(b) TTFS/latency coding:** needs a
*real-valued magnitude* to compress into a spike time — an LGN bit has no magnitude (W4/W1).
**(c) Rate coding:** strictly worse than an exact 1 bit (adds latency for no gain). And "spikes
over time" in a binary net is *definitionally* the recurrent/sequential LGN you already build —
RDDLGN (2508.06097) + P1/P2 own the temporal axis. Two adjacent lines are already occupied and
any P2 phrasing must distinguish sharply: **Boolean gates *from* spikes** for biocomputation
(AAAI 2020; npj Unconv. Computing 2026 already does spiking latches/registers) and the
**BNN↔SNN bridge** (Lu & Sengupta, Front. Neurosci. 2020). → **Harvest:** one related-work
sentence in P2 — *"why not spikes: our sequential bits already are events; why not Hebbian:
there is no synaptic weight to update, only a categorical over 16 gates."*

### 2. Hebbian *trains* the LGN — NO-GO, conf 0.85
Split by the two honest mappings of "wire together, fire together":
- **(b) "wire together" = correlation-driven connectivity/wiring learning — OWNED.** Now a
  *crowded* lane: LILogic (2511.12340), Mommen (2507.06173), Operand-Selective LGN (OSLGN),
  "Scalable Interconnect Learning in Boolean Networks" (2507.02585). Dead.
- **(a) "fire together" = correlation/MI selects the *gate*.** The one place discreteness
  *helps* (the categorical gate parameter has an exact closed-form local optimum = argmax
  MI/corr over the 16 gates given a local target — dodges W4). **But it reinvents Oliveira &
  Sangiovanni-Vincentelli, NeurIPS 1993** (greedy MI-maximizing Boolean-function learning; the
  ID3 / logic-synthesis lineage), it needs an *external* local-target mechanism (Forward-
  Forward / local-loss / target-prop — a large crowded real-valued lane), and pure unsupervised
  Hebbian **collapses**: deep hidden gates have no teaching signal (the exact credit-assignment
  problem backprop already solves), and forgoing cross-layer co-adaptation loses accuracy and
  *widens* the discretization gap. Also collides with Light DLGN's AND-OR/residual init (W3),
  which needs *no* data and already biases toward the same canalizing gates an MI-argmax would
  pick. → **Harvest:** at most, MI gate-attribution as a *post-hoc interpretability/ablation
  probe* in P2 (largely duplicates the planned FSM/L\* read-out) + a defensive Oliveira'93 cite.
  Not a trainer, not a paper.

### 3. e-prop / RTRL / STDP — local temporal credit assignment for the recurrent LGN — FOLD → C2, conf 0.90
The most *seductive* angle (it targets P2's real pain, BPTT vanishing/exploding gradients),
and the source of the C2 payoff below. The clean transfers fail on **substrate and on purpose**:
eligibility traces are real-valued low-pass filters of membrane potential/synaptic current —
neither exists in a gate-choice unit (W4) — and e-prop / every bounded-rank online estimator
works by **dropping the recurrent-Jacobian term**, which is *exactly the long-range-memory term
P2 is built to preserve*. The one genuinely LGN-specific hook nobody has stated: **fan-in-2
fixed wiring makes the RTRL influence matrix structurally near-diagonal**, so SnAp-1's
approximation (lossy for dense RNNs) could be near-exact here. But it is **W2-fatal now**: P2's
benchmarks (copy T≤50, psMNIST-784 already chunked) fit BPTT in memory — the measured pain is
the *discretization gap* (a forward-pass issue C3 targets), not BPTT cost; and 2603.15195
(2026, *agent-surfaced — verify*) argues the online-learning redundancy is near-*isotropic*, not
near-diagonal, undercutting the hook. → **Park** SnAp-1-on-sparse-wiring as explicit post-P2
future work, revisited **only if** a measured BPTT-cost/vanishing wall appears on the long
benchmarks (psMNIST-784, adding-problem). Its *failure* feeds C2 (§ below).

### 4. Target-Propagation-Through-Time (TPTT) — NO-GO, conf 0.88
The one bio-plausible trainer that is *not* a fixed-point/implicit method (so it sidesteps C2)
and natively passes signal through hard/discrete steps. Killed anyway by **W4**: TP is defined
via *learned real-valued inverse autoencoders*, and a layer of fixed-wired 2-input Boolean
gates is **maximally non-invertible** (AND/OR collapse preimages) — the exact regime the TP
literature says vanilla TP fails; booleanizing a learned inverse onto categorical gate-choice
units is unproven. **W2**: the discretization gap is a *forward-pass* property — 2603.14157
*(verified)* shows Hard-ST forward yields **zero selection gap with any backward estimator** —
so a *backward* method like TP cannot be the gap-closer it's pitched as. TP also doesn't scale
past MNIST (Bartunov et al. 2018) and LRA-RNNs still vanish (2504.13531, *agent-surfaced*). →
**Harvest:** cite TPTT (Manchev & Spratling, JMLR 2020) and Difference Target Prop (1412.7525)
as a *foil* in C2 — "even the one non-implicit bio-plausible trainer re-meets the memory
obstruction as inverse non-injectivity at overwrite events."

### 5. Neuromorphic energy / async LGN — NO-GO / cite, conf 0.85
Apples-to-oranges (W2 + W4). The SNN energy win (Loihi/SpiNNaker) is **sparsity-driven** —
energy scales with #spikes, harvested from the **MAC + weight-memory** bottleneck (Delta
Networks, DeltaKWS). An LGN has **neither MACs nor weight memory** (it eliminated them by
construction), so delta/event-driven skipping has nothing to save — and the per-node
change-detection (XOR-vs-previous + register + enable) to *skip* a ~fJ gate **costs more than
evaluating it**. The one genuinely prior-art-free adjacency — an **asynchronous/clockless
(QDI) sequential LGN** — is a *hardware-design* contribution (no learning novelty), roughly
doubles gates (dual-rail + completion detection), forfeits the deterministic <10 ns settle
guarantee, and **conflicts with P2/P3's deliberate clocked-FPGA-register mapping** (BitLogic
2602.07400 is the field doubling down on *synchronous dense* logic). → **Harvest:** an honest
two-column **energy-story paragraph/figure in P3's motivation** (SNN = sparse/temporal/stateful,
~mJ–nJ, ~ms latency; LGN-sequential = dense/one-shot/clocked, ~tens-of-ns) to pre-empt the
"why not neuromorphic?" reviewer at zero risk. Park "async LGN" as a one-line P3 future note.

### 6. Learnable-delay / temporal-code primitive — FOLD-INTO-P2, conf 0.90
A trainable *k*-step delay line *is* a chain of *k* D-flip-flops — already inside P2's
primitive set — so it is not a separable contribution. Learnable delays are owned in the SNN
world (DCLS, ICLR 2024, 2306.17670; DeNN 2501.10425), where their value is spike-coincidence
detection over *real firing times* (W4); Maass & Schmitt 1999's "delays add expressivity"
theorem is about *continuous-time* spiking neurons and vanishes on a clocked binary LGN. Pure
delay+XOR chains are **GF(2)-linear** (shift registers/LFSRs) and collapse the nonlinearity
(W1). → **Harvest:** one sentence in P2's gate-set discussion (delay = D-FF chain; cite DCLS +
Maass-Schmitt; use W4 to explain why TTFS latency-coding has no LGN analog).

---

## ★ The real payoff — a red-team pass on C2 (P2's moat)

**Context.** C2 (workmap §Tier-3, lines 136–146 & 175–178) claims: *perfect 1-bit memory ⇒
unit Jacobian eigenvalue ⇒ `(I−Jₚ)` singular ⇒ IFT/DEQ/Almeida–Pineda ill-posed*, with
non-uniqueness and the separatrix `ρ(Jₚ)>1` as secondary points. Sweeping the **entire**
non-BPTT trainer family surfaced **three issues + four missing citations**. *None kills C2* —
but fixing them converts a challengeable headline into an airtight one, at the cost of ~a page
of rewrite and **zero new experiments**.

**Issue 1 — the singularity is scoped to the *lossless/identity* hold, which is P1's carousel,
and is in tension with C3.** The workmap's own derivation (line 137) gets `(I−Jₚ)=0` by taking
the ideal NOR–NOR hold to be the **identity** map `q←q` (eigenvalue *exactly* 1). That
identity-hold is mathematically the **marginal constant-error carousel** — the workmap itself
labels `∂Q⁺/∂q = 1 in hold (carousel)` on line 157 — i.e. the **P1** object, not a *restoring*
bistable well. But C3's selling point (lines 24–31) is that the latch **restores** a drifted
soft value to a clean {0,1} each step (that is *how* it "closes the discretization gap gating
could not"). A restoring latch is a **contraction toward its wells**, eigenvalue **|λ|<1**,
where **`(I−Jₚ)` is invertible and the implicit gradient EXISTS** (Bai et al. 2021,
arXiv:2106.14342 *(verified)*: a convergent equilibrium has spectral radius `ρ(J)<1`). So the
same primitive is described two ways — *marginal* (λ=1, singular → supports C2) and *restoring*
(|λ|<1, invertible → supports C3). A sharp reviewer pulls that thread: **"either the latch
holds as an identity, so it doesn't clean the bit (no C3 gap-closing); or it restores, so
`(I−J)` is invertible (no C2 singularity)."**

**Issue 2 — the airtight, regime-independent obstruction is NON-UNIQUENESS, which you already
have but bury.** Two coexisting stable states make the selected fixed point a **non-unique,
history-dependent, discontinuous-in-parameters** object — true for **both** the lossless and
the restoring latch, and it is the *published* EP/RBP limitation: Almeida 1987 / Pineda 1988,
and explicitly **Bal & Sengupta, IJCAI 2023, arXiv:2209.09626 *(verified)*: EP requires the
input static in both phases ⇒ LSTM/GRU-like sequential-state models are impossible under EP.**
The workmap states non-uniqueness (line 140–141, 177) but subordinates it to the singularity
headline. **Fix: promote non-uniqueness / basin-selection to C2's *primary* mechanism; demote
"`(I−Jₚ)` singular" to the marginal-hold (P1-carousel) + separatrix boundary cases.** This
reconciles C2 with C3 and removes the only real handle a reviewer has.

**Issue 3 — don't universalize to "every non-BPTT trainer."** The empirical line ("nobody uses
fixed-point gradients; everyone uses unroll-BPTT," 143–146) is fine. But *every cheap/local
trainer fails* is false: **exact RTRL** carries the full recurrent Jacobian and is **not**
blocked (Zucchet et al. 2023, arXiv:2305.19044 *(verified)* — only its O(n⁴) cost rules it
out, not degeneracy); **unrolled temporal predictive coding** approximates *truncated BPTT*
(Tang & Millidge 2024), so it isn't cleanly in the equilibrium class. Keep the claim as *"the
**fixed-point / implicit / equilibrium** family is ill-posed at a persistent bistable memory"* —
true and defensible — not *"all non-BPTT methods fail."*

**Bonus — your "cheap negative experiment" is largely already published.** Workmap line 183–184
plans to *show* the DEQ/implicit gradient destabilizes on a 1-bit latch. **Laydevant, Marković
& Grollier, "Training an Ising machine with EqProp," Nat. Commun. 2024** already shows ON/OFF
(bistable) units resist equilibrium training. Cite it; a run on *your* SR primitive is still
worth one figure, but frame it as **confirmation, not discovery**.

**Citations to add to C2's D-refs (all verified real this session):**
- **Bai et al. 2021, arXiv:2106.14342** — Stabilizing DEQ via Jacobian regularization; the
  `ρ(J)<1`-at-convergent-equilibrium fact you must engage to defend the (corrected) C2.
- **Bal & Sengupta 2023, IJCAI, arXiv:2209.09626** — EP needs static input ⇒ the *published*
  "why not EqProp for sequential memory," and a *foil* (they build a Hopfield-attention
  workaround that *does* sequence tasks — cite carefully).
- **Laydevant et al. 2024, Nat. Commun.** — Ising-EqProp; bistable units resist equilibrium
  training (your negative result, pre-existing).
- **Zucchet et al. 2023, arXiv:2305.19044** — exact RTRL is non-BPTT yet not degenerate (scopes
  the universal).

**Net:** C2 survives and gets *stronger*. The corrected version — **non-uniqueness primary,
singularity scoped to the marginal/P1 boundary case, universality scoped to the equilibrium
family, +4 cites** — is harder to attack and now explicitly answers the "why not
EqProp/e-prop/predictive-coding?" question a reviewer will raise. This is the honest,
high-value output of the whole SNN/Hebbian scout.

---

## Bonus prior art (newer than the last scout; all touch P1/P2 — verify against your claims)

> **Update 2026-07-01: all three read in FULL — claim-by-claim comparison in
> [14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md).** Headline: none
> scoops P2's core; 2603.14157 is a *gift* (its selection/computation gap split frames C3),
> 2605.24649 is the closest neighbor (cite+distinguish on C3), 2605.08657 narrows claim-#2 to
> the characteristic-equation reduction. The "verify before citing" caveat below is now
> discharged for these three (verified real + read).

1. **"Align Forward, Adapt Backward: Closing the Discretization Gap in Logic Gate Networks"**
   — arXiv:2603.14157 *(verified)*. Proves the **selection/discretization gap is a FORWARD-pass
   property**: **Hard-ST forward → zero selection gap *by construction*, with any backward
   estimator** (proposes CAGE = Confidence-Adaptive Gradient Estimation; 98% MNIST / 58%
   CIFAR-10, zero gap across temperatures). **Action for P2/C3:** this closes the *per-step
   selection* gap (soft-mix-of-16 vs argmax-1); your C3 targets the *accumulated temporal-drift*
   gap over long sequences (soft/hard trajectories diverge step-by-step). They are
   **complementary, not competing** — but you must cite it and draw the distinction explicitly,
   or a reviewer thinks the gap is already solved. It likely also strengthens C3's framing
   ("per-step selection gap is closed by Hard-ST; the *remaining* long-sequence gap is temporal
   drift, which a bistable latch removes architecturally").

2. **"On the Stability and Realizability of Recurrent Polynomial Surrogate Ternary Logic Gate
   Networks" (R-DTLGN)** — arXiv:2605.24649 *(verified)*, submitted May 2026. A **third**
   recurrent-LGN entrant (after RDDLGN and DiffLogic CA): Kleene 3-valued {−1,0,+1}, recurrent,
   poly-surrogate → hardens to a ternary circuit, aimed at **runtime monitors in safety-critical
   systems**, with **provable degradation guarantees**. **Two flags:** (i) still *no latch
   primitives, no local learning* — trained by surrogate+unroll — so P2's gap holds; but (ii)
   its **"stability and realizability of *recurrent* LGN"** analysis brushes C2's turf. **Read it
   and confirm it does not pre-empt any part of C2's obstruction or C1's realizability claim.**

3. **"Fitting Multilinear Polynomials for Logic Gate Networks"** — arXiv:2605.08657 *(surfaced,
   title verified via search)*. Directly adjacent to C2's **multilinear characteristic-equation
   reduction** (your *only other* claimed-new item besides C2). **Read it** and confirm the
   multilinear-surrogate reduction you claim as new isn't anticipated.

*Minor landscape notes:* the connectivity-learning lane is now **≥4 papers** (LILogic, Mommen,
OSLGN, 2507.02585) — map-(b) is thoroughly dead. WARP-LUTs (2510.15655, Walsh-assisted LUT
relaxation) is a fresh data point for the parked Fourier angle #3.

> **Caveat on IDs.** Papers tagged *(verified)* were confirmed live this session. Items tagged
> *(agent-surfaced / verify)* (e.g. 2603.15195, 2504.13531, the AAAI-2020 / npj-2026
> biocomputation refs) came from the sub-agents' web+recall and should be checked before you
> cite them in a paper — treat them as leads, not facts.

---

## What to harvest (concrete, all low-cost, none diverts P1/P2)

- **P2 §C2 rewrite (do this — highest value):** re-anchor on non-uniqueness; scope the
  singularity claim; scope the universality to the equilibrium family; add the 4 citations
  above. Pre-empts the sharpest reviewer objection and answers "why not EqProp/e-prop/PC?".
- **P2 §C3:** cite **2603.14157** and distinguish *per-step selection gap* (closed by Hard-ST)
  from *long-sequence temporal-drift gap* (C3's target). Read **2605.24649** and **2605.08657**
  against C1/C2's novelty.
- **P2 related-work:** the one-sentence "why not spikes / why not Hebbian" framing (§1) + the
  Oliveira'93 defensive cite (§2) + TPTT/DTP as a *foil* (§4) + delay = D-FF-chain, cite DCLS
  (§6).
- **P3 motivation:** the honest two-story SNN-vs-LGN energy paragraph/figure (§5). Park "async
  LGN" as a one-line future note.
- **Future work (parked, gated):** SnAp-1 online training of a recurrent LGN on maximally-sparse
  fan-in-2 wiring — revisit **only if** a *measured* BPTT wall shows up on psMNIST-784 / adding.

## Reading-list additions (for [12_reading_sequential_memory.md](12_reading_sequential_memory.md))

Bai et al. 2021 (2106.14342, DEQ stability) · Bal & Sengupta 2023 (2209.09626, EP sequence
learning) · Laydevant et al. 2024 (Ising-EqProp, Nat. Commun.) · Zucchet et al. 2023
(2305.19044, RTRL limits) · Scellier & Bengio 2019 (1711.08416, EqProp≡RBP — *already in
workmap D-refs*) · Bellec et al. 2020 (e-prop, Nat. Commun.) · Menick et al. 2020 (2006.07232,
SnAp) · Manchev & Spratling 2020 (TPTT, JMLR) · Lee et al. 2015 (1412.7525, Difference Target
Prop) · Oliveira & Sangiovanni-Vincentelli 1993 (greedy-MI Boolean learning).

---

## Bottom line / steer

**Do NOT open an SNN/Hebbian sub-project** — the transfers are relabels (spiking = your
sequential LGN), owned lanes (Hebbian-wiring; training-method), or substrate-mismatched (W4,
no membrane/synapse to update). Keep the gating(P1) → latch(P2) → FPGA(P3) plan intact. The
scout's real deliverables are **(1) a red-team hardening of C2** — reframe on non-uniqueness,
scope the singularity, +4 cites — which strengthens P2's moat and pre-empts its sharpest
objection; and **(2) three must-check 2026 papers** (2603.14157 gap-closing, 2605.24649
recurrent-ternary-LGN stability, 2605.08657 multilinear-polynomial LGN) that touch C1/C2/C3
directly. Act on (1) at P2 write-up time; verify (2) before they become a race problem.

*ETH race note: the training/reparam/surrogate lane (W3) and BitLogic (2602.07400) sit behind
every SNN transfer here, and 2603.14157/2605.24649 read as more of the same group's output —
consistent with the standing tripwire on the Wattenhofer/Bührer arXiv feed.*
