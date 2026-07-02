# Paper #2 — Work Map: "Sequential Logic Gate Networks"

_Created 2026-07-01. The standing plan for Paper #2 (latch / flip-flop primitives —
Angle #2, the A\* anchor). Builds on [05_my_angles.md](05_my_angles.md) (Angle #2 GO
verdict), [06_paper_plan.md](06_paper_plan.md) (3-paper roadmap + the 4-way comparison),
[08_paper1_checklist.md](08_paper1_checklist.md) (what P1 established that P2 inherits),
[09_training_speed_scout.md](09_training_speed_scout.md) (parallel-scan → P2 only),
[10_fpga_scout.md](10_fpga_scout.md) (FPGA demo folds INTO P2), and the reading list in
[12_reading_sequential_memory.md](12_reading_sequential_memory.md). Synthesizes a full
codebase audit + a differentiable-feedback math sweep + a curated non-paper resource sweep
(2026-06-30 → 07-01)._

---

## A. The thesis, sharpened — READ FIRST (it changes the framing)

The prior framing ([06_paper_plan.md](06_paper_plan.md)) asks *"does **holding** state in a
latch beat **recomputing** it (RDDLGN)?"* — the right floor, but it undersells the paper.
Two findings (one from re-reading [`../seqlgn/cells.py`](../seqlgn/cells.py), one from the
math sweep) point to a sharper, higher-tier thesis:

1. **`gated` (P1) is *already* a soft gated D-latch.** `h' = s·h + (1−s)·c` **is** the
   multiplexer-hold `Q⁺ = e·D + (1−e)·Q`. So "latch vs rddlgn" partly re-runs a fight P1
   already won. The unclaimed fight is **`latch` vs `gated`**: a *bistable* hold vs a
   *soft-multiply* hold.
2. **The bistable hold closes the discretization gap that gating could not.** P1's one
   remaining wall was copy-50: soft solves (0.88), discrete stuck (0.37, **gap +0.50**),
   written off as "a general difflogic property, orthogonal to gating." It is **not**
   orthogonal — it is *caused by soft-multiply drift*: with `s<1`, `s·h` bleeds the held
   value off {0,1}, so the soft and hard trajectories diverge over time. A **true bistable
   register restores the state to a clean bit every step** → zero drift → the gap collapses.
   The math sweep confirms the recommended latch's forward pass settles to an exact bit,
   giving **"zero discretization gap at inference."**

   **Formal backing (Kim 2026, arXiv:2603.14157 — full read
   [14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md)).** Kim splits the
   train/inference gap into a **selection gap** (which-gate mismatch; closed by *Hard-ST
   forward* training, for any backward temperature) + a **computation gap** (soft-vs-hard
   *values* for the chosen gate; irreducible by training, provably **=0 iff inputs are
   binary**, and 50–75% for values near the 0.5 boundary). Our copy-50 +0.50 is a
   **computation gap** — the tell is that it **grows with sequence length** (0 at L20 → +0.50
   at L50; a selection gap is *length-independent* for a shared recurrent cell). The bistable
   latch **re-binarizes the recurrent state each step ⇒ recurrent inputs become binary ⇒
   computation gap → 0, architecturally** — the one thing Kim's training-side methods cannot
   do. This is the precise, citable statement of C3 and it **pre-empts the "isn't the gap
   already solved?" reviewer**: the *selection* half is solved (Hard-ST), the
   *computation-over-time* half is not. (Nuance: at copy-50 soft is 0.88, not 1.0, so ~0.12 is
   capacity/under-solving and the +0.50 soft→hard is the computation gap — quantify the split,
   see §Pre-commit sanity check.)

So P2's contribution is a **triad**, not a single comparison:

| # | Contribution | Type | Why it's anchor-grade |
|---|---|---|---|
| **C1** | Bistable sequential primitives (D-FF, gated D-latch, **SR latch**, T-FF) in difflogic's vocabulary, with a custom STE through the feedback | Method | Nobody has stateful primitives in an LGN; turns the net into a true clocked circuit → FPGA flip-flops |
| **C2** | **The memory-degeneracy obstruction**: perfect 1-bit memory ⇒ unit Jacobian eigenvalue ⇒ `(I−∂F/∂q)` singular ⇒ implicit/DEQ gradients ill-posed *by construction* | Theory | A citable theorem for *why* the whole diff-logic field (DiffLogic CA, RDDLGN, diff-FSM, NeuroSAT) uses unroll-BPTT and nobody uses fixed-point gradients. The moat. |
| **C3** | Bistable hold **closes the long-sequence discretization gap** + exact long-range recall + **FSM induction** + direct FPGA mapping | Empirical | Closes the **computation gap** (Kim 2603.14157) that *no training method* can — architecturally, via a re-binarized bit; distinct from R-DTLGN's post-hoc distillation (2605.24649). Falsifiable on the copy-50 numbers we already have. |

The difference between "we added latch primitives" (incremental) and "we explain *why*
differentiable memory is hard and solve it three ways" (main-track).
**Working decision: lead with latch-vs-gated + the gap-closing claim; keep `rddlgn` as the
floor control.** (Open decision H1 — not yet locked.)

---

## B. What we need (and what already exists)

Infrastructure reality: **~80% of the build is done by P1.** The `latch` arm is a
*localized change in [`../seqlgn/cells.py`](../seqlgn/cells.py)* — the interface is already
reserved there as a `NotImplementedError` stub, and [`../seqlgn/docs/design.md`](../seqlgn/docs/design.md)
§6 already sketches it.

| Already built (reuse as-is) | New to build for P2 |
|---|---|
| Pluggable cell (`rddlgn/gated/lstm/gru_cell`) + `models.py` unroll loop | **`latch` mechanism** in `cells.py` (3 tiers, §B.1) |
| 4-way comparison harness, discrete-locked eval | **Custom `autograd.Function`** for the SR latch (hard settle fwd / surrogate bwd) |
| Training recipe: keep-bias, skip-step, cosine LR, grad-clip (`train.py`) | **T-flip-flop primitive** (`Q⁺=T⊕Q`) — the parity demonstrator |
| Benchmarks: smnist / psmnist / copy / parity + `--delay` / `--chunk` (`data.py`) | **FSM / automata induction task** (new loader — §C) |
| `grad_norm_through_time`, `count_gates`, `collate.py`, `plot.py` | **Adding-problem regression head** (the one real data gap, flagged in `data.py`) |
| Equal-gates fairness protocol, multi-seed (`docs/experiments.md`) | **Minimal RTL emitter** (D-FF→register) for the FPGA demo (see [10_fpga_scout.md](10_fpga_scout.md)) |
| DUST cluster 2×2080 Ti ([[dust-cluster-deployment]]) + local 2080S | (optional) **permutation / barrel-shift arm** (VSA-scout 5th mechanism, [08_vsa_crosspollination_scout.md](08_vsa_crosspollination_scout.md)) |

### B.1 The `latch` cell — three tiers, minimal diff

Add to the existing `if mechanism ==` chain in `LogicRecurrentCell.forward`:

```
# Tier 1 — D-FF (on-ramp / plumbing check): write-enabled register, identity-gradient delay.
# Tier 2 — gated D-latch: == `gated`, but pass the held state through a bistable restore.
# Tier 3 — SR latch (the contribution): learned logic drives set/reset lines; bistable hold.
s  = self.set_net(z)          # S line  (write-enable AND candidate)
r  = self.reset_net(z)        # R line  (write-enable AND NOT candidate)
h' = SRLatch.apply(s, r, h)   # custom Function: hard NOR-NOR settle fwd, surrogate bwd
```

`SRLatch` forward = the exact discrete settle **seeded by the stored bit `h`**
(hardware-faithful; resolves bistability exactly as real silicon does); backward = the
multilinear surrogate of §D. In **hold** (`s=r=0`) the gradient is exactly 1 — the
constant-error carousel as a *primitive*, not an init trick. Compute is cheap (the
recommended gradient is O(1) backward, no unrolling), which fits the shared-GPU budget;
synthetic tasks (copy / parity / FSM) run in minutes.

---

## C. Data / benchmarks

Everything is wired except the FSM task and the adding head. Mapped to what each *isolates*:

| Task | Status | Isolates | Expected: latch vs gated vs rddlgn |
|---|---|---|---|
| **copy(T=20,35,50,100)** | ✅ wired | Exact long-range recall + **the discretization gap** | **Headline**: latch gap≈0 at T=50 where gated had +0.50; rddlgn dead |
| **parity(L)** | ✅ wired | 1-bit running state = **exactly a T flip-flop** | latch (T-FF) solves in *one gate*; rddlgn recomputes XOR-chain (vanishes) → the "right primitive makes it trivial" plot |
| **FSM / automata induction** | 🔨 new loader | Learning a clocked state machine end-to-end | **Conceptual home turf** — read out the learned DFA, synthesize it. Tomita-7 / mod-N counter / "1011" detector |
| **psMNIST(784, chunked)** | ✅ wired | Hard standard long-range benchmark | credibility benchmark; chunk to keep wall-clock sane (full 784 ≈ 20–40 h/run) |
| **delayed-MNIST(d=0..100)** | ✅ wired | Airtight recall (P1's clean win) | carry over; latch should match/extend gated's hold |
| **adding-problem(T)** | 🔨 needs reg. head | Long-range credit assignment (classic LSTM test) | optional; only real infra add on the data side |
| **FPGA D-FF demo** | 🔨 new emitter | "true sequential circuit → hardware", *measured* | ns-latency on a cheap board; the P3-into-P2 demo |

The **FSM induction task is the most valuable addition** — it's the natural showcase ("we
learn an FSM-as-circuit"), it's cheap, and it unlocks an *interpretability* story (extract
the learned state-transition table via Weiss et al.'s L\* algorithm — see
[12_reading_sequential_memory.md](12_reading_sequential_memory.md) §F) plus the *hardware*
story (synthesize the FSM to flip-flops). It ties C1+C2+C3 together on one benchmark.

---

## D. The math — existing vs new

Bottom line from the toolkit sweep: **most of it exists; the new content is narrow, and it
should be claimed narrowly.** Three tiers:

**Tier 1 — D flip-flop = unit delay. No new math.**
`q[t+1]=D[t]`, `∂q[t+1]/∂D[t]=1` — the constant-error carousel (Hochreiter & Schmidhuber
1997) as a *primitive*, trained by BPTT (Werbos 1990).

**Tier 2 — gated D-latch = 2:1 MUX hold. No new math.**
`Q⁺ = e·D + (1−e)·Q`; already inside difflogic's multilinear algebra and identical to P1's
`gated`. `∂Q⁺/∂D=e`, `∂Q⁺/∂Q=1−e`, `∂Q⁺/∂e=D−Q`.

**Tier 3 — SR / cross-coupled latch = combinational feedback fixed point. The real problem
— and the contribution.**

*Existing math you'd reach for, and why it fails.* The fixed point `q*=F(q*;s,r)` invites
the implicit-gradient toolkit — **IFT**, **Deep Equilibrium Models** (Bai/Kolter/Koltun
2019), **Almeida–Pineda recurrent backprop** (1987; Liao et al. 2018), **Equilibrium
Propagation** (Scellier & Bengio 2017). All compute the same gradient via `(I−∂F/∂q)⁻¹`:

```
(I − Jₚ) dq*/d(·) = ∂F/∂(·),   Jₚ := ∂F/∂q|_{q*}          (implicit gradient)
```

**All are degenerate at a bistable latch.** For the NOR SR-latch in hold (`s=r=0`),
`NOR(0,x)=1−x` gives `q ← 1−(1−q) = q`, so `Jₚ = +1` and `(I−Jₚ)=0` is **singular**. This
is the *math signature of memory*: a perfect 1-bit store is infinitely sensitive along the
storage direction (unit Jacobian eigenvalue). Consequences: IFT undefined in hold; a leaky
latch has loop gain `~1/ε` (the exploding-gradient face of memory); two stable fixed points
violate DEQ/monDEQ **uniqueness**; the separatrix has `ρ(Jₚ)>1` so the adjoint iteration
*diverges*; at the set/reset threshold the stable state jumps basins → a **Dirac** gradient
(the spiking-neuron non-differentiability). **This is exactly why every diff-logic system
with real state uses fixed-unroll BPTT and nobody uses an implicit fixed-point gradient**
(confirmed across DiffLogic CA, RDDLGN, differentiable FSM/FST, NeuroSAT/CircuitSAT — §D
refs). **That obstruction is contribution C2.**

**⚠ Scope C2 before publishing (2026-07-01 red-team, see
[13_snn_hebbian_scout.md](13_snn_hebbian_scout.md) §"real payoff" +
[14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md)).** The
"`(I−Jₚ)` singular" headline is exact **only for the lossless/identity hold**
(`∂Q⁺/∂q=1` — the *marginal carousel*, which is P1's object). A **restoring** bistable
latch — the one that delivers C3 by cleaning the bit each step — is a *contraction*,
`|λ|<1`, where `(I−Jₚ)` is **invertible** and the implicit gradient *exists*
(Bai et al. 2021, arXiv:2106.14342). So a sharp reviewer pulls the thread *"identity-hold ⇒
no bit-cleaning (no C3); restoring ⇒ no singularity (no C2)."* Fixes: **(i)** lead C2 on
**fixed-point NON-UNIQUENESS / basin-selection** (two coexisting stable states ⇒ the selected
attractor is a non-unique, discontinuous-in-parameters object — Almeida/Pineda; and
explicitly **Bal & Sengupta, IJCAI 2023, arXiv:2209.09626: EP requires static input ⇒
LSTM/GRU-like sequential-state models are impossible**), and demote the singularity to the
marginal-hold + separatrix boundary cases (both already in the derivation above). **(ii)**
**Do not universalize to "every non-BPTT trainer"** — exact RTRL carries the full recurrent
Jacobian and is *not* blocked (Zucchet et al. 2023, arXiv:2305.19044); scope the claim to the
*fixed-point / implicit / equilibrium* family. **(iii)** The planned "implicit gradient
destabilizes" negative experiment is largely **pre-existing** — **Laydevant, Marković &
Grollier, Ising-EqProp, Nat. Commun. 2024** already shows bistable ON/OFF units resist
equilibrium training; run yours on the *SR primitive* as *confirmation*, not discovery.
**Terminology guard vs R-DTLGN (arXiv:2605.24649):** their "stability" is forward-dynamics
Tarski/Kleene fixed-point *existence* + graceful degradation (they even call state→unknown
"degenerate memory") — a **different object** from our *gradient* well-posedness; contrast it
explicitly so the two aren't conflated.

*The recommended gradient (custom STE — stable, hardware-faithful, O(1)).* Collapse the
feedback loop to the textbook **characteristic next-state equation** `Q⁺ = S + R̄·Q` and
relax it multilinearly:

```
FORWARD (hard):  exact NOR–NOR settle seeded by stored q  →  q[t+1] ∈ {0,1}
BACKWARD (surrogate = Jacobian of the multilinear char. eq. Q⁺ = s + (1−r)q − s(1−r)q):
    ∂Q⁺/∂s = 1 − q + r·q
    ∂Q⁺/∂r = −q·(1 − s)
    ∂Q⁺/∂q = (1 − r)(1 − s)      ← = 1 in hold (carousel), = 0 at set/reset
```

Check at legal corners (`s,r,q∈{0,1}`, `s·r=0`): `(0,0,q)→q` hold; `(1,0,·)→1` set;
`(0,1,·)→0` reset — exact. The surrogate lives in the *same multilinear algebra as
difflogic's 16 gates*, so all three tiers are **one consistent relaxation**. It's justified
two ways: (i) it is the Jacobian-free / phantom-gradient approximation of the loop
(`(I−Jₚ)⁻¹≈I` ⇒ "differentiate one application of the gate at the fixed point"); (ii) it is
the SNN surrogate-gradient recipe (Neftci/Mostafa/Zenke 2019), where Zenke & Vogels 2021
show the surrogate *shape* barely matters (but is *necessary* — zero without it). Optionally
wrap the write decision in a temperature-annealed soft enable and anneal `τ` (deterministic
annealing, Rose 1998) to close any residual gap.

**Energy framing (optional color, not the main method).** The cross-coupled NOR latch maps
to a 2-spin Ising **double well** (two minima = the two stored bits); gives a
temperature-annealed soft→hard schedule and an Equilibrium-Propagation route that *sidesteps
the singular inverse* (two relaxations instead of `(I−Jₚ)⁻¹`). Keep as a theoretical bridge.

**What to claim as new (and ONLY this):**
1. **The memory-degeneracy obstruction (C2)** — perfect memory ⇒ singular `(I−Jₚ)` ⇒
   IFT/DEQ/Almeida–Pineda ill-posed; bistability ⇒ non-unique fixed point ⇒ violates
   monDEQ. A clean obstruction analysis (corroborated empirically: nobody does it).
   **← but lead on the non-uniqueness half, not the singularity half — see the ⚠ Scope C2
   note above (the restoring latch has `|λ|<1`, so `(I−Jₚ)` is invertible).**
2. **The reduction** — replacing the feedback relaxation with the **multilinear
   characteristic-equation surrogate**, unifying all three tiers + the existing 16 gates
   into one relaxation and turning a fixed-point problem into a feedforward neuron + STE.
   **Scope (2026-07-01, see [14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md)):**
   the multilinear *representation* of the 16 gates and *coefficient-space STE training* are
   **occupied** (Petersen; **Kim 2026, arXiv:2605.08657**; **IWP / Rüttgers et al.** corner
   basis) — cite them and claim as new **only** the *characteristic-equation / feedback-
   fixed-point → feedforward reduction* (the sequential move nobody has made). Kim's **`c_ab`
   interaction-coefficient starvation** (the `ab` term gets gradient on only ~25% of samples)
   likely also hits the latch surrogate's `q·(...)` interaction terms — watch for it and
   consider CovJac-style coupling if it bites.

The gradient estimators themselves are **imported** — say so. A cheap *negative-result
experiment* (show the DEQ/implicit gradient destabilizes, confirming C2) strengthens the
theory at almost no cost — but note it is **largely pre-existing** (Laydevant Ising-EqProp
2024; see the ⚠ Scope C2 note).

### D-refs (author/year + arXiv/DOI)
DEQ: Bai/Kolter/Koltun 2019, arXiv:1909.01377 · monDEQ: Winston & Kolter 2020,
arXiv:2006.08591 · JFB: Fung et al. 2021, arXiv:2103.12803 · phantom grad: Geng et al. 2021,
arXiv:2111.05177 · Almeida 1987 (IEEE ICNN) / Pineda 1987 (PRL 59:2229) / Liao et al. 2018,
arXiv:1803.06396 · STE: Bengio/Léonard/Courville 2013, arXiv:1308.3432 · surrogate grads:
Neftci/Mostafa/Zenke 2019, arXiv:1901.09948; Zenke & Vogels 2021 (Neural Comp.) · Gumbel:
Jang/Gu/Poole 2017, arXiv:1611.01144 · EqProp: Scellier & Bengio 2017, arXiv:1602.05179;
equivalence to RBP: Scellier & Bengio 2019, arXiv:1711.08416 · Hopfield 1982 (PNAS 79:2554),
1984 (PNAS 81:3088); det. annealing Rose 1998 (Proc. IEEE 86:2210) · CEC/LSTM: Hochreiter &
Schmidhuber 1997 (Neural Comp. 9:1735); BPTT Werbos 1990; grad flow Pascanu et al. 2013,
arXiv:1211.5063 · difflogic: Petersen et al. 2022 (arXiv:2210.08277), 2024 (arXiv:2411.04732)
· prior stateful diff-logic (BPTT-unroll only): DiffLogic CA arXiv:2506.04912, RDDLGN
arXiv:2508.06097, diff-FSM (Mordvintsev et al. 2022), recurrent ternary/Kleene LGN
arXiv:2605.24649, CircuitSAT (Amizadeh et al. 2019, ICLR), NeuroSAT arXiv:1802.03685.

**2026 deep-read refs (added 2026-07-01, see [14_recurrent_lgn_2026_deepread.md](14_recurrent_lgn_2026_deepread.md)):**
DEQ convergent-equilibrium `ρ(J)<1` ⇒ `(I−J)` invertible: **Bai et al. 2021, arXiv:2106.14342**
· EP needs static input ⇒ sequential-state impossible: **Bal & Sengupta 2023 (IJCAI),
arXiv:2209.09626** · Ising-EqProp, bistable units resist equilibrium training (= the C2
negative experiment, pre-existing): **Laydevant, Marković & Grollier 2024 (Nat. Commun.)** ·
exact RTRL non-BPTT yet not blocked (scopes the C2 universal): **Zucchet et al. 2023,
arXiv:2305.19044** · selection/computation gap decomposition (formalizes C3): **Kim 2026,
arXiv:2603.14157** · multilinear coefficient-space gate training (occupies claim-#2 surface —
cite, don't claim): **Kim 2026, arXiv:2605.08657**; **IWP/Rüttgers et al.** · R-DTLGN — recurrent
ternary, trajectory-distillation gap fix + Myhill–Nerode realizability bound + DFA-extraction/
gated-variant future work (closest C3 neighbor, cite+distinguish): **arXiv:2605.24649**.

**RL-scout refs (added 2026-07-01, see [15_rl_lgn_scout.md](15_rl_lgn_scout.md)):** LGN-as-RL-policy
occupier — DWC "Differentiable Weightless Controllers" (Kresse & Lampert, ISTA, ICML'26,
**arXiv:2512.01467**; feedforward/memoryless, SAC, Artix-7 ~2 nJ/action — cite as the predecessor of
any LGN-RL-policy demo and distinguish P2's *clocked-sequential latch*) · Petersen's own early LGN-RL
— **"Efficient RL Agents with DLGNs," CoRL'24 DiffOpt workshop** (behavioral cloning, feedforward —
verify scope) · joint gate+wiring "adaptive resampling," beats connectivity SOTA — CompactLogic
(Vechev/ETH-SRI, **arXiv:2602.05830**) · training-infra: Decoupled STE **arXiv:2410.13331** (rejects
REINFORCE on DLGN); the pathwise-dominates-score-function result (Mohamed et al., JMLR 2020) is why
RL-as-LGN-trainer is a NO-GO.

**Cross-pollination / verification refs (added 2026-07-01, see [16_crosspollination_and_robotics.md](16_crosspollination_and_robotics.md)):**
⚠ **ISTA feedforward LGN verification — "Logic Gate Neural Networks are Good for Verification"**
(Kresse, Yu, Lampert, **Henzinger**, ISTA, NeuS'25 Disruptive-Idea award, **arXiv:2505.19932**):
the base P2's *sequential/temporal* verification extends — and the **#1 scoop threat** (same lab as
DWC + connectivity 2507.02585; sequential case = their obvious v2). Cite + distinguish
feedforward-vs-clocked. · logic/LUT-net **bit-flip resilience** (DWN team, **arXiv:2603.22770**) —
edge/drone robustness · **Crazyflie Artix-7 XC7A15T** learned-controller flight platform
(**arXiv:2403.18703**) — the drone-capstone sim-to-real template · **"Illusion of State in SSMs"**
(**arXiv:2404.08819**, ICML'24) — the TC⁰ theorem that kills the associative-scan/"O(log n) training"
temptation (cite to explain why the Mamba-style scan sirens, incl. RDDLGN's own future-work
direction, don't buy structured memory).

---

## E. Gates to include

**Keep all 16 combinational gates** — they *drive the latch control lines* and must stay
universal (NAND/NOR = the cross-coupled pair for the SR latch; XOR builds the T-flip-flop;
TRUE-gate id 15 is the keep-bias hook; FALSE-gate id 0 is the input-closed hook). No
pruning. **Add a small sequential-primitive vocabulary:**

| Primitive | Char. equation | Role in P2 | Priority |
|---|---|---|---|
| **D flip-flop** | `Q⁺ = D` | On-ramp / plumbing check (identity gradient) | Must |
| **Gated D-latch** | `Q⁺ = e·D + (1−e)·Q` | Bridge from P1 (= `gated`); isolates bistable-restore from soft-hold | Must |
| **SR latch** | `Q⁺ = S + R̄·Q` | **The novelty core** — cross-coupled, custom STE, hardware-faithful | Must |
| **T flip-flop** | `Q⁺ = T ⊕ Q` | **The parity demonstrator** — one primitive solves a task rddlgn can't | Must |
| **JK flip-flop** | `Q⁺ = J·Q̄ + K̄·Q` | Universal FF (set/reset/toggle/hold), no forbidden state — capability escalation | Optional |
| **Permutation / barrel-shift** | cyclic shift register | VSA-scout 5th arm (shift-register memory) | Optional |

**Mechanism comparison matrix** (all modifications to the *same* base cell so only the
memory mechanism varies — this is the experimental design):

| Variant | Memory mechanism | Role |
|---|---|---|
| `rddlgn` | concat-recurrence, recompute each step | floor control (RDDLGN / DiffLogic CA) |
| `gated` | soft 2:1 MUX hold | P1 arm = soft gated D-latch |
| **`latch`** | **bistable SR primitive holds the bit** | **P2 core** |
| `combo` | latch + MUX gating | P1 ⊕ P2 — the "logic-native LSTM as a clocked circuit" |
| (`tff` / `perm`) | toggle / shift-register | task-matched demonstrators (parity / shift) |

---

## F. Build order (cheapest falsifying tests first)

- **M0 — D-FF on-ramp.** Implement Tier-1 register; verify gradient=1; confirm it trains on
  copy-20. *If even the trivial delay doesn't beat rddlgn, stop and rethink.*
- **M1 — T-FF on parity.** The cleanest win: one toggle primitive solves parity that rddlgn
  can't. Fast, interpretable — a great Figure 1.
- **M2 — SR latch + custom STE.** The contribution. Validate the surrogate against
  finite-difference (at non-degenerate operating points) + against an unroll-`k`+BPTT latch
  on a 1-bit recall toy.
- **M3 — The headline: copy-50 discretization gap.** `latch` vs `gated` vs `rddlgn`,
  multi-seed. Show latch gap≈0 where gated had +0.50. This is C3.
- **M4 — FSM induction + read-out.** Tomita / counter / detector task; extract the learned
  DFA; the home-turf result + interpretability.
- **M5 — psMNIST credibility run + minimal FPGA D-FF demo.** The hard benchmark + the "true
  sequential circuit → silicon" proof point. The RTL emitter doubles as the *first public
  LGN→sequential-RTL* artifact — widens the moat vs ETH's unreleased BitLogic
  ([10_fpga_scout.md](10_fpga_scout.md)).

C2 (the obstruction theorem + the "implicit gradient fails" negative result) is written
alongside M2–M3 — it's analysis, not a run.

---

## G. Compute & training notes

- GPU: DUST cluster (2×RTX 2080 Ti, sm_75, shared → ~4–5 GB/card) + local 2080S. Build the
  `difflogic_cuda` kernel inside the DUST work volume; only `/home/jovyan/work` persists.
  See [[dust-cluster-deployment]] for the hard operational constraints.
- The recommended STE gradient is O(1) backward with **no settling unroll**, so Tier-3 costs
  no more than `gated` per step — the shared-GPU budget is not a blocker for the synthetic
  tasks. psMNIST is the only heavy run (chunk it).
- Training stability: carry P1's recipe (keep-bias, skip-step on non-finite grad norm, cosine
  LR decay). New knob: temperature-anneal on the soft write-enable if a residual gap remains.
- Fairness: multi-seed (≥3) on headline numbers; equal-gates control (size `rddlgn`/`gated`
  up to match `latch`'s gate count) so "latch wins" can't be dismissed as "more gates."

---

## H. Open decisions (recommendations in **bold**; NOT yet locked)

1. **Framing** — lead with **latch-vs-gated + discretization-gap-closing** (anchor-grade),
   `rddlgn` as floor? Or the documented latch-vs-rddlgn (safer, weaker)? → **the former.**
   *(This doc is written to the former; flip §A if H1 resolves the other way.)*
2. **C2 as a contribution** — include the degeneracy obstruction + the "implicit gradient
   fails" negative result? → **yes; cheap, and it's the moat.**
3. **Benchmark scope** — add the **FSM induction task** (rec: yes) + build the
   adding-problem head (optional)? FPGA demo INTO P2 (per fpga-scout) or defer to P3?
   **Optional capstone (2026-07-01, from [15_rl_lgn_scout.md](15_rl_lgn_scout.md)):** a
   **memory-required POMDP "logic-DRQN" demo** — a tiny partially-observable task (T-maze /
   memory-length / flickering-observation, **NOT** MuJoCo continuous control = DWC's turf) where a
   *feedforward* logic controller (DWC-style, arXiv:2512.01467) provably fails and P2's stateful
   latch provably fixes it, belief-state held in FPGA registers at ns/nJ. This is the single most
   compelling "why a learnable sequential logic primitive must exist" argument and the P2→P3 bridge;
   its novelty is 100% the latch (fold in, do not spin out — scoop-exposed to ISTA/ETH). **Gate it on
   a trainability check first**: policy-gradient through the BPTT-unrolled discrete-relaxed latch
   compounds the C2 feedback-training risk — prove on a tiny POMDP before making it the headline.
   Pairs with an **execute-a-model-check verification demo** (the thing R-DTLGN & DWC only *named*).
   **Physical embodiment — the Kyushu drone opportunity (2026-07-01, [16_crosspollination_and_robotics.md](16_crosspollination_and_robotics.md)):**
   this same capstone can be grounded on a **real nano-drone** (Kyushu robotics group; advisor likely
   Danilo V. Vargas). The novel angle is {P2 latch}×{real drone genuine partial-observability
   (gust/dropout/occlusion)}×{belief-state in clocked FPGA registers} — where feedforward DWC provably
   fails. Sim→real ladder, **distill from a PID/MPC teacher (never RL-on-a-real-drone)**: Rung0
   feedforward stabilization (de-risk, now) → Rung1 FPGA-synth+measure → **Rung2 recurrent latch on a
   hidden-state task = the publishable rung** → Rung3 model-check safety envelope → Rung4 fly on the
   Crazyflie Artix-7 deck (2403.18703). Frame on **energy/verifiability, NOT speed** (W2 trap). Same
   blocking gate as this capstone: **latch trainability first.**
   **The `latch → clocked FSM → exact model-check` chain is P2's moat vs ISTA's feedforward-only
   verification (2505.19932)** — make sequential/temporal verification an explicit P2 differentiator,
   not just future work.
4. **Scope vs P1** — P2 standalone main-track, or fold P1's gating in as the `combo` arm and
   ship one combined "Sequential Logic Gate Networks" paper? Notes lean **combined for the
   A\* shot**; the matrix in §E supports either.
5. **Venue/timeline** — see [07_venues_timeline.md](07_venues_timeline.md) (ICLR 2027 main
   was the pencilled target for the latch paper). Race tripwire: Bührer/Wattenhofer arXiv
   feed (RDDLGN + BitLogic same lab; window 3–12 mo per [10_fpga_scout.md](10_fpga_scout.md)).
   **⚠ ADD a second tripwire (2026-07-01): ISTA — Kresse / Lampert / Henzinger** (DWC control +
   feedforward LGN-verification 2505.19932 + connectivity 2507.02585). They are now the front-runner
   on the control+verification axis; the recurrent/sequential-verification result P2 targets is their
   obvious next step. Time-to-publish matters — see [16_crosspollination_and_robotics.md](16_crosspollination_and_robotics.md) §B.

---

## Pre-commit sanity check (before locking H1)

The cheapest test of the discretization-gap thesis: re-read the existing copy-50 `gated`
results in [`../seqlgn/results/`](../seqlgn/results/) — is the gap **drift-over-time**
(soft and hard trajectories diverge as `t` grows → supports the thesis, latch fixes it) or
**capacity-bound** (under-solving even the soft objective → weakens it, latch won't
help)? [08_paper1_checklist.md](08_paper1_checklist.md) §1 notes a capacity bump lifted
discrete 0.37→0.75 at hidden 2048, which is *partly* capacity — so quantify the split before
making gap-closing the headline.
