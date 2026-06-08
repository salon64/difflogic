# 09 — Faster LGN Training: Novelty Scout

**Scouted:** 2026-06-07. **Trigger:** "LGN training is slow — is there a novel technique to
train faster? Read xLSTM, maybe something transfers." Adversarial novelty check.

**Question:** a *paper-shaped* technique to cut LGN **training** cost (wall-clock / epochs /
memory) — NOT inference (solved) and NOT the discretization gap (owned by ETH).

---

## Verdict: NO-GO as a "train LGN faster" paper (conf 78%)

The training-speed lane is **saturated** and the xLSTM angle doesn't transfer cleanly.

### Prior art (the lane is full — ~4 papers in 12 months, ETH-dominated)
- **Light DLGN** (arXiv:2510.03250, ETH) — reparametrizes the gate neuron: **8.5× fewer steps,
  45% faster backward, 4× less memory**. *This is the training-speed paper.*
- **Mind-the-Gap / Gumbel LGN** (2506.07500, ETH) — Gumbel-softmax + straight-through: faster
  convergence, **fewer epochs**, flatter loss landscape.
- **Polynomial Surrogate Training** (2603.00302, Mar 2026) — degree-(2,2) polynomial neuron,
  **2,187× fewer params, 2–3× faster** (ternary).
- **Sinkhorn-Constrained Spectral Selection** (2601.13953, Jan 2026) — Boolean-Fourier basis +
  Birkhoff-polytope routing; alt gate parametrization (also noted in `08`/verdicts as #3-occupier).

### Why xLSTM doesn't rescue it
- **Feedforward LGN is already layer-parallel** → xLSTM's parallel/chunkwise **scan buys nothing**
  there. Scan only matters for *recurrent* LGN.
- **Exp gating + matrix memory are real-valued/linear** mechanisms — matrix memory = a linear
  associative memory, not a gate circuit; exp gating doesn't booleanize. **Dead on transfer.**

### The one literal opening — parallel/associative-scan training of *recurrent* LGN — and its two walls
- **Technique is already prior art** (outside LGN): differentiable FSMs (Google-Research 2022),
  GraphFSA (2408.11042), Symbolic-Feedforward for PFA (2509.10034, "unroll PFA → parallel
  feedforward"), data-parallel FSMs (ASPLOS'14). So only the *LGN application* would be new.
- **Wall 1 — linearity vs expressivity (same wall as the binding scout in `08`).** Parallel scan
  needs an *associative* combine → cheap only if the recurrence is **linear** (mLSTM=matmul,
  Mamba=linear SSM). A logic-gate recurrence over a *d*-bit state is an FSM; composing two steps is
  a 2^d→2^d transition object (exponential, no closed form). Force GF(2)-linear gates so composition
  is cheap → collapse to XOR/XNOR/permutation (LFSR/linear-CA), losing the nonlinearity that makes
  LGN worth using. **Nonlinear ⇒ no cheap scan; cheap scan ⇒ trivial gates.**
- **Wall 2 — wrong bottleneck.** At current recurrent-LGN lengths (RDDLGN caps at **16 tokens**),
  BPTT depth O(16) vs O(log 16)=4 is *not* the cost — the per-step **dense 16-gate softmax** is, and
  Light/Gumbel already kill that. So scan doesn't address the actual slowness.
- **Note:** RDDLGN (2508.06097) itself flags "longer training times" + vanishing gradients, but
  routes its own fix to **weight reparametrization** (i.e., Light), not scan.

### The only surviving sliver (NOT a speed paper)
Parallel-scan as an **enabler for *long-sequence* recurrent LGN** — training *at all* over
100s–1000s of steps (unclaimed per `lgn-landscape-key-facts`), not "train the same thing faster."
Belongs in **P2 (recurrent architecture)**, gated on resolving Wall 1 — likely a **chunkwise hybrid**
(sequential across chunks, structured/factored state within) or a deliberately **linear (GF(2))
recurrent core with nonlinear read-in/read-out**. Competes with RDDLGN's stated next step.

**REMAINING GATE (if revived):** can a recurrent LGN keep a *cheaply-composable* (≈linear/factored)
state core yet stay expressive via nonlinear gates only at read-in/read-out? Yes → real
long-sequence recurrent-LGN paper. No → the wall holds.

---

## Bottom line / steer
**Do NOT pursue "faster LGN training" as a thesis** — saturated, and xLSTM doesn't transfer. The
training-cost pain is real but already owned by ETH (Light/Gumbel) + the 2026 reparametrization
wave. The only thing worth carrying forward is the **long-sequence enabler** framing, and that's a
P2 architecture question, not a speed project. Keep the gating(P1)→latch(P2) plan.

See also: [06_paper_plan.md](06_paper_plan.md), [08_vsa_crosspollination_scout.md](08_vsa_crosspollination_scout.md),
memory `lgn-recurrent-scout-verdicts`, `lgn-landscape-key-facts`.
