# 08 — VSA/HDC × LGN Cross-Pollination: Novelty Scout

**Scouted:** 2026-06-06. **Trigger:** read 3 papers from the Rachkovskij/Osipov/Kleyko/Gayler
HDC–VSA cluster and asked "what can LGNs borrow / what's publishable at the LGN×VSA seam?"

**Source papers (the 3):**
1. **WSP** — Schlegel, Rachkovskij, Osipov, Protzel, Neubert, *Learnable Weighted Superposition
   in HDC*, IJCNN 2024. (full text read)
2. **MMPerc** — Rachkovskij, Osipov, Volkov, De Silva, Kleyko, *Multiclass Linear Perceptrons
   With Multiplicative Margins*, Neural Computation 38(4):602–650, 2026. (abstract-level only —
   full text not accessible)
3. **GVFA** — Kahawala, De Silva, Osipov, Rachkovskij, Gayler, *Graph Vector Function
   Architecture*, Neural Networks 197 (2026) 108416, CC BY. (full text read)

**The connective insight:** a *binary* GVFA reduces to bitwise logic — binding = XOR,
bundling/superposition = majority/popcount, `Λ`=sign = hard threshold, role-tagging = fixed
permutation (GVFA Appendix B.4 says this explicitly). So a fixed binary VSA encoder *is* a
hand-wired logic-gate circuit, and an LGN is the *learnable* version of the same object. That
reframing is what makes the seam interesting.

---

## Verdicts (5 ideas)

| # | Idea | Verdict | Conf |
|---|------|---------|------|
| 1 | **HDC/boolean encoder → differentiable LGN classifier** (graph/set/seq inputs, all in popcount HW) | **CONDITIONAL GO — the one genuinely new paper-shaped direction** | 65% |
| 2 | **Permutation-as-recurrent-state** for recurrent LGN (fixed bit-shuffle carry + learned gate transition) | **FOLD INTO P2** as a 5th state-mechanism arm | 60% |
| 3 | LGN as learned COMBINE Φ over VSA graphs ("learnable VSA") | **NO-GO standalone** (binding-learning occupied) | 40% |
| 4 | WSP weighted-superposition + iterative prune-and-reinit for LGN input gating | **NO-GO as paper** (pruning/wiring occupied); keep the `exp(γ)` trick | 30% |
| 5 | Multiplicative-margin (MMPerc) readout vs GroupSum→argmax | **MINOR** — ablation/half-section only | 45% |

---

## Idea 1 — HDC encoder → LGN classifier  (CONDITIONAL GO, 65%)

**Gap (survives):** no paper couples an HDC/VSA encoder to a *differentiable logic gate network*.
The boolean-classifier neighborhood is occupied but adjacent, not on it:
- **DWN** (Differentiable Weightless Networks, ICML'24, arXiv:2410.11112), **DW-Controllers**
  (arXiv:2512.01467), **WARP-LUTs** (arXiv:2510.15655), LogicNets/PolyLUT/NeuraLUT/SparseLUT —
  learnable boolean/LUT classifiers, FPGA-grade, but fed by **thermometer encoding, not HDC**.
- **GVFA, VS-Graph** (arXiv:2512.03394), **GraphHD / MoleHD / RelHD / Hypernode** — HDC graph
  encoders, but classified with **Ridge / SVM / centroid / MLP**, never logic gates.
- **FeFET Logic-in-Memory HDC encoder** (arXiv:2512.20302) — HW HDC encoder + HDC associative
  memory classifier, not an LGN.

**Pitch:** training-free HDC/VSA encoder (XOR-bind / popcount-bundle / fixed permutation / sign)
→ binary hypervector → LGN classifier, for graph/set/sequence inputs vanilla LGNs can't ingest;
end-to-end in gate/popcount hardware.

**REMAINING GATE (could still kill it):** sweep the LUT-NN/weightless literature to confirm
nobody fed **HDC / random-projection binary features into a LUT/weightless net** — if they did,
#1 collapses to "HDC features + DWN." Softer gate: need a benchmark where an LGN head beats HDC's
native linear/centroid classifier (likely answer: learned nonlinear boolean decision + single HW
substrate). If gate clears → ~80%.

**Resource fit:** needs `difflogic_cuda` (GPU); graph benchmarks mid-compute. **NOT feasible on
the CPU laptop — needs cluster.**

## Idea 2 — Permutation-as-recurrent-state  (FOLD INTO P2, 60%)

**Occupied (real-valued):** permutation/binding as sequence memory is Frady/Kleyko/Sommer
(arXiv:1803.00412) and Rachkovskij recursive binding (arXiv:2201.11691) — but **linear, not
logic gates**. **Open:** permutation carry + *learned logic-gate* transition. Neither recurrent
LGN uses it — DiffLogic CA (arXiv:2506.04912) = combinational recurrence over a binary cell
vector, no permutation; RDDLGN (arXiv:2508.06097) = learned encoder-decoder recurrence.

**Catch:** a fixed permutation over a bit-vector is a barrel-shifter of 1-bit registers — i.e. a
**cousin of the already-GO latch/flip-flop substrate** (P2). More a *mechanism arm* than a paper.
**Action:** add as a 5th arm to P2's comparison {sequential / latch / gated / combo / **perm-state**}.
Fits existing `mlgn/seqlgn/` infra → CPU-dev-able.

## Idea 3 — Learnable VSA via LGN  (NO-GO standalone, 40%)
Learnable/differentiable binding is occupied: **HLB / Walsh-Hadamard linear VSA** (NeurIPS'24),
**LARS-VSA** (arXiv:2405.14436), **Generalized HRR** (arXiv:2405.09689), trainable HV encoders.
"16-gate difflogic circuit *as* the binding op" is a narrow unclaimed sliver but weak motivation,
far from the recurrent focus. → at most a paragraph inside #1.

## Idea 4 — WSP pruning for LGN  (NO-GO as paper, 30%)
difflogic already prunes redundant gates; wiring/sparsification owned by **LILogic Net**
(arXiv:2511.12340), **Mommen et al.** (arXiv:2507.06173), **Scalable Interconnect Learning in
Boolean Networks** (2025); generic differentiable pruning mature (DiffPrune, SequentialAttention++).
**But pocket the `exp(γ)` temperature trick** from WSP — directly targets the "readout converges
before gate structure settles" risk in the gating plan. Use it as training infra, don't paper it.

## Idea 5 — Multiplicative-margin readout  (MINOR, 45%)
LGN readout is uniformly GroupSum (bit-count) → /τ → softmax-CE; no margin/mistake-bound readout
found, so technically open — but it's a readout swap orthogonal to the recurrent thesis.
Possible half-section/ablation ("does a multiplicative-margin head shrink the discrete-eval gap?").
**Gate:** read MMPerc full text first (currently abstract-level only).

---

## Bottom line / steer

Only **#1** is a new, defensible, paper-shaped direction — but it pulls into the VSA neighborhood
and needs cluster GPU. **#2 belongs inside P2** (reinforces latch, doesn't compete). **#3–#5 are
not standalone papers**; harvest #4's `exp(γ)` and maybe #5's margin head as free ingredients.

**Recommendation:** do NOT pivot now. Keep the gating(P1)→latch(P2) plan (real moat, CPU-dev-able).
**Park #1 as "Paper 4 / post-ICLR"** — it survives novelty so it keeps. Fold #2 into P2. Re-open #1
only if P1/P2 stall or cluster time is secured and a structured-input thrust is wanted.

**Open gate if/when #1 is revived:** LUT-NN/weightless sweep for HDC-features-into-LUT-net.

---

## LGN → VSA (reverse direction)  — scouted 2026-06-07

All of the above is VSA→LGN (what LGN borrows). This section is the reverse: **what LGN
contributes to the VSA/HDC field.** The seam is asymmetric — VSA gives LGN encoders for
structured inputs; the reverse is thinner because LGN's unique powers (learning discrete
functions, gate/popcount HW) partly overlap with what VSA already half-solves *linearly*.

Four candidates, ranked:

**A. Trainable *nonlinear* classifier head, in-substrate — the real surviving contribution.**
This is **Idea 1 flipped**. From VSA's side, the perennial weakness is the *readout*: HDC
pipelines end in centroid / Ridge / SVM / linear dot-product — and the current field trend
(XL-HD arXiv:2605.24788; trainable-encoder line, ACM TODAES 10.1145/3665891) is to learn the
*encoder/projection*, binarize, then do a **linear** dot-product, leaving the nonlinear-decision
hole wide open. LGN fills it with a learnable *nonlinear boolean* head that stays in popcount/gate
HW — matching HDC's whole hardware ethos. Framing ("HDC finally gets a trainable nonlinear head
that never leaves the logic substrate") is arguably stronger + better-venued (Neural Computation /
Kleyko–Rachkovskij cluster) than #1's LGN-centric pitch. Status unchanged: still Paper 4 /
post-ICLR, still needs cluster GPU. Net effect: a *better motivation* for #1, not a new project.

**B. Learned boolean binding/bundling operator — NO-GO (scout 2026-06-07, conf 72%).**
Pitch was: difflogic 16-gate cell *as* a learnable, HW-cheap, purely-boolean binding op
(vs the linear HLB / LARS-VSA / GHRR; vs linear-bundling WSP). The literal gap is **open**
(nobody has done gate-learned binding) but the idea is **killed on theory + motivation**:
- *Invertibility self-defeat:* binding exists to be invertible (unbinding is the whole point;
  XOR is special as its own inverse). A freely-learned 2-input gate is generically non-invertible
  → destroys the property that justifies binding. Constrain to reversible gates → space collapses
  to ≈XOR/XNOR → nothing left to learn. Either *useless* or *empty*.
- *Against the current:* the field already answered "fixed ops hurt" by learning the **encoder**,
  treating explicit binding as replaceable — so "keep binding but learn it as a gate" swims upstream.
- *Bundling* (no inverse needed) is the only clean sub-part, but it's occupied by **WSP** and
  majority-bundle is already near-optimal + HW-trivial.
This sharpens file 08 Idea 3's NO-GO: not just "binding-learning is occupied" but "boolean-gate
binding is *self-defeating*." **Drop it.** Lone reopener: learned binding *restricted to the
reversible/permutation subset* in a resonator/factoring loop — but that's just Idea 2 (perm-as-
state, already folded into P2), not standalone.

**C. Logic-circuit *probe* for distributed VSA codes — unscouted, weak.** Train an LGN to read
hypervectors and emit an interpretable boolean rule for what structure a code carries. Genuinely
different from #1 and hits a real VSA pain point (opacity), but value is speculative and far from
the recurrent thesis. Not scouted; park.

**D. Distillation / capacity** — compress a hand-built VSA encode→bundle→classify pipeline into a
minimal learned logic circuit. Minor, orthogonal.

**Bottom line:** only **A** moves anything, and only as a stronger framing for the already-parked
Idea 1. **B is dead.** C/D are not standalone. No reason to pivot; reverse direction does not beat
the gating(P1)→latch(P2) plan.

---

See also: [05_my_angles.md](05_my_angles.md), [06_paper_plan.md](06_paper_plan.md),
memory `lgn-recurrent-scout-verdicts`, `lgn-landscape-key-facts`.
