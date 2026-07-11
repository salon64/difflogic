# The Certificate Lives at the Readout: Machine-Checked Guarantees for Trained Sequential Logic Gate Networks

**Skeleton v0.1 (2026-07-11) — Paper #3a (verification track; research/20 §D1, executed 07-10/11).**
Author: Malcolm Malle `[affiliation]`. Solo paper.
Working title above; alternates: *"Attractors and Certificates: Formal Verification of Trained
Recurrent LGNs"* / *"Provably Robust by Training: Dose-Responsive Certificates for Learned
Sequential Circuits"*. `[TODO pick after abstract settles]`
Venue options (decide by ~Sep '26): **arXiv first in all cases** (speed-to-arXiv is the race
defense — ISTA does not need P2 to do this, and R-DTLGN named DFA-extraction-for-model-checking
as future work). Then: NeurIPS 2027 main (May '27) as default; **CAV 2027 (~Jan '27)** if the
framing tilts formal-methods (the recipe + engine study carry a CAV paper; the ML story may not);
SaTML 2027 if tilted safety/robustness. `[TODO venue scout closer to date]`
Citations are pandoc keys into `references.bib` (`[TODO: seed from ../p2/references.bib` — shares
~15 keys — `then add the verification cluster]`). `[TODO]`/`[FIG]`/`[VERIFY]` mark gaps.
Code & artifacts: `mlgn/netlist/` — every claim traces to a committed artifact (BLIF + ABC log +
`report.json`); see the Evidence Map (§E).

---

## Abstract `[TODO compress to ~250 words once results freeze]`

Deployed logic gate networks are exact discrete circuits, and their sequential variants are exact
clocked finite-state machines — yet they are validated like neural networks, by test accuracy.
We close that gap: we present the first pipeline from trained recurrent-LGN checkpoints to
gate-level netlists to an industrial model checker (ABC), with every step bit-exactness-gated
(the rebuilt circuit must reproduce the recorded test accuracy exactly and match the training
framework bit-for-bit over millions of state bits), and we use it to prove — not estimate —
what the trained circuits do. Four findings. **(1) Mechanism:** learned registers are attractor
systems, not latches. After a write, the state evolves 9–28 further steps before settling into a
per-symbol fixed point or limit cycle; discrete accuracy is *attractor-count-quantized*
(observed 3/8 → 0.38, 7/8 → 0.87, 8/8 → 1.00 across mechanisms, tasks, and training regimes),
and seeds at identical 1.000 accuracy learn qualitatively different solution families
(fixed-point vs. permanent oscillator) that only verification distinguishes. **(2) Spec:**
hold-type properties ("the state freezes") separate these families and fail even on robust
circuits; decode-type properties ("the readout is correct, forever") are provable across
families — the certificate lives at the readout, not the state bits. **(3) Method:** at ~1,000
latches, push-button IC3 and k-induction time out even on true invariants; a simulation-derived
settle bound followed by temporal decomposition and inductive sweeping (`tempor→scorr→pdr`)
proves them in seconds, and we exhibit a strict verification hierarchy — random testing missed a
readout corruption that exhaustive 200k-state enumeration also missed and directed bounded model
checking found in minutes. **(4) Result:** provable robustness is *dose-responsive in training
pressure*: circuits trained on clean copy carry no distractor certificate; trained with 8
distractors they provably tolerate any distractor sequence after settling; trained with 20, they
carry the full theorem — any legal write followed by any distractor tokens at any time yields a
correct readout at every step, forever. A yosys synthesis of a verified circuit (901 LUTs, 885
FFs — 9% of a $30 FPGA) closes the loop from checkpoint to certified hardware object.

---

## 1. Introduction

`[TODO: 1 page. Hook = the validation mismatch: the deployed object is a circuit; circuits have
verification tools; nobody connects them. Then the "what does 87% accuracy MEAN" question —
answer: exactly 7 of 8 correct attractors, a sentence no test set can produce.]`

**Contribution 1 — the pipeline (checkpoint → netlist → theorem), correctness-gated.**
Eval-time sequential LGNs are exact clocked FSMs (binary state by induction; cite P2 §C2 reframe
`[@malle2026clatch]`). We ship the exporter (wiring reconstruction for legacy checkpoints via RNG
replay + self-contained checkpoints going forward), a netlist IR with bit-exact numpy semantics,
an in-netlist GroupSum/popcount argmax head with the framework's exact tie-break, BLIF/Verilog
emitters, and the discipline that makes proofs trustworthy: full-test-set accuracy-gate equality,
bit-exact equivalence over state trajectories, counterexample re-simulation, BFS↔MC
cross-validation, and **non-vacuity controls** (a corrupted-shadow variant must fire exactly at
the armed frames — a vacuously-true property also "proves"; ours caught a real bug).
`[VERIFY final counts: 14 checkpoints, 12 verified, ~40 verdicts, 0 gate failures]`

**Contribution 2 — mechanism: accuracy is attractor structure.**
Transients (decode is *wrong* during them — a minimum-delay operational envelope invisible to
test sets), per-symbol fixed points and limit cycles (periods 2/6/12 observed), solution families
that vary by seed at identical accuracy (fixed-point / progressive-crystallization / permanent-
oscillator across a training curriculum), and quantization: every sub-1.0 discrete accuracy we
verified equals (correct attractors)/8 up to test noise. `[FIG: ladder table + crystallization
curve 48→32→0→0 cycling inputs across c8→c50]` `[FIG: transient decode traces]`

**Contribution 3 — spec + method for proving it.**
Hold-type certificates are the wrong deliverable: they CEX on every circuit that wiggles yet
decodes perfectly (including the strongest robust ones). Decode-type certificates prove across
all families — including permanent oscillators whose orbit decodes correctly forever. Proving
them at ~1,000 latches: naive `pdr` and k-induction time out **on true invariants** (stuck at
the arming frontier); the working recipe is simulation-derived settle bound K → `tempor -F K+2`
→ `scorr` → `dc2` → `pdr` (seconds; one proof required a genuine 120-clause inductive invariant).
For *false* properties the recipe is inverted: simulation-guided depth, then bounded `bmc3`. We
demonstrate a strict hierarchy on one circuit: random testing (0/224 frames) < exhaustive
enumeration (escaped a 200k-state cap, zero wrong states found) < directed bounded MC (CEX at
frame 30 in 276 s, replay-confirmed). `[FIG: engine study table]`

**Contribution 4 — the dose-response result.**
| training pressure | decode after settling (fs0) | decode during settling (fs1) |
|---|---|---|
| copy (0 distractors) | CEX @ 30 | CEX @ 16 |
| distcopy d8 (3 mechanisms) | **PROVED** | CEX |
| distcopy d20 | **PROVED** | **PROVED** |
The d20 theorem is total functional correctness on the trained task at unbounded length: any
legal write + ANY distractor sequence at ANY time ⇒ readout equals the written symbol at every
frame ≥ K. Training with distractors bought a *machine-checked* guarantee, graded by pressure.
Meanwhile `distractor_hold` CEXes on every circuit ever tested — trained or not.

**Integrity statement.** `[TODO, p2-style. Candidates: hold-was-our-first-spec-and-it-was-wrong
(we report the journey); pdr fails honestly at this scale (the recipe is the contribution, not
engine magic); fs1-robustness only demonstrated for the d20-combo circuit (n=1 at that cell —
more seeds queued); parity/T-FF checkpoints trained poorly (0.58/0.61) and are excluded, not
hidden; all negative verdicts (hold CEXes) are load-bearing results, not failures.]`

---

## 2. Related Work

`[TODO ~0.75 page. Clusters:]`
- **Feedforward LGN verification:** ISTA `[@kresse2025verification]` (SAT certification,
  feedforward, award paper — our sequential complement; THE reference to position against);
  BNN-verification complexity `[@bnnverif2026]` (2606.18918).
- **Recurrent LGNs:** RDDLGN, DiffLogic CA, R-DTLGN (ternary, STL monitors; **names
  DFA-extraction for model checking as future work** — quote it), P1/P2 `[@malle2026gating;
  @malle2026clatch]`. Nobody exports or checks a trained sequential LGN.
- **Model checking:** IC3/PDR `[@bradley2011ic3]`, k-induction, temporal decomposition +
  signal correspondence `[@abc; @tempor; @scorr]` — we use them off-the-shelf; the contribution
  is the sim-guided composition and the evidence it is *necessary* at this scale.
- **RNN verification / extraction:** DFA extraction (Weiss et al.), QBN — extraction
  *approximates* real-valued RNNs; our object IS the circuit, so verification is exact — no
  abstraction gap. `[TODO 2604.12092 UMD ternary temporal behavior trees — tripwire cluster]`
- **Certified robustness:** interval/abstract-interpretation certificates vs. our exact
  temporal certificates on discrete state; verified envelopes as deployment metadata.

---

## 3. Preliminaries
`[TODO ~0.5 page: LGN relaxation, seqlgn cell zoo (gated/combo/clatch deploy to the SAME
MUX-register circuit class at argmax — table), GroupSum head, the binary-state induction
(one lemma), tasks (copy/distcopy protocols as input automata).]`

## 4. From Checkpoint to Netlist (the pipeline)
`[TODO ~1 page:]`
- 4.1 wiring: the RNG-replay reconstruction for legacy checkpoints + gates (accuracy equality on
  the full test set; bit-exact trajectories; survived a torch-version change); self-contained
  checkpoints (buffers) going forward.
- 4.2 netlist IR + simulator; 16-gate vocabulary → BLIF covers; MUX decomposition.
- 4.3 the head: popcount adder trees + comparators + first-max tie-break (exactness tests incl.
  engineered ties; 11,735 gates for the (8,128) head).
- 4.4 property harnesses as circuit transformations: warm-up shift register (settle window K),
  legality latch, shadow register, input-automaton gating (blank-forcing; distractor alphabet
  by aliasing multi-hot to blank — soundness argument).
- 4.5 verification discipline: non-vacuity controls, cex replay, BFS↔MC cross-validation.
  `[Include the caught-bug anecdote: vacuous pass → control fired → layer-slice bug found.]`

## 5. Properties
`[TODO ~0.5 page: table of comb_hold / seq_hold / protocol_hold(+anyx0) / protocol_decode /
distractor_hold / distractor_decode × fs0/fs1, each with its question in one line and its
input automaton. Emphasize hold vs decode as the spec-design axis.]`

## 6. Proving and Refuting at 1,000 Latches (method)
`[TODO ~1 page:]`
- 6.1 engine study on a true invariant: pdr timeout @ arming frontier (900 s), k-induction
  timeout (204k conflicts at the step), scorr-alone timeout, **tempor(K)+scorr+dc2+pdr ~2 s**;
  ablation shows the sim-derived K is the unlock. `[FIG table, from round-1 logs]`
- 6.2 closed vs. free inputs: blank protocols collapse under scorr (0 AND/0 latch residue);
  free-input proofs leave 200–800 latches and need real pdr work (the 120-clause invariant).
- 6.3 false properties: bmc3-first; simulation-guided depth (the frame-29-vs-30 story); the
  hierarchy result (random < enumeration < MC) with the sym5 corruption.
- 6.4 exhaustive simulation as a proof engine for cue-then-closed systems (512-case analysis;
  when MC is confirmatory vs. necessary).

## 7. Results
`[TODO ~2 pages; all numbers exist in committed artifacts:]`
- 7.1 attractor mechanics + quantized accuracy (12-checkpoint falsifier table + ladder study:
  three families at 1.0000; crystallization; settle depths don't shrink with curriculum —
  orbit structure changes).
- 7.2 blank-channel theorems: protocol_decode PROVED on every disc-1.0 circuit (11 circuits,
  6–23 s each) incl. permanent oscillators; CEX localizing the wrong attractor on every
  sub-1.0 circuit (frame = arming; count matches accuracy).
- 7.3 the dose-response table (headline) + BFS closure sizes (near-frozen 1–32 states trained,
  87k/escaped untrained) + what fs1-CEX means for d8 circuits (settle-window fragility).
- 7.4 hardware corollary: yosys synth_xilinx 901 LUT / 885 FDRE ≈ 9%/4.4% of XC7A15T; iverilog
  golden-vector equivalence incl. post-synthesis; `[TODO head-included synthesis run]`.

## 8. Discussion
`[TODO: certificates as deployment metadata ("correct iff channel silent" is an ENVELOPE, not a
failure); spec design lesson (readout invariants); what training does mechanically (freezes
closures / prunes wrong attractors) — verification as a lens on optimization; limits: 1k latches
/ 9-bit alphabet / synthetic tasks; MNIST-scale sequential circuits untested; scorr behavior
under free inputs is the scaling frontier; single-seed cells in the dose-response table.]`

## 9. Future Work
`[TODO one paragraph each: (a) the CAN-bus IDS wedge (stateful stream + ISO-21434 — research/22
flagship; these certificates are exactly what that domain buys); (b) engine breadth (AIGER →
rIC3/Pono/&pdr); (c) training FOR provability (certificate-guided regularization — does adding
the theorem as a training signal shrink K / freeze closures?); (d) T-FF/parity once checkpoints
train; (e) P3b: Fmax/board + head-in-fabric.]`

---

## E. Evidence Map `[keep updated; every number → artifact]`

| claim | artifact |
|---|---|
| pipeline gates (12/12 exact, bit-exact equiv) | `mlgn/netlist/out/*/report.json` (`accuracy_gate`, `equivalence`) |
| engine study (pdr/ind/scorr fail; recipe proves) | `out/ckpt_cp50A_curr_c35/*.log`, `ind_try.log`, `scorr_pdr_try.log`, `tempor_pdr_try.log` |
| attractor mechanics + families + quantization | `out/ladder_summary.{md,json}`, `out/v_*/report.json` |
| blank-channel decode theorems | `out/v_*_decode/`, `out/v_*/report.json` (`protocol_decode`) |
| hierarchy result (sym5, frame 30) | `out/distractor_study/combo/` (`bmc3deep` log + cex), `report.md` §2 |
| dose-response table | `out/distractor_study/dc_*/circuit_report.json`, `report.md` round-3 addendum |
| 120-clause invariant proof | `out/distractor_study/dc_clatch_d8/` (fs0; foreground log noted in circuit_report) |
| synthesis numbers | `mlgn/netlist/synth/report.md` + `synth/log/` |
| non-vacuity discipline & caught bug | `out/a1_gate/gate_report.json`; experiment log 2026-07-10 (pm) |

## R. Remaining experiments before submission `[none block the arXiv skeleton]`

- [ ] seeds: second seed for dc_combo_d20 (fs1 theorem is n=1) + dc_gated/clatch d20 decode cells → DUST queue v2
- [ ] distcopy d40: does the dose-response continue or saturate?
- [ ] head-included yosys synthesis (deployment-realistic LUT count)
- [ ] T-FF parity: train to >0.95 (running-target recipe needs tuning), then the parity_decode property (running-XOR shadow, ~10 lines)
- [ ] cross-engine sanity: one theorem re-proved in nuXmv or rIC3 (defends against "ABC quirk" review)
- [ ] `[VERIFY]` sweep: re-run every number in §7 from the committed artifacts before draft v1
