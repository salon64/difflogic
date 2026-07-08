# 20 — Full-program validation report (2026-07-08)

_81-agent validation workflow (12 validators + adversarial verification of every critical/major
finding; ~30 independent recomputations of headline numbers from the 191 result JSONs). Scope:
P1 draft + evidence, P2 locked claims (workmap §A0'), code/pipeline audit, future-work plans
(P3a/P3b/P4), fresh web race check, reviewer red-team, completeness sweep. Every finding below
survived an independent verifier that tried to refute it against the actual files/data unless
marked otherwise._

## VERDICT (TL;DR)

1. **The numbers are real.** Every quantitative claim in the P1 draft and in workmap §A0'
   reproduces **exactly** from the raw JSONs — no fabrication, no mis-transcription, no
   test-peeking, fair paired seeds, no result-biasing bug found in cells.py/train.py/data.py.
   The never-write-collapse evidence chain is complete and verified end-to-end.
2. **But two of P2's three locked edges have framing/validity problems** a competent reviewer
   would find: the stability census is config-confounded and its flagship "matched pair" is
   provably not matched (keep_bias 3 vs 1); the 4.4× gap edge is partly by-construction
   (clatch's "soft" eval already runs at hard_alpha=1.0) — the log admits both, A0' does not.
3. **P1 is numerically honest but stale**: the delayed-MNIST table quotes what is now the
   best-of-3 seeds (d100 0.34 → 3-seed mean 0.30); 14 completed runs (06-24..07-01) are
   recorded **nowhere**; the promised grad-norm-through-time figure was never made though the
   data sits in 47 JSONs. Workshop pick deadline ≈ Jul 11.
4. **Future work: directions sound, schedule incoherent.** One missing artifact — the
   **netlist/AIGER/RTL exporter** — is simultaneously P2's beat-4 demo, P3a's critical
   dependency, P3b's seed, and the Kyushu pitch's backing, and it is scheduled *after*
   everything that needs it. P4's park verdict survives but its "dead on accuracy" premise is
   an extrapolation (a live 2-way gated⊕clatch carrier exists).
5. **Race check (today): all novelty windows still open.** RDDLGN is published (EdgeFM@MobiCom)
   but still v1/no sequential follow-up; Bührer is doing combinational image compression. Zero
   hits on registers/write-enable/verification/FPGA for sequential LGNs.

---

## A. P2 — the three locked claims under audit

### A1. Confirmed (recomputed exactly from JSONs)
- **Accuracy tie**: psMNIST kb0 clatch 0.6340 {.5978,.6475,.6566} vs gated 0.6018
  {.5191,.6322,.6542} — edge is exactly the gated-s2 outlier (Welch p≈0.52). distcopy d20
  0.8739 vs 0.8734 (s1 is a 4-decimal per-seed tie); gated ahead at d8 (0.9357 vs 0.8770,
  clatch s1 collapsed 0.754). "ZERO tasks where clatch beats gated" — confirmed.
- **copy-50**: clatch+ds 3/3 disc=1.000; without ds at chance; gated ds-only 2/3;
  gated+margin+ds 3/3 → method (deep-sup) evidence, not primitive. Confirmed.
- **Stability numerator**: 0 non-finite steps in **all 75** register-family runs (clatch 23,
  latch 25, combo 27); the only 5 runs anywhere with n_skipped>0 are all gated
  (16695, 13339, 1477, 903, 2082). rddlgn/lstm/gru_cell all 0. Confirmed (and stronger than 0/72).
- **Gap numbers**: clatch {+.0087,+.0232,+.0273} mean +.0197 vs gated {+.0384,+.0766,+.1459}
  mean +.0870; non-overlap margin 0.011; ratio 4.41×. Numbers exact.
- **Honest scope caveats** ("gated fine at L=101/112/128", "instability clusters at high
  lr/margin/hard-long") — confirmed, 4/5 explosions fit the taxonomy (L50cap loosest fit).
- Parity-dense (only tff moves, 0.57-0.59), inventory ("all figures have JSONs") — confirmed
  for beats 1-3.

### A2. Problems in the locked claims (all verifier-CONFIRMED)
1. **"Matched selcopy-L100 pair" is NOT matched** — gated kb=3.0 vs clatch kb=1.0 (+ clatch-only
   anneal); everything else identical. keep_bias is the project's own documented explosion
   driver (draft §3.4; log 06-10 Jacobian argument), and the log's 07-07 entry itself lists this
   exact mismatch as a **verified confound** (04_log:77-78) on a task it calls structurally unfit
   (K=1 OR-solvable, ~24% ill-posed deep-sup). Gated also *wins the pair on accuracy* (0.499 vs
   0.375). These are the only two selcopy-L100 files — no kb-matched pair exists.
   **Fix:** reword ("same config at each mechanism's task-tuned kb") or 2 reruns
   (clatch@kb3 + gated@kb1, ~40 min each).
2. **The census is config-confounded** (reviewer kill-prob ~85% as framed): 4 of 5 gated
   explosions occurred at configs the register family **never ran** (lr 0.01 ×2, hidden 2048,
   margin_reg 0.1); all 75 register runs are lr 0.003 / hidden ≤1024 / margin 0. Also ~32/75
   register runs sit at/near chance (dead nets don't explode), and rddlgn is equally 0-skip —
   "never destabilizes" doesn't separate register from the prior-art control.
   **Fix:** stratify the census ("within lr=0.003/no-margin: gated 1/50 exploded"), lean on the
   trained-hard clatch runs (copy-50 1.000, distcopy 0.88-1.0, psMNIST 0.66, all 0 skips).
3. **The 4.4× gap is partly by-construction** (kill-prob ~75% as framed): train.py:348-349
   forces `hard_alpha=1.0` before clatch's soft eval, so clatch's gap measures only gate
   selection while gated's also includes activation drift. The 07-08 log concedes this verbatim
   ("partly BY CONSTRUCTION… NOT independent evidence", 04_log:26-27); A0' headline doesn't.
   Also: asymmetric outlier handling (s2 discounted for the accuracy tie, included in the gap
   mean — without s2 the ratio is ~2.9×), and the "at equal accuracy" qualifier needs the same
   s2 discount. **Fix:** reframe as a **design property** — "clocking the enable makes the
   training forward deploy-consistent, at no accuracy/trainability/gate cost" — which IS the
   paper's actual thesis and is defensible.
4. **A0' self-contradiction on gap sign**: L33 declares the gap-SIGN axis REFUTED; L47
   re-asserts "clatch never leaks upward" *unscoped*. Globally false by the project's own data
   (clatch +0.1121 selcopy-L100, +0.041 psMNIST kb1, +0.009..+0.027 kb0, +0.018 pargen). True
   only on distcopy. **Fix:** add "on distcopy" inline at workmap:47.
5. **distcopy tie is n=2 seeds** — violates the project's own ≥3-seed headline rule
   (workmap §G:455); the log even mislabels it "single-seed". Per-seed the tie is robust, so a
   third seed is insurance, not rescue (4 runs).
6. **"Deep supervision closes the gap" over-generalizes**: evidence is saturating copy-50;
   the own distcopy run contradicts it off-saturation (gated ds=0.2 → gap +0.122); psMNIST
   headline runs never use ds. **Fix (writing only):** scope to "per-step supervision fixes
   50-step credit assignment on hold tasks, dissolving the never-write plateau".
7. **Beat-4 has zero artifacts** ("verifiable deployable register"): no RTL emitter, netlist,
   synthesis number, or model-check script anywhere in repo or git history; A0's "all
   figures/tables have JSONs" is false-by-omission for beat 4; workmap H3 still lists the
   demo as an *open* decision while M5 still promises it. The word **"verifiable" in the locked
   headline currently has no in-paper evidence** behind it.
8. **clatch deploys to the identical circuit as gated** (one `_ste_round` on the gate signal
   that argmax binarizes anyway; cells.py:350 comment: "alpha=0 => this IS gated"). A reviewer
   from the Mind-the-Gap school will note the "primitive" is a train-time STE placement choice.
   **Fix:** own it — "round the control, never the state" as a placement result, with the
   never-write collapse as its proof; stop calling clatch a *new deployable* primitive.
9. Smaller: census denominator stale (0/72 → 0/75 current; 67 unique stems; 8 duplicate JSON
   pairs from a cp50A queue race; 13 early-June gated/rddlgn JSONs predate the n_skipped
   counter); psMNIST gated-s0 is the 2026-06-18 run (older schema) mixed with July s1/s2; the
   07-08 log's "4-way table" rddlgn ~0.63 traces to no matched run.

### A3. Remaining reviewer attacks (red-team, verified)
- **No positive accuracy result anywhere** — "competitive, not superior" is literally "never
  better, sometimes worse". Defense: make the integrity spine the explicit contribution + land
  beat-4 as a measurable artifact; report d8/selcopy losses in the main table.
- **No float RNN baseline in the entire repo** (copy_lstm/gru files are P1's *logic-native*
  ablations, single-seed, mostly dead; harness can't even train nn.GRU today). A ~50k-param
  float GRU on copy-50/distcopy/psMNIST-28 will near-solve everything; the paper must own that
  (LGN value = gates/energy/verifiability). Also rename the task **psMNIST-28 (chunked)**
  everywhere — no run uses the standard 784-step version (published LSTMs ~0.90 there vs 0.63 here).
- **Collapse-as-strawman risk** (beat 1): scope it — combo does NOT collapse on psMNIST
  (0.624, healthy gates); collapse is a hold-task/high-kb regime phenomenon; kb6 init is a
  co-cause (drives set/reset nets to 100% FALSE). The gate-histogram evidence is figure-ready.

---

## B. P1 — draft validation ("Gating Enables Long-Range Recall…")

**Every table cell reproduces exactly** (copy 0.96/0.79/0.33 3-seed means; control ≈0.25 with
soft at chance & grad-norm[t=0]=0; delayed-MNIST seed-0 column; psMNIST kb sweep; eqgates
0.655 vs 0.632; "0 skipped steps"; LSTM dC'/dC ≈ 0.58). Citations check out. Issues:

1. **(major)** Delayed-MNIST table is seed-0 = what is now best-of-3 (d100 0.339 vs mean
   0.3001±0.061; d50 0.362; d0 0.715). Seed 0 was run *first* (06-19), so not cherry-picked —
   but re-base on means before submission. "~3× chance through 100 blank steps" survives at
   exactly 3.00× mean.
2. **(major)** 14 completion runs (s1/s2 at d0/50/100; kb{0,3} at d50; d25/d75 ×3 seeds) exist
   in results/ but are in **no log entry** (log has nothing between 06-21 and 07-02), the
   checklist still lists them as "REMAINING TO RUN", and the draft still says "[TODO: seeds]".
   The kb sweep completes the task-dependence story (d50: kb0 0.177 / kb3 0.293 / kb6 0.369 —
   recall wants HIGH kb, the mirror of psMNIST). d75 mean 0.274 < d100 0.300 (within noise).
3. **(major)** Plan's min-experiment set unmet: adding(T) intentionally omitted
   (data.py:34 — fine, but say so), **plain sMNIST never run**, copy stops at T=50, and the
   **grad-norm-through-time carousel figure** — which 06_paper_plan:33-35 says is what stops
   the "incremental bolt-on" review — was never produced though grad_profile arrays sit in 47
   JSONs (e.g. gated L50 [109.8, 98.2, …] vs rddlgn grad_profile[0]=0.0). Cheapest rigor item left.
4. Minor: abstract/captions say control "collapses to chance" — discrete is 0.25 = 2× chance
   (only soft is at chance); fix wording or plot the soft curve. Priority claim needs one
   RDDLGN-memorization contrast sentence (their shift-≤12 recall predates "memory in a learned
   logic circuit"; "long-range" is the defensible qualifier). "FPGA/ASIC" → FPGA only.
   L35 seed spread {0.88, 0.50, 1.00} is bimodal — show spread. kb1-vs-kb2 ordering is noise.
5. Free wins already in results/: copy eqgates rddlgn control (h2048, still dead 0.126-0.248)
   converts the §5.1 gate-count defense from argument to data; the d25/d75 curve; the d50 kb
   sweep. And soften "LR decay + step-skipping fix exploding" for consistency with P2's census
   (same recipe skips 2082/20000 at selcopy-L100).
6. **Deadlines**: workshop pick ~Jul 11 (3 days); submission ~Aug 29. No bibtex yet.

---

## C. Code & pipeline audit

**Clean where it matters** (verifier-confirmed): gap = test_soft − test_acc on the same
best-val checkpoint, same split; no Gumbel/dropout/BN anywhere so soft eval is deterministic;
model selection on val, test touched once; psMNIST permutation fixed (seed 1234) independent
of run seed; paired seeds across arms; task generators correct (no off-by-one/label leak);
gated/clatch/combo/latch exactly gate-matched (4000/4096 both arms); state re-init per
sequence, full BPTT, no cross-batch leakage; clatch hold is bit-exact with unit Jacobian;
gated's deploy path is exactly binary (A0 KEY INSIGHT confirmed in code); rddlgn is a faithful
concat-recurrence control.

**Footguns / debt:**
1. **(major)** `--soft-state` is silently ignored for clatch (cells.py:344-350 never consults
   hard_state; combo/latch do) yet train.py records hard_state for it — a future ablation
   would no-op and mislead. Honor the flag or assert.
2. **(major)** **LogicLayer wiring (indices) is not in state_dict** — checkpoints are
   RNG-path-dependent; cross-seed loads succeed silently and compute garbage. Blocker for
   P3a/P3b (a checkpoint alone cannot reconstruct the circuit). Register indices as buffers
   before any emitter work.
3. JSONs omit load-bearing args: --distractors (d8 vs d20 lives only in filenames!),
   --alphabet, --chunk, --delay, --sel-flag, --init-from, --entropy-ramp, tag. 2-line fix in
   train.py's record dict.
4. run_queue.sh GPUS=(0 1) round-robin voided the curriculum ladder's ordering guarantee
   (warm-start *did* engage — c50 disc=1.000 impossible fresh — but evidence lives only in
   non-archived DUST logs; train.py:241 silently falls back to fresh start). Commit the queue
   logs for headline runs.
5. Hygiene: 8 duplicate JSON pairs (cp50A double-execution; identical except train_minutes) +
   9 CLI-smoke JSONs pollute any glob census; plot.py has no dedup and its per-group "highest
   kb wins" is post-hoc selection; scheduler advances on skipped steps; docs/design.md §6
   still says latch = PARKED/NotImplementedError; MNIST-task "gap" includes an
   input-binarization component; distcopy distractors can equal the target (effective
   overwrite pressure ~7/8 — one docstring line).

---

## D. Future work — P3a / P3b / P4

### D1. P3a (sequential-verification framework) — directionally sound, NOT sound-as-planned
- **Premise code-true but family-wide**: at eval *every* mechanism (gated, rddlgn included)
  argmaxes to an exact-binary deterministic clocked FSM — so the verification-*enabling*
  property is not clatch-specific and **ISTA does not need P2 to do sequential verification**.
  P2's real contribution to P3a is quantitative (accuracy retention + stability), not enabling.
  Restate the moat as "first + accuracy-retention + stability + only group with trained
  sequential-LGN checkpoints"; speed-to-arXiv is the actual defense.
- **No spec**: the entire plan is 2 sentences (workmap:89-90, 17_concepts:143) + one PURSUE row
  (BMC/IC3-PDR, ABC/nuXmv, AIGER). Missing: exporter, property list per task, task suite
  (parity vignette is dead — nothing learned parity), tool choice (BDDs hopeless at ~1000
  latches; IC3 at that size untested), GroupSum/popcount head encoding, contribution beyond
  push-button MC (which 16_crosspollination itself calls "thin"), schedule.
- **Dependency inversion**: the netlist/AIGER exporter has zero code and is implicitly
  scheduled inside P3b (Apr–Sep'27) — *after* P3a's "instant" start (~Feb'27). P3a's realistic
  solo window is ~2 months; "full framework" is not credible in it.
- **Cheapest falsifying test (do before committing)**: export one existing disc=1.000 copy-50
  clatch checkpoint → ABC `pdr`/`bmc3` on a "writes-only-on-cue"/hold invariant. One afternoon;
  de-risks or reframes the whole paper. (Note: BMC unroll depth is 28, not 784 — all psMNIST
  artifacts are chunk-28.)
- **P2-boundary conflict unresolved**: A0 locks a "compact clocked-verification DEMO" as P2
  beat-4; its trigger failed, host task died, A0' is silent. Decide explicitly: demo-in-P2
  (best host: copy-50/distcopy hold invariant) vs. explicit descope sentence.
- **Threats**: ISTA (Kresse/Yu/Lampert/Henzinger, 2505.19932 — still not in 02_papers.md!);
  **R-DTLGN group named DFA-extraction-for-model-checking as future work** — add
  Damera/Puranic/Baras/Belta (UMD/BU, 3 papers in 4 months) to the tripwire list.

### D2. P3b (FPGA RTL + Kyushu drone thesis) — strategically sound, operationally incoherent
- Method-gated moat intact; clatch actually *strengthens* it (write-enabled register → 1:1
  FF-with-clock-enable, cleaner than SR).
- **The prescribed D-FF demo never entered P2** (zero RTL artifacts ever, per git history);
  beat-4 quietly became mapping-on-paper; H3 still "open", M5 stale — same document says both.
  The hardware-first race defense (10_fpga:133-135) is currently forfeited.
- **Calendar dead**: FCCM'27 (~Jan) / FPL'27 (~Mar-Apr) deadlines precede the Apr–Sep'27
  emitter build → de facto FCCM/FPL'28, a year past the scout's own 3–12-month ETH window
  (expires ~Jun'27). Re-baseline venues (FPL/FCCM'28 or NeurIPS'27 demo track) and pull a
  minimal emitter + yosys/Vivado synthesis report into Q1'27 as the hardware timestamp.
- **Drone gate not run**: 16_crosspollination's own blocking gate (tiny-sim-POMDP closed-loop
  trainability, "BEFORE any drone commitment", prescribed "immediately" on 07-01) has zero
  code and zero runs. All trainability evidence is open-loop. Favorable alignment nobody wrote
  down: **distillation-from-PID provides per-step targets = exactly the deep supervision P2
  proved necessary** — record it, then run the gate before confirming the thesis topic.
- **Kyushu pitch over-promises**: 3 of 5 named contributions (emitter, FPGA synthesis,
  verification demo) have no code and no build slot before Apr'27, pitched to a lab Risk-5 says
  can't scaffold them. De-scope the pitch or front-load emitter v0.
- Size feasibility recomputed here and fits (4k gates + 1k FFs vs XC7A15T 10.4k LUT6/20.8k FF)
  — but no doc does this arithmetic, and all ns/nJ figures are imported from feedforward papers.
- Stale docs: 07_venues still says ICLR'27 for P2; 06_paper_plan has no P3a/P3b split.

### D3. P4 (learnable latch vocabulary) — park verdict SURVIVES, three corrections
- The psMNIST/distcopy tie premise **holds** on recomputation (if anything conservative —
  gated ahead at d8).
- **(1) "Dead on accuracy" is an extrapolation**: the tie is between *homogeneous fixed* cells;
  no learned-mix run exists; in DARTS-RNN the searched cell beats both endpoints. And the
  repo's own numbers expose a live, P3b-independent carrier — **the 2-way gated⊕clatch blend
  (malcolm's actual 17_concepts §2b framing, which the scout never scoped)**: gated soft mean
  0.689 vs best hard 0.634 = ~3.5pt harvestable headroom; a mix holding gated's soft ceiling at
  clatch's +0.02 gap deploys ~0.67 > both. Testable on existing infra, zero synthesis.
- **(2) Split the gate**: tier-(a) workshop is gated on P2-arXiv + a cheap heterogeneous-softmax
  trainability spike (runnable today), NOT on P3b; tier-(b)'s decisive ASIC measurement is
  outside P3b's documented FPGA-only scope (no yosys/OpenLane anywhere) — restate as FPGA
  LUT/slice/Fmax win, or add an open-source ASIC step to P3b.
- **(3) Prior-art fix**: RDDLGN does **not** "explicitly name flip-flops/latches as future
  FPGA work" (10_fpga:37-40's verified full-text read: analogy only; the correct evidence is
  BitLogic's "stateful modules" + shared first author). Also: the scout miscites §A0' (it's
  §A0, which still says "P2.5/P3" — add a one-line P4-verdict pointer to the workmap), and
  R-DTLGN is ternary-PST, not a 16-gate softmax. 02_papers.md has none of the scout's cites.

### D4. Race check (fresh, 2026-07-08) — ALL WINDOWS OPEN
- **P1/P2/P3a/P3b/P4 all clear.** Only recurrent/sequential LGN papers remain RDDLGN and
  R-DTLGN (ternary STL monitoring, explicitly no registers/latches). Zero hits: sequential-LGN
  verification, sequential-LGN-on-FPGA, write-enable/clocked state, recurrent discretization
  gap/stability.
- RDDLGN: published at EdgeFM@MobiCom (ACM 10.1145/3737902.3768357), arXiv still v1; Bührer's
  current project is GIC-DLC (combinational image compression) — **the feared ETH sequential
  follow-up has not appeared**. No DiffLogic-CA follow-up; no new Petersen recurrent work.
- New cites to add: **Dhayalkar, arXiv:2604.01228** (LG-TS-FFN, differentiable logic gates +
  recurrence for exact AFA simulation, IEEE Access 2026 — ADJACENT, cite in P2/P3a related
  work); arXiv:2606.18918 (BNN robustness-verification complexity — P3a background);
  2604.12092 (UMD/BU ternary temporal behavior trees). Main medium-term P3a risk =
  the UMD/BU ternary group.

---

## E. Missed things & quick wins

1. **Doc-18's #1-ranked insurance separator (n-bit memorization) was never built** even though
   the contingency it insured against (both core separators failing) occurred — the "ZERO
   tasks separate" verdict rests on 4 of the doc's 5 tasks. Run one pair or record the skip
   decision explicitly.
2. **selcopy/distcopy are documented in NO seqlgn doc** — the two tasks carrying the locked
   claims exist only in data.py docstrings + run_queue comments; doc 18 never got its round-5
   correction (its predictions are all falsified; "see research/18" pointers dangle).
3. **Orphan anomalies**: selcopy --sel-flag ablation's discrete metrics are *exactly* identical
   (4 decimals) to the no-flag run while being a genuinely different model — analyzed nowhere,
   contradicts doc-18's re-saturation prediction; copy eqgates rddlgn L20 anomaly unanalyzed.
4. **Free figures from existing JSONs**: (a) gate-count + train-time table — clatch costs the
   same gates and ~same minutes as gated (supports "the stable register costs nothing"), and
   gate count does NOT discriminate (kills a lingering 07-04 hope); (b) state_mushy_by_t drift
   curves (3 JSONs) → the beat-2 mechanism figure; (c) P1's grad-norm carousel plot (47 JSONs).
5. **Open decisions never formally closed**: Gumbel+STE/IWP "bake-in" promise (obsoleted by
   deep-sup — say so in one backlog line); WMT'14 head-to-head decision; workmap §H still
   headed "NOT yet locked" with H1/H2/H3 contradicting A0/A0'.
6. **Provenance**: scratchpad evidence artifacts cited by the log (collapse_*, scoping_out.txt,
   validate_*) are not in the repo; the retracted "distcopy L20→L40 CPU smoke" has no original
   log entry; two silently-refuted early conclusions (06-11 "0.75 capacity ceiling", 06-10 "gap
   orthogonal to gating") carry no correction annotation.
7. **Branch hygiene**: local main is 29 commits behind p2; origin/main another behind. Merge
   p2→main + push before arXiv; tag the commit the paper numbers come from.
8. **No P2 draft skeleton exists** despite A0' "LOCK TRACK B, WRITE".

## F. Ranked action list

**Writing-only (no GPU), highest value:**
1. Patch workmap A0': drop/scope "matched" (pair), scope "never leaks upward" to distcopy,
   update 0/72→0/75 (state the counting rule), reframe the gap edge as a design property,
   scope the deep-sup headline to hold tasks, resolve the beat-4 demo (build-or-descope), close
   §H, add the P4-verdict pointer.
2. P1: re-base delayed-MNIST on 3-seed means, ingest the 14 orphan runs (one catch-up log
   entry + checklist ticks), fix "chance" wording, add the RDDLGN-contrast sentence and the
   two citation fixes. Venue pick by ~Jul 11.
3. Start the P2 draft skeleton with the corrected claim structure.

**Cheap code/tooling (no GPU):**
4. **Netlist/AIGER exporter + ABC smoke + yosys synthesis report on an existing clatch
   checkpoint** — the single artifact that simultaneously fixes P2 beat-4, de-risks P3a,
   re-arms the P3b hardware timestamp, and backs the Kyushu pitch. Prereq: persist LogicLayer
   indices (buffers) — checkpoints alone can't reconstruct the circuit.
5. Add missing fields to train.py's record dict; guard clatch+--soft-state; dedup-aware plotting.
6. P1 grad-norm-through-time figure from existing grad_profile arrays.

**Cheap GPU (~40 min/run on DUST), in order of claim-repair value:**
7. selcopy-L100 kb-matched pair: clatch@kb3 + gated@kb1 (2 runs) → repairs the flagship pair.
8. distcopy d8/d20 seed-2 (4 runs) → headline meets the ≥3-seed rule.
9. psMNIST kb0 +2 seeds/arm (6 runs, optional) → stabilizes the non-overlap; symmetric outlier
   policy.
10. Float GRU baseline on copy-50/distcopy/psMNIST-28 (needs small nn.GRU harness addition) —
    price the accuracy-for-hardware trade before a reviewer does.
11. Optional: one n-bit-memorization pair (closes the doc-18 hole); psMNIST kb0 + ds>0 ×3
    (tests the method claim on an integration task).

**Before Kyushu/thesis commitments:**
12. Run the tiny-sim-POMDP closed-loop gate (T-maze/flickering-obs + distilled-PID head);
    record the distillation≡deep-sup alignment in 16/workmap; re-baseline P3 venues to '28 or
    NeurIPS'27 demo track; de-scope or back the Kyushu pitch.
