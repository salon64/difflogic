# Experiment Log

Newest first. One block per run/idea. Keep it terse and honest (record failures too).

Template:
```
## YYYY-MM-DD — <short title>
- Hypothesis:
- Setup: (script, dataset, model, key hyperparams, seed, hardware)
- Result: (val/test acc, gates, train time)
- Read: (what it tells me / next step)
```

---

## 2026-07-10 (pm) — P3a round 2 (8-agent build+verify): decode THEOREM proved; combo provably NOT distractor-robust; 901-LUT synthesis; self-contained checkpoints
- **Setup:** multi-agent workflow (4 build/experiment chains, each adversarially reviewed by an
  independent agent that re-ran every gate; 455k tokens, ~2h). All results below survived review.
- **(1) In-netlist GroupSum head + the full-correctness theorem.** New `mlgn/netlist/head.py`
  (popcount adder trees + comparators + exact torch first-max argmax; 11,735 gates for the (8,128)
  head; bit-exact vs numpy on 20k+ cases incl. engineered ties + exhaustive small heads).
  **`protocol_decode` PROVED on combo copy-35** (19k gates / 1,049 latches; recipe ~15 s): for every
  legal write and EVERY readout time ≥ 16, the deployed circuit's readout equals the written symbol —
  write→settle→readout correctness at arbitrary delay as ONE model-checking query. Bonus: the
  mandated non-vacuity control caught a latent same-layer-read bug in protocol_hold/seq_hold
  (invisible to ABC, garbage under python sim) — fixed; lesson: ALWAYS ship a corrupt-shadow control.
- **(2) Distractor campaign (`mlgn/netlist/out/distractor_study/report.md`) — the P3a free-input
  experiment.** Copy-trained combo is **provably not distractor-robust in ANY variant**: hold CEX at
  the first armed frame (one echo of the written symbol moves the state); decode-during-settling CEX;
  headline = decode-after-settling **CEX @ frame 30 (sym5) found by BFS-guided bmc3 -F 40 in 276 s
  after exhaustive enumeration ESCAPED its 200k cap with zero wrong states and random testing found
  nothing** — random sim < enumeration < directed bounded MC, a strict hierarchy on the same circuit.
  Gated fails everything at frame 22 without distractors. Method split confirmed: tempor→scorr→pdr
  proves TRUE properties; with free inputs scorr stops collapsing (806 latches survive, pdr stalls) —
  FALSE properties need simulation-guided bounded MC. 9/9 cexes replay-confirmed bit-exact;
  BFS↔MC cross-validation 0 contradictions. So the combo certificate is precisely: "correct at any
  delay iff the channel stays silent" — an envelope statement no L=35 test set could expose.
- **(3) RTL + synthesis (`mlgn/netlist/synth/report.md`) — hardware timestamp re-armed with real
  numbers.** `verilog.py` emitter; iverilog golden-vector equivalence 8/8 symbols (327,680 state
  bits, 0 mismatches; post-synth netlist re-passes); yosys synth_xilinx: **901 LUTs + 885 FDRE**
  (BLIF cross-flow 1,020/1,019) = **8.7–9.8% of XC7A15T LUTs / 4.3–4.9% FFs, zero BRAM/DSP/carry** —
  the §D2 "4k gates + 1k FF fits" estimate confirmed w/ ~4× margin. OSS CAD Suite in WSL. No P&R/Fmax
  yet; head-included synthesis pending.
- **(4) Footgun #2 CLOSED upstream:** `difflogic/difflogic.py` LogicLayer.indices → property backed
  by persistent conn_a/conn_b buffers; old ckpts load under strict=True unchanged (replay semantics
  kept); NEW checkpoints are self-contained and `--init-from` now restores wiring too; CUDA backward
  helper refreshed on wiring change (stale-scatter-gradient hazard found and averted). 7/7 tests +
  end-to-end falsify regression + reviewer probes (deepcopy/pickle/prefix handling) all pass.
- **(5) Ladder study COMPLETE (9 ckpts, all gates 1.0000; `mlgn/netlist/out/ladder_summary.md`):
  three solution families at IDENTICAL discrete accuracy, distinguishable only by verification.**
  Seed 2: fixed points from the first rung — hold + anyx0 PROVED at c8/c20/c35 (2–5 s each).
  Seed 0 (cp50A): progressive crystallization — cycling inputs 48 → 32 → 0 along the curriculum;
  hold provable from c20, anyx0 only at c35. Seed 1 (cp4_s1): NEVER crystallizes — period-2/6
  limit cycles persist through c35 (48 → 28 → 40 cycling inputs), hold genuinely FALSE (CEX at the
  first armed frame, sim-confirmed; an apparent sim↔MC contradiction en route was my misread —
  max-settle printed over the settled subset only). **BUT the decode theorem PROVES on the
  oscillator seed at every rung (~1 s + bmc3 clean 131+ frames): the state never stops moving for
  5/8 writes, yet every reachable orbit state decodes to the written symbol forever.** ⇒ Hold-type
  certificates separate solution families; the decode-type certificate (needs the in-netlist head)
  is the right deployment spec — a P3a thesis statement. Also: settle depth does NOT shrink along
  the curriculum (12–15 everywhere); curriculum changes orbit STRUCTURE, not transient length; the
  L=8 rungs read out at t=7, before their own settle — early-curriculum circuits are orbit decoders.
- **Read:** P3a now has its method core (recipe + hierarchy + envelope certificates), its
  quantitative contrast (combo vs gated), its hardware leg, and a natural headline experiment
  (train distcopy → does training buy provable robustness?). Next: clatch + distcopy checkpoints
  from DUST; head-included synthesis; Fmax.

## 2026-07-10 — P3a falsifier EXECUTED: exporter works; two theorems PROVED; naive IC3 fails, tempor recipe closes it
- **Hypothesis (research/20 §D1 gate):** export one disc=1.000 copy checkpoint → ABC
  `pdr`/`bmc3` on a hold invariant; one afternoon de-risks or reframes P3a.
- **Setup:** NEW `mlgn/netlist/` (extract / ir / sim / props / blif / falsify).
  Target `ckpt_cp50A_curr_c35.pt` (combo copy-35, h1024, seed 0, disc test 1.000 —
  no clatch ckpt exists locally; at eval combo/clatch/gated deploy to the same
  MUX-register circuit). ABC built in WSL (`make ABC_USE_NO_READLINE=1`). Wiring
  reconstructed by RNG replay (indices aren't in state_dict — footgun #2): seed →
  the two `randperm`s per LogicLayer in module order; synthetic tasks consume zero
  global-CPU draws so the stream position is exact. CPU host, torch 2.10.
- **Gates passed:** rebuilt disc acc **1.0000 == recorded** (full 10k, survives the
  DUST→local torch change); netlist **bit-exact** vs torch (9.2M state bits, 0
  mismatches). Netlist: 9 PIs, 1024 latches, 7168 gates (candidate MLP + gate MLP +
  3-gate MUX/bit).
- **Finding 1 — the learned register is an ATTRACTOR, not a latch:** after the cue
  write, the state keeps changing 9–15 blank steps (legal) before hitting a
  per-symbol fixed point; decode is WRONG during the transient (defaults to class
  2). Exhaustive closed-system sim of ALL 512 first inputs (post-write inputs are
  forced blank ⇒ complete case analysis): every input settles, garbage ≤ 18 steps,
  slowest = multi-hot conflicting writes ({1,7}), ZERO limit cycles.
- **Finding 2 — verdict table on the true K=16 hold invariant (1041 lat / 8183 and):**
  `pdr` timeout 900 s @ frame 16; `ind` timeout 300 s @ frame 17 (204k conflicts);
  `scorr;dc2;pdr` timeout 600 s; `bmc3` clean to 310 frames (bounded only);
  **`tempor -F 20; scorr; dc2; pdr` → PROVED in ~2 s** (property cone collapses to
  const-0). anyx0 variant (K=19): PROVED in ~3 s. **Recipe = settle bound by
  simulation → temporal decomposition past it → inductive sweeping.** The ablation
  shows tempor (with the sim-derived K) is the unlock, not scorr.
- **Theorems (machine-checked, deployed circuit):** (i) every legal write is held
  FOREVER after 16 blank steps + fixed points decode correctly (8/8) ⇒ functional
  correctness at ARBITRARY delay ≥ 16; (ii) NO first input (incl. garbage) can
  cause a non-converging orbit (settles ≤ 19, then fixed forever).
- **Finding 3 — contrast run `ckpt_cpB_gated_oracle` (gated copy-50, disc 0.3803)
  EXPLAINS the accuracy mechanistically:** the gated circuit is ALSO provably
  stable (all 512 x0 settle ≤ 21, zero cycles; both K=22 hold theorems PROVED by
  the same recipe in <1 s; naive pdr again dies at the arming frontier, frame 22)
  — but only **3/8 fixed points decode correctly** (syms 0,2,6), and 3/8 = 0.375
  ≈ 0.3803: the "38%" is a deterministic 3-of-8-attractors map, not noise. Both
  ckpts have test_soft ≈ 0.88 → soft accuracy can't see any of this. **Stability
  is architectural (both mechanisms); correctness = which fixpoints the write
  dynamics select.**
- **Read (P3a verdict): DE-RISKED with a reframe.** Push-button IC3 at ~1000
  latches genuinely fails (as research/20 feared) — but the paper doesn't need it:
  the method contribution is the sim+tempor recipe, and the real MC territory is
  FREE-INPUT properties (distcopy "no distractor corrupts the register" — needs the
  popcount/GroupSum comparator head, the known missing encoding) plus the
  stability-vs-correctness cross-mechanism contrast. Full details + artifacts:
  `mlgn/netlist/README.md`, `mlgn/netlist/out/*/`. P2-boundary decision (beat-4
  demo vs descope) now has its artifact and stays open for malcolm.

## 2026-07-10 — P1 hardening queue (p1f_*, 23 runs) ingested → draft v1.1; P1 experiments CLOSED
- **Setup:** `mlgn/paper/p1/run_queue_p1.sh` (Tiers A+B) on DUST. All 23 complete, **0 skipped
  steps in all 23**. Addresses validation §B (20_program_validation.md). Ingested together with
  the P2-backfill spillover that touches P1 (psm gated-kb0 s3/s4, eqgates dupes — entry below).
- **Results (now in `mlgn/paper/p1/p1_draft1.md` v1.1):**
  - **psMNIST-28 §5.3 boundary, all arms seeded:** control eqgates (h2000) **0.674±0.017** (n=3),
    control h1000 0.629±0.012 (n=3), gated kb0 **0.616±0.056** (n=5, incl. the P2-queue s3/s4;
    gap mean +0.079 vs control's +0.034). Control ahead 5.8pt; **note:** gated-s3 (0.6555) edges
    control-s0 (0.6552) by 3e-4 ⇒ strict seed non-overlap is FALSE — worded as "no gated seed
    reaches the control mean," NOT non-overlap. gated kb4 n=3: 0.431±0.080 (sweep endpoint firm).
  - **dMNIST-50 EQGATES CONTROL (new, h2000 = 4k gates):** disc 0.117, soft 0.1135 (majority
    baseline), **grad_profile[0] = 0 exactly** — the zero-write-step-gradient failure is
    gate-count-independent ON THE HEADLINE TASK (previously only argued from copy).
  - **dMNIST-50 kb sweep n=3:** kb0 {0.177,0.135,0.152}=0.155±0.021, **soft = 0.1135 in 3/3
    (cold start replicates)**; kb3 {0.293,0.241,0.411}=0.315±0.087 — kb3≈kb6 within noise at
    D=50; the necessary ingredient is kb>0.
  - **copy controls n=3 + eqgates-L20 anomaly RESOLVED:** narrow 0.25±0.00/0.21±0.07/0.25±0.01
    (L20/35/50), eqgates 0.21±0.07/0.25±0.00/0.29±0.07; s0's eqgates-L20 0.126 was seed noise
    between dead circuits. Soft pinned at chance (0.118–0.130) in ALL 18 control runs.
    ⚠ eqgates-L50-s1 discretizes to **0.369 from a chance soft** — rounding a dead circuit can
    scatter high ⇒ control disc = *scatter around a degenerate predictor* (0.13–0.37), never a
    learned level (and gated's deployed L50 margin over that scatter is small — gap-bound, §5.5).
- **Also this session:** canonical per-figure scripts `mlgn/paper/p1/fig{1,2,3}_*.py` +
  `figstyle.py` (Okabe-Ito, seed error bars, P1 hygiene filter — mechanisms ∩ no ds/margin/anneal,
  same-seed dupes collapsed — enforced in the loader; outputs `figs/*.{png,pdf}`), incl. the
  long-owed **grad-norm carousel figure (Fig 3)**; `seqlgn/plot.py` got the same filter (doc-20
  debt). §3.4 stability caveat re-worded to the kb×horizon driver (consistent w/ 07-09 pair).
- **Read:** P1 experiments CLOSED (Tier-C extras stay commented in run_queue_p1.sh). Next:
  workshop pick (~Jul 11) → condense to 4pp → verify bib author names → submit (~Aug 29).

## 2026-07-09 — P2 backfill queue RAN (all 27): stability edge REFUTED by the matched pair; gap edge HARDENED at 5 seeds
- **Setup:** `mlgn/paper/p2/run_queue_p2.sh` on DUST (both GPUs; logs in `mlgn/paper/p2/logs/`; results commit 9e52847):
  distcopy s2 ×4, selcopy-L100 kb-MATCHED pair ×2, psMNIST kb0 s3/s4 ×6 + rddlgn-eqgates s1/s2 + gated+ds probe ×3,
  parity-L16 panel ×10. Archive now 218 JSONs.
- **(1) THE PAIR FLIPPED THE STABILITY STORY (headline):** `sc_clatch_L100_kb3` destabilized — **8,044/20,000 skips**
  (disc 0.497) — while `sc_gated_L100_kb1` was CLEAN (0 skips) and set the best selcopy-L100 disc ever (**0.609**).
  ⇒ instability follows **keep-bias × horizon, NOT mechanism**; the hard-identity hold does not protect (the
  enable/candidate nets still get kb-shaped gradients over the unroll). **A0'/Track-B edge #1 (register stability)
  is DEAD as an edge** — kept as a *diagnosis*. Census now: gated 5/93 destabilized, clatch 1/31, latch 0/28,
  combo 0/29, rddlgn 0/27; every destabilization (either mechanism) at high kb / aggressive config on long horizon.
  (Validation report A2.1/A2.2 called this confound; the controlled pair confirmed it was real, not cosmetic.)
- **(2) Gap edge HARDENED:** psMNIST-28 kb0 @ 5 seeds/arm — clatch gaps +.009..+.027 (mean +.020) vs gated
  +.038..+.146 (mean +.079): **non-overlap HOLDS** (margin .011), ratio 3.9× (3.1× excl. gated-s2). Accuracy tie
  stands (clatch .643 vs gated .616, Welch t≈0.97; excl-s2 .643 vs .641). combo gaps ±.025 (mean −.005).
- **(3) distcopy n=3 — the outliers are SYMMETRIC:** d8 gated mean **0.955** > clatch 0.875 (clatch-s1 0.754);
  d20 clatch mean **0.916** > gated 0.874 (clatch-s2 hit 1.000); all other seed pairs tie to 3 decimals.
  ⇒ "ZERO tasks where clatch beats gated" is superseded by "**no consistent separation in EITHER direction**."
  Gap-sign story fully dead both ways (gated d8-s2 gap −0.120).
- **(4) ds HURTS integration (new negative control for the method scope):** psMNIST gated kb0 +ds(0.2) =
  **0.549** (3 seeds) vs 0.616 without — **−6.8pt**, as §6.1's scoping predicted (per-step final-label supervision
  is wrong off hold tasks). Strengthens, not weakens, the deep-sup claim by bounding it.
- **(5) parity-L16 panel:** same shape as L32 — **tff sole mover** (disc 0.576/0.607/0.607, soft up to 0.936),
  gated/clatch/rddlgn at chance. FIG-5 material (L∈{16,32}).
- **(6) rddlgn equal-gates control seeded (n=3): disc mean 0.674** — the concat-recurrence control is the MOST
  ACCURATE psMNIST-28 model (mirrors P1's equal-gates finding; gaps +.030..+.038). Honest Table-3 row.
- **Net:** the draft (`mlgn/paper/p2/p2_draft1.md`) was rewritten to the corrected claim structure — register's
  one verified edge = deploy-consistency (by construction, 5-seed non-overlap); stability = corrected finding
  (kb×horizon, mechanism-independent); accuracy = symmetric no-separation. The integrity spine got STRONGER: we
  ran the controlled test our own census implied and report the refutation. No further GPU runs planned; remaining
  pre-arXiv items are code/figures (FIG 1–3, exporter decision, float-GRU baseline).

## 2026-07-08 (combo completes the psMNIST table) — restore family viable off the long-hold regime; discretization edge is partly by-construction
- **Setup:** `psm_combo_kb0_s0/1/2` (combo = gated + hard-STATE restore), matched to the gated/clatch/rddlgn kb0 rows.
- **4-way psMNIST kb0 (chunk28, hidden1000, 20k; chance 0.10):** clatch mean **0.634** (gaps +0.009..+0.027),
  combo **0.624** (gaps −0.025..+0.021), rddlgn ~0.63, gated **0.602** (gaps +0.038..+0.146; s2 outlier 0.519).
- **Read:** (1) **combo does NOT collapse here** (0.624, no skips, healthy writing gates — no FALSE/TRUE collapse),
  despite dying at copy-50 via the never-write collapse. Honest cause = **task- not just length-dependent**:
  psMNIST is integration/classify-at-end, so the "never-write/hold-zero" attractor that killed combo on the
  copy-50 HOLD task doesn't exist here. ⇒ the bistable restore is a **viable mechanism on a real integration task**;
  the copy-50 failure was specific to long hold/recall (+ high keep-bias), not the restore per se.
  (2) **Discretization-cleanliness holds across the whole restore family** (clatch AND combo gaps ~±0.02, some
  negative; gated leaks +0.04..+0.15) — corroborates Track-B claim #2 for Table 2, **BUT partly BY CONSTRUCTION**
  (both round during training, so disc≈soft is baked in; NOT independent evidence, and NOT the refuted gap-sign axis).
- **Net:** doesn't change the Track-B verdict (accuracy still a wash: clatch≈combo≈rddlgn≥gated, no separation).
  Adds a clean 4th column for the psMNIST mechanism table + the "restore viable outside the pathological long-hold
  regime" point. No new runs prompted.

## 2026-07-08 (DECIDER: corrected separators RAN) — Track A DEAD; P2 headline LOCKED to Track B
- **Setup:** GPU. The two CORRECTED separators + fair psMNIST (round-5): `pd_*` parity-dense (--running-target),
  `dc_*` distcopy (matched kb3), `psm_*_kb0`. Analyzed by a 31-agent workflow w/ adversarial verification
  (22 confirmed / 3 partial / 0 refuted). See workmap **§A0'** for the locked plan.
- **Result (all verified):**
  - **PARITY-DENSE:** the running-XOR supervision WORKS (breaks the flat-gradient wall) but only **tff** moves
    off chance (disc 0.572/0.582/0.586, soft up to 0.828) — the toggle primitive on its home task. **clatch &
    gated stay at CHANCE** (0.50), rddlgn chance. **Misses the 0.9 win bar by ~0.32.** Not a clean separator;
    and the sole mover is tff, not the clatch the headline rested on.
  - **DISTCOPY (corrected hold separator):** **accuracy TIE at d20** (gated 0.8734 vs clatch 0.8739), gated
    AHEAD at d8 (0.936 vs 0.877; clatch s1 collapsed to 0.754). A faint gap-trend (gated gap 0.000→+0.067 with
    distractors, clatch ≤0) — but **single-seed & the gap-sign axis was REFUTED** (clatch's neg gaps = a worse
    soft optimum rounding recovers at tied disc). No primitive accuracy win.
  - **psMNIST kb0 (fair, matched keep-bias):** **accuracy TIE** (clatch 0.634 vs gated 0.602, the edge is a
    gated-s2 outlier 0.519) — REVERSES the earlier "clatch loses psMNIST" (that was a kb artifact). AND clatch's
    discretization gap is **tighter & non-overlapping** across 3 seeds (max clatch +0.027 < min gated +0.038,
    ~4.4× tighter) at equal accuracy.
- **VERDICT (VERIFIED): Track A ("the primitive wins on accuracy") is DEAD — zero tasks where clatch beats
  gated on accuracy.** P2 pivots to **Track B**: deep-supervision = the training method that closes the
  recurrent-LGN discretization gap; `clatch` = a **stable, cleanly-discretizing, verifiable deployable register,
  COMPETITIVE not superior on accuracy** (integrity spine). Two surviving edges carry it: (1) numerical
  **stability** (register family 0/72 non-finite steps; matched selcopy-L100 gated 2082 vs clatch 0 — scope it
  honestly, NOT "gated always explodes"); (2) **tighter discretization at matched psMNIST kb0** (non-overlapping).
- **CORRECTIONS to my own earlier reads (important):** length-generalization is **DROPPED** as a load-bearing
  edge — the only GPU length-gen runs are parity (all chance); the "distcopy L20→L40 gap-0" was a **single
  un-reproduced CPU smoke**, no GPU JSON (I over-leaned on it 07-07). Gap-SIGN axis **refuted**. copy-50 3/3 =
  **method** evidence (deep-sup), not a primitive carry.
- **NEXT: LOCK TRACK B, WRITE. No GPU runs needed** — all figures/tables have JSONs (copy_* method; psmnist_*_kb0
  discretization; distcopy_*/selcopy_*_L100 stability census; parity_*_pd_* mechanism panel). Optional nice-to-have:
  one L=16 parity mechanism-panel sweep (tff sole mover); skippable. Do NOT run distcopy-d40. ICML'27 buffer intact.

## 2026-07-07 (cp4 confirm + BOTH separators) — no clean clatch>gated on ANY task; separators broke on fixable+structural bugs
- **Setup:** GPU. Multi-seed copy-50 confirmations (`cp4_*`) + the two purpose-built SEPARATORS (parity
  length-gen `bh_parity*`/`bh_pargen*`, selective-copy `bh_selcopy*`) + psMNIST credibility (`bh_psm28*`).
  Analyzed by a 42-agent workflow with adversarial verification (21 confirmed / 14 partial / 0 refuted).
- **Result:**
  - **copy-50 CONFIRMS (but saturates, cannot rank):** clatch+deep-sup **3/3 disc=1.000**, gated+deep-sup
    **2/3** (s1=0.882), combo+curriculum 3/3. Caveat: clatch's 3/3 is disc-only (seed-0 soft 0.754,
    gap -0.246 — soft not tracking discrete). **deep-sup, not the primitive, closes the gap** (both need it).
  - **PARITY = DEAD for ALL mechanisms** (gated/tff/clatch/rddlgn all ~0.50 chance at L32 AND L128) — incl.
    **tff which computes parity by construction.** Length-gen runs are MOOT (byte-identical gate_totals =
    same untrained L32 model re-evaluated long). Cause (verified): **final-timestep-only supervision on the
    flat/deceptive XOR gradient** (every proper prefix-XOR is uncorrelated with the final label). NOT a
    GroupSum/fan-out wall (that story was refuted; fan-out is easy for a logic net).
  - **SELECTIVE-COPY = confounded AND structurally wrong:** gated L50 disc **0.626** > clatch **0.494**;
    gated wins L100 too. Three verified confounds, all favoring gated: (1) **mismatched keep-bias** (gated=3
    vs clatch=1 on a HOLD task where kb helps → clatch handicapped on hold); (2) **--deep-sup ill-posed**
    (~24% of steps t<pos demand an impossible target since the symbol sits at a random pos in [0,L/2));
    (3) **K=1 selcopy is OR-solvable** — one nonzero token among blanks never exercises hold-vs-overwrite,
    so it CANNOT test the thesis. Also: gated L50 shows **~0 discretization gap** (it did NOT leak),
    against the core "soft-MUX decays" prediction (caveat: 0.626 is a lossy partial solve, not a proven hold).
  - **psMNIST (clean, trained, goes AGAINST clatch):** clatch kb1 disc **0.570** (gap 0.041, tightest of any
    variant) vs gated kb0 **0.632**, rddlgn 0.62-0.66. At each mechanism's best config gated wins; clatch's
    shortfall is soft-CEILING (0.611 vs 0.709) not discretization. tff kb1 = 0.200 (partial collapse). Use
    psMNIST as a **no-collapse sanity check**, not a separator (integration task structurally favors the MUX).
- **Read (VERIFIED, honest):** we have **ZERO tasks where clatch beats gated on accuracy.** The one surviving
  asymmetry is **training STABILITY** (clatch n_skipped=0 everywhere; gated explodes — 2082/20000 skips at
  selcopy-L100, +0.50 gap at copy-50 w/o deep-sup). The stronger "clatch neg-gap holds / gated pos-gap leaks"
  claim was **REFUTED** (clatch shows +gaps on selcopy-L100 +0.112 and psMNIST +0.041) → defensible axis is
  STABILITY, not gap-sign. Separator failures are **mixed, leaning (B) the primitive doesn't separate** — with
  parity-dense the one unexhausted shot at (A).
- **FIXES BUILT (this session):** (1) **`--running-target`** (train.py) = per-step running-XOR deep-sup for
  parity (the dense-gradient fix); (2) **`distcopy`** task (data.py) = the CORRECTED hold separator (cued
  target at t=0 + `--distractors` non-cued tokens to hold THROUGH; deep-sup stays valid; not OR-solvable).
- **FORK (recommendation):** run 2 decisive corrected experiments this week — `parity-dense` (does tff/clatch
  cleanly beat gated with ~0 gap? = the Track-A gate) + `distcopy` (does clatch hold through distractors where
  gated leaks?). **Start writing Track B (obstruction-forward: never-write collapse + gated-already-binary
  reframe + deep-supervision as the training method + clatch as a stable, gap-tight, DEPLOYABLE register on
  the verification/hardware axis) in parallel — it's already supported by existing JSONs and is the honest
  default if the two reruns don't separate.** ICML'27 (~Jan/Feb'27) has months of buffer.

## 2026-07-04 (round 3 + Path-A RESULTS) — the gap IS closeable at copy-50; the fix is DEEP SUPERVISION, not the primitive
- **Setup:** GPU/DUST. Round-3 head-to-head (`cpB_*`, the two arms from the 5-agent insight) + Path-A
  rescues (`cp50A_*`). copy-50, alphabet 8 (chance 0.125), hidden 1024, 20k iters. gated baseline = disc
  0.33 / soft 0.83 / gap +0.50.
- **Result — THREE routes reach discrete 1.000; three fail:**

  | config | disc | soft | gap | note |
  |---|---|---|---|---|
  | `gated + deep-sup + margin` s0/s1/s2 | **1.000 / 1.000 / 1.000** | 1.0/1.0/0.879 | ~0 | robust, 3 seeds |
  | `gated + deep-sup ONLY` | **1.000** | 1.000 | 0 | deep-sup is the active ingredient |
  | `clatch + deep-sup` | **1.000** | 0.754 | −0.25 | primitive works *with* deep-sup |
  | `combo + length curriculum` c8→c20→c35→c50 | **1.000** all rungs | — | ~0 | 2nd independent route |
  | `clatch` ALONE s0/s1/s2 | 0.126 / 0.247 / 0.248 | ~0.12 | — | **FAILED** — didn't train even softly |
  | `gated + margin ONLY` | 0.126 | 0.126 | 0 | **DEAD — 903 skips** (margin exploded grads) |
  | Path-A rounded `combo`/`latch` kb0/kb1 | 0.126–0.247 | ~0.12 | — | never-write collapse, unrescued |

- **Read — the decision gate resolved, and NOT the way we bet:**
  - **The failure was never the primitive or the round — it was CREDIT ASSIGNMENT over 50 steps.** copy-50
    only supervises output-vs-input-50-steps-ago; the "write early, hold" signal is buried 49 steps deep and
    the easy escape is "write nothing" (the collapse). **Deep per-timestep supervision** (supervise the state
    at EVERY t) gives a local write/hold signal and dissolves the plateau. **Length curriculum** (learn
    copy-8 first, extend) fixes the SAME problem a 2nd way. **Rounding the state fixes neither** → all Path-A
    rounded variants stayed at chance.
  - **`clatch` alone did NOT close the gap** (0.126/0.247/0.248 across all 3 seeds; soft ~0.12 — it didn't
    even train softly). Our copy-8 clatch smoke did NOT scale. clatch only hit 1.000 *once we added deep-sup*
    — and so did plain `gated` (no latch at all). **The primitive is not the trainability win; deep-sup is.**
  - **`margin` loss alone is HARMFUL** (exploded, 903 skipped batches). The "margin+deep-sup" bundle is deep-sup
    carrying it. Drop margin from the recipe.
  - **Oracle:** gated best-ckpt state is barely mushy (mushy-fraction t0 0.008 / mid 0.031 / t−1 0.008), and no
    FALSE/TRUE gate collapse (gate.1 TRUE 27%, diverse) — consistent with "gated already ~binary; the gap is a
    small activation drift," which deep-sup removes.
- **THE CATCH (what we're missing): copy-50 SATURATES — it cannot rank the methods.** Everything that trains
  hits *exactly* 1.000, including plain gated+deep-sup with NO latch. So copy-50 accuracy CANNOT show clatch >
  gated. To justify a "clatch primitive" headline we need a **discriminating axis**: (i) a harder memory task
  where they separate (longer delay / distractors / a real POMDP-T-maze), or (ii) the **deployment axis** —
  gate count, exact-hold-under-perturbation, verifiability — where clatch's register structure is provably
  cleaner *regardless of the accuracy tie*. Copy-50 proved the gap is CLOSEABLE; it can't say by-what-is-best.
- **Caveat:** the winners we care about (`clatch+ds`, `combo+curriculum`) are **single-seed**; only
  `gated+deep-sup` is 3-seed-robust. → round-4 confirmation queue (below) before any headline lock.
- **Headline implication:** the clean "clatch trains cleanly on its own" claim is **falsified**. Honest arc =
  never-write collapse → gated-already-binary reframe → **deep supervision is what trains it** → clatch is the
  clean *deployable register* (verification/hardware axis), not the trainability hero. Leans toward the
  scoping panel's "obstruction-forward + deep-sup fix" fallback over the "primitive-as-hero" headline.

## 2026-07-03 (5-agent ideation) — the round was UNNECESSARY and CAUSED the collapse; escapable
- Ran a 5-lens workflow (`scratchpad/collapse_{lenses,synth}.txt`) on the never-write collapse.
  **Verdict: ESCAPABLE, not fundamental.**
- **The angle we missed (agents verified in difflogic.py): plain `gated` ALREADY deploys an
  exactly-binary state** — argmax->one-hot gate, MUX-of-binaries is binary, copy inputs binary,
  h_0=0. So the exact-binary / FPGA-flip-flop goal is met BY CONSTRUCTION **without `_ste_round`**.
  `_ste_round` was added only to force the SOFT TRAINING trajectory binary (deployment never needed
  it) — and that hard-forward is what MANUFACTURES the collapse. So `gated`'s gap (0.83->0.33) is a
  train->eval TRAJECTORY drift = Kim's COMPUTATION gap, attacked at the ACTIVATION level by nobody
  (entropy-reg only touches the gate-SELECTION gap).
- **Root cause (cross-lens, code-verified): the `(1-s)` candidate-starvation valve** — keep_bias
  forces s~=1 => dQ/dc=(1-s)~=0 starves the write net; init sits INSIDE the never-write attractor;
  partial writes punished below chance = a "moat". **CORRECTION to our own story:** the STE-round
  HOLD is error-CORRECTING (a super-attracting fixed point) — holds do NOT drift; the barrier is the
  partial-WRITE moat + flat plateau, not compounding drift.
- **Top fixes:** (0) FREE oracle — histogram h_t vs t on a gated checkpoint (drift => escapable);
  (1) **activation-margin loss `h*(1-h)` on plain gated (no round)** — closes drift without ever
  rounding a write; (2) **deep per-timestep supervision** (GroupSum head at every t vs the
  time-invariant copy label) + candidate-gradient bypass — kills the flat plateau + the 49-step
  delayed credit; (3) **stochastic (Bernoulli) state rounding** if a rounded state is truly needed;
  (4) **input-clocked latch: round the ENABLE not the VALUE** = a write-enabled register,
  exact-by-construction, no moat.
- **STRATEGIC FORK:** #1/#2 (drop the round) may DISSOLVE the latch primitive C1 => P2 becomes a
  training-method paper (novelty vs ETH Mind-the-Gap = TBD). **#3 (round-the-enable) PRESERVES C1**
  (a learnable write-enabled clocked register — a *cleaner* latch story than SR; a write-enable IS
  what real flip-flops have) AND is trainable — likely the best strategic answer. Decision pending.

## 2026-07-03 (round 2) — `combo` ALSO fails at copy-50; gate tracking DIAGNOSES a "never-write collapse"
- **Setup:** round-2 queue (combo copy-50 x3 + noanneal + latch-kb6) + round-1 re-run WITH gate tracking.
- **Result:**
  - **combo (gated write + bistable restore) ALSO dead/near-chance** — s0 0.126, s1 0.122, s2 0.252,
    noanneal 0.246 — NOT gated's soft 0.83. So the go/no-go is **NEGATIVE for combo too**. gated baseline
    reconfirmed (soft 0.83, disc 0.33, gap +0.50). latch all dead incl kb6.
  - **Gate distributions diagnose it (the key finding, from malcolm's gate-tracking idea):** hard-state
    causes a **"never-write collapse"**:
    - latch: `set_net`/`reset_net` 2nd layer collapse to **FALSE** (74-82%; **kb6 -> FALSE 100%**) ->
      S=R=0 -> holds 0 forever -> chance.
    - combo: `gate` net collapses to **TRUE (84%)** -> s=1 -> MUX always keeps -> never writes -> chance.
    - gated (WORKING) has a HEALTHY dist: `candidate`=identity/negation (B/A/!A, writing), `gate`=keep.
      The contrast proves the collapse. **keep-bias DRIVES it** (kb6 -> FALSE 100%, worse than kb3).
- **Read: the bistable restore closes the gap BY CONSTRUCTION but is in FUNDAMENTAL TENSION with
  trainability at length.** Hard-rounding the recurrent state over 50 steps makes "never write / always
  hold" the loss-minimizing attractor (a write near 0.5 gets round-flipped -> corrupts memory -> worse
  than chance -> model retreats to hold-0), and keep-bias pushes it there. Copy-8 worked (8 steps don't
  compound); copy-50 doesn't. Anneal + entropy did NOT rescue it. **C3-via-hard-state-training is not
  demonstrated and looks structurally hard.** OPTIONS: (1) cheap justified test — hard-state + **LOW
  keep-bias (kb0/1)** for combo & latch (does removing the collapse-driving bias let writes survive?),
  + a much-slower/later anneal (soft until ~80%) and/or a **length curriculum** (short->long transfer);
  (2) if those also collapse, **REFRAME** the paper around the OBSTRUCTION (why exact bistable memory is
  untrainable at length -> connects to C2) + the gate-collapse diagnosis, NOT gap-closing. The gate
  tracking turned a black-box failure into a crisp, figure-worthy mechanism.

## 2026-07-03 — COPY-50 GPU GO/NO-GO: pure SR latch FAILS to train at scale; `combo` is the fix
- **Setup:** `run_queue.sh` copy-50 slice on DUST (1x2080Ti). copy alphabet 8 (chance .125), hidden
  1024, 20k iters, lr 0.003->0.0003 cosine, kb 3; gated vs latch:sr (annealed) x3 seeds + rddlgn +
  ablations (soft-state/no-anneal/+entropy). `train.py` wired for latch/anneal (this session).
- **Result:**
  - **gated (3 seeds): soft 0.83±0.06, discrete 0.33±0.07, gap +0.50.** Trains softly, big
    discretization gap — exactly as P1 predicted. The working baseline WITH the gap.
  - **latch:sr — DEAD AT CHANCE (soft = discrete ≈ 0.124 = 1/8) in ALL FOUR configs** (annealed,
    no-anneal, v0 soft-state, +entropy). n_skipped=0 → *stuck/cold-start, not exploding*. Its
    "gap≈0" is trivial (chance=chance), NOT a gap-close.
  - rddlgn control ≈ chance (soft 0.12 / disc 0.26).
- **Read: NEGATIVE — the copy-8 CPU-smoke success did NOT transfer to copy-50 at scale.** The pure
  SR latch can't learn copy-50. Diagnosis (STRUCTURAL, not hyperparameter — all 4 configs fail
  identically & stuck): the SR latch **entangles write-value + write-enable in its S/R lines**,
  whereas gated cleanly separates `candidate`(value) + `gate`(enable). Copy = write-then-hold, so
  gated's separation is *why it trains* and the SR latch can't learn the 8-way write at length 50.
  **Fix = `combo`** (implemented in cells.py + train.py): gated's write path + the bistable restore
  on the hold — `h' = ste_round(s·h + (1−s)·c)` — should train like gated AND close the gap like the
  restore (the workmap's #1⊕#2 strategic synthesis). **Next:** queue `combo` copy-50 x3 as the new
  go/no-go; a pure-latch kb=6 / more-iters retry as a long shot. **CORRECTION to the 2026-07-02
  (v1)/(anneal) entries below: those copy-8 results are real but do NOT establish C3 at scale — the
  pure latch does not reach copy-50, so C3 now rides on `combo` (or a fixed latch).**
  Also added **per-layer gate-usage tracking** to every results JSON (`gate_totals` +
  `gate_distribution`, via `utils.GATE_NAMES`) + a "top 4/16" log line. Early hint (tiny CPU run):
  the SR latch's set/reset nets lean on the constant **FALSE** gate — a "dead-gate" collapse that
  would explain the write failure; round-2 (`cp50_latch_kb6`, `cp50_combo_*`) will confirm it and
  contrast a working cell's gate distribution (a real C1/interpretability ingredient).

## 2026-07-02 (anneal) — soft->hard anneal: bistable latch CLOSES the gap (copy-8: discrete 1.000, gap 0)
- **Hypothesis:** annealing the bistable restore `alpha` 0->1 over training fixes v1's hard-from-scratch
  fragility (cold-start/plateau) and closes the state-drift gap.
- **Setup:** generalized `_ste_round(x, alpha)` — forward `(1-a)x + a*round(x)`, backward = identity for
  ALL a (carousel preserved throughout); added `cell.hard_alpha` (train loop sets per-epoch) +
  `utils.hard_anneal_alpha(progress, start, end)` (linear 0->1 ramp = deterministic annealing, Rose'98).
  `scratchpad/validate_anneal.py`; copy-8 sr, hidden 16, tau 0.5, lr 0.02, 40 ep, window [0.1,0.6], CPU, seed 0.
- **Result (copy-8 sr, anneal ON):**
  - **kb=3: soft 1.000 -> DISCRETE 1.000, gap_state 0, gap_gate 0 — ZERO discretization gap.** The bistable
    latch closes the gap COMPLETELY (v0 had +0.49; v1-no-anneal kb=1 got 0.76). **← the C3 demonstration.**
  - kb=1: 0.762 (~ no-anneal). kb=2: hardstate 0.762 but discrete 0.492 (a gate-selection blip — single-seed
    noise). No-anneal kb=3 @40 ep reached 0.75 (hard-from-scratch is slow/unreliable, not impossible).
- **Read:** **the anneal delivers the clean C3 result** — a bistable-latch config (kb=3 annealed) reaches
  **discrete 1.000 with ZERO gap** on copy-8, matching a clean discretizer but via the learnable bistable
  primitive; the soft->hard schedule is what escapes the plateau. **Honest caveat:** single-seed CPU smoke is
  NOISY across kb (0.49-1.0) -> which kb/window is RELIABLE needs multi-seed + GPU (this is a
  can-close-the-gap demonstration, not yet a robustness claim). Remaining: (1) multi-seed + **copy-50 on
  DUST** (the real length, v0 gap +0.50); (2) `--entropy-reg` (already built, `utils.gate_entropy`) for the
  residual GATE-selection tail (parity); (3) wire latch + anneal schedule into `train.py` CLI. CPU smoke only.

## 2026-07-02 (v1) — hard-state STE latch: correct + C3 mechanism validated, but hard-from-scratch is fragile
- **Hypothesis:** v1 = STE-round the latch STATE (bistable restore) closes the discretization gap (C3),
  which v0's soft latch showed.
- **Setup:** implemented `_ste_round(x) = x + (x.round()-x).detach()` + `hard_state`(default True)/
  `hard_control` flags in `cells.py`/`models.py`. **Design locked by a 3-lens workflow panel**
  (`scratchpad/v1_design.txt`): round the STATE only, identity STE = the workmap §D characteristic-eq
  reduction realised in 2 lines (NOT the literal "NOR-settle", which would round control lines / risk
  C2); preserves the carousel exactly, no fixed-point iteration. Validation `scratchpad/validate_v1*.py`,
  **CPU**. Unit checks: STE identity ✓, char-eq Jacobian == analytic (dS=1-q+Rq, dR=-q(1-S), dQ=(1-R)(1-S)) ✓,
  eval-state-binary ✓.
- **Result** (gap-source decomposition: `gap_state` = acc[soft-gate,soft-state] − acc[soft-gate,HARD-state];
  `gap_gate` = acc[soft-gate,HARD-state] − acc[hard-gate=discrete]):
  - **The gap has TWO components** — `gap_state` (state drifts over time; v1 CLOSES it) + `gap_gate`
    (softmax-mixture ≠ argmax-gate; needs entropy-reg/Gumbel, NOT v1).
  - **copy-8 is 100% state-drift** (v0: gap_state +0.49, gap_gate ~0). **v1 improves discrete 0.51→0.76**
    (kb=1) — the bistable restore works. BUT trainability is FRAGILE: only kb=1 trains; **kb=0 cold-starts,
    kb≥2 plateaus** (hard round + hold-bias = decision-boundary plateau, as the panel predicted). STE-bias
    residual leaves 0.76 not 1.0.
  - **parity-8: v1 removes state-drift (gap_state→0) but EXPOSES a large gate-selection gap** (gap_gate +0.53);
    net discrete drops 0.62→0.47. Parity's residual is bit-exact gate-selection (XOR near-tie gate) → entropy-reg.
- **Read:** **v1 is correct and the C3 mechanism is validated** (bistable restore closes the state-drift half),
  **but naive hard-from-scratch is fragile** (narrow keep_bias window + STE-bias residual) and does NOT touch
  the gate-selection half. **Next: (1) soft→hard ANNEAL** (temperature-anneal the round / deterministic
  annealing — the workmap §D anticipated this) for robust training; **(2) `--entropy-reg`** (already built)
  for the gate-selection component; **(3) GPU/DUST**: copy-50 (v0 gap +0.50, long drift ⇒ v1 advantage should
  be larger/cleaner) + proper width + multi-seed + the v0-vs-v1 ablation. CPU smoke only.

## 2026-07-02 (b) — T-FF solves PARITY (control can't); keep-bias is task-dependent; disc. gap reconfirmed
- **Hypothesis:** M1 — a T flip-flop (`Q⁺ = T ⊕ Q`) solves parity (running XOR) that the recompute
  control (rddlgn) can't. Added `latch_kind='tff'` alongside `'sr'` in `cells.py` (+ `LATCH_KINDS`,
  threaded through `models.py`).
- **Setup:** `scratchpad/smoke_parity{,2}.py`; parity L∈{4,8,16}, hidden 16–24, tau 0.5, Adam lr .02–.03,
  cell_layers 2, seed 0, **CPU python impl** (laptop).
- **Result:**
  - *First run (L=16, kb=3): ALL mechanisms stuck at chance* (tff pinned at exactly ln2, zero movement).
    Diagnosed: (i) parity is SGD-hard (flat plateau / no partial credit), L=16 too long for a cold start;
    (ii) **kb=3 biases the toggle line toward HOLD → never toggles → ~0 gradient** (wrong bias for an
    integrate task).
  - *Retry (kb=0, length ladder, hidden 24): **T-FF SOLVES parity*** — L=4 in **2 ep**, L=8 in **7 ep**
    (soft acc 1.000). **rddlgn control dead at chance (0.53)**. SR latch also stuck at chance (toggle is
    not its native op).
  - **Discretization gap: tff soft 1.0 → discrete val 0.53 @ L=8** (0.70 @ L=4) — hardened circuit does
    NOT inherit the soft solution; parity is the most bit-sensitive task ⇒ worst-case gap.
- **Read:** **M1 delivered — one toggle primitive does what recompute can't** (clean Figure-1), once the
  hold-bias is removed. **keep-bias is TASK-DEPENDENT for latches too**: LOW for toggle/integrate (parity),
  HIGH for hold/recall (copy) — same axis as P1's psMNIST finding; document as a gotcha. **Primitive↔task
  fit:** T-FF↔parity, SR↔copy (SR didn't learn toggle). **The disc. gap now shows on copy AND parity ⇒
  v1 = hard NOR-settle forward + STE is unambiguously the load-bearing next step** (v0's soft forward learns
  the fn but the deployed circuit doesn't discretize — catastrophic on bit-exact parity). CPU smoke only;
  real runs (width, length sweep, multi-seed, equal-gates) on DUST.

## 2026-07-02 — P2 GATE 0 CLEARED (CPU smoke): the bistable SR-latch TRAINS
- **Hypothesis:** the bistable SR-latch recurrence — the object C2 says has ill-posed
  fixed-point gradients — can be trained under BPTT via the multilinear characteristic-equation
  reduction. I.e. is the P2 primitive trainable at all? (The single blocking gate for P2 → ICML'27.)
- **Setup:** new `mechanism='latch'` in `seqlgn/cells.py` (v0): `S=set_net(z)`, `R=reset_net(z)`,
  `Q⁺ = S + (1−R)Q − S(1−R)Q` (soft multilinear SR characteristic eq.; autograd backward, **no
  custom STE yet**). keep-bias analog = bias S,R → FALSE ("hold") so the carousel is on at init.
  Smoke: `scratchpad/smoke_latch.py`; copy L=8, alphabet 4 (chance .25), hidden 16, cell_layers 2,
  tau 0.5, kb 3.0, Adam lr .02, 25 ep, n_train 2048, seed 0, **CPU python impl** (laptop; NOT a real run).
- **Result:** latch **trains — soft train_acc 1.000 by ep20** (loss 1.45→0.115), no NaN/instability.
  Beats control: **rddlgn dead at chance (~0.25)**; gated reference solves by ep6. **Discretization
  gap: latch discrete val_acc 0.512** (soft 1.0) vs **gated discrete 1.000** (zero gap at L=8).
  [First tau run was flat — tau=30 (calibrated for hidden≈1024) squashed logits at hidden=16; tau=0.5 fixed it.]
- **Read:** **GATE 0 PASSED — the SR primitive is trainable and the characteristic-equation reduction
  works** (BPTT flows through the bistable recurrence; the `∂Q⁺/∂Q=(1−R)(1−S)` carousel does its job).
  Beating the recompute control on a memory task = the core P2 thesis in miniature. The discretization
  gap is EXPECTED for v0 (soft forward, no bistable restore) → **v1 = hard NOR-settle forward + STE
  backward is load-bearing for C3, not polish.** Caveat: tiny CPU smoke (hidden 16, 1 seed, copy-8) =
  green light only; the thesis lives at copy-50 on GPU/DUST (gated's gap opens, latch should close it).
  **Next:** (a) v1 hard-settle+STE; (b) parity via T-FF (`S=T·Q̄, R=T·Q`); (c) DUST: width + copy-length
  sweep + multi-seed + equal-gates control.

## 2026-06-21 — STORY FLIPS POSITIVE: keep-bias is task-dependent; gating wins recall AND classification
B + C results (fig: `results/curves_bc.png`). The psMNIST "loss" was a keep-bias artifact.

**B — psMNIST-28, gated, keep-bias sweep (vs rddlgn 0.620 / soft 0.652):**
| kb | test | soft | gap |
|---|---|---|---|
| 0 | **0.632** | **0.709** | +0.077 |
| 1 | 0.547 | 0.660 | +0.112 |
| 2 | 0.541 | 0.668 | +0.126 |
| 4 (old) | 0.389 | 0.659 | +0.270 |
→ **low keep-bias rescues integration:** soft 0.66→0.71, gap 0.27→0.08, and gated **kb0
beats the control** (0.632 > 0.620; soft 0.709 > 0.652). The earlier "gating loses
psMNIST" was keep-bias=4 *over-holding* (under-writing) on a task that needs to absorb
inputs. (Caveat: gated kb0 = 4k gates vs rddlgn 2k → firm up with an equal-gates control;
but soft superiority + the recall win below make the claim solid.)

**C — delayed-MNIST recall (1-step encode + delay), gated kb6 vs rddlgn:**
| delay | gated | rddlgn |
|---|---|---|
| 0 | 0.700 | 0.554 |
| 50 | **0.369** | **0.114 (chance)** |
| 100 | **0.339** | **0.114 (chance)** |
→ **decisive recall win:** the control collapses to chance at any delay (can't even learn
softly; grad ratio 0 = total vanishing), gated holds ~3× chance through 100 blank steps
(grad ratio 1.4e4). Clean real-data demonstration of the carousel.

**KEY INSIGHT:** keep-bias is **task-dependent** — HIGH for *recall* (hold state), LOW for
*integration* (absorb inputs).

**CORRECTION (equal-gates control, same day):** rddlgn at hidden 2000 = **4,000 gates**
(matching gated kb0) gets psMNIST-28 **test 0.655 > gated 0.632** (soft 0.694 vs 0.709).
So **gating does NOT win classification at equal gates** — my mid-day "gated beats control
on psMNIST" was a gate-count artifact (gated had 2×). low-keep-bias makes gated
*competitive* (0.63 vs 0.66) but the MUX discretization gap (0.077 vs 0.038) keeps it
behind. **Honest scope: gating helps long-range RECALL, not classification.**

**FINAL P1 story (scoped, airtight):** the gated carousel enables **long-range recall**
where concat-recurrence *completely fails* — copy (0.96 vs dead) and **delayed-MNIST
(control → chance at any delay; gated holds ~3×chance through 100 steps).** On
classification (psMNIST) gating gives no benefit at equal gates (control ties/wins).
Secondary: keep-bias is task-dependent; + the training recipe (keep-bias/lr-decay/skip-step).
This is the workshop contribution. Experiments DONE.
Full sweeps done (fig: `mlgn/seqlgn/results/curves.png`, `plot.py`).

**copy (synthetic recall), test acc:** gated **0.96 / 0.79 / 0.33** (L20/35/50, 3 seeds);
rddlgn dead ~0.25; lstm & gru_cell cold-start to chance at L≥35 (only work at L20, < gated).
→ on pure recall, **gated wins decisively**, and GRU > LSTM/gru_cell (simpler is better).

**psMNIST (real, chunked), test acc:** **rddlgn (control) BEATS gated at EVERY length:**
0.62/0.61/0.60/0.52 (L28/49/56/98) vs gated 0.39/0.38/0.30/0.28/0.32 (L28..112). NO crossover.

**Diagnosis — two effects, both against gated on psMNIST:**
1. **MUX discretization gap.** Soft models ≈ tied (~0.65 short), but gated gap +0.18..0.27
   vs rddlgn +0.03..0.05. The MUX `s·h+(1−s)·c` blends → analog hidden state → discretizes
   badly when the task isn't fully solvable; rddlgn's logic-recompute stays near-binary.
2. **keep-bias ⊥ integration.** gated's *soft* also degrades faster (0.66→0.50 by L98 vs
   rddlgn 0.65→0.57): keep-bias 4 makes it KEEP (under-write), which helps recall but hurts
   classification (needs to absorb every input). And at 28–98 steps rddlgn's vanishing
   (grad 1e-3) isn't fatal → gating's gradient-flow edge buys nothing here.

**Conclusion:** **the carousel helps long-range RECALL, not classification/integration.**
"Gating wins for recurrent LGNs" is NOT supported in general — only on pure-recall tasks.
This confirms the old scout read: P1 (gating) is an honest *workshop characterization*, not
a main-conf "our method wins"; **P2 (latch) is the real anchor.** Options + decision in chat.
copy, hidden 1024, lr 0.003→3e-4, 20k iters. Discrete test acc:

| seq | rddlgn (control) | **gated (GRU)** | lstm (kb4, fixed) |
|---|---|---|---|
| 20 | 0.25 | **1.00** | 0.76 |
| 35 | 0.26 | **0.88** | 0.13 ✗ cold-start |
| 50 | 0.26 | 0.38 (soft 1.0) | 0.13 ✗ cold-start |

(gated@50 = **0.75 at hidden 2048** — capacity extends the frontier; control `soft`≈0.12 =
genuinely dead at all lengths.)

**Findings:**
1. **GRU ≫ control at every length** (1.0/0.88/0.38 vs flat ~0.25). The headline figure.
2. **LSTM input-gate-closed fix WORKED at seq-20** (0.13→0.76; grad ratio 1.2e-3→1.1e+2 =
   carousel engaged). But **cold-starts again at seq-35/50** (kb4 carousel≈0.84, 0.84⁵⁰≈2e-4
   → vanish). LSTM needs ever-stronger keep-bias as length grows.
3. **GRU > LSTM cleanly** — even where LSTM works (seq-20) it's worse (0.76 vs 1.00) and far
   more finicky. → **ablation justifies the GRU**: one complementary MUX gate beats two
   independent gates needing coordinated init.

**Paper message locked:** the simple logic-GRU cell is the sweet spot — robust long-range
memory, beats the recompute control decisively, beats the more complex LSTM.

**Next (§4 rigor):** 3 seeds on the GRU points + equal-gates control (size rddlgn up to
gated's gate count). Optional: lstm kb6 @20/35 to firm up "LSTM harder even w/ strong init."

## 2026-06-11 (pm) — LSTM cold-starts (worse than GRU); fix = also close the input gate at init
First `lstm` run (copy-20, hidden 1024, keep-bias 3, lr-decay): **total failure** — flat at
chance (0.13), loss 2.08 all 20k, grad@t=0 = 8e-7 (**vanishing**, ratio 1.2e-3).

**Diagnosis:** LSTM carousel `∂C'/∂C = f·(1 − i·C̃)`. We keep-biased `f` (≈0.78) but `i`
(input gate) + `C̃` are random at init → `i·C̃ ≈ 0.25` **eats the carousel** → ∂C'/∂C ≈ 0.58
(vs GRU's `∂h'/∂h = s ≈ 0.78`). 0.58²⁰ → vanish → cold-start. So the LSTM's *separate*
forget/input gates make init harder than the GRU's single MUX gate.

**Fix:** standard LSTM init — keep-bias forget AND **close the input gate** (`bias_gate_closed`
→ FALSE logit) so `i·C̃≈0`, ∂C'/∂C ≈ f (strong), write path preserved. Applied in `cells.py`
lstm branch (both scaled by `keep_bias`). At keep_bias 4: f≈0.89, i≈0.11, carousel≈0.84.

**Note for the LSTM-vs-GRU framing:** even with the fix, this already shows the **GRU is
more robust to train** (single complementary gate vs LSTM's two independent gates needing
coordinated init) — a legitimate ablation point *for* the GRU as the recommended cell.

**Next:** re-run copy-20 lstm with `--keep-bias 4` (sweep 6 if still vanishing). Watch
grad@t=0 stops vanishing + loss drops below 2.08.

## 2026-06-11 — §1b entropy reg = NEGATIVE result; LR decay fixes stability; 0.75 is a discrete ceiling
copy-50, gated, hidden 2048, lr 0.003→3e-4 cosine, entropy-reg 0.05, 30k iters.
- ✅ **LR decay fixed stability:** skip=0 all 30k, no NaN (vs L50cap NaN @19k). Keep this.
- ✅ entropy reg committed gates: `ent` 1.6→0.03 (near one-hot).
- ❌ **Gap did NOT close.** val **0.75** (test 0.754), soft dropped **1.0→0.886**. Discrete
  unchanged vs L50cap (0.75); entropy reg pulled *soft DOWN* to discrete, not discrete up.

**Finding:** entropy/commitment reg is the WRONG tool — soft's 100% solution uses the gate
*mixture*; forcing one-hot destroys it and lands on the same 0.75 discrete circuit. We need
discrete→soft, not soft→discrete. **0.75 discrete is a capacity-bound ceiling** at hidden
2048 for copy-50 (recalls ~6/8 symbols), not a regularizable soft/hard mismatch.

**To lift discrete:** (a) more capacity (hidden 4096 / cell_layers 3), or (b) **STE / Mind
the Gap (§1c)** — hard forward + soft backward, optimizes the discrete circuit directly.

**Strategic call:** mechanism is VALIDATED (seq-20 100%, seq-50 75% discrete / soft-solvable,
control ~12.5%). Closing seq-50 to ~100% is orthogonal plumbing w/ diminishing returns.
**Recommend: bank this, pivot to core paper experiments** (lstm arm, length sweep, ≥3 seeds,
equal-gates). STE/capacity = optional polish, do only if a clean seq-50 is wanted.
Keep `--lr-min` (decay) as default-on infra going forward; drop `--entropy-reg`.

## 2026-06-10 (pm4) — §1a capacity bump: discrete 0.37→0.76; isolates a true residual gap
copy-50, gated, kb=3, lr 0.003, **hidden 2048** (2×), 30k iters.
**Result: test 0.757** (was 0.380 at hidden 1024) — chance 0.125, control ~0.125. Big win.

Two clean splits:
- **Capacity closed the *under-solving*:** discrete 0.37→0.75. Confirms §1a hypothesis.
- **A genuine ~0.25 discretization gap remains:** soft → **1.000** (iter 15k+) while
  discrete **plateaus at 0.75** (stable iters 12k–18k). So now it's a *true* gap (soft
  solved, discrete lags), not under-solving → **§1b entropy reg now properly motivated.**
- **NaN returned @ iter 19k** (loss had dropped to 0.07 → sharp gates → gradients spike
  again even at lr 0.003). Dead-weights early-stop fired (best ckpt 0.75 kept). So
  stability must hold through the *confident* late phase.

**Standing result for P1:** copy-50 gated **75.7%** vs control ~12.5%. Decisive; path to
~100% = close the residual gap (entropy reg) + stabilize the sharp phase.

**Next:** (i) §1b gate-entropy reg to close the 0.25 gap; (ii) late-phase stability — LR
decay (0.003→~3e-4) or `--grad-factor 0.5`. These pair (entropy reg sharpens gates →
needs the stability). Bigger capacity (4096) is a fallback but the gap, not under-solving,
is now the wall.

## 2026-06-10 (pm3) — lr=0.003 FIXES stability; only the discretization gap remains
copy-50, gated, kb=3, **lr 0.003**, 30k iters: **skip=0, no NaN** the whole run (explosion
fully fixed). **soft reaches 0.876 (hit 1.000 @ iter 13k)** but **discrete stuck at 0.37,
gap +0.50**, discrete flat from iter 14k → more training won't help.

**So all three bottlenecks resolved in order:** vanishing→keep-bias, exploding→lower LR,
**discretization gap = the sole remaining wall.** The gap is a *general difflogic* property
(orthogonal to our gating contribution), NOT specific to recurrence — and at seq-20 it was
**0** (fully solved discretizes perfectly). So the seq-50 gap is partly under-solving at
that length.

**Next steps to close it (cheapest first — NOT yet done):**
1. **More capacity** (bigger `--hidden`, `--cell-layers`) so seq-50 fully solves like
   seq-20 did → gap likely closes on its own (no special method).
2. **Gate-entropy regularizer** (push gate distributions to one-hot; `utils.gate_entropy`
   sketch noted) — cheap, CUDA/CPU-agnostic.
3. **Gumbel+STE (Mind the Gap, 2506.07500)** — proven heavy-artillery; only if 1–2 fail.
Likely don't *need* (3); it's borrowed plumbing, not our contribution. Try 1, then 2.

## 2026-06-10 (pm2) — skip-step shows failure is "tip into dead region" → prevention (lower LR)
Re-ran seq 35/50 with skip-step. It **learns then dies**: seq-50 soft hit **0.875** @ iter
6k, then one update poisons the weights → from there ~100% of steps skip and it spins on
NaN. Skip-step catches the aftermath but can't un-poison. seq-35: best_val 0.511 (soft
0.76) then dead @ ~4k; seq-50: best_val 0.489 (soft 0.875) then dead @ ~6k.

**Mechanism:** a finite-but-huge gradient slips through one step; Adam rescales it (small
2nd-moment → giant effective step) → a weight overflows → `softmax(inf)=nan` → dead. So
clipping/skip (magnitude safety nets) can't fix it; the fix is **prevention** — keep the
weights out of that region. Standard lever: **lower LR** (0.01 is aggressive for a 50-step
unroll). Also added a **dead-weights early-stop** (whole window skipped → break) so failed
runs die in seconds, not 30 min.

**Status for the paper:** seq-20 fully solved (100%, gap 0); seq-50 *soft* reaches 87.5% —
gating clearly enables 50-step memory; only the optimization is unstable at long lengths.

**Next:** seq-50 with `--lr 0.003` (then 0.001 / `--grad-factor 0.5` / `--keep-bias 2` if
needed). Watch `skip=` stays low and soft/val climb to convergence.

## 2026-06-10 (pm) — clip 1.0 insufficient; soft model hits 87%@seq50 → skip-step fix
Re-ran seq 35/50 with `--grad-clip 1.0`. **Clip did NOT prevent the NaN** — both still blew
up (NaN guard stopped them early: L35clip 6.2 min, L50clip 17.5 min).

| seq | best_val (disc) | test | **test_soft** | gap | grad@t=0 | outcome |
|---|---|---|---|---|---|---|
| 35 | 0.511 | 0.510 | 0.631 | 0.12 | 0.88 | NaN'd again (~iter 4k) |
| 50 | 0.363 | 0.373 | **0.867** | 0.49 | **341** | NaN'd (~iter 6k) — but soft was at 87%! |

**Key insight: the soft model nearly SOLVES copy-50 (87%) before exploding.** So gating
*can* do 50-step memory; the only blocker is numerical stability. grad@t=0 = 341 confirms a
real exploding gradient (~17000× early-vs-late).

**Why clip 1.0 failed:** `clip_grad_norm_` runs *after* backward — once a single backward
overflows to inf, clipping it yields nan (post-hoc clip can't rescue an overflowed grad).

**Better fix added: SKIP the optimizer step when the global grad norm is non-finite.** The
blow-up batch never touches the weights, and the model is kept OUT of the NaN basin (the
steps that would push it unstable are exactly the ones skipped). Tracks `skip=`/`skipped=`;
auto-suggests lower `--lr`/`--grad-factor 0.5` if >20% skipped. `train.py`.

**Next:** re-run seq 35/50 (skip-step is automatic). If it stalls (high skip count),
add `--lr 0.003` and/or `--grad-factor 0.5`. Expect seq-50 to finish solving (soft was 87%).

## 2026-06-10 — gated SOLVES copy-20 (gap=0); seq≥35 NaNs (exploding grad) → added clipping
copy, gated, keep_bias 3, hidden 1024 (RTX 2080S).

| seq | best_val | test | gap (soft−disc) | outcome |
|---|---|---|---|---|
| **20** | **1.000** | **1.000** | **0.000** | **SOLVED** — perfect, zero discretization gap |
| 35 | 0.511 | 0.510 | +0.12 | learned to 51%/76%-soft then **loss=NaN @ iter 4000** |
| 50 (50k it) | 0.124 | 0.125 | +0.26 | soft 0.38 @ 2k then **NaN @ iter 4000** (wasted 149 min) |

**Two findings:**
1. **Clean win at seq-20:** a logic GRU holds a symbol over 20 blank steps at 100%, and
   **gap→0** — the earlier 37%@seq-50 wasn't a fundamental gap; a fully-solved length
   discretizes perfectly. Quotable result.
2. **seq≥35 blocker is NaN = exploding gradients, not the gap.** keep-bias makes the
   recurrence Jacobian ≈ s ≈ 1 (fixes vanishing) but over 35–50 steps it creeps >1 →
   explodes. No gradient clipping was in the loop. We've now hit BOTH classic RNN
   pathologies: vanishing (→ keep-bias) and exploding (→ clipping). Coherent "how to train
   recurrent LGNs" story.

**Fix added:** `--grad-clip` (default 1.0, global grad-norm clip) + NaN early-stop guard +
`gnorm` logged each eval + `grad_clip` in JSON. `train.py`.

**Next:** re-run seq 35 & 50 with clipping (now default); watch `gnorm` (if pinned at 1.0
while loss still high → raise to 5–10). Expect the frontier to extend like seq-20.

## 2026-06-09 — keep-bias fixes the cold-start; bottleneck moves to discretization gap
copy, seq 50, hidden 1024, 20k iters (RTX 2080S).

| run | mech | keep_bias / gradf | best_val (discrete) | test | train loss | grad ratio | read |
|---|---|---|---|---|---|---|---|
| keepbias | gated | kb=3 | **0.379** | 0.373 | ~0.52 | 2e+4 | cold-start GONE; learns; beats control |
| fair | rddlgn | gradf=2 | 0.258 | 0.251 | 2.08 (flat) | 4e-4 | still DEAD — grad-factor 2 didn't save it |

**Verdict: directional claim validated.** gated (37%) ≫ fair control (25% ≈ chance) on a
50-step memory task; cold-start fixed (loss 2.08→0.52, gradient now reaches t=0, ratio 2e4
vs 7e-8 before). The control is genuinely dead even with grad-factor 2 → clean contrast.

**But not *solved* (37%, not 80%+).** Split: training **loss ~0.5** (soft model learning)
vs **discrete val ~0.37** → the **difflogic discretization gap** is now the dominant
bottleneck, plus discrete val was still noisy-climbing at 20k (under-trained). Cold-start
✅ → bottleneck is now (a) discretization gap, (b) under-training.

**Instrument added:** `evaluate(..., discrete=False)` → every eval now prints `soft` acc
and `gap = soft − discrete`; `test_soft`/`discretization_gap` in results JSON. Lets us
quantify the gap directly next run.

**Next:** (1) length sweep gated kb=3 {20,35,50} to find the clean-win regime + map the
frontier (the headline acc-vs-length plot); (2) more iters (50k) at seq50 to separate
"gap" from "under-trained"; (3) if soft ≫ discrete, implement Gumbel+STE (Mind the Gap,
arXiv:2506.07500) — the principled gap fix. Also probe keep_bias=2.

## 2026-06-08 — First GPU validation (copy task): cold-start found, keep-bias fix added
Hardware: RTX 2080S. Task: copy/recall (chance = 12.5%, alphabet 8), hidden 1024,
cell_layers 2, 20k iters. **No keep-bias yet (effectively keep_bias=0).**

| run | seq | mech | best_val | test | grad ratio (early/late) | read |
|---|---|---|---|---|---|---|
| sanity | 8 | rddlgn | 0.868 | 0.874 | — | learns, but only after a long plateau (breaks ~iter 18k) |
| sanity | 8 | gated | **1.000** | **1.000** | — | learns **instantly** (val=1.0 by iter 1k) |
| val | 50 | rddlgn | 0.258 | 0.251 | 4e-20 | dead (catastrophic vanishing) |
| val | 50 | gated | 0.252 | 0.246 | 7e-8 | **never started** — loss flat at log(8)=2.08 |

**Reads:**
- **Positive:** at seq-8 gating *dominates* (instant 100% vs rddlgn's struggling 87%);
  gated gradient flow is **~12 orders of magnitude** better than the control (7e-8 vs
  4e-20). The carousel works mechanically.
- **Negative:** at seq-50 BOTH fail. gated's flat loss = a **cold start**, not slow
  learning: the gate isn't keep-biased at init, so the symbol decays before the gate can
  *learn* to keep it (chicken-and-egg), and there's no gradient signal to bootstrap.
- **Diagnosis = the known LSTM cold-start.** Fix = positive forget/keep-gate bias at init
  (Gers et al. 2000) ≡ difflogic residual init (Petersen 2024).

**Action taken:** implemented `keep_bias` (adds to the TRUE-gate logit of the gate's final
layer → carousel ON at init, write path preserved). CLI `--keep-bias` (default 3.0),
applies to `gated` (gate) and `lstm` (forget). `bias_gate_keep` in `seqlgn/cells.py`.

**Next (GPU):** re-run copy-50 `gated` with `--keep-bias 3` (sweep {2,3,5} if needed);
expect it to break the plateau. Then rddlgn with `--grad-factor 2` for a fair control.

## 2026-06-04 — seqlgn infra built + smoke-tested (Paper #1 ready to run)
- Built `mlgn/seqlgn/`: pluggable recurrent cell (`rddlgn` control / `gated` Paper#1 /
  `latch` stub), `SequenceClassifier`, benchmarks (smnist/smnist-pixel/psmnist/parity/
  copy), CLI `train.py` w/ discrete-locked eval + grad-norm-through-time, docs.
- **CPU enablement:** difflogic does `import difflogic_cuda` at module top → unimportable
  on this CPU-only laptop. Added `seqlgn/_cpu_compat.py` (stub injection) so dev/debug
  works on CPU (`device cpu`, python impl, slow). Also removed a dead debug-print block in
  `difflogic/difflogic.py` `forward_python` that spammed stdout every forward.
- **Smoke test (CPU, parity seq=8, hidden=20):** both mechanisms run end-to-end; results
  JSON + LOG-LINE emitted; `--grad-analysis` + `--show-gates` work.
- **Early instrument signal (untrained, tiny — NOT a result):** `rddlgn` control grad
  ratio earliest/latest ≈ 9e-12 → severe vanishing through time, exactly what `gated`'s
  carousel should fix. Confirm with trained GPU runs.
- Next: real runs on a GPU box — psmnist + parity/copy sweep, `gated` vs `rddlgn`,
  equal-width AND equal-gates. Protocol: `seqlgn/docs/experiments.md`.

## 2026-06-04 — Baselines reproduced (pre-research-program)
- **`mnist_test.py`** — paper-style FC DLGN, 6×`LogicLayer(64_000)`, `GroupSum(k=10,
  tau=30)`, Adam lr 0.01, 100k iters, bs 128, inputs `.round()` at eval.
  - Result: **val 98.04% / test 98.24%** (discrete/locked gates).
- **`secuential.py`** — `LogicRNNCell` (2-layer logic cell) over **28 MNIST rows** as
  timesteps, hidden 16k, `GroupSum(k=10, tau=30)`, Adam lr 0.01, 100k iters.
  - Result: best **val 98.04%** (test ~98.2% region). First working **recurrent LGN** on
    sequential-MNIST in this fork.
  - Note: this is the prototype overlapping Recurrent DDLGN [5] but on
    sequential-image classification rather than translation — see
    [03_open_problems.md](03_open_problems.md) §B.

## Standing setup notes
- difflogic needs CUDA + CUDA Toolkit; `implementation='cuda'` only on GPU.
- Inference modes: `PackBitsTensor` (GPU) / `CompiledLogicNet` (compile to C/.so).
- Eval must binarize inputs (`.round()`) + `model.eval()` to lock argmax gates, or the
  reported number is the soft (cheating) accuracy.
- For deep nets raise `grad_factor` (~2) to fight vanishing gradients.

## Backlog — aligned to the 3-paper plan ([06_paper_plan.md](06_paper_plan.md))
Scout done (2026-06-04): #1 gating = CONDITIONAL GO (race), #2 latch = GO (anchor).

Shared infra (do first):
- [x] Generalize `LogicRNNCell` → pluggable memory mechanism (rddlgn/gated; latch stub). → `mlgn/seqlgn/`
- [x] Sequential benchmark harness: sMNIST, smnist-pixel, psMNIST, parity(L), copy(L). (adding=TODO, regression head)
- [ ] Bake in Gumbel+STE [3] and IWP [4] as default training infra.
- [x] Logging: acc / gates / train-time / grad-norm-through-time per variant (`train.py`).
- [x] CPU-dev enablement (`_cpu_compat.py`) + smoke-tested.

P1 — gating (fast, plant flag):
- [ ] MUX-gated cell vs rddlgn control on psMNIST + parity/copy(L). ← next, NEEDS GPU.
      Equal-width AND equal-gates; ≥3 seeds; +`--grad-analysis`. (infra ready)

P2 — latch (anchor):
- [ ] D-flip-flop primitive (trivial delay) vs baseline on copy/parity.
- [ ] gated D-latch / SR latch + custom STE backprop (feedback stability is the risk).
- [ ] 4-way comparison: sequential / just-latch / gated / combo.

Deferred:
- [ ] P3 FPGA synthesis (future work, like RDDLGN & DiffLogic CA).
- [ ] #3 Fourier — analysis/cite only (method layer occupied by arXiv:2601.13953).
- [ ] #4 hard-attention/CAM — scout before committing.
