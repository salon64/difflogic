# 18 — Sequential/memory benchmarks that SEPARATE clatch from the gated foil

Result of a 6-family fan-out benchmark hunt (2026-07-04, 68-agent workflow), motivated by the copy-50
finding that **copy-50 saturates** — every trainable route (gated+deep-sup, clatch+deep-sup,
combo+curriculum) hit discrete **1.000**, so copy accuracy cannot show clatch > gated. We need a task
that *discriminates* the primitive from the foil. Searched 61 unique candidates across synthetic-memory,
RL-POMDP, formal-language, seq-classification, hardware-edge, and Long-Range-Arena; scored each against
5 constraints (memory-required / binary-input-friendly / tractable on 1 GPU / citable / **separates
clatch>gated**). See `04_experiment_log.md` 2026-07-04 for the copy-50 result that triggered this.

## TL;DR — recommended suite (2 core + 1 parked)

| Role | Task | Citation | Where to get it |
|---|---|---|---|
| **(a) HEADLINE-SEPARATOR** (show clatch > gated) | **Selective Copying** — content-based copy with distractors | Gu & Dao, *Mamba*, arXiv:2312.00752 (2023) §3.1; lineage Jing et al. *GORU*, AAAI 2019 | Synthetic. ~20-line extension of `_make_copy` in `mlgn/seqlgn/data.py`. Ref gens: HazyResearch/zoology, MinhZou/selective-copying-mamba |
| **(b) CREDIBILITY-MATCH + verify** | **Parity** (n-bit running XOR), regular-bucket task of the Chomsky-hierarchy suite | Delétang et al., *NNs and the Chomsky Hierarchy*, ICLR 2023 (oral), arXiv:2207.02098; support Hahn, TACL 2020 | **Already implemented** — `_make_parity` in `data.py`, wired into `train.py`; prior result JSONs exist |
| **(c) DEPLOYMENT-CAPSTONE** (parked, cheap) | Extract argmaxed LGN from Parity → verify it equals the canonical **1-XOR + 1-T-FF**; report length-gen to arbitrary L | Same Delétang cite; extraction lineage Weiss et al. ICML 2018 | Built on (b); no new task, just extraction + equivalence check |

**Why this pairing:** Selective Copying is the single best *structural* match for the clatch thesis (a
per-step, content-dependent write-enable = exactly what clatch IS). Parity is already coded, is a famous
named benchmark ETH-DISCO/ISTA reviewers respect, gives a clean **train-short/test-long** length-gen
axis, and its solution *is* a T flip-flop you can formally verify — folding credibility-match and
verification-capstone into one cheap task.

**Honest caveat:** both core tasks carry the same saturation risk that killed copy-50. Neither is
*guaranteed* to separate — the separation must be *found* by dialing difficulty. Insurance = the **n-bit
memorization** task (Martens & Sutskever ICML 2011 / H&S 1997), a near-free third separator on the pure
long-hold axis (dial = blank-gap length T).

## The separator: why clatch should beat gated on Selective Copying, and what to sweep

**Mechanism.** Selective copy forces a per-step content-dependent write-enable: "is this token signal
(write) or noise (hold)". That decision is the clatch primitive.
- **clatch** rounds the enable → writes on a detected real token, holds the stored value **bit-exact**
  across every noise step. Hold is T-independent by construction.
- **gated** (soft-MUX) approximates the enable in (0,1), so on *every* noise step it blends a fraction of
  the noise token into the register. Across long variable-length distractor runs and multiple slots, the
  leak **compounds** and corrupts stored symbols by the recall phase.

This is exactly Mamba's argument for *selective* (input-dependent) state → maps 1:1 onto "clean register
vs leaky MUX". Separation should show on **both** axes: accuracy-at-gap-length AND the deployment axis
(exact hold, verifiable "writes-only-on-cue" invariant, lower gate count).

**Dials to sweep, in priority order:**
1. **Kill the explicit write-flag.** If the model gets a per-token "this is real" cue bit, gated can also
   learn near-perfect not-write and saturate to 1.000 (copy-50 redux). Force **content-based selection**
   (model must *infer* real-vs-noise). **The single most important knob for opening a gap.**
2. **Gap length** (noise tokens between real tokens) — the leak-accumulation axis. Push until gated cracks.
3. **Number of stored slots K** — multiplies per-step leak across more registers.
4. **Train-short / test-long** — exact register generalizes, soft-MUX drifts.

Sweep #1+#2 first. Target figure: **discrete test-acc vs gap length**, gated rolling off while clatch
stays flat — the exact plot copy-50 could not produce.

## Ranked top-8 candidates

| # | Name | Role | Fit | Separates? | Binary fit | Compute | Citation |
|---|---|---|---|---|---|---|---|
| 1 | **n-bit memorization** (5/20-bit) | headline-separator | 84 | High (pure long-hold; best single store-and-hold) | Native bits | Low; length-sensitive BPTT | Martens & Sutskever ICML 2011; H&S 1997 |
| 2 | **Parity** (n-bit XOR) | credibility-match (+verify) | 83 | Med (high on length-gen/deploy, low-med in-dist) | Native (1 bit/step) | Trivial (already runs) | Delétang et al. ICLR 2023 |
| 3 | **Selective Copying** | headline-separator | 82 | High (best structural match for clatch) | 1-hot + noise channel | Trivial | Gu & Dao (Mamba) 2023 |
| 4 | **MQAR** (multi-query assoc recall) | headline / credibility | 80 | High *if* register-bank + addressing built | 1-hot, small vocab | Cheap at small scale | Arora et al. (Zoology) 2023 |
| 5 | **Temporal order / distractor** | headline-separator | 80 | High (write-on-rare-event, hold thru distractors) | 1-hot, ~6-8 sym | Trivial | Hochreiter & Schmidhuber 1997 (Exp 6) |
| 6 | **Chomsky Hierarchy suite** (regular bucket) | credibility-match | 80 | Med (concentrated in finite-state tasks) | Excellent | Low for regular bucket | Delétang et al. ICLR 2023 |
| 7 | **Bounded Dyck-(k,m)** | credibility-match (+verify) | 74 | Med-high (needs stack scaffold; Goldilocks m) | 1-3 bits/step | Trivial | Suzgun et al. 2019; Hewitt et al. 2020 |
| 8 | **MLRegTest** (subregular langs) | credibility + verify | 70 | Med (drift-forgiving classification) | 2-6 bits/symbol | OK if sliced to SP/TSL long-string subset | van der Poel et al. JMLR 2024 |

## Rejected / low-fit (don't revisit)

- **Copy / copy-memory:** saturates to 1.000 for all routes — the anchor that *created* this hunt.
- **Repeat copy / NTM:** payload storage dominates, copy-like; needs external memory to be interesting.
- **All RL (T-Maze, POPGym, Memory Gym/Maze, Numpad, bsuite, MuJoCo-POMDP):** need a from-scratch RL-LGN
  stack we don't have; reward variance drowns the register-vs-MUX signal. Passive T-Maze is the *eventual*
  capstone but wrong first move — its memory core is long-delay copy you can probe **supervised** instead.
- **Counter langs aⁿbⁿ, binary add/mult:** exercise per-step *update* not exact *hold*; carry writes every
  step so gated ≈ clatch; both extract to identical hard gates.
- **Sequential parity as capstone / Tomita / Embedded Reber:** toggle/short FSM; saturate or preview the
  T-latch (not clatch); good verification vignettes only.
- **psMNIST / sMNIST / seq-CIFAR:** integration/accumulation — structurally favor the soft-MUX, run
  *against* the clatch thesis. psMNIST also ~40h/run.
- **All LRA (ListOps, Text, Retrieval, Pathfinder, Path-X, Image):** classification/aggregation, no
  exact-hold event; 1K-16K-step BPTT is compute-hostile; S4's home turf where an LGN looks weak.
- **Audio/sensor (KWS, wake-word, SHD/SSC, FSDD, UCI-HAR, MIT-BIH, DVS128, deep-soli, DCASE-AD):**
  feedforward-solvable or evidence-integration; violate memory-required; binarization front-end confounds.
- **WMT/enwik8/PG-19:** generative LM, no clocked exact-hold; huge embeddings dilute gate-count story.
- **Adding problem:** MSE regression, needs a real-valued head; confounds hold with arithmetic.
- **Game of Life / DiffLogic-CA:** spatial persistence not sequence memory; contested lane; new CA infra.
- **MAD suite:** only its selective-copy member helps — get that from #3 directly.

## Implementation notes (easiest-first)

**Zero-build — run today:**
1. **Parity length sweep + train-short/test-long eval.** `_make_parity` and the `tff` mechanism already
   exist; prior JSONs present. ✅ DONE (2026-07-04): the length-gen split is now wired — `--test-seq-len`
   in `train.py`/`data.py` builds the TEST set at a different length than train/val (model selection stays
   on train-length val; test_acc is the length-gen number). Recorded as `test_seq_len` in the result JSON.
   Queued: `bh_parity_*_L32/L128` (in-dist probe) + `bh_pargen_{gated,tff,clatch}_t128/t256` (train L=32,
   test L∈{128,256}) → the gated-rolls-off-while-tff/clatch-stay-flat curve. Least-effort credibility-match.

**Low-build — the headline separator:**
2. **Selective Copying generator** — ✅ DONE (2026-07-04): `_make_selective_copy` + `selcopy` task in
   `data.py`, `--sel-flag` ablation in `train.py`. Implemented as the **K=1** variant (one data symbol at a
   random early position among blanks, output at the end) to fit the single-label GroupSum head; **no cue
   bit by default** (content-based). Gets `--test-seq-len` length-gen for free. Queued: `bh_selcopy_*` —
   gated(kb3) vs clatch(kb1,anneal), both +deep-sup, at L∈{50,100}, plus the `_flag` ablation. WIN = clatch
   discrete >> gated discrete. (K>1 multi-symbol would need a per-position output head — future work.)
3. **(Insurance) n-bit memorization** — same harness family: emit k value symbols, then T blanks, then a
   cue; target = the k symbols. Main added work = long-T support in the BPTT loop + sweep T. Pure long-hold
   separator if selective-copy's content-selection turns out too hard for the LGN to learn.

**Medium-build — verification capstone (after a separation appears):**
4. **Argmax-to-circuit extraction + equivalence check.** The argmaxed recurrent LGN over discrete state
   bits *is* a Boolean FSM — extraction is exact and cheap (no RNN state-clustering). For Parity, verify
   equivalence to the canonical 1-XOR + 1-T-FF, report gate count + length-gen to arbitrary L. Reusable P2
   infra and the ISTA/FPGA moat, riding on a task we already run.

**Parked (don't build for this paper):**
5. Any RL task. If P2 later wants a POMDP capstone, first cheaply extract its memory core as a
   **supervised** hold-1-bit-for-long-L recall (L into hundreds/thousands). If gated doesn't leak *there*,
   the RL version won't discriminate either.

## Risk summary (be honest)

- **Primary (both core tasks): saturation.** gated may hit 1.000 wherever the LGN trains, forcing
  difficulty up until BPTT cost — not the mechanism — is the binding constraint. Mitigation: content-based
  selection (no write-flag) for selective-copy; length-gen split for parity, where an exact register
  structurally wins and a soft-MUX cannot.
- **Secondary: no LGN prior art** on selective-copy or n-bit memorization → the head-to-head there is
  internal (clatch vs gated) with only RNN/SSM external reference numbers. **Parity carries the *external*
  credibility** (Delétang's 15-task suite, named baselines).
- **What success looks like:** one plot of discrete test-acc vs gap-length (selective copy) or vs
  test-length (parity) where the gated curve rolls off and the clatch curve stays flat — the figure
  copy-50 could not produce.
