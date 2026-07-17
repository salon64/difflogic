<!--
p1_writing_prep.md — pre-writing dossier for P1 (assembled 2026-07-14).

Purpose: reload full context before writing the camera-ready. Three parts:
  §1-3  what we did, where the code is, what changed vs upstream difflogic
  §4-5  reading list — the sources (all 18 bib entries link-verified 2026-07-14) + non-academic material
  §6    academic-writing guides, curated for "condense an existing draft to 4 pp"
  §7    open items & flags from the adversarial verification pass

Provenance: assembled from a 6-agent research workflow + a completeness critic that
spot-checked ~25 file:line anchors and re-derived headline numbers from the raw results
JSONs; the critic's corrections are applied inline. Companion docs: mlgn/paper/p1/p1_draft1.md
(the draft itself), mlgn/research/08_paper1_checklist.md, mlgn/research/20_program_validation.md.
-->

# P1 pre-writing dossier — context, code, reading, writing

**Paper:** *Gating Enables Long-Range Recall in Recurrent Logic Gate Networks* — draft v1.1 at
`mlgn/paper/p1/p1_draft1.md`, targeting a NeurIPS 2026 workshop (non-archival), submission ~Aug 29,
venue decision due now (list dropped ~Jul 11 — see `mlgn/research/07_venues_timeline.md`).

---

## 1. What we did — the P1 story

### 1.1 The question and why it matters

P1 asks whether *gating* — the constant-error carousel that made RNNs trainable on long
dependencies — can be realized in pure Boolean logic, and what it buys a recurrent Logic Gate
Network. The gap is real and named by the field itself: all three existing recurrent/stateful
LGNs (RDDLGN, DiffLogic CA, R-DTLGN) carry state by concat-recurrence — recompute the hidden
state from scratch every step — and two of the three explicitly list gating as future work
(`mlgn/paper/p1/p1_draft1.md` §1). The plan rated this a CONDITIONAL-GO "fast workshop
flag-plant" with real scoop risk, survivable only if it carried (a) a long-range benchmark where
vanilla recurrence *fails* and gating *fixes* it, and (b) a gradient-flow analysis
(`mlgn/research/06_paper_plan.md`). The answer delivered is deliberately specific: gating buys
**long-range recall** (50–100 steps), a regime where concat-recurrence's gradient at the write
step is *exactly zero*; it does **not** buy classification. First gated recurrent LGN; first
50–100-step recall in a learned logic circuit (draft abstract).

### 1.2 The cell design, in plain words

Everything is built from one block — a `LogicMLP` (two `LogicLayer`s) over `z = [x_t; h]` — so
only the memory mechanism varies (`p1_draft1.md` §3.1). The **control** (`rddlgn`) recomputes:
`h′ = LogicMLP(z)`, 2H learned gates. The **gated cell** runs two parallel LogicMLPs — a
candidate net `c` and a gate net `s` — into a fixed per-bit multiplexer `h′ = s·h + (1−s)·c`,
which at binary values is exactly the Boolean MUX `(s∧h) ∨ (¬s∧c)`; 4H learned gates plus 3
*fixed* gates per bit at deployment. The recurrence Jacobian is `∂h′/∂h = diag(s)`: kept bits
pass gradient back unattenuated — the LSTM/GRU carousel in logic (§3.2). Crucially the mechanism
lives at the *cell* level, not in the 16-gate pool (`mlgn/research/17_concepts_and_journey.md`
§1–2b). Two dual-state ablations (`lstm` 10H, `gru_cell` 8H) complete a 2×2 over gate structure ×
state separation. **Keep-bias** adds a constant to the TRUE-gate logit of the gate net's final
layer so `s` starts high (carousel on): simultaneously the LSTM forget-gate bias (Gers 2000) and
the *temporal* twin of Petersen's residual initialization — the framing
`08_paper1_checklist.md` §"free wins" prescribes (credit residual-init, claim the temporal
generalization). It is necessary, not cosmetic: kb = 0 cold-starts every recall task.

### 1.3 The experiments

Shared config: Adam, lr 3e-3 cosine-decayed to 3e-4, 20k iters, batch 128, τ = 30, hidden
1000/1024; discrete-locked eval (argmaxed gates, binarized inputs), model selection on discrete
val, test touched once, ≥3 paired seeds on headlines (`p1_draft1.md` §4).

- **copy(L)** — one-hot symbol (alphabet 8, cue bit) then L−1 blanks; chance 12.5%. Gated disc:
  **0.96 ± 0.07 / 0.79 ± 0.26 / 0.33 ± 0.08** at L = 20/35/50. Control dead at both widths (1024
  and equal-gates 2048): soft pinned at chance (0.118–0.130) in all 18 control runs; disc is
  scatter 0.13–0.37 around a degenerate predictor. lstm/gru_cell hit ~0.75 at L20 but cold-start
  to 0.13 at L ≥ 35. Capacity (h2048) lifts copy-50 disc to 0.757 (§5.1, App. A.1).
- **delayed-recall MNIST** (whole 784-px image at t = 0, D blanks) — the headline. Gated kb6
  disc: **0.715 ± 0.013 (D0), 0.489 ± 0.028 (D25), 0.362 ± 0.045 (D50), 0.274 ± 0.035 (D75),
  0.300 ± 0.061 (D100)** = 3.0× uniform chance at 100 steps. Control: **exactly 0.1135** — the
  majority-class baseline — soft *and* discrete at every D ≥ 25, with write-step gradient
  identically zero. The new equal-gates control (h2000 = 4,000 gates, D50): disc 0.1165 (the
  draft body rounds this to 0.117 — JSON and v1.1 header say 0.1165; don't stall on the
  mismatch), soft 0.1135, `grad_profile[0] = 0` exactly — the gate-count defense is now data on
  the headline task (`04_experiment_log.md` 2026-07-10).
- **psMNIST-28 (chunked)** — the integration control-task, explicitly flagged nonstandard
  (28 px × 28 steps, not the 784-step benchmark). Gated kb0 **0.616 ± 0.056** (5 seeds) vs
  equal-gates control **0.674 ± 0.017** vs same-width control 0.629 ± 0.012 (§5.3).

Fig 1 = length sweeps, Fig 2 = keep-bias dial + recall-vs-delay, Fig 3 = the grad-norm carousel
figure (gated copy-50 profile `[109.8, 98.2, …]` vs control `[0.0, 0.0, …]`), rebuilt from
`mlgn/paper/p1/fig{1,2,3}_*.py` with the hygiene-filtered loader `figstyle.py` (App. A.3).

### 1.4 The negative/boundary results, phrased honestly

- **No classification win at equal gates** (§5.3): control ahead by 5.8 points; no gated seed
  reaches the control mean — and because the best gated seed (0.6555) edges *past* the worst
  control seed (0.6552) by 0.0003, strict seed non-overlap is *not* claimed (draft header v1.1).
  "Gating buys nothing on integration tasks."
- **Keep-bias inverts across tasks** (§5.2): psMNIST falls 0.616 (kb0) → 0.431 ± 0.080 (kb4);
  dMNIST-50 rises 0.155 ± 0.021 (kb0, cold-start 3/3 seeds at soft = 0.1135) → 0.362 ± 0.045
  (kb6). The draft even turns its own early mistake (kb4 on psMNIST) into a cautionary tale.
- **The discretization gap** (§5.5): copy-50 soft 0.83 vs disc 0.33; the gap grows with length
  (0.00 at L20 → +0.50 at L50), placing it as Kim's *computation* gap, per the stance in
  `08_paper1_checklist.md` (Mind-the-Gap's mechanism cited cautiously). The entropy-regularizer
  negative result (soft 1.0 → 0.886, disc unchanged at 0.75) supports this. "We deliberately do
  not claim a fix."
- **Exact wording discipline**: "collapses to chance" was made precise per validation — copy
  control soft at chance but discrete ≈ 2× chance; dMNIST control exactly the majority baseline
  (`20_program_validation.md` §B4).
- **Priority scoped**: RDDLGN's memorization probe already shows short-shift recall (≤12
  positions, 16-step sequences), so the claim is scoped to *long-range* (§1). No float-RNN
  baseline; owned in §7 ("a small float GRU would likely solve these tasks — the LGN value
  proposition is deployment cost").

### 1.5 The training recipe and its scoping

Both classic RNN pathologies appeared and each got one fix (§3.4, §5.4). **Vanishing →
keep-bias**: found 2026-06-08 (gated copy-50 flat at loss log(8) = 2.08, grad ratio 7e-8; fix
implemented same day, `04_experiment_log.md`). **Exploding → prevention, not clipping**:
grad-clip 1.0 failed (clipping an `inf` gradient yields `nan` — observed directly, 06-10);
skip-step alone catches the aftermath but the run is already poisoned (13–17k of 20k steps
skipped); cosine LR decay fixed it — every headline run finished with **zero skipped steps**.
The claim is scoped: "sufficient for every configuration reported here," with the honest caveat
that heavy skipping reappears out-of-scope under high keep-bias × long unroll (P2's
selcopy-L100 clatch run skipped 8,044/20,000 — the §3.4 wording was recalibrated for
consistency, log 07-09/07-10).

### 1.6 Claim-by-claim status

- **Long-range recall (contribution 2): 3-seed-firm end-to-end.** All copy cells 3-seed (both
  control widths); dMNIST gated 3-seed at all five delays; the equal-gates control exists on
  *both* recall tasks. Firmed by the `p1f_*` 23-run hardening queue, 07-10.
- **Keep-bias dial (contribution 3): firm at the endpoints** (psMNIST kb0 n=5, kb4 n=3;
  dMNIST-50 kb0/kb3/kb6 n=3); kb1/kb2 single-seed with their ordering explicitly called noise;
  kb3≈kb6 overlap disclosed.
- **Classification boundary (contribution 5): firm** (5 vs 3 seeds); non-overlap deliberately
  NOT claimed (3e-4 overlap, see §1.4).
- **Dropped**: "gated soft is better" on psMNIST (did not survive 3 seeds — draft header);
  "gating wins classification" (killed same-day 06-21 by the first equal-gates control, log);
  the imprecise "collapses to chance."
- **Scoped**: recipe (see above); dual-state ablation numbers are single-seed and §7 admits
  "not exhaustively tuned"; control dMNIST single-seed is justified (degenerate variance —
  exactly 0.1135). The 81-agent validation confirmed every table cell reproduces exactly from
  the raw JSONs (`20_program_validation.md` §B, TL;DR 1), and its P1 fixes (3-seed re-basing,
  orphan-run ingestion, Fig 3, wording) are all in v1.1.

### 1.7 What remains before submission (~Aug 29)

Per the draft header (`p1_draft1.md` lines 31–35) and the checklist status update
(`08_paper1_checklist.md`, 2026-07-10 — "EXPERIMENTS CLOSED… remaining is writing/admin only"):

1. **[TODO-VENUE]** Pick the workshop (list dropped ~Jul 11 — decision is now due; the repo's
   own venue analysis is `mlgn/research/07_venues_timeline.md`), condense to 4 pp in its
   template, add affiliation + acknowledgements.
2. **[TODO-BIB]** Verify author given names and final venue strings against arXiv — **now done,
   see §4 below**; three entries need edits (miotti title, yue title, bacellar author).
3. **[TODO-OPT]** No runs required; Tier-C extras (lstm kb6, capacity seeds, sMNIST row, d100
   eqgates control, float-GRU harness) stay commented in `mlgn/paper/p1/run_queue_p1.sh`.
4. Program-level hygiene before anything goes public: tag the commit the paper numbers come
   from — the repo currently has **no tags**; the p2→main merge and push are already done as of
   2026-07-14 (`20_program_validation.md` §E7).

---

## 2. Code map — where every P1 concept lives

All paths repo-relative. Line numbers verified 2026-07-14 (critic re-opened ~25 of them).

### 2.1 Concept → code table

| Paper concept | Location | What's there |
|---|---|---|
| Gated cell: per-bit 2-to-1 mux (contribution) | `mlgn/seqlgn/cells.py:384` | `Q = s * h + (1.0 - s) * c` — the soft MUX / GRU-style update; full branch `cells.py:371-392`; module construction (separate candidate + gate `LogicMLP`s) `cells.py:295-306`; boolean-MUX equivalence + carousel argument in the docstring `cells.py:29-39` |
| Keep-bias initialization (logic forget-bias / residual init) | `mlgn/seqlgn/cells.py:158-172` | `bias_gate_keep` adds `strength` to the TRUE-gate (id 15) logit of the gate MLP's final layer (`cells.py:172`); applied to `gated` at `cells.py:306`, to lstm forget gate `cells.py:319` (+ `bias_gate_closed` on the input gate, `cells.py:143-155`, applied `cells.py:320`), gru_cell `cells.py:334`. CLI knob `--keep-bias` `mlgn/seqlgn/train.py:144-148` |
| Concat-recurrence control (`rddlgn`) | `mlgn/seqlgn/cells.py:284-286` (single `update` LogicMLP over `[x_t;h_t]`), forward `cells.py:368-369`; provenance (Bührer / DiffLogic CA) in docstring `cells.py:12-17` |
| Latch / clatch variants (P2 machinery, mentioned as sibling) | SR/T-FF latch construction `cells.py:336-358`, forward (characteristic equation `Q⁺ = S + (1−R)Q − S(1−R)Q`) `cells.py:416-439`; `clatch` = rounded write-enable `cells.py:374-380`; STE round `_ste_round` `cells.py:114-135`. P1 fig scripts *exclude* these mechanisms (`figstyle.py:52`, `85-88`) |
| LSTM / gru_cell ablations (dual-state arms) | construction `cells.py:308-334`, forward `cells.py:394-414`; OR-as-addition stand-in `_or` `cells.py:107-111` |
| Discretization (soft→hard) eval + gap | `mlgn/seqlgn/train.py:61-80` — `evaluate(discrete=True)` = `model.eval()` (argmax gates, `train.py:67`) + `x.round()` (`train.py:72`); soft vs discrete both run at `train.py:481-482`; `"discretization_gap": test_soft - test_acc` recorded `train.py:587` |
| Write-step gradient-norm instrumentation | `mlgn/seqlgn/utils.py:104-131` `grad_norm_through_time` (L2 of `dL/dh_t` per step); enabled by `retain_grad()` on per-step carousel states in `mlgn/seqlgn/models.py:93-100`; `carousel_state` helper (C for lstm/gru_cell, h otherwise) `cells.py:455-459`; triggered by `--grad-analysis` at `train.py:522-531`; saved as `"grad_profile"` `train.py:615` |
| LR decay + non-finite-step skipping | cosine schedule (only if `0 <= --lr-min < --lr`) `train.py:372-375`; grad clip + measure `train.py:442-444`; **skip `optimizer.step()` when norm non-finite** `train.py:445-448` (rationale comment `439-441`); dead-weights early stop when a whole eval window skipped `train.py:470-474` |
| Copy task generator | `mlgn/seqlgn/data.py:78-84` `_make_copy` (one-hot symbol + cue bit at t=0, L−1 blanks); wired at `data.py:327-336`. Siblings: parity `data.py:71-75`, selcopy `data.py:87-110`, distcopy `data.py:113-138` |
| Delayed-recall MNIST (d25–d100) | blank-step append in `_SeqMNIST.__getitem__` `data.py:212-215`; `seq_len = image_steps + delay` `data.py:241`; CLI `--delay` `train.py:173-175`. Headline config = `--task smnist-pixel --chunk 784 --delay D` (image is 1 step of 784 pixels) |
| psMNIST-28 chunked | fixed permutation (seed 1234) `data.py:228-229`; chunked reshape `data.py:205-211`; task branch `data.py:314-315`; `--chunk` `train.py:170-172` (`--chunk 28` → 28 steps of 28 pixels) |
| Results-JSON writer | `train.py:548-624`; record dict `train.py:558-621` (persists every arg per the research/20 §C3 rule); filename `{task}_{mechanism}_{tag}_{stamp}.json` `train.py:556` into `mlgn/seqlgn/results/` (`train.py:53`) |
| Gate-count parity / equal-gates control | draft §3.1 "Gate accounting" + `mlgn/seqlgn/docs/experiments.md` §"Capacity fairness" (~line 101) | Control width doubles (1000→2000) so the rddlgn control matches the gated cell's 4H learned gates; `utils.py:59-61` `count_gates` (= Σ LogicLayer.out_dim) computes it, printed as `logic gates=` at `train.py:356`. ⚠ `cells.py:49-58` is the **P3b ff-vs-gated** parity rule (`H_ff = 2·H` for the stateless arm) — do NOT cite it for P1's fairness paragraph |
| GroupSum readout head + unroll | `models.py:83` (head), unroll loop `models.py:94-102`, `hidden_dim % num_classes` assert `models.py:59-63` |
| Gate-entropy regularizer (used in some runs; figs filter it) | `utils.py:86-101`, applied `train.py:405-409` |
| CPU-only import shim (dev convenience) | `mlgn/seqlgn/_cpu_compat.py:27-65` `ensure_cpu_importable` stubs `difflogic_cuda` |

### 2.2 Entry points & reproduction

**One training run** (everything is one script, mechanism is the flipped variable):
`python -m mlgn.seqlgn.train --task <copy|smnist-pixel|psmnist> --mechanism <gated|rddlgn|lstm|gru_cell> ...`
(usage header `train.py:5-25`). Every run writes a JSON to `mlgn/seqlgn/results/` and prints a
`LOG-LINE:` summary (`train.py:625-627`).

**Hardening queue** (`mlgn/paper/p1/run_queue_p1.sh`): job list at lines 45-132 — Tier A:
psMNIST-28 control seeds (A1/A2, lines 51-56), the *missing* dMNIST-d50 equal-gates control with
`--grad-analysis` (A3, line 63), dMNIST-d50 kb0/kb3 seeds (A4, lines 68-71), psMNIST-28 kb4
seeds (A5, lines 76-77); Tier B copy controls lines 85-101; Tier C commented out (103-131).
Mechanics: skip-if-JSON-exists (line 145, so re-running after disconnect is safe), round-robin
over `GPUS=(0 1)` (lines 155-163), each job runs `python -m mlgn.seqlgn.train $job` logging to
`logs/<tag>.log` (line 149). Launch: `nohup bash mlgn/paper/p1/run_queue_p1.sh > logs/queue_p1.log 2>&1 &`
from repo root on DUST (header lines 17-20). All clean-sweep runs share
`--lr 0.003 --lr-min 0.0003 --iters 20000`.

**Figures** (canonical, from repo root; outputs → `mlgn/paper/p1/figs/*.{png,pdf}`,
`figstyle.py:154-158`):

- `python mlgn/paper/p1/fig1_length_sweeps.py` — copy(L) sweep + psMNIST-28 wrong-keep-bias
  cautionary panel; series specs `fig1_length_sweeps.py:29-70`.
- `python mlgn/paper/p1/fig2_keepbias_recall.py` — keep-bias dial (opposite slopes) +
  delayed-recall headline; majority baseline 0.1135 at line 21; equal-gates diamond hooks the
  `p1f_dm50_rddlgn_eqgates` run at lines 53-60.
- `python mlgn/paper/p1/fig3_gradnorm.py` — grad-norm-through-time panels; reads `grad_profile`
  by filename glob, `PANELS` at lines 19-24, display floor `1e-8` line 17.

Data hygiene is enforced centrally in `figstyle.load_records` (`figstyle.py:77-91`): P1
mechanisms only (`P1_MECHS`, line 52), drops any run with `deep_sup`/`margin_reg`/`anneal`
(line 87); clean-sweep filter `CLEAN = dict(lr_min=0.0003, iters=20000)` in each fig script
(fig1:20, fig2:20); seed stats collapse same-seed duplicates then take ddof=1 s.d.
(`figstyle.py:109-123`). Quick table without figures:
`python -m mlgn.seqlgn.collate [--task copy] [--csv]` (`collate.py:9-13`, columns 27-44).
`mlgn/seqlgn/plot.py` is the informal working-figure tool (same hygiene filter, `plot.py:30-45`).

### 2.3 Architecture in 10 lines (train.py)

1. `build_args()` (`train.py:115`) → `get_task()` (`train.py:292`, dispatch in `data.py:256`) →
   `SequenceClassifier` (`train.py:340`), which owns one `LogicRecurrentCell` + `GroupSum` head.
2. Loss = CE; optimizer = Adam; optional cosine LR if `--lr-min` set (`train.py:366-375`).
3. Infinite `cycle(train_loader)` for `--iters` steps (`train.py:384`).
4. Forward: model unrolls the cell one timestep at a time over `x:[B,T,F]` (`models.py:94-102`);
   `return_hidden=True` only when margin/deep-sup regularizers need per-step states
   (`train.py:396-400`).
5. Loss = CE on final-step logits (`train.py:401`) + optional gate-entropy (`405-409`),
   activation-margin (`412-415`), deep supervision (`421-434`) terms.
6. Backward, clip grad norm, **skip the step if the norm is non-finite** (`train.py:436-448`) —
   the NaN-basin guard.
7. Every `--eval-freq`: discrete AND soft val accuracy, print the gap, checkpoint
   best-by-discrete-val (`train.py:452-467`); abort if a whole window was skipped (`470-474`).
8. Final: reload best checkpoint, report discrete + soft test acc = the deployed-circuit numbers
   (`train.py:477-482`).
9. Optional analyses: state-drift oracle (`508-520`), `--grad-analysis` grad profile (`522-531`),
   gate-usage histogram (`540-546`).
10. Dump the full run record (all args + metrics + grad_profile + gate distribution) as JSON to
    `mlgn/seqlgn/results/` (`train.py:548-624`) — the sole data source `collate.py`, `plot.py`,
    and the fig scripts consume.

Protocol prose backing the experiments lives in `mlgn/seqlgn/docs/experiments.md`
(capacity-fairness rules §"Capacity fairness" line 101; definition of "P1 is working" line 149)
and `mlgn/seqlgn/README.md` (quickstart line 138, GPU setup line 59).

---

## 3. Changes vs upstream difflogic

The fork is 144 commits ahead of `upstream/main` (Felix-Petersen/difflogic). Only **two upstream
library files are modified** — `difflogic/difflogic.py` and `difflogic/cuda/difflogic_kernel.cu`
— across three commits, all by the repo owner (git identities `malco` and `salon64` are the same
person):

- `81a2b76` "init" (2026-06-05, malco)
- `b4812d1` "cuda update" (2026-06-08, salon64)
- `be02ffd` "difflogic: persist LogicLayer wiring in checkpoints (conn_a/conn_b buffers)" (2026-07-10, malco)

### 3.1 `difflogic/difflogic.py`

**1. Dead debug block removed from `forward_python` (81a2b76).** Upstream had an
`if self.indices[0].dtype == torch.int64` block that printed index dtypes and reassigned
`indices` — the condition is always true (indices are created as int64), so it printed to stdout
on every forward call on the CPU/python path. Removal is a pure cleanup; no numeric or
behavioral change.

**2. Random wiring persisted in checkpoints (be02ffd).** Upstream stores the layer's random
connectivity (`self.indices`) as a plain attribute, so `state_dict()` saved only gate weights —
the wiring had to be replayed from construction-time RNG state, making checkpoints non-portable.
The fork turns `indices` into a **property backed by two registered int64 buffers
`conn_a`/`conn_b`**: new checkpoints carry the wiring, `load_state_dict` restores it (overriding
constructed wiring), and buffers follow `.to(device)`. Supporting machinery: a setter that
shape-validates before mutating (preventing silent broadcast or half-updated a/b corruption) and
recomputes the CUDA backward helper indices; a custom `_load_from_state_dict` that (a) drops
`conn_a`/`conn_b` from `missing_keys` when **both** are absent, so old-format checkpoints still
load under `strict=True` with exactly the pre-change semantics, while a half-present checkpoint
still errors as corrupt, and (b) refreshes `given_x_indices_of_y` after wiring restore on the
CUDA path. The backward-helper precomputation was factored into
`_compute_given_x_indices_of_y()` (same logic as upstream). Motivation (from code comments +
timing, landing during P3a): trained models must be reloadable exactly for netlist
export/verification and cross-machine resume. Regression test:
`mlgn/netlist/test_indices_buffers.py`. **Feedforward impact: additive** — checkpoint files gain
two keys; forward/backward math is untouched; old checkpoints remain loadable.

### 3.2 `difflogic/cuda/difflogic_kernel.cu`

All changes are **compile-compatibility fixes for modern PyTorch/CUDA (the cu12 Docker image)**
with zero behavioral change: `x.type().is_cuda()` → `x.is_cuda()` in `CHECK_CUDA` (81a2b76),
then `x.type()` → `x.scalar_type()` at all six `AT_DISPATCH_*` sites (b4812d1) — the
`Tensor.type()` dispatch API is deprecated/removed in PyTorch 2.x. Also in
`groupbitsum_kernel`, a C99-style designated union initializer (`{.signed_scalar = ...}`) was
replaced with a two-step assignment, which nvcc rejects pre-C++20. Same commit added
`mlgn/seqlgn/_cpu_compat.py`, a stub-injection shim that makes the package importable on
CPU-only machines (upstream hard-imports `difflogic_cuda` at module top even though a
pure-Python path exists).

### 3.3 New top-level additions by project

**P1 sequential LGN** — `mlgn/seqlgn/`: `cells.py` (recurrent cells incl. the gated mux cell),
`models.py`, `data.py` (copy task, delayed-recall MNIST, permuted MNIST), `train.py`,
`collate.py`, `utils.py`, `plot.py`, `run_queue.sh`, `docs/` (api/design/experiments/benchmarks),
plus `_cpu_compat.py` (above). Paper assets in `mlgn/paper/p1/`: `p1_draft1.md`,
`references.bib`, figure scripts `fig1_length_sweeps.py`/`fig2_keepbias_recall.py`/
`fig3_gradnorm.py`, `figstyle.py`, `run_queue_p1.sh`.

**P2 FPGA synthesis** — `mlgn/netlist/synth/`: exporters `export_fsm.py`/`export_top.py`,
Verilog sources and synthesized variants (`fsm.v`, `top.v`, `*_synth.v`, `fsm_pnr_wrap.v`),
testbenches `tb_*.v`, flow scripts `run_all.sh`/`run_pnr.sh`/`run_top.sh`/`build_chipdb.sh`,
`rand_probe.py`, `report.md`. Paper: `mlgn/paper/p2/p2_draft1.md` + `references.bib` +
`run_queue_p2.sh`.

**P3a netlist verification** — `mlgn/netlist/*.py`: `ir.py` (netlist IR), `extract.py`
(trained-model → netlist), `blif.py`/`verilog.py` (exporters), `props.py` (properties),
`sim.py`, `falsify.py`, `head.py`, `run_distractor_study.py`, `run_queue_p3a.sh`, tests
`test_head.py`/`test_can_extract.py`/`test_indices_buffers.py`. Paper:
`mlgn/paper/p3a/p3a_skeleton.md`.

**P3b applications** — CAN-bus IDS: `mlgn/seqlgn/can_data.py`, `test_can.py`,
`run_queue_c0g.sh`, `mlgn/seqlgn/data/can/README.md`. Flight-sim gate: `mlgn/flightgate/`
(`env.py`, `mock_env.py`, `teacher.py`, `student.py`, `trainer.py`, `encode.py`, `gate_eval.py`,
`cli.py`, tests, `run_queue_d1.sh`). Paper: `mlgn/paper/p3b/findings.md`.

**Knowledge base & roadmap** — `mlgn/research/00–24_*.md` (landscape, papers, scouts, workmaps,
roadmap) + `results_table.md`; parked idea stubs `mlgn/paper/p4..p14/init.md`; product stubs
`mlgn/product/{lgn-toolchain,can-ids,bearing-monitor,hft-inference}/init.md` + `00_README.md`.

**Infra** — `Dockerfile`/`.dockerignore` (the `salon64/difflogic-jupyter:cu12` cluster image the
CUDA fixes target), `.gitignore`, `images/` (figure PNGs).

**Doesn't fit the buckets:** `mlgn/secuential.py` (411-line June-05 prototype of a
`LogicRNNCell` — concat-recurrence over stacked `LogicLayer`s on MNIST; the predecessor
superseded by `mlgn/seqlgn/`, note the "secuential" typo) and `mlgn/mnist_test.py` (a plain
**feedforward** MNIST training/sanity script with gate-name inspection and best-model saving —
used to validate the upstream library and fork setup, not part of any paper pipeline). Both are
historical scaffolding.

### 3.4 Suggested reproducibility/code statement bullets

- Code is a fork of the official `difflogic` library (Petersen et al.); the core library is
  unchanged except for (i) a backward-compatible extension that persists each layer's random
  wiring in checkpoints (`conn_a`/`conn_b` buffers) and (ii) build fixes for PyTorch 2.x /
  CUDA 12. No training math, kernels, or numerics were modified.
- The gated cell and all baselines are compositions of unmodified `LogicLayer`s — no custom CUDA
  kernels were written for P1.
- All P1 headline configurations are documented as runnable commands (draft App A.2) with
  archived result JSONs in `mlgn/seqlgn/results/`; the 23-run hardening queue is
  `mlgn/paper/p1/run_queue_p1.sh` (idempotent, skip-if-JSON-exists). ⚠ Do **not** claim
  push-button full reproduction: the original headline job lines are no longer in
  `mlgn/seqlgn/run_queue.sh` (it was rewritten for later projects) — reconstruct a dedicated
  full-repro queue first if the paper is to make that claim.
- Released checkpoints are self-contained (weights + wiring), so reported models can be reloaded
  bit-exactly, including for discretization and netlist export.
- A pinned `Dockerfile` reproduces the GPU environment; a CPU import shim
  (`mlgn/seqlgn/_cpu_compat.py`) allows smoke-testing without the CUDA extension.

---

## 4. Sources — annotated bibliography

All 18 bib entries verified against canonical pages (arXiv abs / DOI / PMLR / ACL Anthology) on
2026-07-14. Every link below resolved and matched title+authors unless flagged. **Bib fixes
needed (resolves the TODO-BIB):** all `TODO(names)` are now filled in below; three entries need
edits — `miotti2025difflogic` (title truncated), `yue2025dlnts` (title wrong),
`bacellar2024dwn` (author missing).

### MUST-REREAD before writing

1. **Bührer, Simon; Plesner, Andreas; Aczel, Till; Wattenhofer, Roger — "Recurrent Deep
   Differentiable Logic Gate Networks" (RDDLGN), EdgeFM'25 workshop @ ACM MobiCom, Hong Kong.**
   [arXiv:2508.06097](https://arxiv.org/abs/2508.06097) ·
   [DOI 10.1145/3737902.3768357](https://doi.org/10.1145/3737902.3768357) (resolves; ACM page
   bot-blocked but title/venue confirmed via search + ETH Research Collection). First recurrent
   LGN: concat-recurrence encoder/decoder for WMT'14 translation, 16-token sequences, with
   reported vanishing gradients on longer sequences. This is *the* baseline P1 argues against
   (write-step gradient exactly zero claim; "carries state by concatenation" framing) — reread
   its memorization-task numbers and limits section closely. **Bib fix:** first author given
   name = **Simon**; venue is the *2nd* International Workshop on Edge and Mobile Foundation
   Models.
2. **Kim, Youngsung — "Align Forward, Adapt Backward: Closing the Discretization Gap in Logic
   Gate Networks" (CAGE).** [arXiv:2603.14157](https://arxiv.org/abs/2603.14157) (verified;
   single author; still preprint). Decomposes the train/deploy gap into a reducible *selection*
   gap and an irreducible *computation* gap (zero iff inputs binary). P1 leans on this
   decomposition in four places (§discretization analysis, the "gap grows with length =
   computation gap" argument, and a footnote noting Kim challenges the Hessian-regularizer story
   of Mind-the-Gap) — the single most load-bearing methods citation.
3. **Miotti, Pietro; Niklasson, Eyvind; Randazzo, Ettore; Mordvintsev, Alexander —
   "Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation", ALIFE 2025
   (ISAL proceedings, isal.a.882).** [arXiv:2506.04912](https://arxiv.org/abs/2506.04912).
   Stateful difflogic CA with pure combinational recurrence; explicitly names "LSTM-like gating"
   and state-forgetting gates as future work — P1's motivation quote. **Bib fix:** bib title
   drops the subtitle "From Game of Life to Pattern Generation"; ALIFE 2025 venue confirmed via
   MIT Press ISAL proceedings.
4. **Damera, Sai Sandeep; Matheu, Ryan; Puranic, Aniruddh G.; Baras, John S.; Belta, Calin —
   "On the Stability and Realizability of Recurrent Polynomial Surrogate Ternary Logic Gate
   Networks" (R-DTLGN).** [arXiv:2605.24649](https://arxiv.org/abs/2605.24649) (verified;
   comments: "Submitted to IEEE", 9 pp). Recurrent *ternary* LGN for STL runtime monitoring;
   delay-register concat-recurrence, no gating (gated variants named as future work — P1 cites
   exactly this). Reread §II-C/§VII to keep the distinguish-language precise; beware its
   unrelated "degenerate memory" terminology. **Bib fix:** given names = **Sai Sandeep** Damera,
   **Ryan** Matheu.
5. **Petersen, Felix; Borgelt, Christian; Kuehne, Hilde; Deussen, Oliver — "Deep Differentiable
   Logic Gate Networks", NeurIPS 2022.** [arXiv:2210.08277](https://arxiv.org/abs/2210.08277).
   The substrate: softmax-over-16 relaxation, GroupSum head, random fixed wiring. P1's cell
   definition and notation must match this exactly — reread §method before writing §2.

### SKIM

6. **Yousefi, Shakir; Plesner, Andreas; Aczel, Till; Wattenhofer, Roger — "Mind the Gap:
   Removing the Discretization Gap in Differentiable Logic Gate Networks", NeurIPS 2025 (main
   track — venue confirmed, update bib note).**
   [arXiv:2506.07500](https://arxiv.org/abs/2506.07500). Gumbel noise + STE closes 98% of the
   feedforward gap. P1 cites it as the training-side gap fix that targets only the *selection*
   component. **Bib fix:** first author given name = **Shakir**.
7. **Petersen, Felix; Kuehne, Hilde; Borgelt, Christian; Welzel, Julian; Ermon, Stefano —
   "Convolutional Differentiable Logic Gate Networks", NeurIPS 2024 Oral.**
   [arXiv:2411.04732](https://arxiv.org/abs/2411.04732). Residual initialization (bias toward
   pass-through gate "A") — P1 frames gate-open init as its *temporal* generalization; also
   source of the 9 ns FPGA number. Skim the residual-init section.
8. **Hochreiter, Sepp; Schmidhuber, Jürgen — "Long Short-Term Memory", Neural Computation
   9(8):1735–1780, 1997.**
   [DOI 10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) (resolves;
   volume/pages match). Constant error carousel — P1's central analogy ("CEC in pure Boolean
   logic") and the copy-task lineage. Skim §CEC to quote it correctly.
9. **Gers, Felix A.; Schmidhuber, Jürgen; Cummins, Fred — "Learning to Forget: Continual
   Prediction with LSTM", Neural Computation 12(10):2451–2471, 2000.**
   [DOI 10.1162/089976600300015015](https://doi.org/10.1162/089976600300015015) (resolves;
   matches). Forget gate + open-init practice; P1's gate-init-open analogy rests on it.

### CITE-ONLY (verified, no reread needed)

10. **Rüttgers, Lukas; Aczel, Till; Plesner, Andreas; Wattenhofer, Roger — "Light Differentiable
    Logic Gate Networks".** [arXiv:2510.03250](https://arxiv.org/abs/2510.03250) — 4-basis
    reparametrization; cited once. **Bib fix:** first author = **Lukas**; still preprint (no
    acceptance found — recheck before camera-ready).
11. **Kim, Youngsung — "Fitting Multilinear Polynomials for Logic Gate Networks".**
    [arXiv:2605.08657](https://arxiv.org/abs/2605.08657) — reparametrization cite alongside #10.
12. **Yue, Chang; Jha, Niraj K.** [arXiv:2508.17512](https://arxiv.org/abs/2508.17512) —
    non-recurrent time-series DLN, cited for contrast. **Bib fix (title wrong):** actual title
    is "**Learning Interpretable** Differentiable Logic Networks for Time-Series
    Classification".
13. **Bacellar et al. — "Differentiable Weightless Neural Networks", ICML 2024.**
    [arXiv:2410.11112](https://arxiv.org/abs/2410.11112) — adjacent LUT paradigm. **Bib fix
    (author missing):** arXiv lists 7 authors — bib omits **Lizy K. John** (between Eugene John
    and Priscila Lima).
14. **Cho et al. — RNN Encoder–Decoder (GRU), EMNLP 2014, pp. 1724–1734.**
    [ACL D14-1179](https://aclanthology.org/D14-1179/) (verified).
15. **Bengio, Simard, Frasconi — IEEE TNN 5(2):157–166, 1994.**
    [DOI 10.1109/72.279181](https://doi.org/10.1109/72.279181) (resolves).
16. **Pascanu, Mikolov, Bengio — ICML 2013, PMLR 28:1310–1318.**
    [PMLR v28](https://proceedings.mlr.press/v28/pascanu13.html) (verified).
17. **Le, Jaitly, Hinton — IRNN / pixel-MNIST provenance.**
    [arXiv:1504.00941](https://arxiv.org/abs/1504.00941) (verified) — source of the
    sequential-MNIST protocol P1's permuted-MNIST-28 derives from.
18. **Arjovsky, Shah, Bengio — Unitary Evolution RNNs, ICML 2016, PMLR 48:1120–1128.**
    [PMLR v48](https://proceedings.mlr.press/v48/arjovsky16.html) (verified) — copy-task
    formalization cite.

### Possible missing citations

- **RDDLGN on OpenReview ([knHHCx1prj](https://openreview.net/forum?id=knHHCx1prj)) — CHECK
  BEFORE WRITING (race-relevant).** Search snippets of this OpenReview version contain abstract
  language *absent from the arXiv/EdgeFM version*: "embeds sequential logic gates, analogous to
  **flip-flops and latches** in hardware" and positioning against "recurrent and **state-space**
  models." This may be an extended ETH resubmission moving toward exactly P1/P2's territory.
  OpenReview is Cloudflare-blocked to automated fetch — open manually and diff against
  arXiv:2508.06097 before finalizing the novelty claims.
- **Damera, Matheu, Puranic, Baras — "Polynomial Surrogate Training for Differentiable Ternary
  Logic Gate Networks" (PST).** [arXiv:2603.00302](https://arxiv.org/abs/2603.00302) (submitted
  to NeuS 2026). The training method R-DTLGN is built on; a one-line cite next to
  `damera2026rdtlgn` makes that entry self-contained.
- **Bührer, Plesner, Aczel, Wattenhofer — "BitLogic: Training Framework for Gradient-Based
  FPGA-Native Neural Networks", TMLR.** [arXiv:2602.07400](https://arxiv.org/abs/2602.07400).
  Feedforward-only unification of deploy-as-logic methods; marginal for P1 (fits the DWN
  "adjacent paradigm" slot) but signals the ETH group's trajectory — cite only if space allows.

No genuinely relevant 2025–2026 work on gating in *binary/quantized* RNNs was found beyond the
above (SigGate 2502.09318 is full-precision; QORNN is 2024) — the gated-logic-cell slot still
appears open.

---

## 5. Beyond the papers — blogs, videos, discussions

All links fetched and verified 2026-07-14. Skipped: Reddit (no substantive r/ML thread on
difflogic surfaced that could be verified) and a standalone 2022-paper HN thread (none with real
engagement exists — the community discussion happened via the CA article and the Clayton post
below).

### 5.1 Difflogic explained (Petersen's own materials + community)

- **[difflogic GitHub README](https://github.com/Felix-Petersen/difflogic)** — repo README,
  ~10 min. The canonical practical intro: the relaxation, `LogicLayer`/`GroupSum` API,
  `CompiledLogicNet`. Notably contains **zero discussion of recurrent/sequential use** — worth
  citing-by-omission when P1 claims the recurrent setting is unexplored in the reference
  tooling.
- **[Convolutional Differentiable Logic Gate Networks — NeurIPS 2024 Oral video](https://www.youtube.com/watch?v=FKQfMwFZvIE)**
  — video, ~5 min. Petersen's own compressed pitch; study how he sells "logic gates as the
  native compute primitive" in 5 minutes — that's the register P1's intro needs.
- **[Felix Petersen at CoRL 2024 Workshop on Differentiable Optimization Everywhere](https://www.youtube.com/watch?v=YEAQlkV9JAw)**
  — talk recording. A longer-form Petersen talk (LGNs + RL angle); useful for hearing which
  failure modes he emphasizes unprompted. His page [petersen.ai](https://petersen.ai/) indexes
  all papers/videos/code in one place.
- **["Using Logic Gates as Neurons"](https://www.youtube.com/watch?v=VulM891DhI4)** — third-party
  paper-explainer video (Feb 2023). Verified to exist and be on-topic; useful mainly to see how
  an outsider re-derives the 16-gate relaxation for a lay audience.
- **[MIT Technology Review: "The next generation of neural networks could live in hardware"](https://www.technologyreview.com/2024/12/20/1109183/the-next-generation-of-neural-networks-could-live-in-hardware/)**
  — press article, ~8 min. The mainstream framing of LGNs (energy, on-chip inference) and the
  one caveat journalists kept: training is hundreds of times slower than GPUs — a caveat P1
  inherits and should pre-empt.

### 5.2 DiffLogic Cellular Automata (closest non-academic prior art)

- ★ **[Differentiable Logic CA — Google Paradigms of Intelligence interactive article](https://google-research.github.io/self-organising-systems/difflogic-ca/)**
  (Miotti, Niklasson, Randazzo, Mordvintsev, Mar 2025) — interactive long-read, ~8–10k words.
  THE model for explaining learned recurrent Boolean circuits with per-cell binary state: Game
  of Life, a 22-gate checkerboard, explorable learned circuits. P1's "state as explicit bits
  updated by learned gates" story should be legible to anyone who read this; steal its habit of
  showing the *tiny discrete circuit* that training found.
- **[HN thread on DiffLogic CA](https://news.ycombinator.com/item?id=43286161)** — discussion,
  469 points / 90 comments. What the community latched onto: compile-to-hardware,
  interpretability vs "magic floating points," FPGA speculation, Levin-style self-organization —
  and recurring skepticism about scalability. A cheap map of which P1 claims will be applauded
  vs challenged.

### 5.3 Gating & long-range memory classics (reread as writing models)

- ★ **[colah, "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)**
  (2015) — blog, ~2.5–3k words. The definitive gate explainer: cell state as "conveyor belt,"
  gates as 0/1 valves. P1's learned mux is literally the Boolean limit of this picture (gate ∈
  {0,1} exactly, carousel error exactly preserved) — reread it to calibrate diagrams and to
  phrase "write-step gradient" intuition the way readers already know it.
- **[Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)**
  (2015) — blog long-read. Writing model more than content: concrete artifacts first, mechanism
  second, honest failure gallery. Its interpretable-neuron section (a cell tracking quote-state)
  is the float analog of P1's inspectable latch bits.
- **[Madsen, "Visualizing Memorization in RNNs"](https://distill.pub/2019/memorization-in-rnns/)**
  — Distill article (2019), interactive, ~15 min. Directly about *long-range recall*, P1's exact
  axis: connectivity visualizations showing which past inputs each architecture actually uses.
  Also a template if you want a "which timestep does the gradient reach" figure.

### 5.4 Hands-on / FPGA-adjacent community material

- ★ **[Isaac Clayton, "Compiling a Neural Net to C for a 1,744× speedup"](https://slightknack.dev/blog/difflogic/)**
  (May 2025) — blog, ~25 min, plus its **[HN thread](https://news.ycombinator.com/item?id=44118373)**
  (296 points / 93 comments). A full reimplementation of DiffLogic-CA-style training with the
  clearest independent walkthrough of the 16-gate relaxation, discretization, and circuit
  extraction (92.7% pass-through gates!). The HN comments are a goldmine of P1-relevant
  objections: fixed-vs-learned wiring, convergence pain, discretization gap, even the difflogic
  patent — exactly the honest-boundary terrain P1 walks.
- **[Tessolve / Verification Futures 2025: "Rethinking AI Inference through Differentiable Logic Gate Networks"](https://www.tessolve.com/verification-futures/vf2025-uk/rethinking-ai-inference-through-differentiable-logic-gate-networks-difflogic/)**
  — industry talk abstract (Georg Meinhardt, DiffLogic Inc). Abstract only (no recording found),
  but proof LGN inference is being commercialized (1.3–40 ns latency, FINN comparisons) — one
  line of P1 motivation, verified.
- **[ijc, "GSoC'24: Differentiable Logic for Interactive Systems and Generative Music"](https://ijc8.me/2024/08/26/gsoc-difflogic/)**
  — blog, ~3.5k words ([HN: 105 points](https://news.ycombinator.com/item?id=41638581)).
  Difflogic in the wild on embedded hardware (Bela/BeagleBone) for bytebeat synthesis — and
  explicitly *stateless*, driven by a binary clock input. Nice evidence that even creative-coding
  users hit the "LGNs have no memory" wall P1 fixes.

---

## 6. How to write it — academic writing guides

All links verified 2026-07-14. Curated for the specific job: compressing an existing ~8-page
draft into a sharp 4-page non-archival workshop paper with one headline claim and an honest
negative result.

### 6.0 First: venue + submission mechanics (repo-specific)

- **Venue decision is the blocking task** ([TODO-VENUE]) — the repo's own venue analysis is
  `mlgn/research/07_venues_timeline.md`; the workshop list dropped ~Jul 11, so the decision is
  due now.
- **Submission mechanics once the venue is picked:** the draft is pandoc markdown with `[@key]`
  citations → it must be converted into the workshop's LaTeX template (typically the NeurIPS
  style file with the workshop's footer). Check the **anonymization policy** early — the draft
  currently names the author *and* the repo path, and non-archival workshops vary
  (single-blind vs double-blind). Expect an **OpenReview** submission flow: account with
  institutional email, possible separate abstract deadline, and check whether the page limit is
  inclusive or exclusive of references.

### 6.1 Canonical general guides

- **Simon Peyton Jones — "How to write a great research paper"**
  ([MSR page + slides](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/),
  [PDF slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/02/simon-peyton-jones_paper.pdf)).
  Slides; 30–45 min. Core idea for you: a paper is a vehicle for **one** clear, reusable idea,
  stated as a single sentence ("gating is a Boolean constant-error carousel; concat-recurrence
  has exactly-zero write-step gradient"). A 4-page paper is that idea plus evidence and nothing
  else — his "identify your key idea / nail your contributions as refutable claims" is the
  condensing criterion.
- **Whitesides' Group: Writing a Paper**
  ([Wiley DOI](https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/adma.200400767),
  [free mirror PDF](https://mason.gmu.edu/~hjing2/advice%20for%20phd%20students/Writing%20a%20paper-George%20Whiteside.pdf)).
  3-page article; 20 min. The outline-first method: don't delete from the 8-page draft — build a
  fresh 4-page skeleton from figures + claims, then port only sentences that earn their place.
- **Mensh & Kording — "Ten simple rules for structuring papers"**
  ([PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619)).
  Paper; 45 min. Rule 1 (one central contribution per paper) and context–content–conclusion at
  every scale (paper, section, paragraph) — the best paragraph-level test for what to cut.
- **Steven Pinker — "Why Academics Stink at Writing"**
  ([PDF, author's site](https://stevenpinker.com/files/pinker/files/pinker_2014_why_academics_writing_stinks.pdf)).
  Essay; 30 min. Classic style + the curse of knowledge: most of the wordage you'll cut is
  metadiscourse and defensive hedging, which he teaches you to see.
- **Jennifer Widom — "Tips for Writing Technical Papers"**
  ([Stanford page](https://cs.stanford.edu/people/widom/paper-writing.html)). Short page;
  15 min. Her five-question intro formula (problem / why interesting / why hard / why prior
  solutions fail / contributions) is exactly the shape of a half-page workshop intro.

### 6.2 ML-specific

- **Jakob Foerster — "How to ML Paper — A brief Guide"**
  ([page → Google Doc](https://www.jakobfoerster.com/how-to-ml-paper)). ~2 pages; 20 min.
  Single-narrative-thread discipline and a paragraph-by-paragraph intro recipe; every claim in
  the intro must map to a specific figure/table. Read before restructuring.
- **Jacob Steinhardt — "Advice for Authors"**
  ([blog](https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors)). Blog post; 20 min.
  Argues local, sentence-level style ("be precise, be concise") matters more than global
  structure — the actionable manual for the line-edit pass that gets you from 5 pages to 4.
- **Neel Nanda — "Highly Opinionated Advice on How to Write ML Papers"**
  ([Alignment Forum, 2025](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)).
  Long post; ~1 hr. Best available treatment of **claim discipline**: match claim type
  (existence-proof vs. systematic vs. hedged) to evidence strength. Directly applicable to
  stating "enables 50–100-step recall" (systematic, on your three tasks) vs. the
  equal-gate-count negative result (a feature, not a confession).
- **Michael Black — "Writing a good scientific paper"**
  ([Perceiving Systems blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper)).
  Long post; 45 min. From a 5× test-of-time-award winner: honestly stated limitations *build*
  reviewer trust — the right frame for your discretization-gap and no-classification-gain
  boundaries. Also strong on Figure-1-as-story.
- **Lipton & Steinhardt — "Troubling Trends in Machine Learning Scholarship"**
  ([arXiv:1807.03341](https://arxiv.org/abs/1807.03341)). Paper; 40 min. Its "failure to
  identify sources of empirical gains" trend is the objection your equal-gate-count control
  preempts — say so explicitly, in their vocabulary.

### 6.3 Workshop-paper specific / condensing

- **Devi Parikh — "Shortening papers to fit page limits"**
  ([Medium](https://deviparikh.medium.com/shortening-papers-to-fit-page-limits-97601318681d)).
  Post; 15 min. The single most operational resource here: shorten by saying the same thing in
  fewer words (straggler lines, local tightening, LaTeX whitespace), not by amputating content.
  Do her 30–90-min pass twice.
- **NeurIPS Paper Checklist** ([neurips.cc guide](https://neurips.cc/public/guides/PaperChecklist)).
  Checklist; 30 min self-audit. Even for a non-archival workshop, run it: "do abstract/intro
  claims match scope?" and "limitations are not penalized" are the venue's own codification of
  claim discipline.

### 6.4 Figures & tables

- **Rougier, Droettboom & Bourne — "Ten Simple Rules for Better Figures"**
  ([PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003833),
  [companion code](https://github.com/rougier/ten-rules)). Paper + repo; 30 min. "Message trumps
  beauty" + self-contained captions. In 4 pages your recall-vs-delay curve (gated flat, concat
  collapsing to chance) *is* the argument — design it to carry the headline unaided.

### 6.5 YouTube

- **"PhD: How to write a great research paper" — Simon Peyton Jones**
  ([YouTube](https://www.youtube.com/watch?v=1AYxMbYZQ1Y), Microsoft Research channel, ~1 hr).
  The talk version of the slides above; watch instead of reading if you prefer.
- **"How to write an okay research paper." — Sasha Rush**
  ([YouTube](https://www.youtube.com/watch?v=qNlwVGxkG7Q), his channel @srush_nlp, ~25–30 min).
  Deliberately anti-perfectionist: a concrete, section-by-section walkthrough of a competent
  NLP/ML paper. The right calibration for a first solo workshop paper — aim for clean and
  correct, not magnum opus.
- **Bill Freeman — "How to Write a Good Research Paper"**
  (in [CVPR18 Good Citizen workshop, Part 1, from 4:08](https://youtu.be/MKUCz_3Ee0A?t=248),
  ~20 min segment; slides indexed at
  [Parikh's Good Citizen page](https://deviparikh.com/citizenofcvpr/)). Vision-flavored but
  general: treat the reviewer as a tired, skeptical reader you must convince in one pass.

### 6.6 Suggested order (next 2 weeks)

**Week 1 — restructure (idea → skeleton).** (1) SPJ talk or slides (1 hr). (2) Foerster guide
(20 min). (3) Widom's intro formula (15 min). Then rebuild the 4-page skeleton Whitesides-style
from figures + claims before touching prose. (4) Mensh & Kording as you draft sections.
(5) Nanda's claim-type section (30 min) while writing the abstract/intro — fix exact claim
wording, including the honest-boundary sentences.

**Week 2 — condense and harden.** (6) Parikh immediately before cut pass one; repeat before
submission. (7) Steinhardt for the line-edit pass. (8) Rougier while finalizing the money
figure. (9) Troubling Trends skim + NeurIPS checklist self-audit on the near-final draft.
Background/optional, over lunch: Rush video, Pinker, Black, Freeman.

---

## 7. Open items & flags (from the adversarial check)

The assembled sections above already incorporate the critic's corrections. What remains open:

1. **RDDLGN OpenReview version ([knHHCx1prj](https://openreview.net/forum?id=knHHCx1prj)) —
   manual check required.** Cloudflare-blocked to automated fetch; its abstract language
   ("flip-flops and latches", "state-space models") suggests an extended ETH resubmission moving
   toward P1/P2 territory. Open it in a browser and diff against arXiv:2508.06097 **before
   finalizing the novelty claims**.
2. **Apply the three bib edits** (§4 header): `miotti2025difflogic` subtitle,
   `yue2025dlnts` title, `bacellar2024dwn` missing author (Lizy K. John) — plus the given names
   now filled in for buhrer/damera/yousefi/ruttgers entries.
3. **Tag the paper commit.** The repo has zero tags; `20_program_validation.md` §E7 calls for
   tagging the commit the paper numbers come from before anything goes public (merge/push
   already done).
4. **Venue decision** (`mlgn/research/07_venues_timeline.md`) — then template conversion +
   anonymization check (§6.0).
5. **Known rounding mismatch**, noted so you don't stall on it: dMNIST-50 equal-gates control
   disc is 0.1165 in the JSON and the v1.1 header; the draft body says 0.117 (round-half-up).
   Pick one convention for the camera-ready.
6. **Reproducibility claim scope**: don't claim push-button full reproduction (§3.4) — the
   original headline job lines were rewritten out of `mlgn/seqlgn/run_queue.sh`; reconstruct a
   dedicated queue first if you want that claim.
