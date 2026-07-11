# Distractor study — free-input verification campaign (P3a)

**Date:** 2026-07-10 · **Runner:** `python -m mlgn.netlist.run_distractor_study` (agent CAMPAIGN)
**Circuits:** combo = `ckpt_cp50A_curr_c35` (copy-35, disc test_acc 1.0, settle K=16) · gated = `ckpt_cpB_gated_oracle` (copy-50, disc test_acc 0.3803, settle K=22)
**Machine-readable results:** `report.json` (this dir) + per-circuit `combo/`, `gated/` (BLIFs, ABC logs, cex dumps, `circuit_report.json`, `bfs_witness_sym*.json`).

Both replay gates passed (combo 1.0000 exact; gated 0.3750 on 8 batches vs 0.3803 recorded, within the 3.9-sigma band).
Every ABC counterexample below was **replay-confirmed bit-exactly** in the numpy simulator on the property netlist
(`cex_replay.confirmed=true` in `report.json`), and BFS-vs-MC cross-validation found **zero contradictions**; both expected
anchors (combo `protocol_decode` PROVED, gated `protocol_decode` CEX) reproduced.

## Verdict table

Distractor alphabet after the legal write: {blank} + {8 non-cued one-hot symbols}, free at every frame.
fs0 = distractors only after arming (t >= K); fs1 = distractors already during the settle window (t >= 1).

| property | combo (K=16) | gated (K=22) |
|---|---|---|
| `protocol_decode` (write, blanks forever, readout==written) | **PROVED** — `tempor -F 18; scorr; dc2; pdr` in 14.9 s (bmc3 clean to 148 fr) | **CEX @ frame 22** — bmc3 5.7 s; write sym1, pure blanks, wrong fixed point |
| `distractor_hold` fs0 (state frozen under distractors) | **CEX @ 16** — bmc3 5.9 s; `WRITE sym0` + single `sym0` token at t=16 moves the state | **CEX @ 22** — bmc3 7.2 s; `WRITE sym1` + `sym7` at t=22 |
| `distractor_hold` fs1 | **CEX @ 16** — bmc3 0.4 s (14-token stream during settling) | **CEX @ 22** — bmc3 0.3 s; recipe 0.7 s (tempor prefix BMC) |
| `distractor_decode` fs0 (readout correct under distractors) | **CEX @ 30** — deep `bmc3 -T 900 -F 40` in 276 s; the mandated `bmc3 -T 180` stopped clean at 29 frames and the recipe was UNDECIDED (pdr stalled at transformed frame 4) | **CEX @ 22** — bmc3 0.3 s (wrong fixed point suffices, no distractors needed) |
| `distractor_decode` fs1 | **CEX @ 16** — bmc3 0.4 s; corrupting stream lives entirely inside the settle window | **CEX @ 22** — bmc3 0.4 s |

## BFS pre-analysis (sim-side ground truth; closure of each write's fixed point under the 9 tokens, cap 200 000)

| | combo | gated |
|---|---|---|
| closure sizes (sym0..7) | 2, 1522, 1867, 87 391, 1240, **>=200 001 (ESCAPED CAP)**, 13 376, 22 | 555, 382, 80, 382, 73 049, 382, 1768, 382 (all complete) |
| symbols with a wrongly-decoding reachable state | sym1 (953 wrong, depth >=15), sym3 (183, depth >=16) | all but sym2 (e.g. sym4: 73 049/73 049 wrong; sym1/3/5/7 share one 382-state all-wrong closure decoding 6) |
| fixed points decoding correctly | 8/8 | 3/8 (sym0, 2, 6) |

Neither circuit is in the trivial "enable fully shut" case (closure 1 everywhere): distractors genuinely move both registers,
combo's sym5 closure defeats enumeration outright, and gated's closures are finite so its fs0 envelope is closable by <=73 k
simulations — the model checker had real work to do on combo and confirmatory work on gated.

## Concrete counterexamples (all replay-confirmed)

- combo hold: `WRITE sym0` -> 16 blanks -> one `sym0` token flips state bits at the first armed frame (echoing the written symbol as a distractor is enough).
- combo decode (ABC, frame 30): `WRITE sym5`, blanks to t=15, then `sym5 sym2 sym4 (two-hot->blank) sym3 sym1 sym4 sym1 sym0 sym2 sym7 sym2 sym7 sym7` -> readout leaves class 5. This corrupts **sym5 — the symbol whose closure escaped the BFS cap with zero wrong states found in 200 k**: bounded MC reached what enumeration could not.
- combo decode (BFS witnesses, `bfs_witness_sym{1,3}.json`): sym1 + 15 tokens (`sym2 ...blanks... sym4 ...blanks... sym2 blank blank sym7`) -> decodes 3, bad at frames 31-32; sym3 + 16 tokens -> decodes 5, bad at frame 32. Both fire in the fs0 AND fs1 property netlists.
- gated: no cleverness needed — 1-4 tokens (or none at all for decode: the written fixed point is already wrong) break everything at frame 22.

## Interpretation

1. **What is now proven about the combo register:** the full write->settle->readout theorem under *blank* delay (`protocol_decode`,
   arbitrary delay >= 16) is machine-checked — but the copy-trained combo register is **not distractor-robust in any sense**: no
   variant of hold or decode survives, so its correctness certificate is exactly "correct iff the channel stays silent after the
   write," a deployment envelope statement no test set at L=35 could have exposed. The gated circuit fails everything including
   the blank-channel theorem, and its 3/8-correct fixed points make even the decode counterexamples distractor-free.
2. **Where MC earned its keep:** (a) the sym5 readout corruption — exhaustive BFS *escaped its 200 k cap* there and had found no
   wrong state; directed bounded MC produced one in 276 s, then simulation re-verified it; (b) the fs1 envelope (distractors during
   settling), which BFS does not cover at all, was killed by bmc3 in <1 s on both circuits; (c) the one genuine proof,
   `protocol_decode`, needs the `tempor->scorr->dc2->pdr` recipe (naive pdr/k-induction time out) and no amount of simulation could
   deliver a for-all-delays theorem. Notably, the builder's earlier *random* distractor smoke test (224 armed frames) fired 0 times
   on combo decode — random simulation, systematic enumeration, and directed search form a strict hierarchy here.
3. **Where MC hit its limits:** on the false-but-deep combo `distractor_decode_fs0`, the mandated `bmc3 -T 180` stopped clean at 29
   frames — one frame short of the shallowest cex — and the proving recipe went UNDECIDED because with free inputs `scorr` no longer
   collapses the cone (806 latches / 12 k ANDs survive vs 0/0 in the blank-input case) and pdr stalled at transformed frame 4. The
   BFS witness (depth 15 => frame 31) both decided the property on the sim side and told us how deep to push the re-run
   (`bmc3 -F 40 -T 900` -> CEX @ 30). **Recipe for false free-input properties: simulation-guided depth, then bounded MC; the tempor
   recipe is for the true ones.**
4. **Soundness discipline paid off twice:** every cex was replayed bit-exactly through the property netlist (catching, along the way,
   a token-indexing bug in the witness extractor before it could mislead), and the BFS<->MC cross-validation closed with zero
   contradictions on 10 property verdicts.

## Deviations / notes

- Expected anchors matched exactly (combo `protocol_decode` PROVED again, 14.9 s; gated CEX @ 22 consistent with 3/8 fixpoints).
- `distractor_decode_fs0` (combo) required one engine beyond the mandated pair: `bmc3 -T 900 -F 40` (logged as `bmc3_deep`, cex + log
  in `combo/`). Without it the ABC verdict is UNDECIDED while the ground truth (BFS witness + replay) is CEX.
- `tempor_pdr` on CEX properties usually burns its full 600 s pdr budget before reporting the counterexample; bmc3 is the right
  first engine for suspected-false properties.
- The campaign is resumable (`--resume`); this run resumed twice after host-side interruptions, reusing completed BFS closures and
  property verdicts (visible as RESUME markers in `campaign.log`).

---

# Round 3 addendum (2026-07-11) — distcopy-TRAINED circuits

Same pipeline, four disc-1.0 circuits trained WITH distractors (DUST `run_queue_p3a`,
new-format self-contained checkpoints; per-circuit artifacts in `dc_*/`). All replay
gates exact; BFS<->MC cross-validation and anchors clean on every circuit.

## The dose-response result

| circuit (training) | protocol_decode | d_hold fs0 | d_hold fs1 | d_decode fs0 | d_decode fs1 |
|---|---|---|---|---|---|
| combo (copy, round 2)  | PROVED | CEX | CEX | CEX @ 30 | CEX @ 16 |
| dc_clatch_d8  | PROVED | CEX | CEX | **PROVED** (pdr invariant: 120 clauses / 90 flops) | CEX @ 19 |
| dc_gated_d8   | PROVED | CEX | CEX | **PROVED** (65 s) | CEX @ 29 |
| dc_combo_d8   | PROVED | CEX | CEX | **PROVED** | CEX |
| dc_combo_d20  | PROVED | CEX | CEX | **PROVED** | **PROVED** |

1. **Provable robustness scales with training-time distractor pressure**: 0 distractors
   -> no distractor theorem; d8 -> post-settle readout robustness (fs0) proved on all
   three mechanisms; d20 -> full robustness incl. the settle window (fs1). The d20-combo
   fs1 theorem is the strongest artifact of the program: ANY legal write + ANY distractor
   token sequence at ANY time => readout equals the written symbol at every frame >= 13.
2. **`distractor_hold` CEXes on every circuit, trained or not**: the state register
   wiggles under distractors even when the readout is provably invariant — the
   certificate lives at the readout (decode-type specs), not at the state bits.
3. BFS ground truth matches: distcopy-trained closures are near-frozen (sizes 1-32)
   vs copy-trained 87k/escaped-cap; `bfs_closure` now seeds from limit-cycle orbits
   (`attractor_period` recorded; dc_combo_d20 sym0 is a correctly-decoding period-2 orbit).
4. dc_clatch_d8's fs0 proof is the first where pdr produced a NON-trivial inductive
   invariant (120 clauses over 90 residual flops) instead of scorr collapsing the cone —
   free-input proofs are genuinely harder, and the recipe still closes them.
   (dc_clatch_d8's decode fs0/fs1 verdicts were re-run in the foreground after repeated
   background host kills; engine provenance in its circuit_report.json.)
