# 26 — Scout: Exact Boolean-function analysis of trained LGNs ("simplicity bias, measured exactly")

**Date:** 2026-07-21. Adversarial novelty scout (4 parallel search agents, ~30 papers screened, ~10 read in full).
**Origin:** brainstorm "would we use LGNs to analyze the manifold hypothesis?" → reframed to the hypercube-native version: exact function-space analysis of trained logic circuits.

---

## NOVELTY SCOUT: precise claim

> Trained LGNs are small exact Boolean circuits, so function-space quantities that can only be *estimated* for ReLU nets/transformers can be computed *exactly* — Fourier/Walsh spectrum, average sensitivity, influences, noise stability (O'Donnell toolkit), #SAT decision-region volumes, exact Hamming robustness radii — and used as a measurement instrument for simplicity bias, implicit regularization, and generalization.

## DECOMPOSITION
- **Observation** ("nets are biased to simple/low-sensitivity functions"): heavily occupied (Valle-Pérez ICLR'19; Mingard Nat. Comms 2025; Novak ICLR'18; Vasudeva ICLR'25; Bhattamishra ACL'23; Abbe NeurIPS'22; Rahaman; Yang&Salman).
- **Method** (exact circuit-level machinery on trained discrete nets): exists for *verification* (SAT on BNNs since Narodytska AAAI'18; #SAT/BDD on BNNs = NPAQ CCS'19, BDD4BNN CAV'21/TOSEM'22; SAT on LGNs = Kresse NeuS'25; SAT on truth-table nets = TTnet IJCAI'24) — never pointed at function-space measurement.
- **Metric** (exact sensitivity/degree/counting as generalization measures): OPEN — all existing generalization-measure suites (Jiang ICLR'20 etc.) use proxies/bounds.
- **Application** (LGN ecosystem): training-side Fourier machinery exists (DWN spectral reg ICML'24; WARP Walsh parameterizations 2510.15655 / 2602.03527; per-gate Fourier stratification 2603.00302) — all per-gate / training-time, never post-hoc end-to-end analysis.

Synonyms searched: Fourier/Walsh–Hadamard spectrum, average sensitivity, Boolean influence, noise stability/sensitivity, spectral/low-degree bias, simplicity bias, Occam razor, model counting, #SAT, quantitative verification, maximal safe radius, decision-region volume, circuit complexity of learned functions.

## KEY PRIOR-ART HITS (all READ unless noted)

**Direct threats:**
- **Kresse, Yu, Lampert, Henzinger — "Logic Gate Neural Networks are Good for Verification," NeuS 2025 (arXiv 2505.19932).** First SAT encoding of trained difflogic LGNs; global robustness + fairness as 2-safety hyperproperties; tabular datasets, 50–300 gates/layer; Disruptive Idea Award. **Owns the "LGNs are verification-friendly netlists" framing.** Does NOT: counting, radii distributions, Fourier, sensitivity, interpretability, image scale. ⚠️ **Emily Yu (outreach-plan PI target) is a coauthor** — collaboration hook and race risk simultaneously.
- **BDD4BNN (CAV'21 + TOSEM'22, arXiv 2103.07224).** Exact BDD compilation of BNNs on Hamming balls: exact adversarial counts, class distributions, **exact locally-maximal Hamming radii**, essential-feature queries. Kills "exact quantitative analysis of a trained discrete net" as new in KIND. Does NOT: LGNs, whole-input-space, function-space/spectral quantities.
- **Mingard et al., "Deep neural networks have an inbuilt Occam's razor," Nature Comms 2025 (2304.06670)** (abstract + secondary). Trained nets + complexity of learned function + generalization — but LZ proxy on n≈7 exhaustive Boolean tables. **The exact-simplicity-bias ceiling to break.**
- **arXiv 2505.24060 (May 2025).** Exact min-DNF complexity of trained depth-2 discrete nets — but n≤7, synthetic tasks, non-SGD learners.
- **arXiv 2603.00302 (Mar 2026), polynomial surrogate ternary LGNs.** Real Fourier analysis inside trained difflogic-style nets — strictly **per-gate** (3-input ternary basis, complexity bands); no end-to-end spectrum, no sensitivity, no generalization link.
- **Vasudeva et al., ICLR 2025 (2403.06925).** Sensitivity↔generalization/robustness on trained transformers/CNNs at real scale — all **sampled**. The estimate-based paper the exact version must beat.
- **TTnet, Benamira et al., IJCAI 2024 (2208.08609).** Truth-table CNN with exact CNF dual; sound+complete local robustness at MNIST/CIFAR scale (~10ms/query); TT-rules sister papers do exact global rule extraction. Different architecture from difflogic; no counting/spectral analysis.
- **Kim 2605.08657** (re-read): gradient/trainability theory only (multilinear polynomial parameterization, interaction-coefficient starvation). Zero function-space analysis of trained nets — does NOT threaten this angle.

**Context / secondary:** NPAQ CCS'19 (approx #SAT on BNNs); Yang et al. SAT 2025 certified BNN counting (Meel group, active); Shih et al. SAT'19 (OBDD compilation of BNNs, region-bounded); Marzari IJCAI'23 #DNN-Verification (exact counting for ReLU exists — #P-complete, tiny nets ⇒ **drop the "impossible for ReLU" rhetoric; correct claim = exact AND global AND cheap**); Serra ICML'18 exact linear-region counting; eXpLogic 2503.09910 (saliency-style XAI on difflogic — informal); DiffLogic CA ALIFE'25 (informal circuit inspection only); Abbe NeurIPS'22 (Boolean-measure↔generalization *theory*, target functions not trained models); Gorji UAI'23 (WHT degree profile of trained ReLU nets, approximate); Nordström Ekstedt 2308.09374 (noise sensitivity of DNNs at init, theory); "Revisiting the Volume Hypothesis" 2606.31282 (Louis-adjacent community moving onto binary nets in 2026 — race signal); ICLR'26 "Noise Stability of Transformer Models" (2602.08287 — the neighborhood is active NOW).

## ANGLE STATUS

| Sub-angle | Status | Evidence |
|---|---|---|
| (a) Exact end-to-end Fourier/sensitivity/influences/noise stability of trained LGNs | **OPEN** | Closest: 2603.00302 (per-gate only); WARP/DWN (training-time only); nobody computes these exactly on any trained net's global function |
| (b) #SAT counting on LGNs (region volumes, probabilistic robustness, disagreement mass) | **OPEN for LGNs / OCCUPIED for BNNs** | BDD4BNN does exact counts+radii on BNNs; zero hits on LGNs ⇒ new in model class only, unless global/longitudinal |
| (c) SAT robustness certification of LGNs | **OCCUPIED** | Kresse NeuS'25 (global, tabular) + TTnet IJCAI'24 (local, image scale). Sliver: exact per-input maximal-radius *distributions*, image-scale Petersen LGNs |
| (d) Circuit-level interpretability of LGNs | **PARTIAL** | eXpLogic (saliency), TT-rules (exact but different arch). Exact prime-implicant/BDD explanations of difflogic unclaimed, framing taken twice |
| (e) Exact simplicity-bias measurement at real-dataset scale, correlated with generalization | **OPEN** | Every exact prior result stuck at n≈7 (Mingard '25, 2505.24060); every real-scale result is sampled (Novak, Vasudeva) — the conjunction is unclaimed |

## SURVIVING GAP

**"Exact function-space measurement science on trained LGNs"** — the conjunction (a)+(b)+(e):
1. Exact O'Donnell quantities of the learned end-to-end function: total/per-variable influence, average sensitivity, noise stability at ρ, degree/spectral-energy profile (via circuit structure, BDD, #SAT — pre-commit which are exact vs. certified-bounded; full 2^n spectra only on small-n task suites).
2. #SAT/BDD global quantities nobody computes anywhere: whole-input-space decision-region volumes, exact decision-boundary size (# Hamming-adjacent crossing pairs — unclaimed for ANY learned model), exact disagreement mass between two trained nets (a counting-based stability measure).
3. **Longitudinal:** track all of it across training epochs, width/depth, label noise, seeds — exact view of spectral-bias/simplicity-bias dynamics (all prior dynamics work is estimated). Correlate with train/test gap → the exact analogue of Novak'18/Vasudeva'25, breaking Mingard's n≈7 ceiling.
4. Bonus tie-in: exact effect of random wiring on sensitivity ↔ parked "why-random-wiring-works" sliver (21_landscape_weakness_scout); test Abbe's Boolean-influence predictions on a real trained model.

**Framing constraints (mandatory):** analysis/measurement-science paper, NOT verification (cite Kresse + TTnet as feasibility baseline); never say "impossible for ReLU" (say exact+global+cheap); address uniform-measure vs. data-distribution explicitly (compute both where possible).

## CONTRIBUTION SHAPE
Analysis paper: "the first deep model class where simplicity bias and the sensitivity–generalization link are measured exactly, at real-dataset scale, across training." Findings > machinery: value concentrates in any result that *diverges* from the ReLU/transformer story (random wiring, gate-type distribution effects) and in the exact dynamics curves.

## RESOURCE FIT
Excellent for solo + DUST tier: small models by design, training sweeps cheap, `mlgn/netlist/` exporter + P3a verification tooling directly reusable, ABC/BDD/#SAT tooling free. No barrier.

## VERDICT: **GO (conditional on framing), confidence ~70%**

Occupied cells: verification-as-headline (Kresse/TTnet), exact-counting-in-kind (BDD4BNN), the expected *finding* (low-degree bias — pre-registered many times). Open cells: everything measurement-science. Race risk HIGH-ish: ISTA (Kresse/Yu/Lampert/Henzinger) can bolt counting onto their pipeline in a weekend; Louis/Mingard lineage moving to binary nets; WARP crowd one step from "Fourier-analyze the trained net." Emily Yu overlap is double-edged — arguably the strongest *collaboration hook* in the whole roadmap (email-with-artifact channel per 25_outreach_plan).

**REMAINING GATE (feasibility falsifier, ~1–2 days):** take an existing trained MNIST-scale LGN netlist and attempt (i) exact total influence/average sensitivity via BDD or #SAT, (ii) one whole-space #SAT class-volume count. If both blow up beyond ~small-hundreds-of-gates models, the "exact at scale" pitch degrades to sampled ≈ Vasudeva and the angle dies back to a small-n-suite paper. PASS ⇒ this is a strong pool candidate; consider slotting into the p4..p14 grab-bag with priority.

## GATE RESULT — 2026-07-21, rung 1 PASSED on laptop

Falsifier run (`mlgn/netlist/exact_gate.py`, report `mlgn/netlist/out/exact_gate/report.json`).
Models: fresh CAN-syn IDS 'ff' LGNs (120 PIs, pos-weighted, non-degenerate): **h64** (128 gates, F1 0.38) and **h512** (1024 gates, F1 0.84). Netlists bit-exact vs torch (equivalence gate 0 mismatches). Engine: `dd.autoref` (pure-Python BDD).

- **h64: PASS in 9.2 s.** Exact class-1 volume **0.40566793796054385**; exact average sensitivity **6.496** (vs n/2 = 60 for a random function → the learned function is ~9× smoother than random — the first exact simplicity-bias number on a trained net at n=120, vs the field's n≈7 ceiling); 84/120 inputs influential; exact per-variable influence profile (max 0.30); decision BDD = 7,230 nodes. MC cross-checks agree (0.398 / 6.73, within sampling error).
- **Key mechanism finding: the gate circuit is trivial (216 BDD nodes, 0.0 s); ALL hardness lives in the GroupSum popcount-comparator head.** Naive natural-order DP blew to 89M nodes; support-local variable ordering + sign-decided DP-state pruning collapsed it to 28k peak (≈3000×). This asymmetry (logic easy, cardinality head hard) is itself a finding — and explains why SAT *decision* verification (Kresse) scales while counting needs care.
- **h512: BLOWUP on pure Python** (node cap 8M at DP bit 72/512, ~18 min). → the scale wall for laptop/pure-Python tooling sits between 64- and 512-bit heads at n=120.
- **DUST rung (next):** same script, `dd.cudd` engine (auto-selected when importable) + dynamic sifting; fallback engines if needed: pysdd (installs fine), exact #SAT (ganak/d4) on a Tseitin+sequential-counter CNF. Build CUDD in the container: `pip download dd --no-deps && tar xzf dd-*.tar.gz && cd dd-*/ && DD_FETCH=1 DD_CUDD=1 pip install .` then rerun the exact_gate command with a larger `--node-cap`.

Verdict update: feasibility gate **PASSED at rung 1** — exact O'Donnell quantities on a real trained LGN are practical (seconds, laptop) once the head is compiled sensibly. GO confidence rises ~70% → ~80%, contingent on the DUST/CUDD rung clearing h512-class models (and later MNIST-scale).

## Relation to roadmap
- Reuses: P3a netlist exporter + formal tooling (20_program_validation).
- Feeds: outreach (Emily Yu artifact), "why-random-wiring-works" theory sliver, P3a interpretability story.
- Does NOT displace P1/P2 — analysis paper, slow-payoff genre; PhD-clock priority unchanged (24_roadmap).
