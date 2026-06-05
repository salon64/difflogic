# Venues & Timeline

_Compiled 2026-06-04. Deadlines verified by web search where possible; ones marked (est.)
are predicted from prior years — RE-VERIFY on the official site before relying on them._

## Honest tiering of the two contributions
- **#1 gating alone = workshop-strength**, not main-track A*. It adds a known mechanism
  (LSTM/GRU gating) to a known architecture (recurrent LGN) and was **publicly suggested
  by Google (DiffLogic CA future work)** → reviewers will read it as the obvious next
  step. Great motivation, double-edged novelty.
- **#2 latch primitives = genuine A\* main-track candidate** (new primitive, new
  capability, hardware story, crisp held-vs-recomputed-state question).

## Target plan
- **#1 → NeurIPS 2026 Workshop** (fast, A*-venue, non-archival flag-plant).
- **#2 → ICLR 2027 main track** (take time, do the 4-way comparison right).
- **Highest-EV A\* arc:** workshop #1 now → fold #1+#2 into ONE strong main-track paper
  ("Sequential/Stateful Logic Gate Networks") for ICLR 2027; the workshop becomes the
  early-results citation.

## Deadline table (AoE)
| Venue | Tier | Submission | Conf. | For | Status |
|---|---|---|---|---|---|
| **NeurIPS 2026 Workshops** | A* (workshop) | **~Aug 29, 2026** (per-workshop, late Aug–Sep; mandatory notif Sep 29) | Dec 6–13 2026, Sydney | **#1 primary** | confirmed window |
| **AAAI 2027** | **A\*** | abstract **Jul 21**, paper **Jul 28, 2026** | Feb 16–23 2027, Montréal | #1 stretch (needs more substance) | confirmed |
| **ICLR 2027** | **A\*** | ~late Sep/early Oct 2026 (est.; '25=Sep 24, '24=Oct 1) | Apr 24–28 2027, Sydney | **#2 primary** | CFP not yet out |
| **MLSys 2027** | A (top systems) | ~Oct–Nov 2026 (est.; '26 was Oct 30 2025) | 2027 | #2 (hardware framing) | est. |
| **NeurIPS 2027** | **A\*** | ~May 2027 (est.) | Dec 2027 | #2 (more runway / +FPGA) | est. |
| **DATE 2027** | A/B (EDA) | ~Sep 2026 (est.) | Apr 2027 | #2/#3 FPGA (later) | est. |
| **DAC 2027** | A* (EDA) | ~Nov 2026 (est.) | Jun 2027 | #2/#3 FPGA (later) | est. |
| ~~NeurIPS 2026 main~~ | — | closed May 6 2026 | — | — | **passed** |
| ~~ICML 2026 main~~ | — | closed ~Jan 2026 | Jul 2026 | — | **passed** |

## NeurIPS 2026 Workshop — official dates (AoE) + the trap
| Milestone | Date | Who it's for |
|---|---|---|
| Workshop Application Open | Apr 20 '26 | organizers |
| **Workshop Application Deadline** | **Jun 06 '26** | **organizers ONLY — NOT paper authors. Ignore.** |
| Workshop Acceptance Notifications | Jul 11 '26 | (workshop list becomes public) |
| **Suggested Submission Date for Contributions** | **Aug 29 '26** | **← MY deadline (paper author)** |
| Mandatory Accept/Reject Notification | Sep 29 '26 | hard backstop |

⚠️ The Jun 06 "2-days" countdown is the *organizer* application deadline (to host a
workshop), not paper submission. As an author my date is **Aug 29** (~12 wks). Each
accepted workshop sets its own exact deadline near Aug 29; can't pick the target workshop
until the list drops **Jul 11**.

Runway: now→Jul 11 build infra + run gating experiment (no workshop dependency) → Jul 11
pick workshop → ~Aug 29 submit.

## Notes for #1 workshop sprint (NeurIPS 2026 W, ~Aug 29)
- Workshop list is announced after **Jul 11, 2026** (acceptance notif). Target one of:
  efficient ML / sparsity in NNs / ML-for-systems / on-device & edge / tiny ML.
  Precedent: RDDLGN went to an Edge & Mobile Foundation Models workshop (MobiCom'25).
- Non-archival = you can publish the extended version at a main conf later. Good.
- **Framing to use:** "DiffLogic CA (Google) explicitly proposed learnable gating /
  state-forgetting as future work; we deliver and evaluate logic-native gating, and show
  where it matters (long-range recall)." Cite RDDLGN as the recurrence baseline.
- **Speed caveat:** the same Google line that motivates this also invites a race — ship
  the workshop fast.
- Min viable result: gated cell vs faithful RDDLGN-design baseline on sMNIST/psMNIST +
  copy/adding, with gradient-norm-through-time showing the carousel path helps.

## If chasing AAAI 2027 instead (Jul 28, ~7 wks — only if results are strong)
Needs main-track substance: a long-range benchmark where vanilla recurrence *fails* +
the constant-error-carousel gradient analysis (lightweight theory). Tighter timeline;
default to the NeurIPS workshop unless early results are unusually strong.

## Action checkpoints
- [ ] By ~Jul 11: scan the NeurIPS 2026 workshop list; pick target workshop(s).
- [ ] Decide #1 home: NeurIPS workshop (default) vs AAAI 2027 main (if results strong).
- [ ] By ~Sep 2026: confirm ICLR 2027 official deadline for #2.
- [ ] Build shared infra now (recurrent cell + sequential benchmark) — both papers need it.
