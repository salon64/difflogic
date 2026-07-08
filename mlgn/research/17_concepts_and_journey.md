# 17 — Concepts & the P2 journey (what we actually built, and why)

A plain-language explainer written 2026-07-03 to fix a few mental-model bugs and capture the
architecture + the reframe before this gets lost. Corrects three common misreadings: (a) latches are
NOT in the 16-gate pool, (b) there is no separate "clock" component, (c) SR-latch instability is
sidestepped by construction (and is not why it failed).

## 1. The architecture has TWO levels — keep them separate

**Level 1 — the gate (LogicLayer).** Each neuron softly picks **one of the 16 two-input Boolean gates**
(AND, OR, XOR, NAND, NOR, …) over fixed random 2-input wiring; at inference it argmaxes to a single raw
gate. **This 16-gate pool is UNCHANGED in P2.** We never added a latch as a "17th gate" you pick from.

**Level 2 — the recurrent cell (the "mechanism").** The recurrent cell is *built from* LogicLayers
(Level 1). The `--mechanism` (rddlgn / gated / latch / combo / clatch) is the **fixed wiring around the
state feedback** — how the carried state `h` is combined with the input each step. **P1's gating and
P2's latch live HERE, at the cell level — a latch is a recurrence PATTERN, not a per-neuron gate.**

So P2 did **not** expand the 16-gate vocabulary. A "latch" is realized as: **16-gate LogicMLPs compute
the control signals** (a select/enable bit, or set/reset lines) **+ fixed arithmetic** (a MUX, the SR
characteristic equation, an optional round) combines them with the carried state. The learned parts are
still just 16-gate pickers; the latch is the fixed structure they plug into. (The original P2 plan
*talked about* adding D-FF/SR/T-FF to the "gate vocabulary" — that framing is aspirational; the
implementation makes them cell mechanisms.)

## 2. The mechanisms (all cell-level)
- **rddlgn** (control): recompute the whole state each step, no keep path. `h' = LogicMLP([x;h])`.
- **gated** (P1): a learned 2:1 **MUX per bit**, `h' = s·h + (1−s)·c` (`s`=gate net, `c`=candidate net).
  This IS the GRU update gate; `s·h` is the constant-error carousel. **lstm / gru_cell** = architectural
  variants (a separate cell state `C`, more gates) — ablations, not the story.
- **latch** (P2, original): an **SR or T flip-flop** cell — set/reset/hold via the characteristic
  equation, with the state **hard-rounded each step** ("bistable restore"). *Failed at scale (§5).*
- **combo**: gated write + the bistable restore on the hold. *Also failed.*
- **clatch** (P2, now — the headline): the **input-clocked latch** — a MUX whose write-ENABLE is
  hard-rounded but whose VALUE is held exactly (§5).

## 2b. HOW the cell is wired (dataflow) — parallel control nets → fixed combiner

The latch is NOT a serial block in a stack ("LGN → latch → LGN"). It is a **fixed per-bit combiner**
fed by **parallel** control LogicMLPs. For `clatch` (H = hidden_dim, D = input_dim):

```
 x_t ─┐
      ├─ z=[x_t; h] ──┬──> gate_net      (LogicMLP, 2 LogicLayers) ──> s ─> round ─> s_hard ─┐
 h ───┘ (prev, H)     └──> candidate_net (LogicMLP, 2 LogicLayers) ──> c ────────────────┐   │
   ▲                                                                                     ▼   ▼
   │       per-bit MUX (FIXED arithmetic):  h'_i = s_hard_i · h_i + (1 − s_hard_i) · c_i
   └──────────────────────────── h' (H bits) ── fed back next tick ─────────────────────────┘
```
- `z=[x_t; h]` (concat, dim D+H) feeds ALL control nets.
- Each control net is a small LGN stack (2 LogicLayers of H 16-gate neurons) → an H-dim output.
- The combiner acts **per bit**: bit i uses `s_i, c_i, h_i` → one 1-bit latch. So it is **H independent
  1-bit latches in parallel**, sharing the same control nets (bit i = neuron i of those nets).
- SR latch: same shape with `set_net`+`reset_net` and the SR characteristic equation as the combiner.
- Only `lstm`/`gru_cell` add a readout LogicMLP AFTER the state update (a real "latch → LGN" step);
  `gated`/`clatch`/`latch` have no post-latch LGN — the state IS the output (read by GroupSum at the end).

**Which latch / how to mix — NOT learned per-bit.** The latch TYPE is a GLOBAL choice (`--mechanism` /
`--latch-kind`): the whole cell is one type (all H bits are `clatch`, or all SR). There is NO softmax
over latch types (unlike the softmax over the 16 gates). What IS learned per-bit = the CONTROL LOGIC
(which 16-gate functions compute enable/set/reset/toggle + the candidate value). "Mixing" (`combo` =
gated + restore; `clatch` = MUX + hard enable) is a FIXED architectural composition YOU write in
`cells.py`, not a learned blend. (The "network softly PICKS its sequential primitive per neuron, like it
picks gates" idea = the original aspirational framing; NOT implemented — logged as **P2 future work**:
extend the per-neuron 16-gate softmax to also include stateful primitives {`clatch`/write-enabled register,
gated-D-latch, D-FF, SR, T-FF, plain-combinational} so each bit LEARNS which memory element it is, rather
than us fixing one cell type. Deferred (dilutes the P2 headline; hard-state primitives are fragile to
train per §5; relaxing over stateful-vs-combinational choices is an open design/stability problem). See
workmap §A0 "FUTURE WORK".)

> **FUTURE WORK (malcolm's framing, 2026-07-08): "Adding latches as TRAINABLE PARAMETERS instead of treating
> them architecturally."** Right now the memory type is a GLOBAL architectural choice (`--mechanism`: the whole
> cell is `gated`, or `clatch`, or `combo`…). The natural next step is a **per-bit LEARNED blend/selection**
> between the soft-gated MUX and the hard-clatch register (a true `gated⊕clatch` hybrid) — each neuron learns,
> via a relaxed selection like the 16-gate softmax, whether it wants gated's higher soft ceiling or clatch's
> clean/exact discretization. This is the concrete "best-of-both" question behind the psMNIST gated-vs-clatch
> tie (gated leads soft, clatch discretizes tighter). **This is a distinct post-P2 direction** (its own paper/
> section), NOT part of the Track-B P2 (see workmap §A0'), because a learned stateful-vs-combinational
> relaxation is an open design+stability problem and would dilute P2's now-locked deploy-a-verifiable-register
> story. The existing `combo` mechanism is NOT this — it is gated + a fixed hard-state restore.

## 3. "Clocked" — there is no separate clock we added
A **combinational** circuit is memoryless (`out = f(in)`). A **clocked sequential** circuit has
**registers (flip-flops)** that update once per **clock tick** and hold their value between ticks.
Our recurrent LGN, unrolled over the T timesteps, **IS a clocked sequential circuit**: **each timestep =
one clock tick**, the hidden state `h` = the register contents, the cell logic = the next-state function.
We didn't wire in a clock signal — **the recurrence itself is the clock.** "Input-**clocked** latch" just
means the write-enable is a hard per-tick decision (this tick: write, or hold).

## 4. SR-latch "instability" — sidestepped by construction (and NOT why it failed)
A *physical* cross-coupled NOR–NOR SR latch has a forbidden input (`S=R=1`) and can be **metastable**
(if you iterate the two NORs' feedback it may not settle to a clean bit). **We never iterate that loop.**
We use the closed-form **characteristic next-state equation** `Q⁺ = S + (1−R)·Q − S·(1−R)·Q` — a one-shot
function of `(S, R, Q)`, no feedback to settle → **no metastability**, and the forbidden `S=R=1` resolves
cleanly to set-wins (=1). (This "collapse the loop to the characteristic equation" is what the workmap
calls the C2 reduction.) So instability was *handled*; the SR latch failed for a completely different,
**optimization** reason (§5), which is why it's now demoted.

## 5. The reframe — the part that wasn't clear (the actual P2 story now)
**Goal:** an EXACT deployed circuit — no *discretization gap* (soft-trained accuracy ≫ the accuracy of
the hard/argmax circuit you actually ship).

1. **First idea (bistable restore):** force the state exactly `{0,1}` *during training* by hard-rounding
   it each step. Intuition: a "bistable" bit that re-cleans itself each step won't drift, so the trained
   and deployed circuits match → gap closes.
2. **It FAILED at long sequences (copy-50):** hard-rounding the state made **"never write / hold zero"**
   the easiest strategy — a partial/uncertain write gets round-flipped and corrupts the memory, which is
   *worse than* just holding, so the optimizer retreats to writing nothing → **chance accuracy**. We
   *saw* this via gate-usage tracking: the write nets collapsed to the constant **FALSE**/**TRUE** gates
   (the "never-write collapse").
3. **THE INSIGHT (verified in the difflogic source):** we never needed to round the state at all.
   **Plain `gated`, at inference, ALREADY produces an exactly-binary state** — the gates argmax to hard
   Boolean functions, and a MUX of binary values is binary, so from `h₀=0` the deployed state stays
   `{0,1}` every step and maps straight to FPGA flip-flops. **The exact-binary/register goal was already
   met — with no rounding.** The round only forced the *soft training* state binary (which deployment
   never needed) and **that** manufactured the collapse.
4. **So the real gap is small and specific:** during *training* the soft state drifts into `(0,1)`;
   the *argmax-eval* state doesn't. A train/eval mismatch = **Kim's "computation gap"** (a values
   problem, not a gate-choice problem).
5. **The fix (two ways, we're testing both):**
   - **(a) `clatch` — the headline.** Round the write-**ENABLE** (a hard write-or-hold decision each
     tick) but **hold the VALUE exactly**. Then held bits *never* drift (exact identity carry), and an
     uncertain write just stores a soft value that argmaxes cleanly at eval — **the value is never
     rounded, so there's no "worse-than-chance" trap, so no collapse.** This is precisely a **learnable
     write-enabled register** = what a real flip-flop is. Clean primitive, exact by construction.
   - **(b) drop the round, close the drift on plain `gated`** with a **margin loss** (`h·(1−h)`, pushes
     the soft state toward `{0,1}`) + **deep per-timestep supervision**. This works but adds a
     hyperparameter and yields no crisp primitive → it's the **foil** the paper compares against, not
     the headline (and standalone it's in ETH's saturated training-method lane).

## 6. Where the papers stand (see workmap §A0 for the full portfolio)
- **P1** = logic-native **gating** for long-range recall + the training recipe → NeurIPS'26 workshop (done).
- **P2** = **"Clock the enable, not the value"** → ICML'27: the 4-beat arc *never-write collapse →
  reframe (gated already deploys binary) → the input-clocked latch (`clatch`) → FPGA/verification payoff*.
  The margin-loss is the internal foil; **C2 (the gradient-obstruction theory) is demoted to a footnote**.
- **P3a** = clocked-sequential **verification** (the ISTA competitor); **P3b** = FPGA RTL emitter + the
  Kyushu POMDP nano-drone.
- **Open decision gate:** does `clatch` train + close the gap at copy-50 (the `cpB_clatch_*` queue runs)?
  YES → lock the primitive headline; STALL → obstruction-forward fallback.
