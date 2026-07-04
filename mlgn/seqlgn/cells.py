"""
cells.py — Recurrent cells for Logic Gate Networks.
====================================================

This module defines the core research object for Paper #1 (gating) and the
infrastructure for Paper #2 (latches, currently parked): a **recurrent cell built
entirely from differentiable logic gates**, with a *pluggable memory mechanism* so that
every variant differs in exactly one thing — how state is carried across time.

Memory mechanisms
-----------------
- ``"rddlgn"``  — the CONTROL. A faithful reimplementation of the recurrence used by
  Recurrent Deep Differentiable Logic Gate Networks (Bührer et al., arXiv:2508.06097)
  and by DiffLogic CA (Miotti et al., arXiv:2506.04912): plain *concat-recurrence*.
  The new state is RECOMPUTED from scratch each step:
      h_{t+1} = LogicMLP([x_t ; h_t])
  There is no explicit "keep" path — the network must relearn to copy state every step.

- ``"lstm"``    — Paper #1 (richer arm). A logic-native LSTM with a dedicated cell state
  ``C`` carried across time and independent forget / input / output stages:
      f  = LGN_forget(z);  i = LGN_input(z);  C̃ = LGN_candidate(z)
      C' = (C AND f) OR (i AND C̃)              # OR = logic stand-in for LSTM's '+'
      o  = LGN_outproj(z)                       # project z to hidden_dim
      h' = LGN_readout([o ; C'])                # 2H -> H
  The carousel lives in the CELL state: ``∂C'/∂C = f`` (≈1 on kept bits). ``h'`` is a
  readout. ~5 LGNs and TWO carried states (h, C) vs the gated cell's 2 LGNs / 1 state —
  use it as the "does the extra forget/input/output machinery earn its cost?" ablation.

- ``"gated"``   — Paper #1's contribution. A logic-native LSTM/GRU-style update where a
  learned **2:1 multiplexer per hidden bit** decides keep-vs-write:
      c_t = LogicMLP_candidate([x_t ; h_t])     # the candidate ("write") state
      s_t = LogicMLP_gate([x_t ; h_t])          # the select bit per unit (in [0,1])
      h_{t+1} = s_t * h_t + (1 - s_t) * c_t      # soft multiplexer  (== GRU update gate)
  At binary values this is *exactly* the boolean multiplexer
      MUX(s, h, c) = (s AND h) OR (NOT s AND c)
  (in hardware: 3 gates per bit). The "keep" branch s_t * h_t is a **constant-error
  carousel**: when the gate keeps a bit (s_t ≈ 1), dh_{t+1}/dh_t ≈ 1, so the gradient
  flows backward through time un-attenuated — the mechanism that lets LSTMs/GRUs beat
  vanilla RNNs, here realised in pure logic.

- ``"latch"``   — Paper #2's contribution. A **bistable SR-latch** memory primitive. Two
  learned logic networks drive the *set* (``S``) and *reset* (``R``) lines; the state is
  updated by the SR-latch **characteristic next-state equation** ``Q⁺ = S ∨ (R̄ ∧ Q)``,
  relaxed multilinearly to ``Q⁺ = S + (1−R)Q − S(1−R)Q``:
      S = LGN_set(z);  R = LGN_reset(z)
      Q' = S + (1 - R) * Q - S * (1 - R) * Q       # set / reset / hold
  ``∂Q'/∂Q = (1−R)(1−S)`` = **1 in hold** (S=R=0, the carousel) and **0 at set/reset**;
  ``∂Q'/∂S = 1−Q+RQ`` gives a clean gradient to *learn* to set even from the held state.
  Crucially this uses the *closed-form characteristic equation* rather than iterating the
  cross-coupled NOR–NOR feedback to a fixed point — the "reduction" that turns the
  fixed-point problem into a feedforward neuron and **sidesteps the memory-degeneracy /
  singular-``(I−J)`` obstruction** (Paper 2 §C2). Distinct from ``gated``: ``gated`` is a
  *soft multiply* hold (``s·h``, which bleeds the bit off {0,1} over time); the SR latch is
  a *bistable* set/reset that restores to a clean bit. (v0 = soft multilinear training
  forward; a hard NOR-settle forward + STE backward for exact-bit inference is v1.) The
  trivial D-flip-flop (``Q⁺=D``) degenerates to ``rddlgn`` and the gated D-latch
  (``Q⁺=e·D+(1−e)·Q``) to ``gated`` — so the SR latch is the genuinely new primitive. See
  ``docs/design.md`` §"Paper 2".

Notes on the difflogic API (see ``docs/api.md`` for the full list)
------------------------------------------------------------------
- ``LogicLayer(in_dim, out_dim)`` holds ``weights`` of shape ``[out_dim, 16]`` (the gate
  logits). In ``.train()`` it outputs a softmax mixture over the 16 gates in [0,1]; in
  ``.eval()`` it argmaxes to a single hard gate and outputs {0,1} (given binary inputs).
- Each ``LogicLayer`` requires a 2D input ``[batch, in_dim]``; we therefore process one
  timestep at a time (each ``x_t`` and ``h`` is ``[batch, dim]``).
- Connectivity constraint: ``out_dim * 2 >= in_dim``. For the first layer of a cell,
  ``in_dim = input_dim + hidden_dim`` and ``out_dim = hidden_dim`` ⇒ **hidden_dim must be
  >= input_dim**. We assert this with a helpful message.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflogic import LogicLayer


MECHANISMS = ("rddlgn", "gated", "lstm", "gru_cell", "latch", "combo", "clatch")
# Bistable primitives available under mechanism='latch' (see the latch docstring above).
LATCH_KINDS = ("sr", "tff")
# Mechanisms that carry a tuple state (h, C) — a separate cell state C from the output h.
_TUPLE_STATE = ("lstm", "gru_cell")


def _or(a, b):
    """Soft logic OR, OR(a,b) = a + b - a*b (difflogic gate id 7). Exact at {0,1};
    stays in [0,1] for a,b in [0,1]. Used as the Boolean stand-in for LSTM's additive
    cell-state update (you can't add bits: 1+1 would leave {0,1})."""
    return a + b - a * b


def _ste_round(x, alpha: float = 1.0):
    """Straight-through (annealed) round: forward = ``(1−α)·x + α·round(x)``, backward = identity.

    This is the **bistable restore** for the Paper-2 latch. Applied to the latch's next-state at
    ``α=1`` it re-binarises the carried bit every timestep so the recurrent input to the next step
    is exactly {0,1} (no soft-value drift over time) — the mechanism C3 claims closes the
    long-sequence *computation* gap. Because the (α-scaled) correction is ``.detach()``-ed,
    ``d/dx ≡ 1`` for **every** α, so any Jacobian factor multiplying ``x`` survives untouched: the
    latch carousel ``∂Q'/∂Q = (1−R)(1−S)`` (=1 in hold) is **preserved exactly**, and the backward
    equals v0's soft multilinear characteristic-equation Jacobian bit-for-bit. It performs NO
    within-step fixed-point iteration, so it does not reintroduce the §C2 obstruction. (Design
    locked 2026-07-02 v1 design panel; = the workmap §D characteristic-eq reduction as a short STE.)

    ``alpha`` in [0,1] **anneals** the restore (see ``utils.hard_anneal_alpha`` for the schedule):
    α=1 → forward=round(x) (fully hard, v1); α=0 → forward=x (soft, v0); 0<α<1 → partially
    restored. Annealing 0→1 over training lets the cell learn the soft solution first, then commit
    the state to {0,1}, fixing the cold-start/plateau fragility of hard-from-step-0 (exp log v1)."""
    if alpha >= 1.0:
        return x + (x.round() - x).detach()
    if alpha <= 0.0:
        return x
    return x + alpha * (x.round() - x).detach()


# Gate ids: 15 = TRUE (always 1), 0 = FALSE (always 0); see utils.GATE_NAMES / functional.
_TRUE_GATE = 15
_FALSE_GATE = 0


def bias_gate_closed(logic_mlp: "LogicMLP", strength: float) -> None:
    """Bias a gate toward outputting 0 ("closed" / don't-write) at init by adding
    ``strength`` to the FALSE-gate logit of its final LogicLayer.

    Used for the **LSTM input gate**: paired with a keep-biased forget gate, this is the
    standard "remember + don't-overwrite" LSTM initialisation. Without it, a random input
    gate makes the cell-state carousel ``∂C'/∂C = f·(1 − i·C̃) ≈ 0.58`` at init (the input
    path eats the keep) → vanishing → cold-start. Closing the input gate gives
    ``i·C̃ ≈ 0`` so ``∂C'/∂C ≈ f`` (strong), while leaving a write path (i>0) the gate can
    learn to open. ``strength=0`` disables it."""
    if strength:
        with torch.no_grad():
            logic_mlp.net[-1].weights[:, _FALSE_GATE] += strength


def bias_gate_keep(logic_mlp: "LogicMLP", strength: float) -> None:
    """Bias a gate network toward outputting 1 ("keep") at initialisation by adding
    ``strength`` to the TRUE-gate logit of its FINAL LogicLayer.

    This is the logic-native analog of the **LSTM forget-gate bias** (Gers et al. 2000)
    and difflogic's **residual initialisation** (Petersen et al. 2024): it switches the
    constant-error carousel ON at init so state — and the gradient through it — persist
    across long sequences. Without it, an unbiased gate (s≈0.5) lets state decay before the
    network can *learn* to keep it (the cold-start we observed: copy-50 stuck at chance with
    a flat loss). A moderate ``strength`` leaves a write path (soft s<1) so the cell can
    still learn when to overwrite. ``strength=0`` disables it (reproduces the cold-start).
    """
    if strength:
        with torch.no_grad():
            logic_mlp.net[-1].weights[:, _TRUE_GATE] += strength


class LogicMLP(nn.Module):
    """A stack of ``num_layers`` LogicLayers mapping ``in_dim -> out_dim``.

    Layer 1 maps ``in_dim -> out_dim``; any further layers map ``out_dim -> out_dim``.
    This is the basic feed-forward logic block we reuse for the candidate/gate/update
    networks inside a cell.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 2,
        device: str = "cuda",
        grad_factor: float = 1.0,
        implementation: str | None = None,
        connections: str = "random",
    ):
        super().__init__()
        assert num_layers >= 1, num_layers
        # difflogic connectivity constraint for the first (widest-input) layer.
        assert out_dim * 2 >= in_dim, (
            f"LogicLayer needs out_dim*2 >= in_dim, got out_dim={out_dim}, in_dim={in_dim}. "
            f"For a recurrent cell this means hidden_dim must be >= input_dim."
        )
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(
                LogicLayer(
                    in_dim=d,
                    out_dim=out_dim,
                    device=device,
                    grad_factor=grad_factor,
                    implementation=implementation,
                    connections=connections,
                )
            )
            d = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LogicRecurrentCell(nn.Module):
    """One recurrent step built from logic gates, with a pluggable memory mechanism.

    forward(x_t, state) -> new_state      (all tensors are [batch, dim])

    State convention (so one model loop drives every mechanism):
      - ``rddlgn`` / ``gated``: state is a single tensor ``h``  [batch, hidden_dim].
      - ``lstm``: state is a tuple ``(h, C)`` — hidden state and cell state.
    Use the helpers ``init_state`` / ``readout_h`` / ``carousel_state`` rather than
    poking at the state shape directly.

    The only thing that changes between the scientific conditions is ``mechanism``; the
    capacity (number/size of logic layers) is kept comparable so the comparison isolates
    the memory mechanism (see ``docs/experiments.md``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mechanism: str = "gated",
        cell_layers: int = 2,
        keep_bias: float = 3.0,
        latch_kind: str = "sr",
        hard_state: bool = True,
        hard_control: bool = False,
        device: str = "cuda",
        grad_factor: float = 1.0,
        implementation: str | None = None,
        connections: str = "random",
    ):
        super().__init__()
        assert mechanism in MECHANISMS, f"mechanism must be one of {MECHANISMS}, got {mechanism!r}"
        assert latch_kind in LATCH_KINDS, f"latch_kind must be one of {LATCH_KINDS}, got {latch_kind!r}"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mechanism = mechanism
        self.cell_layers = cell_layers
        self.keep_bias = keep_bias
        self.latch_kind = latch_kind
        # v1 (Paper 2): hard_state=True re-binarises the latch state each step via a
        # straight-through round (the bistable restore that closes the long-sequence
        # discretization gap; see _ste_round). hard_state=False reproduces the v0 soft latch
        # (the ablation). hard_control (round the S/R/T lines too) is the fully-hard workmap
        # variant — OFF by default: at eval the gates are already argmax-binary so it buys no
        # eval correctness, and it adds STE bias + a decision-boundary plateau during training.
        self.hard_state = hard_state
        self.hard_control = hard_control
        # Bistable-restore anneal coefficient in [0,1], read live by the latch forward. Default
        # 1.0 = fully hard (un-annealed v1). To anneal, the training loop sets this per-epoch from
        # utils.hard_anneal_alpha(epoch/total) so the state hardens 0->1 gradually (fixes the
        # hard-from-step-0 cold-start/plateau; see experiment log 2026-07-02 v1). Ignored unless
        # hard_state=True. A plain attribute (not a Parameter/buffer) — set it like cell.hard_alpha=a.
        self.hard_alpha = 1.0

        cat_dim = input_dim + hidden_dim
        mlp_kwargs = dict(
            num_layers=cell_layers,
            device=device,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
        )

        if mechanism == "rddlgn":
            # CONTROL: recompute the full state each step from [x_t ; h_t].
            self.update = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)

        elif mechanism in ("gated", "combo", "clatch"):
            # Paper #1 ('gated'): separate candidate and gate networks (mirrors GRU's separate
            # weight matrices for the candidate and the update gate). 'combo' (Paper #1 + #2) is
            # the SAME cell but applies a bistable restore to the hold (see forward + hard_state).
            # 'clatch' (INPUT-CLOCKED LATCH) rounds the write-ENABLE not the value = a learnable
            # write-enabled clocked register: hold is EXACT identity (no drift), the written value
            # is never rounded (no worse-than-chance "moat"), so it is exact-at-deploy AND trainable
            # (the never-write collapse of hard-rounding the state is avoided). See forward.
            self.candidate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.gate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            # Keep-bias the update gate s so the carousel is ON at init (avoids cold-start).
            bias_gate_keep(self.gate, keep_bias)

        elif mechanism == "lstm":
            # Paper #1 (richer arm): independent forget / input / candidate stages over z,
            # a dedicated cell state C with an OR-combine (logic stand-in for LSTM's '+'),
            # and an output path that projects z to hidden_dim then reads out from [o ; C].
            self.forget = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.input = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.candidate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.out_proj = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            # Standard LSTM init: keep-bias the forget gate (remember) AND close the input
            # gate (don't-overwrite), so the cell-state carousel ∂C'/∂C ≈ f is strong at
            # init instead of being eaten by a random input path (which cold-starts it).
            bias_gate_keep(self.forget, keep_bias)
            bias_gate_closed(self.input, keep_bias)
            # readout sees [o ; C] (2*hidden_dim) -> hidden_dim; 2H >= 2H satisfies the
            # difflogic out*2>=in rule exactly (this is why we project z->o first instead
            # of feeding [z ; C] directly, which would violate it).
            self.readout = LogicMLP(2 * hidden_dim, hidden_dim, **mlp_kwargs)

        elif mechanism == "gru_cell":
            # 2x2 ablation: a dedicated cell state C (like LSTM) updated by the GRU's single
            # complementary MUX gate (like `gated`), with a separate output readout. Tests
            # whether decoupling memory (C) from output (h) helps, with the robust gate.
            self.candidate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.gate = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.out_proj = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
            self.readout = LogicMLP(2 * hidden_dim, hidden_dim, **mlp_kwargs)
            bias_gate_keep(self.gate, keep_bias)  # MUX carousel on C: ∂C'/∂C = s, no leak

        elif mechanism == "latch":
            # Paper #2: a bistable memory primitive. v0 uses a soft multilinear
            # characteristic-equation forward — fully differentiable, so autograd handles the
            # backward pass (the "reduction" that collapses the cross-coupled fixed point into
            # a feedforward neuron and sidesteps the §C2 obstruction).
            if latch_kind == "sr":
                # SR latch: learned set (S) and reset (R) lines drive Q⁺ = S ∨ (R̄ ∧ Q).
                self.set_net = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
                self.reset_net = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
                # keep-bias analog: bias BOTH lines toward 0 ("hold") so the carousel
                # ∂Q'/∂Q = (1−R)(1−S) ≈ 1 at init, while ∂Q'/∂S = 1−Q still gives a clean
                # gradient to LEARN to set. keep_bias=0 reproduces the unbiased cold-start.
                bias_gate_closed(self.set_net, keep_bias)
                bias_gate_closed(self.reset_net, keep_bias)
            elif latch_kind == "tff":
                # T flip-flop: a learned toggle line T drives Q⁺ = T ⊕ Q. One toggle bit
                # computes a running XOR = parity — the M1 demonstrator ("one primitive solves
                # what rddlgn can't"). ∂Q'/∂Q = 1−2T = 1 in hold (T=0, carousel), −1 on toggle
                # (|grad|=1, still flows un-attenuated).
                self.toggle_net = LogicMLP(cat_dim, hidden_dim, **mlp_kwargs)
                # keep-bias analog: bias T toward 0 ("hold / don't toggle") so the carousel is
                # on at init; ∂Q'/∂T = 1−2Q still gives a clean gradient to LEARN to toggle.
                bias_gate_closed(self.toggle_net, keep_bias)

    def forward(self, x_t: torch.Tensor, state):
        # Unpack the (possibly tuple) state into the hidden vector h used to form z.
        h = state[0] if self.mechanism in _TUPLE_STATE else state
        z = torch.cat([x_t, h], dim=-1)  # [batch, input_dim + hidden_dim]

        if self.mechanism == "rddlgn":
            return self.update(z)

        if self.mechanism in ("gated", "combo", "clatch"):
            c = self.candidate(z)              # candidate / "write" value, in [0,1]
            s = self.gate(z)                   # select / write-enable bit per unit, in [0,1]
            if self.mechanism == "clatch":
                # INPUT-CLOCKED LATCH: round the write-ENABLE (not the value). hold (s->1) => Q=h
                # EXACTLY (no drift, unit Jacobian); write (s->0) => Q=c (value never rounded, so an
                # uncertain write just stores a soft value that argmaxes cleanly at eval — no moat,
                # so no never-write collapse). The gap reduces to a single-step gate-SELECTION gap.
                # Annealed by hard_alpha (alpha=0 => this IS gated; alpha=1 => fully hard enable).
                s = _ste_round(s, self.hard_alpha)
            # Soft 2:1 multiplexer == GRU update gate. At binary {0,1} values this is the
            # exact boolean MUX (s AND h) OR (NOT s AND c). The s*h branch is the
            # constant-error carousel: gradient ~1 through kept bits.
            Q = s * h + (1.0 - s) * c
            # combo (Paper #1 write-path + Paper #2 bistable hold): the MUX writes, then the
            # bistable restore re-binarises the held bit each step (annealed by hard_alpha) so the
            # gated hold stops drifting off {0,1} over time — closing the discretization gap that
            # `gated` alone leaves at long sequences (the soft s*h bleeds the bit; the round cleans
            # it). `gated` returns the soft MUX unchanged.
            if self.mechanism == "combo" and self.hard_state:
                return _ste_round(Q, self.hard_alpha)
            return Q

        if self.mechanism == "lstm":
            _, C = state
            f = self.forget(z)                 # forget gate (keep-mask), in [0,1]
            i = self.input(z)                  # input gate (write-enable), in [0,1]
            c_tilde = self.candidate(z)        # candidate cell value, in [0,1]
            # C' = (C AND f) OR (i AND C̃): the OR is the logic stand-in for LSTM's
            # additive C_t = f⊙C_{t-1} + i⊙C̃. ∂C'/∂C = f (≈1 on kept, non-overwritten
            # bits) -> the cell-state carousel.
            C_new = _or(C * f, i * c_tilde)
            o = self.out_proj(z)               # project z to hidden_dim (output-gate-like)
            h_new = self.readout(torch.cat([o, C_new], dim=-1))  # [o ; C'] (2H) -> H
            return (h_new, C_new)

        if self.mechanism == "gru_cell":
            _, C = state
            s = self.gate(z)                   # single complementary gate (keep-biased)
            c_tilde = self.candidate(z)
            C_new = s * C + (1.0 - s) * c_tilde  # GRU MUX on the cell state; ∂C'/∂C = s
            o = self.out_proj(z)
            h_new = self.readout(torch.cat([o, C_new], dim=-1))
            return (h_new, C_new)

        if self.mechanism == "latch":
            if self.latch_kind == "sr":
                S = self.set_net(z)            # set line, in [0,1]
                R = self.reset_net(z)          # reset line, in [0,1]
                if self.hard_control:          # fully-hard workmap variant (off by default)
                    S, R = _ste_round(S, self.hard_alpha), _ste_round(R, self.hard_alpha)
                # SR-latch characteristic next-state equation Q⁺ = S ∨ (R̄ ∧ Q), relaxed
                # multilinearly: Q⁺ = S + (1−R)Q − S(1−R)Q. Exact at {0,1}: set (S=1)→1,
                # reset (R=1,S=0)→0, hold (S=R=0)→Q. ∂Q⁺/∂Q = (1−R)(1−S) = 1 in hold (the
                # bistable carousel), 0 at set/reset. Closed-form ⇒ no fixed-point iteration,
                # so autograd differentiates it directly (the §C2-sidestepping reduction).
                Rbar_Q = (1.0 - R) * h
                Q = S + Rbar_Q - S * Rbar_Q
                # v1 bistable restore (annealed by self.hard_alpha): re-binarise the held bit each
                # step (identity STE) so the state stops drifting off {0,1} across time — closes the
                # long-sequence gap (C3). alpha ramps 0->1 over training (utils.hard_anneal_alpha).
                return _ste_round(Q, self.hard_alpha) if self.hard_state else Q
            # tff: T flip-flop Q⁺ = T ⊕ Q, relaxed multilinearly XOR(T,Q) = T + Q − 2TQ.
            # toggle (T=1) → 1−Q, hold (T=0) → Q. ∂Q⁺/∂Q = 1−2T (carousel at T=0).
            T = self.toggle_net(z)             # toggle line, in [0,1]
            if self.hard_control:
                T = _ste_round(T, self.hard_alpha)
            Q = T + h - 2.0 * T * h
            return _ste_round(Q, self.hard_alpha) if self.hard_state else Q

        raise RuntimeError(f"unhandled mechanism {self.mechanism!r}")

    # --- state helpers (keep the model loop mechanism-agnostic) -----------------------
    def init_state(self, batch: int, device, dtype=torch.float32):
        h = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        if self.mechanism in _TUPLE_STATE:
            C = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
            return (h, C)
        return h

    def readout_h(self, state) -> torch.Tensor:
        """The hidden vector used for classification (and fed back next step)."""
        return state[0] if self.mechanism in _TUPLE_STATE else state

    def carousel_state(self, state) -> torch.Tensor:
        """The state carrying the long-range gradient highway: the cell state C for
        tuple-state mechanisms (lstm/gru_cell), the hidden state h otherwise. Used by the
        grad-norm-through-time analysis."""
        return state[1] if self.mechanism in _TUPLE_STATE else state

    def extra_repr(self) -> str:
        kb = f", keep_bias={self.keep_bias}" if self.mechanism in ("gated", "lstm", "gru_cell", "latch", "combo", "clatch") else ""
        if self.mechanism == "latch":
            lk = (f", latch_kind={self.latch_kind!r}, hard_state={self.hard_state}, "
                  f"hard_control={self.hard_control}")
        elif self.mechanism == "combo":
            lk = f", hard_state={self.hard_state}"
        else:
            lk = ""
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"mechanism={self.mechanism!r}, cell_layers={self.cell_layers}{kb}{lk}"
        )
