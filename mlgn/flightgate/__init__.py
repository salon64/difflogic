"""
mlgn.flightgate — P3b gate D1: closed-loop POMDP distillation for recurrent LGN cells.
======================================================================================

The minimal thesis-gate flight harness (research/23 §D1): a PID teacher flying a
hover task is distilled — closed-loop, DAgger-style, with per-step action targets
(≡ deep supervision, the P2-proven method) — into seqlgn recurrent cells (``gated``
and ``clatch`` arms) and a feedforward LogicMLP control arm, on thermometer-encoded
observations, under an observation-dropout wrapper that makes memory REQUIRED.

Gate bar: memory cell ≳ teacher under occlusion where the feedforward student
degrades (and matches it in the no-blackout control — non-vacuity).

This ``__init__`` is deliberately EMPTY of imports: ``mock_env``/``encode``/``env``/
``teacher``/the collection half of ``trainer`` are numpy-only and must stay
importable inside the isolated sim venv (``.venv-flight/``, no torch). Torch enters
only via ``student.py`` (which injects the difflogic CPU stub through
``mlgn.seqlgn`` before importing ``difflogic``). See README.md.
"""
