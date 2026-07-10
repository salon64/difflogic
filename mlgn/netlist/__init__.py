"""
netlist — checkpoint → gate-level netlist → formal verification (and, later, RTL).
==================================================================================

The shared artifact identified by the 2026-07-08 program validation (research/20 §E.4):
one exporter that simultaneously (a) provides P2's compact clocked-verification demo,
(b) de-risks P3a (sequential verification of trained LGNs), (c) re-arms P3b's
hardware timestamp, and (d) backs the Kyushu pitch.

Pipeline:  extract.rebuild_model (RNG-replay wiring reconstruction, accuracy-gated)
        →  ir.extract_netlist   (eval-time FSM as an explicit gate netlist)
        →  sim                  (numpy simulator; bit-exact equivalence vs torch)
        →  props + blif         (property miters, BLIF emission)
        →  falsify              (ABC sat / pdr / bmc3, verdict report)

See README.md for the falsifier design and current verdicts.
"""
