"""
blif.py — emit a Netlist as BLIF for ABC (and later yosys).
===========================================================

BLIF because ABC ingests it natively with sequential elements and the 16-gate
vocabulary maps 1:1 onto ``.names`` covers (see ir.BLIF_COVER) — no AIG decomposition
needed on our side (``strash`` does it). Latches use the 3-token form
``.latch <input> <output> <init>``. Constant nodes: an empty cover is const-0, a
zero-input cover ``1`` is const-1. Gates whose argmax op is FALSE/TRUE are emitted as
constants (their fanins are irrelevant by definition).
"""

from __future__ import annotations

from .ir import BLIF_COVER, Netlist


def signal_name(net: Netlist, s: int) -> str:
    if s == 0:
        return "c0"
    if s == 1:
        return "c1"
    if s < 2 + net.n_pi:
        return f"x{s - 2}"
    if s < 2 + net.n_pi + net.n_state:
        return f"q{s - 2 - net.n_pi}"
    return f"g{s - 2 - net.n_pi - net.n_state}"


def emit_blif(net: Netlist, path: str, model: str = "lgn") -> None:
    nm = lambda s: signal_name(net, int(s))  # noqa: E731
    out = [f".model {model}"]
    out.append(".inputs " + " ".join(f"x{i}" for i in range(net.n_pi)))
    out.append(".outputs " + " ".join(f"o{i}" for i in range(len(net.outputs))))

    used = set()
    for g in range(net.n_gates):
        if int(net.ops[g]) not in (0, 15):          # constants ignore their fanins
            used.add(int(net.src_a[g]))
            used.add(int(net.src_b[g]))
    used.update(int(s) for s in net.next_state)
    used.update(int(s) for s in net.outputs)
    if 0 in used:
        out.append(".names c0")                      # empty cover = constant 0
    if 1 in used:
        out.append(".names c1")
        out.append("1")

    for i, ns in enumerate(net.next_state):
        out.append(f".latch {nm(ns)} q{i} {net.init[i]}")

    for g in range(net.n_gates):
        op = int(net.ops[g])
        cover = BLIF_COVER[op]
        if op in (0, 15):
            out.append(f".names g{g}")
        else:
            out.append(f".names {nm(net.src_a[g])} {nm(net.src_b[g])} g{g}")
        out.extend(cover)

    for i, s in enumerate(net.outputs):              # buffer each output signal
        out.append(f".names {nm(s)} o{i}")
        out.append("1 1")

    out.append(".end")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")
