"""
sim.py — pure-numpy simulator for the extracted netlist + equivalence checks.
=============================================================================

Two jobs:

1. ``equivalence_check`` — the netlist must match the torch model BIT-FOR-BIT at eval
   semantics: identical state trajectory at every timestep and identical predictions,
   over real task data. Together with extract.check_accuracy this closes the loop:
   checkpoint → rebuilt torch model → netlist all agree, so whatever ABC proves about
   the netlist is a statement about the deployed model.

2. ``analyze_protocol`` — deterministic trajectory analysis for the copy protocol.
   After the cue step the input is all-blank, so the state trajectory per symbol is
   deterministic; we report when (if ever) it reaches a fixed point and whether the
   GroupSum argmax decodes to the right symbol along the way. This tells us *in
   advance* which hold property can possibly be proved (e.g. settle depth > 1 ⇒ the
   strict step-1 hold invariant is false and pdr will return a counterexample).
"""

from __future__ import annotations

import numpy as np
import torch

from .ir import GATE_FN, Netlist


def step(net: Netlist, x: np.ndarray, state: np.ndarray, sig: np.ndarray | None = None) -> np.ndarray:
    """One clock tick. x: [B, n_pi] bool, state: [B, n_state] bool -> new state."""
    B = x.shape[0]
    if sig is None:
        sig = np.empty((B, net.n_signals), dtype=bool)
    sig[:, 0] = False
    sig[:, 1] = True
    sig[:, 2:2 + net.n_pi] = x
    sig[:, 2 + net.n_pi:2 + net.n_pi + net.n_state] = state
    base = 2 + net.n_pi + net.n_state
    for (s, e) in net.layers:
        a = sig[:, net.src_a[s:e]]
        b = sig[:, net.src_b[s:e]]
        out = np.empty_like(a)
        ops = net.ops[s:e]
        for code in np.unique(ops):
            m = ops == code
            out[:, m] = GATE_FN[int(code)](a[:, m], b[:, m])
        sig[:, base + s:base + e] = out
    return sig[:, net.next_state].copy()


def run_sequence(net: Netlist, x_seq: np.ndarray, return_trajectory: bool = False):
    """x_seq: [B, T, n_pi] bool. Returns final state [B, n_state] (and all states if asked)."""
    B, T, _ = x_seq.shape
    state = np.tile(np.asarray(net.init, dtype=bool), (B, 1))
    sig = np.empty((B, net.n_signals), dtype=bool)
    states = []
    for t in range(T):
        state = step(net, x_seq[:, t], state, sig)
        if return_trajectory:
            states.append(state)
    return (state, states) if return_trajectory else state


def head_scores(net: Netlist, state: np.ndarray) -> np.ndarray:
    """GroupSum class scores: per-group popcount over the state bits. [B, k] int."""
    k, gs = net.head
    return state.reshape(state.shape[0], k, gs).sum(-1)


def predict(net: Netlist, x_seq: np.ndarray) -> np.ndarray:
    return head_scores(net, run_sequence(net, x_seq)).argmax(-1)


# ---------------------------------------------------------------------------------
# Equivalence vs the torch model
# ---------------------------------------------------------------------------------
@torch.no_grad()
def _torch_trajectory(model, x: torch.Tensor):
    """Replicate SequenceClassifier.forward at eval, keeping every state. x binary."""
    model.eval()
    cell = model.cell
    state = cell.init_state(x.shape[0], device=x.device, dtype=x.dtype)
    states = []
    for t in range(x.shape[1]):
        state = cell(x[:, t, :], state)
        states.append(cell.readout_h(state))
    logits = model.head(cell.readout_h(state))
    return logits, states


@torch.no_grad()
def equivalence_check(model, net: Netlist, loader, max_batches: int | None = None,
                      trajectory_batches: int = 2) -> dict:
    """Netlist vs torch model on real data: identical predictions on every batch, and
    identical per-timestep state bits on the first ``trajectory_batches`` batches."""
    n = mismatched_preds = 0
    traj_bits_checked = traj_bits_equal = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xb = x.round()
        logits, states = _torch_trajectory(model, xb)
        torch_preds = logits.argmax(-1).numpy()

        x_np = xb.numpy().astype(bool)
        if i < trajectory_batches:
            final, sim_states = run_sequence(net, x_np, return_trajectory=True)
            for st_t, ss_t in zip(states, sim_states):
                tb = st_t.numpy().astype(bool)
                traj_bits_checked += tb.size
                traj_bits_equal += int((tb == ss_t).sum())
        else:
            final = run_sequence(net, x_np)
        sim_preds = head_scores(net, final).argmax(-1)
        mismatched_preds += int((sim_preds != torch_preds).sum())
        n += len(torch_preds)
    return {
        "samples": n,
        "mismatched_predictions": mismatched_preds,
        "trajectory_bits_checked": traj_bits_checked,
        "trajectory_bits_mismatched": traj_bits_checked - traj_bits_equal,
        "bit_exact": mismatched_preds == 0 and traj_bits_checked == traj_bits_equal,
    }


def exhaustive_x0(net: Netlist, alphabet: int, horizon: int = 256) -> dict:
    """Characterize the closed post-write system for EVERY possible first input.

    After t=0 the protocol forces blank inputs, so the system is deterministic and
    closed: the entire reachable set is the union of 2^n_pi trajectories. Batched
    simulation of all of them is therefore a COMPLETE case analysis — an exhaustive
    proof for whatever it establishes (settle bounds, decode correctness), with no
    model checker involved. Reports settle-time distribution split into legal
    protocol writes (cue=1, one-hot symbol) vs the rest (garbage), plus any
    trajectories that fail to reach a fixed point within the horizon (checked for
    limit cycles)."""
    n = 1 << net.n_pi
    x0 = ((np.arange(n)[:, None] >> np.arange(net.n_pi)[None, :]) & 1).astype(bool)
    sym_bits = x0[:, :alphabet]
    legal = x0[:, alphabet] & (sym_bits.sum(1) == 1)
    if net.n_pi > alphabet + 1:
        legal &= ~x0[:, alphabet + 1:].any(1)

    state = np.tile(np.asarray(net.init, dtype=bool), (n, 1))
    state = step(net, x0, state)                      # the write step
    blank = np.zeros((n, net.n_pi), dtype=bool)
    settle = np.full(n, -1, dtype=np.int64)
    seen_hashes: list[dict] = [dict() for _ in range(n)]  # state-hash -> t (cycle detection)
    unsettled = np.ones(n, dtype=bool)
    cycles = {}
    for t in range(1, horizon):
        nxt = step(net, blank, state)
        just = unsettled & (nxt == state).all(1)
        settle[just] = t
        unsettled &= ~just
        if not unsettled.any():
            state = nxt
            break
        for i in np.nonzero(unsettled)[0]:
            hsh = hash(state[i].tobytes())
            prev = seen_hashes[i].get(hsh)
            if prev is not None and i not in cycles:
                cycles[i] = (prev, t)                 # revisited: limit cycle of length t-prev
            seen_hashes[i][hsh] = t
        state = nxt
    decode = head_scores(net, state).argmax(-1)
    sym_of = sym_bits.argmax(1)

    def _dist(mask):
        s = settle[mask]
        return {"n": int(mask.sum()), "settled": int((s >= 0).sum()),
                "max_settle": int(s.max()) if (s >= 0).any() else None,
                "unsettled": int((s < 0).sum())}

    legal_ok = legal & (settle >= 0) & (decode == sym_of)
    return {
        "legal": _dist(legal),
        "legal_decode_correct": int(legal_ok.sum()),
        "garbage": _dist(~legal),
        "limit_cycles": sorted({t1 - t0 for (t0, t1) in cycles.values()}),
        "n_cycling": len({i for i in cycles if settle[i] < 0}),
        "settle_hist_legal": np.bincount(settle[legal][settle[legal] >= 0]).tolist(),
        "settle_hist_garbage": np.bincount(settle[~legal][settle[~legal] >= 0]).tolist(),
    }


# ---------------------------------------------------------------------------------
# Copy-protocol trajectory analysis
# ---------------------------------------------------------------------------------
def analyze_protocol(net: Netlist, alphabet: int, horizon: int = 256) -> list[dict]:
    """For each symbol: cue+one-hot at t=0, then blanks. Deterministic trajectory ⇒
    find the settle time (first t with state[t+1] == state[t]; a fixed point is
    permanent because the input is constant-blank from t=1 on) and whether the decoded
    class is correct at settle and at every step from settle to the horizon."""
    results = []
    for sym in range(alphabet):
        x0 = np.zeros((1, net.n_pi), dtype=bool)
        x0[0, sym] = True
        x0[0, alphabet] = True                      # cue bit
        blank = np.zeros((1, net.n_pi), dtype=bool)
        state = np.tile(np.asarray(net.init, dtype=bool), (1, 1))
        state = step(net, x0, state)                # t=0: the write step -> state h_1
        settle_t, decode_at = None, []
        for t in range(1, horizon):
            nxt = step(net, blank, state)
            decode_at.append(int(head_scores(net, state).argmax(-1)[0]))
            if (nxt == state).all():
                settle_t = t                        # h_{t+1} == h_t: fixed point reached
                break
            state = nxt
        final_decode = int(head_scores(net, state).argmax(-1)[0])
        results.append({
            "symbol": sym,
            "settled": settle_t is not None,
            "settle_step": settle_t,                # steps AFTER the write step; 1 = immediate
            "decode_correct_at_settle": final_decode == sym,
            "decode_trajectory_head": decode_at[:8],
        })
    return results
