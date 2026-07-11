"""
run_distractor_study.py — the free-input verification campaign (agent CAMPAIGN).
================================================================================

The experiment that decides whether model checking earns its keep beyond
exhaustive simulation on the copy-task circuits: distractor robustness.

Per checkpoint (combo = ckpt_cp50A_curr_c35, gated = ckpt_cpB_gated_oracle):

1. Rebuild + extract; quick replay gate (extract.check_accuracy, 8 batches) —
   must match the recorded accuracy (full-set gates already passed).
2. BFS pre-analysis (pure numpy — the sim-side ground truth): for each of the 8
   legal writes, simulate write + blanks to the fixed point, then compute the
   CLOSURE of that fixed point under the 9 allowed inputs
   {blank} ∪ {8 non-cued one-hot symbols} (BFS over distinct states, cap
   200 000 per symbol, packbits-hash dedupe). Records closure size, whether
   every visited state GroupSum-decodes to the written symbol, and cap escapes.
   NOTE the BFS envelope is EXACTLY the reachable armed set of the
   from_start=False distractor properties (blanks are forced through the settle
   window, so the armed frontier is the fixed point); for from_start=True it is
   a reachable SUBSET (blank distractors during settling are allowed), so BFS
   witnesses refute from_start=True proofs but BFS completeness does not
   certify them.
3. Build + emit + model-check 5 properties: protocol_decode(settle),
   distractor_hold / distractor_decode x from_start ∈ {False, True}.
   Engines per property: (i) the proving recipe
       read_blif F; strash; tempor -F <settle+2>; scorr; dc2; pdr -T 600
   and (ii) bmc3 -T 180 as an independent counterexample hunter (its cex frame
   is ground truth; write_cex dumps the PI trace). If the recipe errors out on
   a counterexample through tempor, plain pdr -T 600 is the fallback.
4. Cross-validate MC verdicts against the BFS ground truth. Any PROVED claim
   contradicted by a BFS witness (or a from_start=False CEX contradicted by a
   COMPLETE clean closure) is a soundness alarm: the run stops with exit 4 and
   the contradiction in the report, never papered over. Additionally every
   ABC counterexample is REPLAYED through the numpy simulator on the property
   netlist (write_cex dump -> PI frames -> eval_netlist): a cex that does not
   assert bad at its claimed frame is itself a contradiction.

Outputs: mlgn/netlist/out/distractor_study/<circuit>/ (BLIFs, ABC logs, cex
dumps, circuit_report.json) and mlgn/netlist/out/distractor_study/report.json.

Run from the repo root:
    python -m mlgn.netlist.run_distractor_study [--circuits combo,gated] [--resume]

--resume reuses, from an existing circuit_report.json, the (expensive) BFS
closures and any property whose engines already reached a final verdict —
the rebuild/replay gate and cross-validation always run fresh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.netlist import blif, props, sim  # noqa: E402
from mlgn.netlist.extract import check_accuracy, rebuild_model, spec_from_json  # noqa: E402
from mlgn.netlist.falsify import parse_verdict, run_abc  # noqa: E402
from mlgn.netlist.ir import extract_netlist  # noqa: E402
from mlgn.netlist.test_head import eval_netlist  # noqa: E402

RESULTS = os.path.join(_ROOT, "mlgn", "seqlgn", "results")
OUTBASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "distractor_study")
ALPHABET = 8
TOKENS = ["blank"] + [f"sym{i}" for i in range(ALPHABET)]

CIRCUITS = {
    "combo": {
        "ckpt": os.path.join(RESULTS, "ckpt_cp50A_curr_c35.pt"),
        "json": os.path.join(RESULTS, "copy_combo_cp50A_curr_c35_20260704-023800.json"),
        "settle": 16,
        # recorded disc test_acc is 1.0 — any subset must reproduce it exactly
        "acc_tol": 1e-9,
        "anchor": {"protocol_decode": "PROVED"},
    },
    "gated": {
        "ckpt": os.path.join(RESULTS, "ckpt_cpB_gated_oracle.pt"),
        "json": os.path.join(RESULTS, "copy_gated_cpB_gated_oracle_20260704-083424.json"),
        "settle": 22,
        # recorded 0.3803 on the full set; the 8-batch (1024-sample) subset has
        # binomial std ≈ 0.015, so a 0.06 band is a > 3.9σ gate (chance = 0.125)
        "acc_tol": 0.06,
        "anchor": {"protocol_decode": "CEX"},
    },
    # ── distcopy-TRAINED circuits (2026-07-11, DUST run_queue_p3a) — the headline question:
    # does training WITH distractors buy provable distractor robustness? settle values and
    # protocol_decode anchors come from the falsify runs in out/v_dc_* (all full-set gates
    # passed exactly). dc_combo_d20 is a MIXED family (7 fixpoints + 1 correctly-decoding
    # orbit, 16 cycling inputs) — its hold props are known-CEX; decode is the question.
    "dc_combo_d20": {
        "ckpt": os.path.join(RESULTS, "ckpt_v_dc_combo_d20_s0.pt"),
        "json": os.path.join(RESULTS, "distcopy_combo_v_dc_combo_d20_s0_20260710-212712.json"),
        "settle": 13, "acc_tol": 1e-9, "distractors": 20,
        "anchor": {"protocol_decode": "PROVED"},
    },
    "dc_combo_d8": {
        "ckpt": os.path.join(RESULTS, "ckpt_v_dc_combo_d8_s0.pt"),
        "json": os.path.join(RESULTS, "distcopy_combo_v_dc_combo_d8_s0_20260710-212633.json"),
        "settle": 14, "acc_tol": 1e-9, "distractors": 8,
        "anchor": {"protocol_decode": "PROVED"},
    },
    "dc_gated_d8": {
        "ckpt": os.path.join(RESULTS, "ckpt_v_dc_gated_d8_s0.pt"),
        "json": os.path.join(RESULTS, "distcopy_gated_v_dc_gated_d8_s0_20260710-192124.json"),
        "settle": 29, "acc_tol": 1e-9, "distractors": 8,
        "anchor": {"protocol_decode": "PROVED"},
    },
    "dc_clatch_d8": {
        "ckpt": os.path.join(RESULTS, "ckpt_v_dc_clatch_d8_s0.pt"),
        "json": os.path.join(RESULTS, "distcopy_clatch_v_dc_clatch_d8_s0_20260710-192106.json"),
        "settle": 19, "acc_tol": 1e-3, "distractors": 8,  # recorded 0.9999
        "anchor": {"protocol_decode": "PROVED"},
    },
}

PROP_ORDER = ["protocol_decode",
              "distractor_hold_fs0", "distractor_hold_fs1",
              "distractor_decode_fs0", "distractor_decode_fs1"]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


# -----------------------------------------------------------------------------------
# stage 2: BFS closure of each per-symbol fixed point under the distractor alphabet
# -----------------------------------------------------------------------------------
def bfs_closure(net, sym: int, cap: int, chunk: int = 4096, settle_horizon: int = 64) -> dict:
    """Write symbol ``sym`` (cue + one-hot), blanks to the fixed point, then BFS the
    closure of that fixed point under {blank} ∪ {non-cued one-hot tokens}."""
    t_start = time.time()
    n_pi = net.n_pi
    blank1 = np.zeros((1, n_pi), dtype=bool)
    x0 = np.zeros((1, n_pi), dtype=bool)
    x0[0, sym] = True
    x0[0, ALPHABET] = True                      # cue
    state = np.tile(np.asarray(net.init, dtype=bool), (1, 1))
    state = sim.step(net, x0, state)            # the write step -> h_1
    # Follow blanks to the post-write ATTRACTOR: a fixed point (period 1) or a limit
    # cycle (mixed-family circuits, e.g. dc_combo_d20 sym0 — period-N orbit). The BFS
    # is seeded with ALL attractor states, so the closure semantics are unchanged:
    # "every state reachable from the settled behavior under the distractor alphabet".
    seen: dict[bytes, int] = {}
    hist: list[np.ndarray] = []
    orbit = None
    settle_t = None
    for t in range(settle_horizon):
        k = np.packbits(state[0]).tobytes()
        if k in seen:
            orbit = np.stack(hist[seen[k]:])
            settle_t = seen[k]                  # first step the attractor is entered
            break
        seen[k] = t
        hist.append(state[0].copy())
        state = sim.step(net, blank1, state)
    assert orbit is not None, f"symbol {sym}: no attractor within {settle_horizon} blank steps"
    period = len(orbit)
    fp = orbit[0].copy()                        # representative state (the fixed point when period 1)
    fp_decode = int(sim.head_scores(net, fp[None, :]).argmax(-1)[0])

    inputs = np.zeros((ALPHABET + 1, n_pi), dtype=bool)   # row 0 = blank; cue always 0
    for i in range(ALPHABET):
        inputs[1 + i, i] = True

    def keys_of(arr2d: np.ndarray) -> list[bytes]:
        return [row.tobytes() for row in np.packbits(arr2d, axis=1)]

    visited = set(keys_of(orbit))
    orbit_dec = sim.head_scores(net, orbit).argmax(-1)
    n_wrong = int((orbit_dec != sym).sum())
    wrong_examples = [{"decoded": int(d), "bfs_depth": 0}
                      for d in orbit_dec[orbit_dec != sym][:3]]
    frontier = orbit.copy()
    moving_tokens_from_fp: list[str] = []
    escaped = False
    depth = 0
    block = max(1, chunk // (ALPHABET + 1))     # frontier states expanded per block
    while len(frontier) and not escaped:
        depth += 1
        new_parts: list[np.ndarray] = []
        for lo in range(0, len(frontier), block):
            fr = frontier[lo:lo + block]
            reps = np.repeat(fr, ALPHABET + 1, axis=0)
            xs = np.tile(inputs, (len(fr), 1))
            nxt = sim.step(net, xs, reps)
            if depth == 1 and lo == 0:
                moving_tokens_from_fp = [TOKENS[j] for j in range(ALPHABET + 1)
                                         if not (nxt[j] == fp).all()]
            new_rows = []
            for j, k in enumerate(keys_of(nxt)):
                if k not in visited:
                    visited.add(k)
                    new_rows.append(j)
                    if len(visited) > cap:
                        escaped = True
                        break
            if new_rows:
                fresh = nxt[new_rows]
                new_parts.append(fresh)
                dec = sim.head_scores(net, fresh).argmax(-1)
                bad = np.nonzero(dec != sym)[0]
                n_wrong += len(bad)
                for bi in bad[:max(0, 3 - len(wrong_examples))]:
                    wrong_examples.append({"decoded": int(dec[bi]), "bfs_depth": depth})
            if escaped:
                break
        frontier = (np.concatenate(new_parts) if new_parts
                    else np.empty((0, net.n_state), dtype=bool))
    return {
        "symbol": sym,
        "settle_step": settle_t,
        "attractor_period": period,
        "fp_decode": fp_decode,
        "fp_decode_correct": fp_decode == sym,
        "closure_size": len(visited),
        "escaped_cap": escaped,
        "bfs_levels": depth,
        "moving_tokens_from_fp": moving_tokens_from_fp,
        "n_wrong_decode": int(n_wrong),
        "all_visited_decode_correct": n_wrong == 0,
        "wrong_decode_examples": wrong_examples,
        "seconds": round(time.time() - t_start, 1),
    }


# -----------------------------------------------------------------------------------
# stage 3: ABC
# -----------------------------------------------------------------------------------
def parse_cex_tokens(path: str, n_pi: int, init: list[int]) -> dict | None:
    """Parse an ABC write_cex dump into per-frame protocol tokens + raw PI frames.

    ABC's format (observed, abc git 2026): one line of '0'/'1' chars = the latch
    initial values (n_latch bits) followed by the PI values of every frame
    (n_pi bits each), terminated by '# DONE'. A cex written AFTER structural
    transformations (tempor/scorr/dc2) lives in the transformed signal space —
    detected via the latch prefix not matching the property netlist's init —
    and is then reported as non-replayable rather than misparsed."""
    if not os.path.exists(path):
        return None
    n_latch = len(init)
    init_str = "".join(str(int(v)) for v in init)
    bits = ""
    with open(path) as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if s and set(s) <= {"0", "1"}:
                bits += s
    if not (bits and len(bits) >= n_latch and (len(bits) - n_latch) % n_pi == 0
            and bits[:n_latch] == init_str):
        return {"raw_bits": len(bits), "frames": None,
                "note": f"cex bits do not start with the {n_latch}-bit latch init "
                        f"followed by whole {n_pi}-bit PI frames — most likely a "
                        f"transformed-space cex; not replayable"}
    pi_bits = bits[n_latch:]
    frames, pi_frames = [], []
    for t in range(len(pi_bits) // n_pi):
        fb = pi_bits[t * n_pi:(t + 1) * n_pi]
        pi_frames.append(fb)
        syms = [i for i in range(ALPHABET) if fb[i] == "1"]
        cue = fb[ALPHABET] == "1"
        if not syms and not cue:
            tok = "blank"
        elif cue and len(syms) == 1:
            tok = f"WRITE sym{syms[0]}"
        elif not cue and len(syms) == 1:
            tok = f"sym{syms[0]}"
        else:
            tok = f"raw({fb})" + ("+cue" if cue else "") + "->aliased"
        frames.append(tok)
    nonblank = [f"t={t}:{tok}" for t, tok in enumerate(frames) if tok != "blank"]
    return {"n_frames": len(frames), "nonblank_frames": nonblank,
            "pi_frames": pi_frames}


def replay_cex(pnet, cex: dict) -> dict | None:
    """Ground-truth an ABC counterexample: drive the property netlist from its own
    init with the cex's PI frames in the numpy simulator and record every frame at
    which the single 'bad' output is 1. Confirmed iff bad=1 at the LAST frame (the
    frame ABC asserted)."""
    if not cex or not cex.get("pi_frames"):
        return None
    state = np.tile(np.asarray(pnet.init, dtype=bool), (1, 1))
    bad_frames = []
    for t, fb in enumerate(cex["pi_frames"]):
        x = np.array([[c == "1" for c in fb]], dtype=bool)
        outs, state = eval_netlist(pnet, x, state)
        if outs[0, 0]:
            bad_frames.append(t)
    return {"bad_frames": bad_frames,
            "confirmed": bool(bad_frames) and bad_frames[-1] == len(cex["pi_frames"]) - 1}


def parse_verdict_seq(log: str) -> dict:
    """falsify.parse_verdict plus the message tempor's built-in prefix BMC prints
    when the property already fails inside the unrolled frames."""
    v = parse_verdict(log, comb=False)
    if v["verdict"] == "UNKNOWN":
        m = re.search(r"[Oo]utput.*?failed in frame\s+(\d+)", log)
        if m:
            v["verdict"] = "CEX"
            v["detail"] = f"counterexample at frame {m.group(1)} (tempor prefix bmc)"
    return v


def abc_property(name: str, pnet, outdir: str, settle: int, timeout: int,
                 abc_path: str = "~/abc/abc") -> dict:
    """Emit BLIF, run bmc3 (cex hunter, with write_cex) and the proving recipe;
    fall back to plain pdr if the recipe errors out. Returns the result record."""
    blif_path = os.path.join(outdir, f"{name}.blif")
    blif.emit_blif(pnet, blif_path, model=name)
    fname = os.path.basename(blif_path)
    rec: dict = {"blif": blif_path,
                 "netlist": {"pis": pnet.n_pi, "latches": pnet.n_state,
                             "gates": int(pnet.n_gates)},
                 "engines": {}}

    def run(eng: str, script: str, sub_timeout: int, cexfile: str | None = None) -> dict:
        log, secs = run_abc(script, outdir, abc_path, True, sub_timeout)
        with open(os.path.join(outdir, f"{name}.{eng}.log"), "w") as f:
            f.write(log)
        v = parse_verdict_seq(log)
        v["seconds"] = round(secs, 1)
        v["script"] = script
        v["_log_has_error"] = ("rror" in log)
        if cexfile and v["verdict"] == "CEX":
            cex = parse_cex_tokens(os.path.join(outdir, cexfile), pnet.n_pi, pnet.init)
            if cex:
                v["cex_tokens"] = {k2: v2 for k2, v2 in cex.items() if k2 != "pi_frames"}
                v["cex_replay"] = replay_cex(pnet, cex)
        rec["engines"][eng] = v
        print(f"    [{eng:10s}] {v['verdict']:14s} ({v['seconds']}s)  {v.get('detail', '')}"
              + (f"  replay_confirmed={v['cex_replay']['confirmed']}"
                 if v.get("cex_replay") else ""), flush=True)
        return v

    # (ii) first the independent cex hunter — its cex frame is ground truth
    run("bmc3", f"read_blif {fname}; strash; print_stats; bmc3 -T 180; "
                f"write_cex {name}.bmc3.cex", 300, cexfile=f"{name}.bmc3.cex")

    # (i) the proving recipe (tempor's built-in prefix BMC catches cexes < K)
    k = settle + 2
    v = run("tempor_pdr",
            f"read_blif {fname}; strash; print_stats; tempor -F {k}; scorr; dc2; "
            f"print_stats; pdr -T {timeout}; write_cex {name}.tempor_pdr.cex",
            timeout + 120, cexfile=f"{name}.tempor_pdr.cex")

    # fallback: plain pdr when the recipe path errors out / stays UNKNOWN on a cex
    if v["verdict"] == "UNKNOWN" or v["_log_has_error"]:
        run("pdr_fallback",
            f"read_blif {fname}; strash; print_stats; pdr -T {timeout}; "
            f"write_cex {name}.pdr.cex", timeout + 120, cexfile=f"{name}.pdr.cex")

    verdicts = {e: r["verdict"] for e, r in rec["engines"].items()}
    proved = any(v == "PROVED" for v in verdicts.values())
    cexed = any(v == "CEX" for v in verdicts.values())
    # replay status over all cex claims: True = at least one confirmed;
    # False = a parsed cex FAILED to replay (alarm); None = no replayable dump
    replays = [r["cex_replay"]["confirmed"] for r in rec["engines"].values()
               if r.get("cex_replay")]
    if replays:
        rec["cex_replay_confirmed"] = any(replays)
        rec["cex_replay_failed"] = not all(replays)
    rec["final"] = ("CONTRADICTION" if (proved and cexed) or rec.get("cex_replay_failed")
                    else "PROVED" if proved
                    else "CEX" if cexed
                    else "UNDECIDED")
    frames = [r["detail"] for r in rec["engines"].values() if r["verdict"] == "CEX"]
    if frames:
        rec["cex_detail"] = frames
    print(f"    => final: {rec['final']}", flush=True)
    return rec


# -----------------------------------------------------------------------------------
# stage 4: cross-validation of MC verdicts against the BFS ground truth
# -----------------------------------------------------------------------------------
def cross_validate(circ: dict) -> list[str]:
    bfs = circ["bfs"]
    verdicts = {p: circ["props"][p]["final"] for p in circ["props"]}
    fp_all_ok = all(s["fp_decode_correct"] for s in bfs)
    complete = not any(s["escaped_cap"] for s in bfs)
    all_visited_ok = all(s["n_wrong_decode"] == 0 for s in bfs)
    closure_all_1 = complete and all(s["closure_size"] == 1 for s in bfs)
    any_wrong_witness = any(s["n_wrong_decode"] > 0 for s in bfs)
    any_growth = any(s["closure_size"] > 1 or s["escaped_cap"] for s in bfs)

    c: list[str] = []
    # PROVED claims vs concrete BFS witnesses (witnesses are definite even if capped;
    # the BFS envelope is reachable in BOTH from_start modes)
    if verdicts.get("protocol_decode") == "PROVED" and not fp_all_ok:
        c.append("protocol_decode PROVED but a per-symbol fixed point decodes wrongly in simulation")
    for p in ("distractor_decode_fs0", "distractor_decode_fs1"):
        if verdicts.get(p) == "PROVED" and any_wrong_witness:
            c.append(f"{p} PROVED but the BFS closure contains a wrongly-decoding reachable state")
    for p in ("distractor_hold_fs0", "distractor_hold_fs1"):
        if verdicts.get(p) == "PROVED" and any_growth:
            c.append(f"{p} PROVED but the BFS closure grows (a reachable armed state-change exists)")
    # CEX claims vs COMPLETE clean closures (from_start=False only: identical envelope)
    if complete:
        if verdicts.get("distractor_decode_fs0") == "CEX" and all_visited_ok:
            c.append("distractor_decode_fs0 CEX but the complete BFS closure decodes correctly everywhere")
        if verdicts.get("distractor_hold_fs0") == "CEX" and closure_all_1:
            c.append("distractor_hold_fs0 CEX but all complete BFS closures are singletons")
    for p, r in circ["props"].items():
        if r["final"] == "CONTRADICTION":
            c.append(f"{p}: one ABC engine PROVED while another found a counterexample")
    return c


# -----------------------------------------------------------------------------------
def run_circuit(cname: str, cfg: dict, cap: int, timeout: int, skip_abc: bool,
                resume: bool = False) -> dict:
    outdir = os.path.join(OUTBASE, cname)
    os.makedirs(outdir, exist_ok=True)
    circ: dict = {"circuit": cname, "ckpt": cfg["ckpt"], "json": cfg["json"],
                  "settle": cfg["settle"]}
    rpath = os.path.join(outdir, "circuit_report.json")

    prev: dict = {}
    if resume and os.path.exists(rpath):
        with open(rpath) as f:
            prev = json.load(f)
        print(f"[{cname}] resume: found circuit_report.json "
              f"(bfs={len(prev.get('bfs', []))}/8 syms, "
              f"props done={sorted(prev.get('props', {}))})", flush=True)

    # 1. rebuild + replay gate ------------------------------------------------------
    print(f"[{cname}] 1/4 rebuild + replay gate", flush=True)
    spec = spec_from_json(cfg["json"], alphabet=ALPHABET,
                          n_distractors=cfg.get("distractors", 8))
    model = rebuild_model(spec, cfg["ckpt"])
    gate = check_accuracy(model, spec, max_batches=8)
    diff = abs(gate["rebuilt_test_acc"] - spec.test_acc)
    gate["abs_diff_vs_recorded"] = diff
    gate["gate_passed"] = diff <= cfg["acc_tol"]
    circ["accuracy_gate"] = gate
    print(f"    acc(8 batches)={gate['rebuilt_test_acc']:.4f} recorded={spec.test_acc} "
          f"|diff|={diff:.4f} tol={cfg['acc_tol']} -> passed={gate['gate_passed']}", flush=True)
    assert gate["gate_passed"], f"{cname}: replay gate FAILED ({gate})"

    net = extract_netlist(model)
    assert net.n_pi == ALPHABET + 1 and net.head == (ALPHABET, net.n_state // ALPHABET)
    circ["netlist"] = {"pis": net.n_pi, "latches": net.n_state, "gates": int(net.n_gates),
                       "head": list(net.head)}
    traj = sim.analyze_protocol(net, ALPHABET)
    circ["protocol_analysis"] = traj
    n_ok = sum(t["decode_correct_at_settle"] for t in traj)
    print(f"    netlist {net.n_pi} PI / {net.n_state} latches / {net.n_gates} gates; "
          f"fixpoints decode {n_ok}/8", flush=True)

    # 2. BFS pre-analysis ------------------------------------------------------------
    if len(prev.get("bfs", [])) == ALPHABET:
        print(f"[{cname}] 2/4 BFS closures: reusing the {ALPHABET} completed closures "
              f"from the previous run", flush=True)
        circ["bfs"] = prev["bfs"]
    else:
        print(f"[{cname}] 2/4 BFS closures (cap {cap} states/symbol)", flush=True)
        circ["bfs"] = []
    for s in range(len(circ["bfs"]), ALPHABET):
        r = bfs_closure(net, s, cap=cap)
        circ["bfs"].append(r)
        print(f"    sym {s}: settle={r['settle_step']:2d} fp_decode={r['fp_decode']}"
              f"{'' if r['fp_decode_correct'] else ' (WRONG)'} closure={r['closure_size']}"
              f"{' ESCAPED-CAP' if r['escaped_cap'] else ''} wrong_decode={r['n_wrong_decode']} "
              f"movers={r['moving_tokens_from_fp']} ({r['seconds']}s)", flush=True)
        save_json(rpath, circ)
    sizes = [r["closure_size"] for r in circ["bfs"]]
    escaped = [r["symbol"] for r in circ["bfs"] if r["escaped_cap"]]
    if all(z == 1 for z in sizes):
        bfs_case = ("all closures are singletons: the enable is fully shut at the fixed points; "
                    "hold-under-distractors is already established by <=72 simulations and MC "
                    "only re-derives it")
    elif escaped:
        bfs_case = (f"closures escape the {cap}-state cap for symbols {escaped}: the reachable "
                    f"set is too large to enumerate — the model checker is doing genuinely new work")
    else:
        bfs_case = (f"closures are finite but non-trivial (sizes {sizes}): distractors move the "
                    f"state; exhaustive enumeration closes the from_start=False envelope in "
                    f"simulation, MC must close from_start=True")
    circ["bfs_case"] = bfs_case
    print(f"    BFS case: {bfs_case}", flush=True)
    save_json(rpath, circ)

    # 3. properties + ABC --------------------------------------------------------------
    K = cfg["settle"]
    builders = {
        "protocol_decode": lambda: props.protocol_decode(net, ALPHABET, settle=K),
        "distractor_hold_fs0": lambda: props.distractor_hold(net, ALPHABET, settle=K, from_start=False),
        "distractor_hold_fs1": lambda: props.distractor_hold(net, ALPHABET, settle=K, from_start=True),
        "distractor_decode_fs0": lambda: props.distractor_decode(net, ALPHABET, settle=K, from_start=False),
        "distractor_decode_fs1": lambda: props.distractor_decode(net, ALPHABET, settle=K, from_start=True),
    }
    circ["props"] = {}
    if skip_abc:
        save_json(rpath, circ)
        return circ
    print(f"[{cname}] 3/4 model checking (settle K={K}, tempor -F {K + 2})", flush=True)
    for pname in PROP_ORDER:
        done = prev.get("props", {}).get(pname)
        if done and done.get("final") in ("PROVED", "CEX"):
            print(f"  {pname}: reusing previous final={done['final']}", flush=True)
            circ["props"][pname] = done
            save_json(rpath, circ)
            continue
        print(f"  {pname}:", flush=True)
        pnet = builders[pname]()
        circ["props"][pname] = abc_property(pname, pnet, outdir, K, timeout)
        save_json(rpath, circ)

    # 4. cross-validation ---------------------------------------------------------------
    print(f"[{cname}] 4/4 cross-validation", flush=True)
    contradictions = cross_validate(circ)
    circ["contradictions"] = contradictions
    anchor_dev = []
    for p, expected in cfg.get("anchor", {}).items():
        got = circ["props"].get(p, {}).get("final")
        if got != expected:
            anchor_dev.append(f"{p}: expected {expected}, got {got}")
    circ["anchor_deviations"] = anchor_dev
    print(f"    contradictions: {contradictions or 'none'}", flush=True)
    print(f"    anchor deviations: {anchor_dev or 'none'}", flush=True)
    save_json(rpath, circ)
    return circ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--circuits", default="combo,gated")
    ap.add_argument("--cap", type=int, default=200000, help="BFS closure cap per symbol")
    ap.add_argument("--timeout", type=int, default=600, help="pdr timeout per property, s")
    ap.add_argument("--skip-abc", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed BFS closures / property verdicts from an "
                         "existing circuit_report.json")
    args = ap.parse_args()

    os.makedirs(OUTBASE, exist_ok=True)
    report: dict = {"campaign": "distractor_study",
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "circuits": {}}
    status = 0
    for cname in [c.strip() for c in args.circuits.split(",") if c.strip()]:
        t0 = time.time()
        circ = run_circuit(cname, CIRCUITS[cname], args.cap, args.timeout, args.skip_abc,
                           resume=args.resume)
        circ["total_seconds"] = round(time.time() - t0, 1)
        report["circuits"][cname] = circ
        save_json(os.path.join(OUTBASE, "report.json"), report)
        if circ.get("contradictions"):
            print(f"\n[STOP] CONTRADICTION on {cname}: {circ['contradictions']}\n"
                  "A proof disagrees with the simulation ground truth — something is "
                  "unsound; NOT proceeding.", flush=True)
            status = 4
            break
    save_json(os.path.join(OUTBASE, "report.json"), report)
    print(f"\nreport -> {os.path.join(OUTBASE, 'report.json')} (exit {status})", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
