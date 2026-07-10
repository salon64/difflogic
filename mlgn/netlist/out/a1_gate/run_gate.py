"""
run_gate.py — integration gate for the in-netlist GroupSum head (agent HEAD).
=============================================================================

a) Rebuild ckpt_cp50A_curr_c35 (combo copy, disc test_acc 1.0), extract the netlist,
   build props.protocol_decode(net, 8, settle=16), and simulate the property netlist
   on all 8 legal write sequences (write, then blanks, 40 frames): the single 'bad'
   output must be 0 at every frame. Also runs two controls:
     - non-vacuity: corrupt the shadow register mid-run (state surgery on the numpy
       state vector) -> bad MUST fire at every armed frame;
     - legality gating: an illegal (two-hot) write -> bad stays 0 forever.
b) Emit BLIF and prove it with ABC inside WSL using the settle-then-hold recipe:
   read_blif; strash; tempor -F 18; scorr; dc2; pdr -T 600  ->  "Property proved".

Logs land in mlgn/netlist/out/a1_gate/. Run from the repo root:
    python <path>/run_gate.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = r"C:\Users\malco\projects\difflogic"
sys.path.insert(0, ROOT)

from mlgn.netlist import blif, props, sim  # noqa: E402
from mlgn.netlist.extract import rebuild_model, spec_from_json  # noqa: E402
from mlgn.netlist.ir import extract_netlist  # noqa: E402
from mlgn.netlist.test_head import eval_netlist  # noqa: E402

CKPT = os.path.join(ROOT, "mlgn", "seqlgn", "results", "ckpt_cp50A_curr_c35.pt")
JSON = os.path.join(ROOT, "mlgn", "seqlgn", "results",
                    "copy_combo_cp50A_curr_c35_20260704-023800.json")
OUTDIR = os.path.join(ROOT, "mlgn", "netlist", "out", "a1_gate")
ALPHABET, SETTLE, FRAMES = 8, 16, 40


def win_to_wsl(p: str) -> str:
    p = os.path.abspath(p)
    return f"/mnt/{p[0].lower()}{p[2:]}".replace("\\", "/")


def run_frames(pnet, x_seq, tamper=None):
    """Simulate the property netlist; returns bad [B, T]. ``tamper(state, t)`` may
    mutate the latch vector between frames (state surgery for the controls)."""
    B, T, _ = x_seq.shape
    state = np.tile(np.asarray(pnet.init, dtype=bool), (B, 1))
    bad = np.zeros((B, T), dtype=bool)
    for t in range(T):
        outs, state = eval_netlist(pnet, x_seq[:, t], state)
        bad[:, t] = outs[:, 0]
        if tamper is not None:
            tamper(state, t)
    return bad


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    report = {"ckpt": CKPT, "json": JSON, "settle": SETTLE, "frames": FRAMES}

    print("[1/4] rebuild + extract")
    spec = spec_from_json(JSON, alphabet=ALPHABET)
    model = rebuild_model(spec, CKPT)
    net = extract_netlist(model)
    assert (net.n_pi, net.n_state, net.n_gates) == (9, 1024, 7168), \
        (net.n_pi, net.n_state, net.n_gates)
    assert net.head == (8, 128), net.head
    traj = sim.analyze_protocol(net, ALPHABET)
    ok8 = all(t["settled"] and t["decode_correct_at_settle"] and t["settle_step"] <= 15
              for t in traj)
    report["protocol_analysis_ok"] = ok8
    print(f"      netlist 9 PI / 1024 latches / 7168 gates; "
          f"8/8 trajectories settle<=15 & decode: {ok8}")
    assert ok8

    print("[2/4] build protocol_decode(net, 8, settle=16) + simulate 8 legal writes")
    pnet = props.protocol_decode(net, ALPHABET, settle=SETTLE)
    report["property_netlist"] = {"pis": pnet.n_pi, "latches": pnet.n_state,
                                  "gates": int(pnet.n_gates)}
    print(f"      property netlist: {pnet.n_pi} PIs, {pnet.n_state} latches, "
          f"{pnet.n_gates} gates")
    x = np.zeros((ALPHABET, FRAMES, net.n_pi), dtype=bool)
    for s in range(ALPHABET):
        x[s, 0, s] = True            # one-hot symbol
        x[s, 0, ALPHABET] = True     # cue
    bad = run_frames(pnet, x)
    report["legal_writes_bad_any"] = bool(bad.any())
    print(f"      bad over 8 writes x {FRAMES} frames: "
          f"{int(bad.sum())} assertions (must be 0)")
    assert not bad.any(), np.argwhere(bad)

    # control 1 (non-vacuity): write symbol 3, then flip the shadow to symbol 5 by
    # state surgery after frame 1 -> decode says 3, shadow says 5 -> bad must fire
    # at EVERY armed frame (t >= settle).
    sh0 = net.n_state + SETTLE + 1   # first shadow latch index in the state vector

    def corrupt(state, t):
        if t == 1:
            state[:, sh0 + 3] = False
            state[:, sh0 + 5] = True

    bad_c = run_frames(pnet, x[3:4], tamper=corrupt)
    ok_c1 = (not bad_c[0, :SETTLE].any()) and bad_c[0, SETTLE:].all()
    report["control_corrupted_shadow_fires"] = bool(ok_c1)
    print(f"      control 1 (corrupted shadow): bad at frames "
          f"{SETTLE}..{FRAMES - 1} = {bool(bad_c[0, SETTLE:].all())} (must be True)")
    assert ok_c1

    # control 2 (legality gating): two-hot write -> legal=0 -> bad never fires
    x_ill = np.zeros((1, FRAMES, net.n_pi), dtype=bool)
    x_ill[0, 0, [1, 7, ALPHABET]] = True
    bad_i = run_frames(pnet, x_ill)
    report["control_illegal_write_silent"] = not bool(bad_i.any())
    print(f"      control 2 (illegal two-hot write): bad any = {bool(bad_i.any())} "
          f"(must be False)")
    assert not bad_i.any()

    print("[3/4] emit BLIF")
    blif_path = os.path.join(OUTDIR, "protocol_decode.blif")
    blif.emit_blif(pnet, blif_path, model="cp50A_curr_c35_protocol_decode")
    print(f"      -> {blif_path}")

    print("[4/4] ABC: tempor -F 18; scorr; dc2; pdr -T 600")
    script = ("read_blif protocol_decode.blif; strash; print_stats; "
              "tempor -F 18; scorr; dc2; print_stats; pdr -T 600")
    inner = f'cd {win_to_wsl(OUTDIR)} && ~/abc/abc -c "{script}"'
    t0 = time.time()
    res = subprocess.run(["wsl", "-e", "bash", "-lc", inner],
                         capture_output=True, text=True, timeout=780)
    secs = time.time() - t0
    log = res.stdout + res.stderr
    log_path = os.path.join(OUTDIR, "protocol_decode.tempor_pdr.log")
    with open(log_path, "w") as f:
        f.write(log)
    proved = "Property proved" in log
    report["abc"] = {"script": script, "seconds": round(secs, 1), "proved": proved,
                     "log": log_path}
    print(f"      ABC finished in {secs:.1f}s, proved={proved}")
    for line in log.splitlines():
        if any(k in line for k in ("i/o", "Property", "Invariant", "prove")):
            print("      | " + line.strip())

    report["gate_passed"] = proved
    with open(os.path.join(OUTDIR, "gate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {os.path.join(OUTDIR, 'gate_report.json')}")
    if not proved:
        print("GATE FAILED: ABC did not print 'Property proved'.")
        return 1
    print("INTEGRATION GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
