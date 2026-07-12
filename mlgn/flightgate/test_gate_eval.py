"""
test_gate_eval.py — the aggregator gate for gate_eval.py (sim-independent, no torch).
=====================================================================================

Synthesizes fake per-run JSONs (the exact shape cli.py writes) into a temp dir and
asserts the three-arm gate table assembles correctly:
  1. grouping/aggregation: right seeds per (arm, blackout) cell, right mean±std;
  2. record-shape handling: student ("rounds"), teacher_masked, teacher_only;
  3. the WIN criterion fires when recurrent >> ff under blackout AND matches under
     control; and does NOT fire (FAIL) when ff matches recurrent under blackout;
  4. newest-wins de-dup on a re-run of the same (arm, blackout, seed).

Run from the repo root:  python -m mlgn.flightgate.test_gate_eval
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from mlgn.flightgate import gate_eval


def _student_json(arm, blackout, seed, ret, acc_d=0.8, gap=0.02, exit_rate=0.0,
                  teacher_ret=250.0):
    """A student run record (has a 'rounds' list; gate_eval reads the last round)."""
    return {
        "task": "flightgate_hover", "mechanism": arm, "backend": "hover",
        "seed": seed, "hidden": 432, "logic_gates": 1728,
        "blackout": ({"p_start": 0.02, "k_min": 6, "k_max": 12, "seed": seed * 7919}
                     if blackout else None),
        "eval_teacher": {"mean_return": teacher_ret, "mean_rms_err": 0.05,
                         "mean_rms_err_blackout": float("nan"),
                         "envelope_exit_rate": 0.0, "n_episodes": 3},
        "eval_teacher_quantized": {"mean_return": teacher_ret * 0.98,
                                   "envelope_exit_rate": 0.0, "n_episodes": 3},
        "eval_teacher_masked": None,
        "quantization_gate_return_ratio": 0.98,
        "rounds": [
            {"round": 0, "train": {"action_acc_discrete": acc_d - 0.1,
                                   "discretization_gap": gap},
             "eval_student_discrete": {"mean_return": ret - 10, "mean_rms_err": 0.2,
                                       "mean_rms_err_blackout": 0.3,
                                       "envelope_exit_rate": exit_rate,
                                       "n_episodes": 3}},
            {"round": 1, "train": {"action_acc_discrete": acc_d,
                                   "discretization_gap": gap},
             "eval_student_discrete": {"mean_return": ret, "mean_rms_err": 0.15,
                                       "mean_rms_err_blackout": 0.25,
                                       "envelope_exit_rate": exit_rate,
                                       "n_episodes": 3}},
        ],
    }


def _masked_json(seed, ret, exit_rate=0.1, teacher_ret=250.0):
    return {
        "task": "flightgate_hover", "mechanism": "teacher_masked", "backend": "hover",
        "seed": seed, "hidden": 432,
        "blackout": {"p_start": 0.02, "k_min": 6, "k_max": 12, "seed": seed * 7919},
        "eval_teacher": {"mean_return": teacher_ret, "envelope_exit_rate": 0.0,
                         "n_episodes": 3},
        "eval_teacher_quantized": {"mean_return": teacher_ret * 0.98, "n_episodes": 3},
        "eval_teacher_masked": {"mean_return": ret, "mean_rms_err": 0.4,
                                "mean_rms_err_blackout": 0.6,
                                "envelope_exit_rate": exit_rate, "n_episodes": 3},
        "quantization_gate_return_ratio": 0.98,
    }


def _write(dirpath, rec, tag, stamp="20260712-120000"):
    base = f"flightgate_{rec['backend']}_{rec['mechanism']}_{tag}_{stamp}"
    path = os.path.join(dirpath, base + ".json")
    with open(path, "w") as f:
        json.dump(rec, f, default=float)
    return path


def _isclose(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_grouping_and_win():
    d = tempfile.mkdtemp(prefix="gate_eval_win_")
    try:
        # CONTROL: all three arms match (~240). BLACKOUT: recurrent hold ~200, ff
        # collapses ~60. a' ceiling ~222 (oracle >= student). Seeds 0,1,2.
        for seed in (0, 1, 2):
            _write(d, _student_json("gated", False, seed, 240 + seed),
                   f"nb_gated_s{seed}")
            _write(d, _student_json("clatch", False, seed, 238 + seed),
                   f"nb_clatch_s{seed}")
            _write(d, _student_json("ff", False, seed, 239 + seed),
                   f"nb_ff_s{seed}")
            _write(d, _student_json("gated", True, seed, 200 + seed, exit_rate=0.0),
                   f"bk_gated_s{seed}")
            _write(d, _student_json("clatch", True, seed, 195 + seed, exit_rate=0.0),
                   f"bk_clatch_s{seed}")
            _write(d, _student_json("ff", True, seed, 60 + seed, exit_rate=0.3),
                   f"bk_ff_s{seed}")
            _write(d, _masked_json(seed, 222 + seed), f"bk_aprime_s{seed}")

        paths = sorted(__import__("glob").glob(os.path.join(d, "*.json")))
        report = gate_eval.build_report(paths)
        agg = gate_eval.aggregate(gate_eval.load_runs(paths))

        # (1) grouping: 3 seeds per (arm, blackout) cell
        for arm in ("gated", "clatch", "ff"):
            for bk in (True, False):
                cell = agg[(arm, bk)]
                assert cell["n"] == 3, (arm, bk, cell["n"])
                assert sorted(cell["seeds"]) == [0, 1, 2], cell["seeds"]
        # (2) mean correctness: gated blackout returns = {200,201,202} -> mean 201
        assert _isclose(agg[("gated", True)]["return"]["mean"], 201.0), \
            agg[("gated", True)]["return"]
        assert _isclose(agg[("ff", True)]["return"]["mean"], 61.0), \
            agg[("ff", True)]["return"]
        # ff control mean {239,240,241} -> 240
        assert _isclose(agg[("ff", False)]["return"]["mean"], 240.0)
        # a' ceiling present under blackout
        assert agg[("teacher_masked", True)]["return"]["mean"] is not None
        # acc_d reads the LAST round (0.8), not round 0 (0.7)
        assert _isclose(agg[("gated", True)]["acc_d"]["mean"], 0.8, tol=1e-6)

        # (3) verdict = WIN (recurrent >> ff under blackout; matched under control)
        v = report["verdict"]
        assert v["gate_win"] is True, v["status"]
        assert v["status"] == "WIN", v["status"]
        assert v["per_arm"]["gated"]["beats_ff_under_blackout"] is True
        assert v["per_arm"]["gated"]["matches_ff_under_control"] is True
        # margin ~ 201-61 = 140 >> spread
        assert v["per_arm"]["gated"]["blackout_margin"] > 100
        # oracle a' ceiling (~223) is above the students -> NO leak flagged
        assert v["ceiling_leak_arms"] == [], v["ceiling_leak_arms"]
        # ...and the leak detector still WORKS: gated 201 sits below ceiling 223
        assert v["per_arm"]["gated"]["above_ceiling"] is False

        # table renders without error and mentions the arms + verdict
        tbl = report["table"]
        assert "gated (recurrent)" in tbl and "ff (stateless)" in tbl
        assert "GATE VERDICT: WIN" in tbl
        print("PASS test_grouping_and_win")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fail_when_ff_matches():
    d = tempfile.mkdtemp(prefix="gate_eval_fail_")
    try:
        # ff matches (slightly beats) recurrent under blackout -> no memory
        # advantage -> FAIL. ff mean 181 >= gated mean 179 / clatch mean 178.
        for seed in (0, 1, 2):
            _write(d, _student_json("gated", True, seed, 178 + seed), f"bk_gated_s{seed}")
            _write(d, _student_json("clatch", True, seed, 177 + seed), f"bk_clatch_s{seed}")
            _write(d, _student_json("ff", True, seed, 180 + seed), f"bk_ff_s{seed}")
            _write(d, _student_json("gated", False, seed, 240 + seed), f"nb_gated_s{seed}")
            _write(d, _student_json("clatch", False, seed, 240 + seed), f"nb_clatch_s{seed}")
            _write(d, _student_json("ff", False, seed, 240 + seed), f"nb_ff_s{seed}")
        paths = sorted(__import__("glob").glob(os.path.join(d, "*.json")))
        v = gate_eval.build_report(paths)["verdict"]
        assert v["gate_win"] is False, v
        assert v["status"].startswith("FAIL"), v["status"]
        print("PASS test_fail_when_ff_matches")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_newest_wins_dedup():
    d = tempfile.mkdtemp(prefix="gate_eval_dedup_")
    try:
        # same (gated, blackout, seed 0) written twice; the later timestamp wins
        _write(d, _student_json("gated", True, 0, 100.0), "bk_gated_s0",
               stamp="20260712-100000")
        _write(d, _student_json("gated", True, 0, 222.0), "bk_gated_s0",
               stamp="20260712-180000")
        paths = sorted(__import__("glob").glob(os.path.join(d, "*.json")))
        rows = gate_eval.load_runs(paths)
        assert len(rows) == 1, f"de-dup failed: {len(rows)} rows"
        assert _isclose(rows[0]["return"], 222.0), rows[0]["return"]
        print("PASS test_newest_wins_dedup")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    test_grouping_and_win()
    test_fail_when_ff_matches()
    test_newest_wins_dedup()
    print("ALL gate_eval tests PASSED")


if __name__ == "__main__":
    main()
