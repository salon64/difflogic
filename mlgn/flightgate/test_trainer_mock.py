"""
test_trainer_mock.py — the sim-independent correctness gate for the D1 harness.
===============================================================================

Run from the repo root (main env, CPU):

    python -m mlgn.flightgate.test_trainer_mock

Proves, on the deterministic 2D point-mass mock (NO simulator required):
  1. the SensorBlackout wrapper masks exactly what it claims (constant-zero
     sentinel on the masked dims, validity bit, untouched dims bit-identical to
     an unwrapped replay), and its mask bookkeeping is consistent;
  2. thermometer encoder + action discretizer are correct and config-replayable;
  3. episode resets are correct: the collector resets the teacher's integral
     state every episode (the mock teacher carries integral state ON PURPOSE so
     a missing reset is detectable), and the recurrent student policy resets its
     hidden state (identical closed-loop replays);
  4. the whole closed-loop DAgger trainer runs end-to-end (3 rounds, gated arm):
     losses finite and decreasing, per-round eval metrics finite, feedforward
     arm trains too.

House discipline: every assertion is a hard gate; the script exits non-zero on
the first failure and prints PASS lines otherwise.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlgn.flightgate import encode, trainer  # noqa: E402
from mlgn.flightgate.env import SensorBlackout  # noqa: E402
from mlgn.flightgate.mock_env import MockPIDTeacher, MockPointMass2D  # noqa: E402


def test_blackout_wrapper():
    """Masked frames are constant-zero + valid=0 on EXACTLY the mask_dims; all
    other dims (and all unmasked frames) are bit-identical to an unwrapped replay
    of the same deterministic episode."""
    rng = np.random.default_rng(123)
    actions = rng.uniform(-1, 1, size=(60, 4))

    for mask_dims, masked_idx in ((None, [0, 1, 2, 3]), (slice(0, 2), [0, 1])):
        plain = MockPointMass2D(max_steps=100)
        wrapped = SensorBlackout(MockPointMass2D(max_steps=100), p_start=0.15,
                                 k_min=2, k_max=4, mask_dims=mask_dims, seed=7)
        obs_p, _ = plain.reset(seed=5)
        obs_w, _ = wrapped.reset(seed=5)
        n_masked = 0
        keep_idx = [d for d in range(4) if d not in masked_idx]
        for t in range(len(actions)):
            valid = obs_w[-1]
            assert valid in (0.0, 1.0), valid
            assert wrapped.mask_history[-1] == (0 if valid else 1), "mask bookkeeping"
            if valid == 1.0:
                assert np.array_equal(obs_w[:4], obs_p), f"t={t}: unmasked frame differs"
            else:
                n_masked += 1
                assert np.all(obs_w[masked_idx] == 0.0), f"t={t}: sentinel not zero"
                if keep_idx:
                    assert np.array_equal(obs_w[keep_idx], obs_p[keep_idx]), \
                        f"t={t}: unmasked DIMS were altered"
                # sentinel must be CONSTANT ZERO, never freeze-last-value
                if np.any(obs_p[masked_idx] != 0.0):
                    assert not np.array_equal(obs_w[masked_idx], obs_p[masked_idx]), \
                        f"t={t}: wrapper leaked the true value"
            obs_p, *_ = plain.step(actions[t])
            obs_w, *_ = wrapped.step(actions[t])
        assert n_masked >= 3, f"blackouts never triggered (n_masked={n_masked})"
        # dynamics must be untouched by the wrapper (observation-only masking)
        assert np.array_equal(plain.true_state(), wrapped.true_state()), \
            "wrapper altered the dynamics"
    # replayability: same wrapper seed => identical mask sequence
    m = []
    for _ in range(2):
        w = SensorBlackout(MockPointMass2D(max_steps=100), p_start=0.15, k_min=2,
                           k_max=4, seed=42)
        w.reset(seed=5)
        for t in range(40):
            w.step(actions[t % len(actions)])
        m.append(list(w.mask_history))
    assert m[0] == m[1], "mask sequence not replayable from wrapper seed"
    print("PASS test_blackout_wrapper")


def test_encoder_and_discretizer():
    enc = encode.ThermometerEncoder(encode.MOCK_RANGES, bits=4)
    assert enc.n_bits == 16 and enc.n_bits_with_valid == 17
    # monotone (thermometer) code: bit i set => all lower-threshold bits set
    x = np.array([0.3, -1.7, 2.9, -3.5])  # includes out-of-range values (clip)
    bits = enc.encode(x).reshape(4, 4)
    for d in range(4):
        b = bits[d]
        assert np.all(b[:-1] >= b[1:]), f"dim {d}: not a thermometer code: {b}"
    assert np.all(bits[2] == 1.0), "value above hi must set all bits"
    assert np.all(bits[3] == 0.0), "value below lo must clear all bits"
    # exact threshold semantics: bit_i = (x > lo + (i+1)*(hi-lo)/(T+1))
    lo, hi = encode.MOCK_RANGES[0]
    expect = [(0.3 > lo + (i + 1) * (hi - lo) / 5.0) for i in range(4)]
    assert np.array_equal(bits[0], np.array(expect, dtype=np.float32)), bits[0]
    # config roundtrip (the exporter needs thresholds bit-exactly)
    enc2 = encode.ThermometerEncoder.from_config(enc.to_config())
    assert np.array_equal(enc2.encode(x), enc.encode(x))
    # validity passthrough
    ev = enc.encode_with_valid(np.concatenate([x, [1.0]]))
    assert ev.shape == (17,) and ev[-1] == 1.0
    ev0 = enc.encode_with_valid(np.concatenate([x, [0.0]]))
    assert ev0[-1] == 0.0

    disc = encode.ActionDiscretizer(n_bins=5)
    a = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    b = disc.to_bins(a)
    assert b[0] == 0 and b[-1] == disc.n_bins - 1, b
    q = disc.quantize(a)
    assert np.all(np.abs(q - a) <= (2.0 / 5) / 2 + 1e-9), (a, q)
    assert np.all(disc.to_bins(q) == b), "bin centers must round-trip"
    print("PASS test_encoder_and_discretizer")


def test_teacher_reset_in_collector():
    """The mock teacher's integral term makes a missing reset detectable: two
    identical collect_episodes calls must produce identical teacher actions, and
    an episode WITHOUT reset must differ from one with."""
    enc = encode.ThermometerEncoder(encode.MOCK_RANGES, bits=4)
    disc = encode.ActionDiscretizer(n_bins=5)
    teacher = MockPIDTeacher()

    def factory(i):
        return SensorBlackout(MockPointMass2D(max_steps=30), p_start=0.1,
                              k_min=2, k_max=4, seed=100 + i)

    runs = []
    for _ in range(2):
        eps = trainer.collect_episodes(factory, teacher, enc, disc, n_episodes=2,
                                       rng=np.random.default_rng(0), seed_base=50,
                                       max_steps=30)
        runs.append(eps)
    for e1, e2 in zip(*runs):
        assert np.array_equal(e1["teacher_action"], e2["teacher_action"]), \
            "teacher actions not replayable => integral state leaked across episodes"
        assert np.array_equal(e1["bits"], e2["bits"]), "observations not replayable"

    # sensitivity control (non-vacuity): the SAME episode flown twice WITHOUT a
    # teacher reset in between must produce different actions.
    env = MockPointMass2D(max_steps=30)
    teacher.reset()
    acts = []
    for _ in range(2):  # no reset between these two episodes
        obs, _ = env.reset(seed=50)
        ep_a = []
        for _t in range(20):
            a = teacher.act(env)
            ep_a.append(a)
            env.step(a)
        acts.append(np.asarray(ep_a))
    assert not np.array_equal(acts[0], acts[1]), \
        "mock teacher integral term failed to make missing-reset detectable"
    print("PASS test_teacher_reset_in_collector")


def test_student_policy_state_reset():
    """A recurrent student policy must give an identical closed-loop replay after
    reset() (state cleared), on the same env seed."""
    from mlgn.flightgate.student import StudentPolicy, build_student
    from mlgn.seqlgn import utils

    utils.set_seed(0)
    enc = encode.ThermometerEncoder(encode.MOCK_RANGES, bits=4)
    disc = encode.ActionDiscretizer(n_bins=5)
    model = build_student("gated", input_bits=enc.n_bits + 1, hidden=40,
                          n_act=4, n_bins=5, device="cpu")
    policy = StudentPolicy(model, disc, device="cpu", discrete=True)

    def factory(i):
        return SensorBlackout(MockPointMass2D(max_steps=20), p_start=0.2,
                              k_min=2, k_max=3, seed=9)

    outs = []
    for _ in range(2):
        env = factory(0)
        ep = trainer._rollout_actor(env, policy, enc, reset_seed=3, max_steps=20)
        env.close()
        outs.append(ep)
        # direct state check: rollout populated the recurrent state; reset clears it
        assert policy._state is not None, "policy state never initialised"
        policy.reset()
        assert policy._state is None, "policy.reset() did not clear the state"
    assert np.array_equal(outs[0]["position"], outs[1]["position"]), \
        "recurrent policy replay differs => state not reset between episodes"
    print("PASS test_student_policy_state_reset")


def test_closed_loop_dagger():
    """End-to-end: 3 DAgger rounds on the mock, gated arm, CPU-smoke sizes.
    Losses finite + decreasing; eval metrics finite; ff arm also trains."""
    from mlgn.flightgate.student import build_student
    from mlgn.seqlgn import utils

    utils.set_seed(0)
    enc = encode.ThermometerEncoder(encode.MOCK_RANGES, bits=4)
    disc = encode.ActionDiscretizer(n_bins=5)
    teacher = MockPIDTeacher()

    def factory(i):
        return SensorBlackout(MockPointMass2D(max_steps=25), p_start=0.08,
                              k_min=2, k_max=4, seed=1000 + i)

    model = build_student("gated", input_bits=enc.n_bits + 1, hidden=40,
                          n_act=4, n_bins=5, device="cpu")
    t0 = time.time()
    rec = trainer.dagger(
        model, factory, teacher, enc, disc, target=np.zeros(2),
        beta_schedule=(1.0, 0.5, 0.25), episodes_per_round=4, eval_episodes=3,
        iters_per_round=60, batch_size=8, lr=0.01, grad_clip=1.0, bptt=0,
        device="cpu", max_steps=25, seed=0, traj_dir=None)
    mins = (time.time() - t0) / 60
    all_losses = [l for r in rec["rounds"] for l in r["train"]["losses"]]
    assert all(np.isfinite(l) for l in all_losses), "non-finite training loss"
    first = float(np.mean(all_losses[:10]))
    last = float(np.mean(all_losses[-10:]))
    assert last < first, f"loss did not decrease: first10={first:.4f} last10={last:.4f}"
    for r in rec["rounds"]:
        ev = r["eval_student_discrete"]
        assert np.isfinite(ev["mean_return"]), ev
        assert np.isfinite(r["train"]["action_acc_discrete"]), r["train"]
        assert r["train"]["n_skipped"] < len(r["train"]["losses"]), "all steps skipped"
    acc_d = rec["rounds"][-1]["train"]["action_acc_discrete"]
    gap = rec["rounds"][-1]["train"]["discretization_gap"]
    print(f"  dagger: loss {first:.4f} -> {last:.4f}, final acc_d={acc_d:.4f} "
          f"gap={gap:+.4f}, {mins:.1f} min")

    # feedforward control arm: must also train (a few steps, loss finite+decreasing)
    ff = build_student("ff", input_bits=enc.n_bits + 1, hidden=40, n_act=4,
                       n_bins=5, device="cpu")
    eps = trainer.collect_episodes(factory, teacher, enc, disc, n_episodes=4,
                                   rng=np.random.default_rng(1), seed_base=0,
                                   max_steps=25)
    stats = trainer.train_student(ff, eps, iters=40, batch_size=8, device="cpu",
                                  rng_seed=1)
    assert all(np.isfinite(l) for l in stats["losses"]), "ff: non-finite loss"
    assert np.mean(stats["losses"][-5:]) < np.mean(stats["losses"][:5]), \
        "ff: loss did not decrease"
    print("PASS test_closed_loop_dagger")


def main():
    t0 = time.time()
    test_blackout_wrapper()
    test_encoder_and_discretizer()
    test_teacher_reset_in_collector()
    test_student_policy_state_reset()
    test_closed_loop_dagger()
    print(f"ALL flightgate mock-gate tests PASSED ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
