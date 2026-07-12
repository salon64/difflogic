# flightgate — P3b gate D1: closed-loop POMDP distillation for recurrent LGN cells

The **minimal thesis-gate flight harness** (research/23 §D1, kicked off 2026-07-11).
A stock PID teacher flying a hover task is distilled — **closed-loop, DAgger-style,
with per-step action targets ≡ deep supervision** (the training method P2 proved
necessary) — into seqlgn recurrent cells (`gated` + `clatch` arms) and a feedforward
LogicMLP control arm, on **thermometer-encoded observations**, under an
**observation-dropout wrapper** (`SensorBlackout`) that makes memory REQUIRED.

**Gate bar:** memory cell ≳ teacher under occlusion where the feedforward student
degrades — with the **non-vacuity control** that feedforward ≈ recurrent in the
no-blackout condition (proves the gap is memory, not capacity).
**Pre-committed pivot on FAIL** (workmap D1): thesis topic becomes "verified logic
flight controller" (feedforward + P3a verification + board); memory claims get scoped.

## Simulator decision (recon 2026-07-12)

**gym-pybullet-drones @ main (v2.1.0)**, env `HoverAviary`, teacher `DSLPIDControl`
(Bitcraze Crazyflie-2.x-tuned cascaded PID — matches the Kyushu/D2 pitch identity).
PyFlyt rejected (numpy<2.0.0 pin conflicts with the torch/numpy-2 stack; QuadX is not
a validated Crazyflie). Crazyflow (utiasDSL JAX sim, 2606.01478) flagged as the
future throughput option only. gym-pybullet-drones is **NOT on PyPI** — git clone +
editable install; **record the commit hash in every run JSON** (the install script
writes it to `.venv-flight/GPD_COMMIT`, which `cli.py` picks up automatically).

## Files

| file | role |
|---|---|
| `mock_env.py` | sim-independent 2D point-mass + scripted integral-PID teacher (numpy-only) — the mandatory correctness gate when no simulator is available |
| `env.py` | `SensorBlackout` (constant-zero sentinel + validity bit; the memory-required variant) + `HoverAdapter`/`hover_env_factory` (12-dim obs — the 60-dim action-history buffer is CUT: hidden memory channel), per-episode jitter, lazy sim imports |
| `teacher.py` | `HoverPIDTeacher` (DSLPIDControl → exact normalized-action inversion, **clipped** to the ±5 % RPM band = the only valid distillation target) + `MaskedObsHoverTeacher` (**a′ ceiling**: PID on the reconstructed masked obs) + `make_teacher` / `make_masked_teacher` |
| `encode.py` | `ThermometerEncoder` (frozen thresholds recorded per run; **hover velocity ranges are PLACEHOLDERS — calibrate before real runs**, see below), `ActionDiscretizer` (uniform bins, CE targets ↔ bin centers), `calibrate_ranges` |
| `student.py` | the three arms via seqlgn APIs: `gated`/`clatch` = `LogicRecurrentCell` + GroupSum action head; `ff` = `LogicMLP` + GroupSum (matched hidden width; gate counts recorded); `StudentPolicy` (deployed-consistent discrete rollouts, `hard_alpha=1.0`) |
| `trainer.py` | numpy-only collection (teacher labels EVERY visited state) + torch training with the train.py guard rails (non-finite-norm step skip, dead-window stop, per-step `hard_alpha` anneal) + `dagger()` loop + `.npz` trajectory persistence |
| `cli.py` | `python -m mlgn.flightgate.cli` — args, seqlgn seeding conventions, JSON records mirroring train.py's shape (+ encoder thresholds, bin edges, β schedule, blackout config, `hover_jitter`, sim versions/commit); `--teacher-masked` runs the **a′ ceiling** |
| `gate_eval.py` | `python -m mlgn.flightgate.gate_eval` — assembles the **three-arm gate table** (gated/clatch vs ff × blackout/control × seeds) from the per-run JSONs + prints the WIN/FAIL **verdict** (numpy-only; runs on the DUST head node) |
| `run_queue_d1.sh` | the **DUST queue** (mirrors `seqlgn/run_queue_c0g.sh`): 22 jobs = 3 arms × {blackout, control} × 3 seeds + a′ ceiling (3 seeds) + calibration; round-robin over 2 GPUs, `--tag` skip-resume |
| `test_trainer_mock.py` | the sim-independent gate: wrapper masking proofs, encoder/discretizer checks, teacher/policy reset proofs, end-to-end 3-round DAgger with decreasing loss |
| `test_gate_eval.py` | the aggregator gate: synthesizes fake run JSONs, asserts grouping/means, the WIN and FAIL verdicts, and newest-wins de-dup (numpy-only) |

## The observation-dropout wrapper (why it is built this way)

* Masked frames are replaced by a **CONSTANT ZERO sentinel + valid=0** — never
  freeze-last-value, which would make the wrapper itself the memory and void the
  memory-required claim.
* The wrapper owns its **own seeded rng** (recorded per run; per-episode seeds are
  derived as `seed + episode_index` so every mask sequence replays exactly).
* Students only ever see the masked 13-dim view (12 kinematic dims + valid bit),
  thermometer-encoded. The teacher reads the **privileged true state**
  (`env.true_state()`, 20-dim) — that is the point of distillation.
* Blackout knobs: hover default `p_start=0.02, k∈[6,12]` at 30 Hz = 0.2–0.4 s
  blackouts, ~4–5 per 240-step episode. Raise until the feedforward arm visibly
  degrades — that separation IS the gate's discriminative condition. Fallback
  variant: mask only position+velocity (`mask_dims`), keep attitude "IMU" alive.

## How to run

### CPU smoke, mock env (any machine, main repo python, no simulator)

```bash
# from the repo root
python -m mlgn.flightgate.test_trainer_mock          # the correctness gate (~2 min)
python -m mlgn.flightgate.cli --backend mock --arm gated --rounds 3 \
    --episodes 6 --eval-episodes 6 --iters 120 --hidden 40 --device cpu --tag smoke
# flip --arm clatch (add --anneal 0.1,0.6) and --arm ff for the other arms
```

### Install the real sim (isolated venv — NEVER into the main environment)

pybullet 3.2.7 has **no Windows wheels and no cp312/cp313 wheels**; the local
machine's native Python 3.13.1 cannot run it. Local real-sim work goes through
**WSL Ubuntu** (python3.10 → prebuilt manylinux wheel); real runs go to **DUST**.

```bash
# inside WSL, from /mnt/c/Users/malco/projects/difflogic
python3 -m venv .venv-flight            # WSL python3.10 (already created)
.venv-flight/bin/pip install "numpy>=2.2,<3" "scipy>=1.15" "gymnasium==1.2.*" \
    "pybullet==3.2.7" pillow "matplotlib>=3.10" "transforms3d>=0.4" "control>=0.10.2"
.venv-flight/bin/pip install "setuptools<81"   # BaseAviary imports pkg_resources,
                                               # removed in setuptools>=81 (hit 2026-07-12)
cd ~ && git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && git rev-parse HEAD \
    | tee /mnt/c/Users/malco/projects/difflogic/.venv-flight/GPD_COMMIT
/mnt/c/Users/malco/projects/difflogic/.venv-flight/bin/pip install -e . --no-deps
# torch (CPU) for closed-loop smoke inside the venv (DUST uses its CUDA torch):
.venv-flight/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Notes: `stable_baselines3` (an upstream dependency) is deliberately **omitted** —
it is only needed for gym-pybullet-drones' own RL examples, not for this harness.
`.venv-flight/` is gitignored. On DUST (Linux, py3.10/3.11) use the same pins in
the image's environment or a venv; **record the GPD commit hash**.

### Real-sim smoke (WSL venv)

```bash
# teacher-only collection + quantization gate (no torch needed):
wsl -e bash -lc 'cd /mnt/c/Users/malco/projects/difflogic && \
  .venv-flight/bin/python -m mlgn.flightgate.cli --backend hover --teacher-only \
  --episodes 3 --eval-episodes 2 --blackout 0.02,6,12 --tag wsl_smoke'

# tiny closed-loop distillation round (needs torch in the venv):
wsl -e bash -lc 'cd /mnt/c/Users/malco/projects/difflogic && \
  .venv-flight/bin/python -m mlgn.flightgate.cli --backend hover --arm gated \
  --rounds 2 --episodes 2 --eval-episodes 2 --iters 30 --batch-size 4 \
  --hidden 216 --bptt 60 --blackout 0.02,6,12 --device cpu --tag wsl_smoke'
```

### Full gate (DUST) — the queue + the table

The whole D1 gate is one queue script + one aggregator. On DUST, `python` MUST have
**both** torch+CUDA **and** gym-pybullet-drones @ `e712698a` (same process: student
training needs torch, the closed-loop sim needs pybullet — install the README pins into
the image env or a venv; record `.venv-flight/GPD_COMMIT`).

```bash
cd ~/work/difflogic && mkdir -p logs
nohup bash mlgn/flightgate/run_queue_d1.sh > logs/queue_d1.log 2>&1 &
tail -f logs/queue_d1.log
# read the three-arm gate table + WIN/FAIL verdict (any time; tolerates missing cells).
# The d1 glob EXCLUDES stale/mock smokes in results/ — only this campaign's runs:
python -m mlgn.flightgate.gate_eval --results-dir mlgn/flightgate/results \
    --glob 'flightgate_hover_*_d1_*.json'
python -m mlgn.flightgate.gate_eval --results-dir mlgn/flightgate/results \
    --glob 'flightgate_hover_*_d1_*.json' --json-out logs/gate_d1.json  # machine-readable
```

The queue is 22 jobs: 3 arms (`gated`/`clatch`/`ff`, **all 1,728 gates @ hidden 432 —
verified `utils.count_gates`**) × {blackout-ON, no-blackout CONTROL} × 3 seeds, plus the
**a′ ceiling** (3 seeds, blackout) and one `--teacher-only` velocity-range calibration
run. `--tag` makes it resumable (a job whose results JSON exists is SKIPPED). Sizes:
`--hidden 432 --episodes 50 --eval-episodes 50 --iters 2000 --rounds 4 --bptt 0`,
blackout `0.02,6,12` (0.2–0.4 s bursts @ 30 Hz). Edit `GPUS=(0 1)` / the `JOBS` array
to change the campaign.

**Gate WIN** = under blackout, recurrent (gated and/or clatch) closed-loop return >> ff
by more than the cross-seed spread, **and** ff ≈ recurrent under the no-blackout control
(the gap is memory, not capacity). `gate_eval` prints this verdict. **FAIL** → workmap
D1 pre-committed pivot ("verified logic flight controller").

## Correctness gates (house discipline — run them, in order)

1. **Mock gate** — `python -m mlgn.flightgate.test_trainer_mock` must PASS.
2. **Teacher sanity (hover)** — teacher-only rollout from default spawn must reach
   and hold `‖pos − [0,0,1]‖ < 0.05` within ~3 s (validates install + the RPM→action
   inversion). Check `eval_teacher` in the run JSON.
3. **Quantization gate** — teacher actions replayed through the action bins must
   still solve the task (`quantization_gate_return_ratio > 0.9` in the JSON, warned
   otherwise). This bounds the student ceiling honestly; more `--n-bins` if it fails.
4. **Calibration** — hover velocity thermometer ranges are placeholders; run
   `--teacher-only --episodes 50` and freeze `calibrated_ranges_preview` into a
   config before any real training run.
5. **Non-vacuity control** — every gate table needs the `--no-blackout` arm where
   feedforward must MATCH the recurrent cells.

## Honest caveats / not-yet-built

* **Velocity thermometer ranges are placeholders** until calibrated (gate 4).
  First WSL evidence (3 jittered teacher episodes, 2026-07-12): angular-rate dim
  `wz` reached a calibrated preview of ≈ [−7.1, +12.0] rad/s — far outside the ±3
  placeholder (yaw-jitter recovery transients). Calibrate on ≥50 episodes.
* **Jitter TEMPERED (deliverable 4, FIXED 2026-07-12).** The recon-§6 jitter (yaw
  U(−π,π), 0.15 rad tilt, z U(0.6,1.4)) drove the RAW PID out of the |roll|,|pitch|>0.4
  envelope — **measured 0.65 exit-rate over 20 WSL episodes** (mean length 102/240),
  which breaks the a′ ceiling and the quantization gate. The default is now
  `env.HOVER_JITTER` (x,y U(−0.10,0.10), z U(0.85,1.15), roll,pitch U(−0.05,0.05),
  yaw U(−π/4,π/4)): **re-measured 2026-07-12 via the shipped cli** (`--teacher-only
  --eval-episodes 20 --no-blackout`) at the three queue seeds — **exit-rate 0.05–0.10
  (seeds 0/1/2 = 0.10/0.05/0.10), raw return ≈438–458** (quantized 418–461, ratio
  ≥0.95). The raw PID now holds the envelope in ~90–95 % of episodes (up from ~35–50 %),
  which is what keeps the quantization gate and the a′ ceiling meaningful. (An earlier
  draft over-stated this as a perfect 0.00 exit / 480 return; the honest reproducible
  residual is the ~0.05–0.10 above — a small fraction of episodes still tip past the
  ±0.4 rad envelope.) Old values kept as `env.HOVER_JITTER_RECON6` (ablation); the
  active jitter is recorded per run in `record["hover_jitter"]`.
* **The a′ ceiling arm IS implemented (deliverable 3).** `--teacher-masked` evals the
  PID fed the SAME masked obs the student sees (`MaskedObsHoverTeacher`). The obs→state
  shim is **VERIFIED bit-exact on unmasked frames** (masked-teacher action == privileged
  action to 0.0 when fed the float64 state-derived obs); the only residual is the
  **float32 precision of the KIN observation** the student itself consumes
  (`obs == float32(state_derived)`, ~3e-7 on the normalized action) — faithful, not an
  approximation. On a masked frame the wrapper zeroes the obs, so the oracle flies from
  a zeroed belief — that IS the occlusion cost (smoke: raw 480 → a′ 143, −336 return).
* **BPTT windows are episode prefixes** (zero initial state = deployment-consistent);
  `--bptt` truncates from the front. Random mid-episode windows are deliberately not
  offered: a window starting inside a blackout has ill-posed targets.
* **Determinism is per-machine**, not cross-host (pybullet builds differ) — hence
  every rollout is saved (`--save-traj`) so datasets and gate numbers replay exactly.
* Mock quantization note: the mock teacher's smooth small actions quantize to
  coarse bins (5 by default) — its quantized-return ratio can dip below the 0.9
  warning line; that is the warning system working, not a harness bug. The hover
  task with 9 bins is the configuration the gate actually uses.
