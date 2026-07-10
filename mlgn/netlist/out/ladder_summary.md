# Curriculum-ladder verification study (2026-07-10)

All 9 combo checkpoints rebuilt exactly (full-test-set accuracy gate: 9/9 at 1.0000 == recorded 1.0).
settled = legal writes reaching a fixed point (of 8); cyc = inputs (of 512) in limit cycles; hold/anyx0 =
settle-then-hold-forever theorems (tempor recipe); decode = forever-correct-readout theorem (protocol_decode
with in-netlist GroupSum head; s1 + c50A_c35 runs, each independently bmc3-clean 131+ frames).
Raw artifacts: ladder_<tag>/ dirs (BLIFs, ABC logs, report.json). cp50A_curr_c35 = out/ckpt_cp50A_curr_c35.

| ckpt | seed | L | disc acc | settled | max settle | cyc (periods) | hold | anyx0 | decode |
|---|---|---|---|---|---|---|---|---|---|
| cp4_curr_s1_c8 | 1 | 8 | 1.0000 | 3/8 | 12 | 48 [2, 6] | CEX@13 | CEX@13 | PROVED |
| cp4_curr_s1_c20 | 1 | 20 | 1.0000 | 6/8 | 15 | 28 [2, 6] | CEX@16 | CEX@16 | PROVED |
| cp4_curr_s1_c35 | 1 | 35 | 1.0000 | 3/8 | 12 | 40 [2, 6, 12] | CEX@13 | CEX@13 | PROVED |
| cp4_curr_s2_c8 | 2 | 8 | 1.0000 | 8/8 | 13 | 0 [] | PROVED | PROVED | not run |
| cp4_curr_s2_c20 | 2 | 20 | 1.0000 | 8/8 | 15 | 0 [] | PROVED | PROVED | not run |
| cp4_curr_s2_c35 | 2 | 35 | 1.0000 | 8/8 | 14 | 0 [] | PROVED | PROVED | not run |
| cp50A_curr_c8 | 0 | 8 | 1.0000 | 7/8 | 12 | 48 [2, 3, 6] | CEX@13 | CEX@13 | not run |
| cp50A_curr_c20 | 0 | 20 | 1.0000 | 8/8 | 14 | 32 [6, 12] | PROVED | CEX@15 | not run |
| cp50A_curr_c35 | 0 | 35 | 1.0000 | 8/8 | 15 | 0 [] | PROVED | PROVED | PROVED |

## Reading

1. **Three solution families, same 1.0000 discrete accuracy, distinguishable only by verification.**
   Seed 2 crystallizes fixed points from the first rung (0 cycles anywhere; hold + anyx0 PROVED at every
   rung, 2-5 s each). Seed 0 crystallizes progressively (cycling inputs 48 -> 32 -> 0 along the curriculum;
   hold provable from c20, anyx0 only at c35). Seed 1 NEVER crystallizes: limit cycles of period 2 and 6
   persist through c35 (48 -> 28 -> 40 cycling inputs), so every hold-type theorem is genuinely false there
   (CEX at the first armed frame, sim-confirmed).
2. **The decode theorem is the right spec.** On the oscillator seed, protocol_decode PROVES at every rung
   in seconds: the state never stops moving for 5/8 writes, yet every reachable orbit state decodes to the
   written symbol, forever. Hold-type certificates separate the solution families; the decode-type
   certificate is what all accuracy-1.0 circuits share -- and what a deployment spec should demand.
3. **Settle depth does not shrink along the curriculum** (max settle 12-15 at every rung); what curriculum
   training changes (seeds 0/2) is orbit structure -- limit cycles collapsing into fixed points.
4. The L=8 rungs read out at t=7, BEFORE their own settle (12-13): early-curriculum circuits are transient/
   orbit decoders; their long-delay correctness is nonetheless machine-checkable (decode PROVED at s1_c8).
