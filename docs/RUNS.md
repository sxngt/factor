# Experiment registry / 실험 기록

Every training run, where its artefacts live, and under which version of the twin it was made.
Paper material: keep this file in sync with the launches (the training-server launcher in `robot-arm/tools/`).

모든 학습 run 의 기록. 아티팩트 위치와 어떤 트윈 버전에서 만들었는지. 논문 자료이므로 런치와 함께 갱신한다.

Artefacts (training server, a 4×RTX 4090 machine, in `robot-arm/logs/factor/` of its checkout; mirrored
on the workstation in a per-server subdirectory of `robot-arm/logs/factor/`): `<run>.log` (full training log), `<run>.pt` (best checkpoint),
`<run>/figs/*.png` (canonical-view grids: `start_*`, `teacher_end_*`, `round<k>_{rgb,clean,mask,overlay}`).

## Twin versions / 트윈 버전

| version | commits | camera | note |
|---------|---------|--------|------|
| A (mirrored) | up to `8114f3b` (2026-10-15 15:3x) | at anchor (−0.006, −0.397, 0.707), anchor yaw ignored | Camera on the wrong side of the mat; canonical views mirrored w.r.t. the robot. Valid as a controlled simulation study; checkpoints not deployable. |
| B (corrected) | from `1dee5ec` (2026-10-15 17:00) | at anchor (+0.072, +0.391, 0.711) = calibrated, anchor yaw −178.96° honoured | Matches the real camera (marker id 5 top-centre in both). Physical in-hand offset, peg entry on the held ring, offset head 0.02 m. |

Common training recipe unless noted: 128 envs, teacher collection 800 steps, 2 DAgger rounds of
300 steps, 60 + 2×30 epochs, batch 512, evaluation 400 steps with belief resampling off,
8 s episodes, appearance gain ±40 %, bias ±0.08, noise 0.05, robot base ±1 cm / ±3°.
"Success" = evaluation success of the best round (episodes ≈ 400–750).

## Server runs / 서버 run

| run | date | twin | anchor err | cam | mask | res | envs | extra | round 0 / 1 / 2 | best | note |
|-----|------|------|-----------|-----|------|-----|------|-------|-----------------|------|------|
| pc_mask | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 128 | 128 | — | 93.0 / 94.0 / — | 94.0 % | batch 1 reference |
| pc_nomask | 10-15 | A | ±8 mm, ±1° | ±3 cm | no | 128 | 128 | — | 93.0 / 94.0 / — | 94.0 % | mask channel ablation |
| pc_mask_e10 | 10-15 | A | ±10 mm, ±2° | ±3 cm | yes | 128 | 128 | — | 91.1 / 92.9 / — | 92.9 % | harder errors |
| pc_mask_r160 | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 160 | 128 | — | 93.7 / 95.1 / — | 95.1 % | resolution |
| ab_nolatch | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 128 | 128 | `--latch_below 0` | 62.6 / — | 62.6 % | ablation: no height latch |
| ab_nonudge | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 128 | 128 | `--nudge 0` | 83.8 / 86.2 / 86.3 | 86.3 % | ablation: no contact search |
| mild_e6 | 10-15 | A | ±6 mm, ±0.5° | ±2 cm | yes | 128 | 128 | — | 95.1 / 95.7 / 94.2 | 95.7 % | mild condition |
| long_r160 | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 160 | 128 | 1500 steps, 100 epochs | 95.0 / 94.1 / 93.0 | 95.0 % | longer training |
| off_mask | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 128 | 128 | offset head (0.1 m) | 93.4 / 93.5 / 94.8 | 94.8 % | held-offset RMS 3.2 mm (label 0) |
| off_nomask | 10-15 | A | ±8 mm, ±1° | ±3 cm | no | 128 | 128 | offset head (0.1 m) | 94.1 / 94.6 / 93.6 | 94.6 % | held-offset RMS 6.5 mm |
| off_e6 | 10-15 | A | ±6 mm, ±0.5° | ±2 cm | yes | 128 | 128 | offset head (0.1 m) | 95.8 / 95.5 / 95.6 | 95.8 % | held-offset RMS 4.5 mm; used for the overlay video |
| off_r160 | 10-15 | A | ±8 mm, ±1° | ±3 cm | yes | 160 | 96 | 1200 steps, 80 epochs | 95.2 / 95.0 / 92.7 | 95.2 % | |
| ho02_mask | 10-15 | A + in-hand offset ±4 mm | ±8 mm, ±1° | ±3 cm | yes | 128 | 128 | offset head 0.02 m, `--use_offset_estimate` | 96.1 / 95.1 / 94.4 | 96.1 % | held-offset RMS 2.7 mm = label spread |
| ho02_nomask | 10-15 | A + in-hand offset ±4 mm | ±8 mm, ±1° | ±3 cm | no | 128 | 128 | same | 95.9 / 96.9 / 95.3 | 96.9 % | state acc 0.907 vs 0.935 with mask |
| skill_off_e6 | 10-15 | A | ±6 mm, ±0.5° | ±2 cm | (frozen off_e6) | 128 | 128 | residual PPO, 400 it planned | baseline 94.6; it 50/100/150: 95.1 / 93.7 / 94.8 | — | stopped at it 150: no gain |
| dep_e6_nomask | 10-15 18:06 | B | ±6 mm, ±0.5° | ±2 cm | no | 128 | 128 | — | 98.9 / 97.9 / 98.9 | 98.9 % | deployment candidate, mild |
| dep_e8_nomask | 10-15 18:08 | B | ±8 mm, ±1° | ±3 cm | no | 128 | 128 | — | 98.4 / 97.9 / 98.6 | 98.6 % | deployment candidate, robust |
| dep_e8_mask | 10-15 18:10 | B | ±8 mm, ±1° | ±3 cm | yes | 128 | 128 | — | 98.1 / 96.8 / 97.9 | 98.1 % | mask-channel counterpart |
| dep_e8_r160_nomask | 10-15 18:12 | B | ±8 mm, ±1° | ±3 cm | no | 160 | 96 | 1000 steps, 80 epochs | 98.5 / 96.9 / 97.6 | 98.5 % | resolution |

## Local runs / 로컬 run (RTX 4080 16 GB)

| run | date | twin | setting | result | artefacts |
|-----|------|------|---------|--------|-----------|
| skill_pc_mask | 10-15 | A | residual PPO, 64 envs, 150 it, frozen pc_mask | baseline 94.7; evals 95.8 / 92.7 / 94.9 / 92.9 / 95.0 / 94.9 | `robot-arm/logs/factor/skill_pc_mask.{pt,log}` |
| yaw_local_e6 | 10-15 17:12 | B | ±6 mm/0.5°, cam ±2 cm, no mask, 64 envs, 800 + 300 steps, 40 + 20 epochs | 98.5 % (used for the first real moves, see SIM2REAL.md) | `robot-arm/logs/factor/yaw_local_e6.{pt,log}`, figs `yaw_local_e6_figs/` — first real-robot candidate |

## Evaluation-only numbers / 평가 전용 수치

`factor.sim.isaaclab.check` on a saved checkpoint (local, twin A unless noted). See
`ARCHITECTURE.md` for the discussion.

| checkpoint | setting | episodes | success | note |
|------------|---------|----------|---------|------|
| off_e6 | ±6 mm, 8 s | 382 | 95.0 % | 18 time-outs, all while holding (insertion) |
| off_e6 | same, + outer search ring | 400 | 97.5 % | |
| off_e6 | same, + lift-and-retry | 420 | 99.5 % | |
| off_e6 | same, 909 episodes | 909 | 98.3 % | 9 bad releases (grip-target entry check) |
| off_e6 | same, ring-based entry | 862 | 97.6 % | 21 time-outs, 0 bad releases |
| off_e6 | same, 14 s episodes | 1148 | 99.9 % | |
| pc_mask | ±8 mm/1°, cam ±3 cm, 8 s | 342 | 96.2 % | before retry |
| pc_mask | same, + retry | 341 | 97.9 % | |
| pc_mask | same, 14 s | 516 | 99.6 % | |
| ho02_mask | ±8 mm/1°, in-hand ±4 mm, 8 s | 748 | 96.8 % | offset estimate not used |
| ho02_mask | same, offset estimate used | 698 | 95.4 % | |
| ho02_mask | same, 14 s | 1000 | 99.9 % | |
| teacher | any | — | 100 % | twin A and B |

