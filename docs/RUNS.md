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

## 12 h campaign (2026-10-15 20:07 → 10-16 08:10) / 12 시간 캠페인

Runner: `robot-arm/logs/factor/campaign/campaign.sh`, queue `queue.txt`, log `results.txt`.
Twin version C: camera aimed at the true anchor (§10 of SIM2REAL), proprioception + belief
render channel (§11), aim jitter ±1.5 mm, blur 30 %, camera ±8 cm, 60 mm pegs, spiral search.
From commit "foveated canonical view" (≈21:00) the view centres on the active peg (version D).

| run | twin | setting | rounds 0/1/2/3 | best |
|-----|------|---------|----------------|------|
| c1_bel_e6 | C | ±6 mm/0.5°, belief | 97.5 / 96.7 / 98.1 / … | |
| c1_bel_e8 | C | ±8 mm/1°, belief | 95.3 / 97.0 / 97.4 / 96.7 | 97.4 % (14 s: 98.8 %) |
| c1_bel_e8_r160 | C | 160 px | 95.4 / 94.9 / 94.6 / 94.8 | 95.4 % |
| c1_both_e8 | C | belief + true mask (sim-only) | 95.4 / 96.8 / 96.1 / … | |
| c1_bel_e10_cam10 | C | ±10 mm/1.5°, camera ±10 cm | running | |
| c1_bel_e8_proprioless | C | belief channel, no proprio vector | running | |

Foveated view (twin D), 14 s evaluation at ±8 mm / 1° / camera ±8 cm, 96 envs × 800 steps (local, 22:55):

| checkpoint | trained at | view | episodes | success |
|------------|-----------|------|----------|---------|
| f1_e6_warm | ±6 mm, warm from c1_bel_e6 | 14 cm | 1273 | **99.5 %** — deployed as `robot-arm/logs/factor/factor_best.pt` |
| f1_e8_warm | ±8 mm, warm from c1_bel_e8 | 14 cm | 1267 | 99.1 % |
| f1_e8_vw10 | ±8 mm | 10 cm | 1144 | 98.3 % |

Server OOM at 21:40 (four runs with 1600 + 3×500-step buffers); f1_both_e8_warm killed, f1_e10_cam10_warm stopped.
Runner v2 caps concurrency and tracks GPUs by memory.

Later campaign runs, same 14 s ±8 mm test (local, ≈1250–1320 episodes each):

| checkpoint | what | success |
|------------|------|---------|
| diag_yaw02_from_best | trained with yaw error ±0.2° (label diagnostic) | 99.3 % |
| kp_e6 | spatial soft-argmax keypoint encoder, from scratch | 98.9 % |

Findings: the ~1.3 mm near-peg RMS floor is the same for both encoders and does not move when the
yaw error (and with it the source-vs-goal label inconsistency) shrinks — it is per-frame image
noise (≈1.2 px), averaged down by the correction filter in closed loop. Warm-started generations
beyond f1 plateau (g2_e6_best 98.6 %). Next lever: ensembles (server supports comma-separated
checkpoints, reports spread).

### Robustness matrix (paper table) / 강건성 행렬

Best model `f1_e6_warm` (twin D: true-centre aim, belief channel + proprio, foveated 14 cm view),
14 s episodes, 64 envs × 600 steps per cell, appearance randomisation at training strength.
Raw output: `robot-arm/logs/factor/eval/matrix_f1_e6_warm.txt`.

| anchor error (xy, yaw) | camera ±3 cm | camera ±8 cm | camera ±12 cm | rule only, camera ±8 cm |
|------------------------|--------------|--------------|---------------|-------------------------|
| ±4 mm, ±1° | 99.6 % (743) | 99.3 % (679) | 99.9 % (700) | 98.2 % (546) |
| ±6 mm, ±1° | 99.4 % (684) | 99.8 % (664) | 98.7 % (606) | 86.3 % (271) |
| ±8 mm, ±1° | 98.7 % (541) | 98.5 % (530) | 97.2 % (469) | 69.8 % (179) |
| ±10 mm, ±1° | 96.7 % (394) | 95.8 % (404) | 94.0 % (350) | 54.1 % (135) |

(episodes in parentheses; fewer episodes = slower episodes, the rule-only baseline spends its time
in the contact search.)

### Ensembles (14 s, ±8 mm, 96 envs × 800 steps) / 앙상블

| ensemble | members | episodes | success |
|----------|---------|----------|---------|
| local label | loc_s5, loc_s6, loc_s8_app20 | 1435 | **99.7 %** — deployed (`factor_loc1..3.pt`, tools/factor_real.sh) |
| standard label | f1_e6_warm, ens_s1, ens_s2 | 1352 | 99.3 % |

More episodes in the same time = faster moves. Local-label runs at ±8 mm also beat standard ones
(loc_e8_s9 97.9 % vs f1_e8_warm 96.5 %; loc_e10 96.4 % vs h1_e10 95.9 % at ±10 mm).
| local label, gen 2 | loc_s5_gen2, loc_s6_gen2, loc_s8_app20 | 1425 | **99.8 %** — deployed (replaces gen 1, 03:30) |

Warm-start generations of the local-label line keep lowering the near-peg error (x: 1.30 → 0.97
(loc_s5) → 0.88 mm (loc_s6_gen2)); the finger silhouette (fingers_from_s6, 0.94 mm) and the height
output (z3_*, height at chance) do not help.

### Robustness matrix, deployed ensemble / 배포 앙상블 강건성 행렬

`factor_loc1..3` (loc_s5_gen2, loc_s6_gen2, loc_s8_app20; local labels), 14 s, 64 envs × 600 steps.
Raw: `robot-arm/logs/factor/eval/matrix_local_ensemble_gen2.txt`.

| anchor error | camera ±3 cm | camera ±8 cm | camera ±12 cm |
|--------------|--------------|--------------|---------------|
| ±4 mm | 99.8 % (806) | 99.9 % (745) | 99.7 % (763) |
| ±6 mm | 99.9 % (709) | 99.6 % (701) | 99.7 % (687) |
| ±8 mm | 98.9 % (635) | 99.5 % (606) | 97.7 % (570) |
| ±10 mm | 95.1 % (410) | 95.6 % (428) | 96.2 % (399) |

Compared with the single best model of the earlier table, every cell up to ±8 mm is equal or
better and 10–15 % more episodes complete in the same time (faster moves). ±10 mm lies outside the
members' training range (±6 mm); loc_e10* models trained at ±10 mm reach 96.4–96.5 % alone.

### Ablation of the final architecture (paper table) / 최종 구조 어블레이션

From scratch, identical recipe (local labels, ±6 mm/0.5°, camera ±10 cm, appearance ×1.3,
1200 + 3×400 steps, 60 + 3×30 epochs), three seeds each (40, 41, 42). Near-peg RMS from the last
validation epoch; success = best training evaluation (8 s episodes). Mean ± std over seeds.

| variant | near-peg x | near-peg y | success |
|---------|-----------|-----------|---------|
| full (belief channel + proprioception) | 1.41 ± 0.04 mm | 1.66 ± 0.05 mm | **98.4 ± 0.4 %** |
| without proprioception vector | 1.43 ± 0.05 mm | 2.00 ± 0.07 mm | 97.9 ± 0.3 % |
| without belief render channel | 3.67 ± 0.07 mm | 2.20 ± 0.13 mm | 89.1 ± 0.3 % |

The belief render channel carries the result (x error ×2.6, success −9.3 points without it);
the proprioception vector mainly adds precision in y, the camera's depth direction. Warm-start
generations bring the full model further (loc_s6_gen2: x 0.88 mm), see above.

### Final ensemble (06:00) / 최종 앙상블

| ensemble | members | episodes | success |
|----------|---------|----------|---------|
| polished 3 | final_s5, final_s6, loc_s8_app20 | 1395 | 99.4 % |
| **5-model** | loc_s5_gen2, loc_s6_gen2, final_s5, final_s6, loc_s8_app20 | 1453 | **99.9 %** — deployed (`factor_loc1..5.pt`) |

14 s, ±8 mm/1°, camera ±8 cm, 96 envs × 800 steps. One time-out and one bad release in 1453
episodes; the most episodes completed in the same time of any configuration.

Last polishing runs (training evaluation, 8 s): final_s8 99.2 %, final_s14 99.0 %, final_s10 98.9 %,
final_s15 98.8 %, final_e8 98.4 %, final_e8b 98.1 %, final_e10 96.5 % (±10 mm). Campaign ended 06:40
(queue drained; runner exits at 08:10). Spare ensemble members for the robot: final_s8, final_s14.

## Robot-recolour runs (2026-10-16 12:51 → 13:29) / 로봇 재색칠 run

All four use `--local_correction --robot_recolor`. The warm starts are 20 epochs each; rc_scratch
is 30 epochs. Training-time evaluation, ±8 mm:

| run | init | recolor / appearance | near-peg RMS x,y (mm) | best success |
|---|---|---|---|---|
| rc_s5 | final_s5 | 0.6 / 1.5 | 0.89, 1.41 | 99.2 % |
| rc_s6 | final_s6 | 0.6 / 1.5 | 0.97, 1.49 | 98.7 % |
| rc_s8 | loc_s8_app20 | 0.8 / 2.0 | 0.88, 1.48 | 99.0 % |
| rc_scratch | — | 0.8 / 2.0 | 1.35, 1.74 | 98.3 % |

Real-frame belief response is the change of the estimate per mm of belief shift. It was measured on
4 live frames with the grip 95 mm above the plate (13:35):

| model | x-shift → x | y-shift → y |
|---|---|---|
| factor_loc1–5 (deployed) | −0.01 | 0.39 |
| rc_s5 / rc_s6 / rc_s8 | 0.11 / −0.09 / 0.02 | 0.47 / 0.72 / 0.65 |
| rc_scratch | 0.13 | 0.60 |
| rc_s5+s6+s8 | 0.01 | 0.62 |

y responds better after recolouring, and x still does not. The test height is far from the pegs,
so it is not decisive; it should be repeated with the grip near a peg. The deployment stays loc1–5.

## Self-calibration, synthetic (2026-10-16) / 자기보정 합성 실험

See SIM2REAL.md §13 for the table: active 0.23 mm, random 0.31 mm, blind 0.45 mm at 60 s (20 seeds).
