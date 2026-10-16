# Results / 결과

Every number in the paper draft (kept outside this repository until review) comes from a file in this folder or
from `docs/RUNS.md`, which records each training run. Large artefacts (checkpoints, recorded real
frames, videos) are not in git. Their locations are listed at the end.

논문 초안의 모든 수치는 이 폴더의 파일이나 학습 기록인 `docs/RUNS.md` 에서 나온다.
큰 산출물(체크포인트, 실기 프레임, 영상)은 git 에 없으며 위치는 아래에 적었다.

## Tables / 표

| paper table | content | source |
|---|---|---|
| Table 2 | deployed 5-model ensemble, 14 s episodes, ±8 mm / 1°, camera ±8 cm: 99.9 % (1453 episodes) | `docs/RUNS.md`, "Final ensemble" |
| Table 3 | robustness matrix: anchor error × camera displacement, learned vs rule only | `raw/eval/matrix_f1_e6_warm.txt`, `raw/eval/matrix_local_ensemble_gen2.txt` |
| Table 4 | ablation over 3 seeds: full 98.4 %, no proprioception 97.9 %, no belief channel 89.1 % | `docs/RUNS.md`, "Ablation of the final architecture" |
| Table 5 | skill rules: outer search ring and lift-and-retry; RL residual gives no gain | `docs/ARCHITECTURE.md`, "Results so far" |
| Table 6 | active self-calibration, 20 synthetic seeds: active 0.23 mm, random 0.31 mm, blind 0.45 mm at 60 s | `raw/sim_compare_20seeds.txt` |
| Table 7 | real robot, 2026-10-16: per-move outcome and time | `real_moves_20261016.csv` (built by `build_real_table.py` from `raw/real/*.log`) |

## Real robot, 2026-10-16 / 실기

The anchor was set once by centring the gripper on the start peg. Moves ran with the estimator
correction off (`--no-correction`), grasp verification off, and NUDGE off in the bold run.
앵커는 시작 기둥에 그리퍼를 한 번 맞춰 정했다. 추정기 보정과 파지 검증은 껐고, 대담 실행에서는
NUDGE 도 껐다.

| run | settings | attempted | seated | mean time per move |
|---|---|---|---|---|
| single (14:48) | cautious | 1 | 1 | — |
| cautious (14:50) | step 3 mm, insert lead 4 mm, grip wait 2 s, NUDGE on | 6 | 5 (1 max_steps) | 104.8 s (n = 4) |
| bold (15:04) | step 10 mm, insert lead 12 mm, grip wait 0.8 s, NUDGE off | 11 | 11 | 65.3 s (n = 9) |

The bold run stopped at an emergency stop before move 12. It was not a motion failure.
Per-step logs: `raw/real/real_20261016_*.json.gz` (grip point, holding, estimator outputs per
tick). Regenerate the table:

```
python results/build_real_table.py > results/real_moves_20261016.csv
```

## Not in git / git 밖

| artefact | where |
|---|---|
| checkpoints (`*.pt`) | training server and the research workstation, `logs/factor/` |
| recorded real frames with hindsight labels (`move_*.npz`, 175 MB) | `logs/factor/real_data/` |
| robot videos (`factor_*.mp4`) | research workstation, `logs/cam/videos/` |
