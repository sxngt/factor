# FACTOR

**Fiducial-Anchored Canonical views for calibration-free Transfer Of precise Robot manipulation**

[English](README.md) · **한국어**

**고정된 3인칭 RGB 카메라 한 대**로 밀리미터 정밀도의 집기·놓기를 수행하며, 학습은 모두 시뮬레이션에서
합니다. 작업대의 피듀셜 매트가 시뮬레이터와 실기의 관측을 같은 규약으로 고정하므로, 카메라를 옮겨도
다시 보정할 필요가 없습니다.

## 실기

<p align="center"><img src="docs/media/real_robot.gif" width="100%" alt="하노이 링을 옮기는 FR3, 4배속"></p>
<p align="center"><img src="docs/media/factor_live.gif" width="100%" alt="라이브 뷰: 믿는 장면과 보정 추정, 정규화 시야, 신호, 2배속"></p>
<p align="center"><sub>ArUco 매트 위의 FAIRINO FR3 + AG-95(4배속) · 루프 라이브 뷰: 믿는 장면과 보정된 장면,
추정기가 보는 정규화 시야와 신호(2배속)</sub></p>

<p align="center"><img src="docs/figures/fig1_overview.svg" width="100%" alt="FACTOR 개요"></p>

## 아이디어

- **피듀셜 앵커 정규화 시야.** 검출된 ArUco 매트를 중심으로 회전 호모그래피를 적용해, 카메라가 어디에
  있든 같은 가상 시야를 만듭니다. Isaac Lab 트윈은 가상 카메라를 같은 방식으로 조준합니다.
- **믿음을 입력으로.** 로봇이 *믿는* 그리퍼 위치를 그 시야에 그려 넣고 고유수용 입력으로도 줍니다.
  네트워크는 실제 핑거와 그려진 믿음의 차이를 읽어 앵커 보정량을 출력합니다.
- **잔차만 학습, 동작은 규칙으로.** 학습은 믿음이 어디서 틀렸는지만 맡습니다. 높이 래치와 들었다 다시
  내리기를 갖춘 규칙 기반 깔때기 플래너가 실제 동작을 맡습니다.

## 왜 하노이탑인가

하노이탑은 목표가 아니라 검증용 과제입니다.

- **정밀도를 증명합니다.** 지름 10 mm 기둥과 링 구멍 사이 공차가 **약 1 mm**라서, 모든 놓기가 밀리미터
  수준의 삽입입니다. 결과도 안착이나 걸림 둘 중 하나로 갈립니다.
- **물체를 정의하기 쉽습니다.** 링, 기둥, 받침판이 치수가 정해진 단순 형상이라 URDF·USD 모델이 정확하고,
  트윈이 실물과 일치합니다.
- **작업 정의가 명확합니다.** 상태는 세 개의 스택이고, 합법적인 수와 목표 위치가 규칙으로 정해지며, 성공
  여부가 분명합니다. 기호 계획기가 정확한 목표를 주므로 남는 오차는 모두 인식이나 제어의 오차입니다.

다른 작업도 같은 장면 명세로 플러그인처럼 끼울 수 있습니다(`src/factor/tasks/`).

## 환경

<p align="center"><img src="docs/media/sim_scene.jpg" width="100%" alt="Isaac Lab의 FACTOR 트윈"></p>
<p align="center">
  <img src="docs/media/canonical_views.jpg" width="49%" alt="무작위화된 정규화 시야">
  <img src="docs/media/occlusion_overlay.jpg" width="49%" alt="가림 상태의 추정 윤곽과 실제 윤곽">
</p>
<p align="center"><sub>Isaac Lab의 FACTOR 트윈(FR3 + AG-95, ArUco 매트 위 하노이탑) · 카메라 위치·조준·외관을 무작위화한
정규화 시야 · 그리퍼가 링을 가린 순간의 추정 윤곽과 실제 윤곽</sub></p>

## 구조

<p align="center"><img src="docs/figures/fig3_estimator.svg" width="100%" alt="믿음 조건부 추정기"></p>
<p align="center"><img src="docs/figures/fig5_realloop.svg" width="100%" alt="실기 폐루프"></p>

## 성과

| 조건 | 결과 |
|---|---|
| 트윈, 앵커 오차 ±8 mm, 카메라 이동 ±8 cm | 성공률 **99.9 %**(1,453 에피소드), 규칙만 69.8 % |
| 어블레이션, 시드 3개 | 전체 98.4 %, 믿음 채널 제거 89.1 % |
| 트윈, 카메라 이동 ±12 cm, 앵커 오차 ±6 mm | 99.7 % |
| 능동 자기보정(합성, 60초) | 기둥 위치 오차 0.23 mm |
| FAIRINO FR3 연속 이동 | 11 / 11 안착, 수당 65초 |

실기 영상에서는 학습 보정에 아직 외관 격차가 있어, 실기 이동은 기하 모드로 수행했습니다.
자세한 내용은 [docs/SIM2REAL.md](docs/SIM2REAL.md), 모든 학습 기록은 [docs/RUNS.md](docs/RUNS.md)에 있습니다.

**논문:** arXiv 링크 공개 예정.

## 설치

```
git clone https://github.com/sxngt/factor && cd factor
uv sync --extra dev --extra real --extra perception
uv run pytest -q
```

Isaac Lab(트윈·학습), ROS 2(로봇 드라이버), FAIRINO SDK는 따로 설치합니다.
시뮬레이션 에셋은 [docs/ASSETS.md](docs/ASSETS.md)를 참고하세요.

## 사용법

```
# 시뮬레이션: 추정기 학습과 평가 (Isaac Lab Python 안에서)
python -m factor.perception.train --scene configs/hanoi.yaml --assets <root> --out estimator.pt --headless
python -m factor.sim.isaaclab.check --scene configs/hanoi.yaml --assets <root> --checkpoint estimator.pt --headless

# 로봇: 드라이버, 카메라, 뷰어, 폐루프를 스크립트 하나로
export ROBOT_IP=192.168.58.2 CHECKPOINT=estimator.pt TORCH_PYTHON=/path/to/python-with-torch
scripts/run_real.sh --fake            # 움직임 없음
scripts/run_real.sh --moves 1         # 한 수
```

전체 설치·실행 안내: [docs/USAGE.md](docs/USAGE.md). 설계 노트: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 인용

```bibtex
@software{factor2026,
  author = {sxngt},
  title  = {{FACTOR}: Fiducial-Anchored Canonical views for calibration-free Transfer Of precise Robot manipulation},
  year   = {2026},
  url    = {https://github.com/sxngt/factor}
}
```

MIT 라이선스. 제조사 메시와 SDK는 각자의 라이선스를 따릅니다.
