# Architecture / 아키텍처

FACTOR performs precise pick-and-place from a single fixed RGB camera by separating *what is
known from models* (robot kinematics, object CAD, a fiducial board) from *what must be
learned* (the residual between the believed and the true scene, and the behaviour in contact).

FACTOR 는 고정 RGB 카메라 한 대로 정밀 픽앤플레이스를 수행하기 위해, 모델로 아는 것(로봇
운동학, 물체 CAD, 피듀셜 보드)과 학습해야 하는 것(믿는 장면과 참 장면의 잔차, 접촉 중 행동)을
분리한다.

## Layers / 층

```
 task plugin      symbolic planner, nominal placements            tasks/hanoi
 ─────────────────────────────────────────────────────────────────────────────
 skills           pick / place behaviour, contact search          skills, planning
 perception       canonical views, state estimator, fusion        perception
 scene            shared specification (robot, camera, fiducials) scene
 ─────────────────────────────────────────────────────────────────────────────
 twin             labelled data, contact simulation               sim
 robot            streaming control, safety envelope              real
```

### Scene / 장면

`SceneSpec` is the single source of truth for both worlds: intrinsics of the fixed camera, the
fiducial layout that defines the **anchor frame**, the robot URDF and tool offset, and every
manipulated object with its asset and grasp opening. The believed anchor pose in the robot base
frame comes from fiducial calibration; its error is what perception estimates.

### Perception / 인식

* **Canonical view** (`perception.canonical`): a virtual camera that shares the real camera's
  centre but looks at a chosen anchor point with the anchor z axis up. Real images are mapped by
  a rotation homography; the twin reproduces the view by aiming its camera. Two scales are used:
  a coarse view of the workspace and a fine view zoomed on the active target.
* **Kinematic render channels**: the robot URDF rendered at the measured joint angles into the
  same virtual camera (mask and depth). They tell the estimator what is occluding what.
* **State estimator** (`perception.model`): canonical RGB + render channels → anchor pose
  correction `(dx, dy, dyaw)`, interaction state (free / between fingers / grasped / seated /
  jammed) and a log-variance. Trained with supervision from the twin, where the sampled anchor
  error and the contact state are known.
* **Fusion** (`perception.fusion`): per-frame corrections are averaged over time and latched
  while the tool occludes the target.

### Skills / 스킬

`PickPlaceSkill` applies the fused correction to the nominal object and goal positions, runs
the **funnel planner** (approach at hover height, descend inside a lateral funnel, final descent
only when aligned) and the **contact search** (NUDGE: shift the goal through a small pattern when
insertion stalls; when the pattern is exhausted, lift to hover so the correction can refresh and
retry). This rule-based skill is the initialisation for reinforcement-learning
fine-tuning in the contact twin, where the correct behaviour under contact is not known a priori.

The fine-tuned part is a **residual policy** (`factor.skills.residual`): a small MLP that adds up
to ±2 mm to each planner step from a 15-dimensional state available on the robot as well (grip
point relative to the corrected object and goal, gripper command, insertion flag, heights, contact
search index and stall count, previous residual). Its mean head is zero-initialised, so the
untrained policy *is* the rule-based skill. `factor.skills.ppo` is a self-contained PPO; the robot
loop runs the exported policy without torch (`factor.skills.residual_np`, `factor-run --skill`).

### Twin / 트윈

`sim.contract.TwinContract` defines what any simulator must provide: reset / step over many
environments and `Frame`s carrying the canonical RGB, render channels, the correction label, the
interaction-state label, the grip point and the holding flag. `Randomisation` lists the sampled
quantities. The Isaac Lab implementation is being migrated from the research repository.

### Robot / 로봇

`real.robot.Robot` and `real.robot.Gripper` are the only interfaces the skill loop uses:
joint state, grip point, inverse kinematics for the grasp posture, and a streaming joint target.
`real.safety.WorkspaceBox` clamps every commanded grip point before inverse kinematics.

## Robustness layer / 강건성 층 (2026-10-16)

- **Belief as input.** The canonical view is anchored by the fiducial detection, so it shows the
  true scene; the robot's belief enters as proprioception (believed grip point) and as a belief
  render channel (the believed finger segment, optionally both fingers, drawn into the view).
  The estimator reads "real fingers vs drawn belief" — no camera calibration involved.
- **Foveation.** The view is centred on the active peg (source while picking, goal while
  holding) through the detected anchor.
- **Session estimation and probing** (`perception/session.py`, `RealLoop.probe`). The belief
  error is a constant of the session. Before acting, the robot looks at the source and goal pegs
  from five hover poses, fuses the estimates with inverse-variance weights (model log-variance +
  ensemble spread + a noise floor), rejects outliers, and acts on the fused value. A move whose
  fused standard error stays above `--max-stderr` (0.8 mm) is refused, not executed.
- **Ensembles.** The estimator server accepts comma-separated checkpoints (same resolution,
  channels, view width, proprio input and label convention), averages them and reports spread.
- **Local labels** (`--local_correction`). The correction is the translation that is exact at the
  peg in view, yaw folded in; probing fuses per peg.
- **Height.** `--anchor_z` adds a height belief error (anchor z, tool length) and a fourth output.
- **Hindsight self-labelling** (`real/recorder.py`, `perception/finetune_real.py`). With
  `factor-run --record DIR` every successful move writes its frames with labels derived from
  where the fingers closed (pick) and where the ring went down the peg (place); the fine-tuning
  script adapts a local-label checkpoint to real images with an L2 pull to the simulation
  weights. No human labelling.

## Learning plan / 학습 계획

1. Estimator: supervised on twin frames (labels = sampled anchor error, contact state).
2. Skills: rule-based planner and NUDGE as initialisation; PPO fine-tuning in the contact twin
   with the estimator in the loop (reward: success, insertion time, contact force).
3. Evaluation: success over anchor error × camera placement × appearance randomisation in the
   twin, then the same grid on the robot.

## Running the twin / 트윈 실행

Inside the Isaac Lab Python environment, with the package installed (`pip install -e .`):

```
# teacher rollout: must be ~100 %; --nominal gives the rule-only baseline
python -m factor.sim.isaaclab.check --scene configs/hanoi.yaml --assets <repo with assets/usd> \
    --num_envs 32 --steps 300 --headless [--nominal] [--no_mask] [--save out_dir]

# supervised estimator training (frames from the twin, labels = sampled belief error)
python -m factor.perception.train --scene configs/hanoi.yaml --assets <repo with assets/usd> \
    --num_envs 96 --steps 1000 --dagger_rounds 1 --out estimator.pt --headless

# PPO fine-tuning of the residual skill with the frozen estimator in the loop
python -m factor.sim.isaaclab.train_skill --scene configs/hanoi.yaml --assets <repo with assets/usd> \
    --estimator estimator.pt --num_envs 128 --iterations 150 --out skill.pt --headless
```

The skill trainer prints the rule-skill baseline first (the zero-initialised policy), then the
training success every five iterations and a deterministic evaluation every `--eval_every`.

Relative asset paths in `extras.sim` resolve against `--assets`. The robot USD, ring USDs and the
fiducial mat currently live in the research repository; they will move into this repository once
their mesh references are made relative.

