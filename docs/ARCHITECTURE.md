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

