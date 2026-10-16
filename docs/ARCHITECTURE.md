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

## Results so far / 현재 결과

**Caveat (2026-10-15 evening).** Every number below was produced with the twin camera on the wrong
side of the mat: the real camera stands beyond the mat (anchor +y) and looks back at the robot,
the configuration had it at −y, and the twin ignored the anchor yaw (≈180°). Canonical views
were therefore mirrored between simulation and robot. The twin now honours the anchor yaw and the
scene file carries the calibrated camera pose; the simulation study (ablations, skill rules,
occlusion heads) remains valid as a controlled experiment, but the checkpoints are not
deployable and are being retrained.

Package twin, 2026-10-15 (a 4×RTX 4090 server, 128 envs, 800 + 2x300 steps, 60/30 epochs):

| run | anchor error | mask channel | res | round 0 | best |
|-----|--------------|--------------|-----|---------|------|
| pc_mask | ±8 mm, ±1° | yes | 128 | 93.0 % | 94.0 % |
| pc_nomask | ±8 mm, ±1° | no | 128 | 93.0 % | 94.0 % |
| pc_mask_e10 | ±10 mm, ±2° | yes | 128 | 91.1 % | 92.9 % |
| pc_mask_r160 | ±8 mm, ±1° | yes | 160 | 93.7 % | 95.1 % |
| ab_nolatch | ±8 mm, ±1° | yes | 128 | 62.6 % | 62.6 % |
| ab_nonudge | ±8 mm, ±1° | yes | 128 | 83.8 % | 86.3 % |
| mild_e6 | ±6 mm, ±0.5°, cam ±2 cm | yes | 128 | 95.1 % | 95.7 % |
| long_r160 | ±8 mm, ±1° | yes | 160 | 95.0 % | 95.0 % |
| off_mask | ±8 mm, ±1° | yes | 128 | 93.4 % | 94.8 % |
| off_nomask | ±8 mm, ±1° | no | 128 | 94.1 % | 94.6 % |
| off_e6 | ±6 mm, ±0.5°, cam ±2 cm | yes | 128 | 95.8 % | 95.8 % |
| off_r160 | ±8 mm, ±1° | yes | 160 | 95.2 % | 95.2 % |
| ho02_mask (in-hand ±4 mm twin, offset head 0.02 m, offset used) | ±8 mm, ±1° | yes | 128 | 96.1 % | 96.1 % |
| ho02_nomask (same, no mask) | ±8 mm, ±1° | no | 128 | 95.9 % | 96.9 % |

Camera randomisation ±3 cm, appearance gain ±40 %, bias ±0.08, noise 0.05. Teacher 100 %,
rule-only (no correction) 29.6 % at ±8 mm. Single-factor sweep of the no-mask checkpoint
(48 envs x 400 steps, belief fixed per episode): full 96.1 %, no camera randomisation 97.1 %,
no yaw 96.8 %, no appearance 97.7 %, all off 98.1 %, ±4 mm 99.5 % — the remaining failures are
time-outs on large-error episodes, not a single randomisation factor. Ablations: removing the
height latch drops success to 62.6 % and removing the contact search to 86.3 %, so both the
occlusion handling and the insertion search carry the result. The mask channel in its
current form (binary robot silhouette) does not change the outcome; the occlusion-aware
estimator planned next (ring-to-gripper offset head, expected-render channel) is the real test.

Ring-to-grip offset head (`off_*` runs, label = ring centre minus grip point in the believed
frame, kinematic attach so the held label was still zero): validation RMS of the held-ring
offset 3.2 mm with the robot-mask channel, 6.5 mm without (4.5 mm in the mild setting); free-ring
RMS about 31 mm in all three (the free range is ±100 mm). The mask channel is the first thing
that moves this number, which is the occlusion case. The twin now attaches a ring with a
physical in-hand offset (the pre-grasp error along the finger axis is kept, across it the fingers
centre the ring) and held-start episodes sample ±3 mm, so the question was whether the
estimated offset, used when placing (`--use_offset_estimate`), recovers the success lost to it.
Teacher check (24 envs, 250 steps): with ±3 mm held-start offsets 100 % both with and without
the offset, with ±5 mm 99.3 % without and 99.2 % with: the contact search absorbs in-hand
offsets of this size. Trained in the in-hand-offset twin with the head rescaled to 0.02 m
(`ho02_mask`), the held-ring offset RMS stays at 2.6–2.8 mm, which is the spread of the labels
themselves (uniform ±4 mm along the finger axis), i.e. the head does not recover the in-hand
offset from a 128 px view under the gripper. Using it in the planner lowers success (same
checkpoint, 96 envs x 600 steps, ±8 mm/1°, in-hand ±4 mm): 96.8 % without the offset estimate,
95.4 % with it. Without the mask channel the same run reaches 96.9 % but its state head is less accurate
(0.907 vs 0.935) and its held-offset RMS worse (3.5 vs 2.7 mm): the robot-mask channel helps the
two occlusion heads consistently, without moving success. Conclusion for this task: the in-hand
offset is handled by contact search and retry, not by perception; the occlusion-relevant perception outputs are the anchor correction
while the gripper hides the scene, and the grasp verdict.

Occlusion overlay videos (`factor.sim.isaaclab.render_video`, 4 environments tiled, the
estimated ring outline in green against the true outline in blue while the gripper hides the
ring): `logs/factor/overlay_pc_mask.mp4` and `logs/factor/overlay_off_e6.mp4` in the research
repository; the estimate stays within 0.5–2.5 mm of the truth through the grasp.

Where the failures are: with the `check` time-out breakdown, every time-out of the trained
estimators happens while holding, i.e. during insertion — the correction latched below the
hover height is off by more than the contact search could reach. Two rule changes in the skill
layer (both in the twin and in the robot skill) fix most of it:

| estimator, setting | before | + outer search ring (±4.5 mm) | + lift-and-retry |
|--------------------|--------|-------------------------------|------------------|
| off_e6, ±6 mm/0.5°, cam ±2 cm | 95.0 % (18 time-outs) | 97.5 % (8) | **99.5 %** (0, 2 bad releases) |
| pc_mask, ±8 mm/1°, cam ±3 cm | 96.2 % (11) | 96.2 % (11) | **97.9 %** (6) |
| off_mask, ±8 mm/1°, cam ±3 cm | 95.7 % (14) | — | — |

(64 envs x 400 steps, local; teacher stays at 100 %.) Larger samples with both rules: pc_mask at
±8 mm/1° with 14 s episodes 99.6 % (516 episodes); off_e6 at ±6 mm 98.3 % (909 episodes) before
and 97.6 % (862 episodes, all 21 failures insertion time-outs, no bad releases) after the peg entry
check was moved from the grip target to the held ring (the in-hand offset now has to clear the peg
physically, which is what the real ring does). With a 14 s episode budget (the robot
has no 8 s limit) and both rules: ho02_mask at ±8 mm/1°, cam ±3 cm, in-hand ±4 mm **99.9 %**
(999/1000), off_e6 at ±6 mm/0.5°, cam ±2 cm **99.9 %** (1147/1148, one drop). Lift-and-retry:
when the whole search pattern has been tried, the held object rises to the hover height, waits five steps so the
correction filter can refresh above the latch, and descends again with the updated estimate.

Residual skill fine-tuning (PPO, 64 envs, 150 iterations of 32 steps, frozen pc_mask
estimator, ±2 mm residual): rule-skill baseline 94.7 %, evaluations at 25-iteration intervals
95.8 / 92.7 / 94.9 / 92.9 / 95.0 / 94.9 % (about 310 episodes each, so ±1.3 % standard error).
On the server (128 envs, frozen off_e6, mild setting) the same picture: baseline 94.6 %, then
95.1 / 93.7 / 94.8 % at iterations 50/100/150 with more bad releases, stopped. No significant
gain in either run: the remaining time-outs are perception errors that a ±2 mm residual without
image input cannot repair in 8 s. The two rule changes above (wider search, lift-and-retry)
recovered far more than the learned residual, so the skill layer stays rule-based for now and the
RL path is kept as an option with a search-rewarding objective.

Grasp verification from the state head (`holding_from_estimate`): with the pc_mask checkpoint,
8 s episodes 96.4 % (vs 95.6 % with the gripper command as the holding proxy) but 14 s episodes
93.9 % (vs 97.6 %), with new bad releases — the 93 % accurate state head produces false
"not held" verdicts that make the planner drop a held ring. Kept opt-in until the state head is
more reliable (temporal smoothing, ring-offset head).

