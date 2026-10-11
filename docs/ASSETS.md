# Simulation assets / 시뮬레이션 에셋

The Isaac Lab twin (`factor.sim.isaaclab`) loads three kinds of USD files. They are not in this
repository: the robot is derived from vendor meshes that we do not redistribute, and the rings
and the mat are generated per setup. This page lists what the twin needs, where it comes from
and how the scene file points at it.

Isaac Lab 트윈은 USD 세 종류를 읽는다. 로봇은 재배포하지 않는 제조사 메시에서 만들고, 링과 매트는 셋업마다
생성하므로 이 저장소에 없다. 무엇이 필요한지, 어디서 오는지, 장면 파일이 어떻게 가리키는지 정리한다.

## How the scene file points at assets / 장면 파일의 에셋 경로

`configs/hanoi.yaml`, section `extras.sim`:

```yaml
extras:
  sim:
    robot_usd: assets/usd/fr3_ag95.usd
    ring_usds: [assets/usd/rings/ring0.usd, ..., assets/usd/rings/ring5.usd]
    mat_usd: assets/usd/mat.usd
```

Relative paths are resolved against the **assets root** given on the command line
(`--assets <dir>` of `factor.sim.isaaclab.check`, `factor.perception.train`,
`factor.sim.isaaclab.train_skill`, `render_video`); absolute paths are used as they are
(`factor.sim.isaaclab.twin.settings_from_scene`). Keep your assets anywhere and pass the
directory that contains `assets/usd/...`, or edit the paths.

상대 경로는 명령행의 `--assets <dir>` 기준으로 풀리고 절대 경로는 그대로 쓰인다.

The table, the tower plate and the three pegs are built from primitives in the twin (dimensions
from `extras.hanoi`), so they need no files.
테이블·탑 판·기둥 3개는 트윈이 기본 도형으로 만든다(치수는 `extras.hanoi`).

## Robot: FR3 + AG-95 / 로봇

| item | requirement |
|------|-------------|
| file | `assets/usd/fr3_ag95.usd`, an articulation with root prim `/fr3_ag95` |
| joints | `j1`..`j6` (revolute) then `ag95_left_finger_joint`, `ag95_right_finger_joint` (prismatic, 0..0.0475 m = half gap), both driven |
| links | `base_link` … `wrist3_link`, `flange`, `ag95_body`, the two fingers, `tcp` |
| tool frame | `tcp` = flange + (0, 0, `tool_length`), the frame used by `factor.robots.fr3.kinematics` |

Source: **`assets/urdf/fr3_ag95.urdf`** in this repository. It combines

* the FAIRINO FR3 arm (`fairino3_v6`) from FAIRINO's ROS 2 package **frcobot_ros2**
  (`fairino_description`), with the joint origins replaced by the controller-fitted DH
  parameters of `robot.kinematics` in the scene file, the controller soft limits and a `flange`
  link;
* a simplified **DH-Robotics AG-95**: the body mesh from DH-Robotics' **dh_gripper_ros**
  (`dh_robotics_ag95_description`), a box collision body and two prismatic fingers with box pads.
  The vendor gripper model is a mimic-linkage tree that PhysX articulations do not handle well;
  the real fingers' arc is ignored and the pads sit at the nominal tool length.

The URDF references meshes as `package://fairino_description/meshes/...` and
`package://dh_robotics_ag95_description/meshes/...`; the meshes themselves are **not**
included. To build the USD:

1. Get the two description packages from the vendors (FAIRINO `frcobot_ros2`, DH-Robotics
   `dh_gripper_ros`, at versions whose mesh file names match the URDF).
2. Make the `package://` URIs resolvable, either
   * build them in a ROS 2 workspace (`colcon build`, then `source install/setup.bash`) so that
     `ros2 pkg prefix` finds them, or
   * write a local copy of the URDF with absolute paths, e.g.
     `sed "s#package://fairino_description#$FAIRINO_DESC#g; s#package://dh_robotics_ag95_description#$AG95_DESC#g" assets/urdf/fr3_ag95.urdf > /tmp/fr3_ag95.urdf`.
3. Import with Isaac Sim's URDF importer (fixed base, position drives on all eight joints,
   import the mimic finger as an independently driven joint), save as
   `assets/usd/fr3_ag95.usd` with default prim `/fr3_ag95`.
4. Check the joint order and the `tcp` pose against `factor.robots.fr3.kinematics` (the
   forward kinematics should agree to about 0.1 mm), e.g. with
   `python -m factor.sim.isaaclab.check --assets <root> --num_envs 4 --headless` (teacher close
   to 100 %).

메시는 포함하지 않는다. 제조사 패키지(FAIRINO `frcobot_ros2`, DH-Robotics `dh_gripper_ros`)를 받아
`package://` 를 풀 수 있게 한 뒤(ROS 워크스페이스 빌드 또는 절대 경로 사본) Isaac Sim URDF
임포터로 `/fr3_ag95` USD 를 만들고, 기구학과 tcp 가 일치하는지 확인한다.

## Rings / 링

Six rigid bodies, `ring0` (largest) … `ring5`, one USD each. Outer diameters and thickness come
from `objects[].extents` in the scene file (50 … 25 mm, 7 mm), masses from `objects[].mass`.
The hole must clear the 10 mm pegs with the real clearance, and the collision shape must keep
the hole open (convex decomposition or an explicit ring of convex pieces; a single convex hull
fills the hole). Semantic class `ring` is not required; the twin tags the prims itself.

링 6개(ring0 가장 큼). 외경·두께·질량은 장면 파일 값. 충돌 형상은 구멍을 막지 않아야 한다
(볼록 분해 또는 볼록 조각 링; 단일 볼록 껍질은 구멍을 메운다).

