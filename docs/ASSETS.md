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

