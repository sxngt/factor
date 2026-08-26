"""Scene specification feeds the task layout and robot parameters. / 장면 명세 → 기하·로봇 파라미터."""

from pathlib import Path

from factor.robots.fr3 import FR3Params
from factor.scene import load_scene
from factor.tasks.hanoi import HanoiLayout

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hanoi.yaml"


def test_layout_and_params_from_scene():
    scene = load_scene(CONFIG)
    layout = HanoiLayout.from_scene(scene)
    assert [r.radius for r in layout.rings] == [0.025, 0.0225, 0.02, 0.0175, 0.015, 0.0125]
    assert abs(layout.hover_height - (0.062 + 0.030 + 0.0035)) < 1e-9
    params = FR3Params.from_mapping(scene.robot.kinematics)
    assert params.a2 == -0.280491 and params.tool_length == 0.2045
    assert scene.robot.grasp["ik_branch"] == [0, 0, 1]
