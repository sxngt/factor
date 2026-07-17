"""Scene specification shared by the simulation twin and the real robot.

장면 명세. 시뮬 트윈과 실기가 같은 명세를 읽어 같은 기하·피듀셜·카메라 규약을 쓴다.
"""

from factor.scene.spec import (
    CameraSpec,
    FiducialLayout,
    FiducialMarker,
    ObjectSpec,
    RobotSpec,
    SceneSpec,
    load_scene,
)

__all__ = [
    "CameraSpec",
    "FiducialLayout",
    "FiducialMarker",
    "ObjectSpec",
    "RobotSpec",
    "SceneSpec",
    "load_scene",
]
