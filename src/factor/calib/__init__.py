"""Session self-calibration: tie the camera, the robot and the anchor board into one frame.

세션 자기보정: 카메라·로봇·앵커 보드를 하나의 좌표계로 묶는다.
"""

from factor.calib.handeye import ActiveHandEye, GripperBoard, average_poses

__all__ = ["ActiveHandEye", "GripperBoard", "average_poses"]
