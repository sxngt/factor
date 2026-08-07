"""Real-robot interfaces. / 실기 인터페이스.

Robot-specific adapters (for example FAIRINO FR3 over ROS 2) implement :class:`Robot`; the
skills and perception layers only see these interfaces. :class:`WorkspaceBox` is the safety
envelope every commanded grip point must stay inside.
"""

from factor.real.fiducial import FiducialDetection, FiducialDetector, anchor_in_base, calibrate_camera_in_base
from factor.real.robot import Gripper, Robot
from factor.real.safety import WorkspaceBox

__all__ = [
    "FiducialDetection",
    "FiducialDetector",
    "Gripper",
    "Robot",
    "WorkspaceBox",
    "anchor_in_base",
    "calibrate_camera_in_base",
]
