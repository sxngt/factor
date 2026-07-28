"""Perception: fiducial-anchored canonical views and model-based state estimation.

인식 층. 피듀셜에 앵커된 정규화 시야를 만들고, 그 위에서 물체 상태(포즈 잔차, 상호작용 상태)를
추정한다. 네트워크 자체는 ``factor.perception.model`` 에 있고 torch 가 있을 때만 import 된다.
"""

from factor.perception.canonical import CanonicalView, VirtualCamera
from factor.perception.fusion import CorrectionFilter

__all__ = ["CanonicalView", "VirtualCamera", "CorrectionFilter"]
