"""Simulation twin. / 시뮬 트윈.

The twin is built from a :class:`factor.scene.SceneSpec` and provides what the learning layers
need: labelled canonical views for the estimator (the anchor pose error is sampled and therefore
known), kinematic render channels, and a contact environment for skill fine-tuning.

The builder depends on Isaac Lab, which is not installable from PyPI. Import it explicitly:

    from factor.sim.isaaclab import TwinBuilder

The reference implementation lives in the research repository and is being migrated here; see
``docs/ARCHITECTURE.md`` for the contract it must satisfy.

트윈은 장면 명세에서 생성되며, 추정기용 라벨 있는 정규화 시야(앵커 포즈 오차를 샘플링하므로 정답을
안다), 운동학 렌더 채널, 스킬 미세조정용 접촉 환경을 제공한다. Isaac Lab 의존이라 명시적으로
import 한다.
"""

from factor.sim.contract import TwinContract

__all__ = ["TwinContract"]
