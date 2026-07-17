"""FACTOR: Fiducial-Anchored Canonical views for calibration-free Transfer Of precise Robot manipulation.

The package is organised by responsibility:

- ``factor.scene``       scene specification shared by simulation and the real robot
- ``factor.perception``  canonical views, temporal fusion, estimator networks
- ``factor.planning``    rule-based motion planning and contact search
- ``factor.skills``      pick / place skills composed from planning and perception
- ``factor.sim``         simulation twin builder (optional dependency on Isaac Lab)
- ``factor.real``        robot, camera and safety interfaces
- ``factor.tasks``       task plugins (Tower of Hanoi ships as the reference task)

패키지는 책임 단위로 나뉩니다. 시뮬과 실기가 공유하는 장면 명세(``scene``), 인식(``perception``),
규칙 기반 계획(``planning``), 스킬(``skills``), 시뮬 트윈(``sim``), 실기 인터페이스(``real``),
태스크 플러그인(``tasks``).
"""

__version__ = "0.1.0"
