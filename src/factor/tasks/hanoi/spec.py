"""Tower of Hanoi geometry. / 하노이 탑 기하.

All positions are expressed in the anchor frame: origin at the middle peg axis on the base-plate
underside, x along the row of pegs, z up. The skill layer transforms them with the believed
anchor pose; the perception layer corrects that pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PEGS = ("left", "middle", "right")
PEG_INDEX = {"left": -1, "middle": 0, "right": 1}


@dataclass(frozen=True)
class Ring:
    """A ring, largest first in :attr:`HanoiLayout.rings`. / 링(큰 것부터)."""

    name: str
    radius: float
    height: float


@dataclass(frozen=True)
class HanoiLayout:
    """Peg row and ring set. / 기둥 열과 링 집합.

    ``plate_thickness`` is the base plate; rings rest on top of it. ``peg_height`` is measured
    from the plate top.
    """

    peg_spacing: float
    peg_height: float
    plate_thickness: float
    rings: tuple[Ring, ...]
    peg_row_offset: tuple[float, float] = (0.0, 0.0)  # middle peg relative to the mat origin / 중앙 페그 보정
    hover_clearance: float = 0.030
    place_clearance: float = 0.002

    @property
    def peg_top(self) -> float:
        return self.plate_thickness + self.peg_height

    @property
    def zone_top(self) -> float:
        """Height below which a held ring is inside the peg zone. / 기둥 구간 윗면."""
        return self.peg_top + self.rings[0].height / 2.0

    @property
    def hover_height(self) -> float:
        """Grip-point height when travelling between pegs. / 이동 시 그립점 높이."""
        return self.peg_top + self.hover_clearance + self.rings[0].height / 2.0

    @classmethod
    def from_scene(cls, scene) -> HanoiLayout:
        """Build the layout from a scene's ``extras.hanoi`` and object extents. / 장면 명세에서 생성."""
        h = scene.extras["hanoi"]
        rings = tuple(
            Ring(name, scene.object(name).extents[0] / 2.0, float(h["ring_height"])) for name in h["rings"]
        )
        return cls(
            peg_spacing=float(h["peg_spacing"]),
            peg_row_offset=tuple(float(v) for v in h.get("peg_row_offset", (0.0, 0.0))),
            peg_height=float(h["peg_height"]),
            plate_thickness=float(h["plate_thickness"]),
            rings=rings,
            hover_clearance=float(h.get("hover_clearance", 0.030)),
            place_clearance=float(h.get("place_clearance", 0.002)),
        )

    def ring(self, name: str) -> Ring:
        for r in self.rings:
            if r.name == name:
                return r
        raise KeyError(name)

    def peg_xy(self, peg: str) -> np.ndarray:
        """Peg axis in the anchor frame. / 앵커 프레임의 페그 축 위치.

        ``peg_row_offset`` is a fixed calibration of the tower relative to the printed mat (the
        tower is attached to the mat, so this is measured once, not randomised).
        """
        return np.array([PEG_INDEX[peg] * self.peg_spacing + self.peg_row_offset[0], self.peg_row_offset[1]])

    def ring_center(self, peg: str, level: int) -> np.ndarray:
        """Centre of the ring at ``level`` (0 = on the plate) on ``peg``. / 기둥 위 ``level`` 번째 링 중심."""
        h = self.rings[0].height
        return np.array([*self.peg_xy(peg), self.plate_thickness + h * (level + 0.5)])

    def goal_center(self, peg: str, level: int) -> np.ndarray:
        """Where a ring should be released to land at ``level``. / 놓기 목표(여유 포함)."""
        return self.ring_center(peg, level) + np.array([0.0, 0.0, self.place_clearance])
