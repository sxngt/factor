"""Hindsight self-labelling from successful real moves. / 성공한 실기 동작에서 사후 라벨 생성.

A successful move reveals where things really were: when the fingers close on the ring the grip
point *is* the ring centre (within the finger clearance), and when a ring drops onto its peg the
grip point *is* above the peg axis. In the believed anchor frame, the difference between those
points and the nominal ones is exactly the local correction an estimator with
``label="local"`` should have predicted while looking at that peg. Every frame of the move is
saved with that label, so the robot collects real training data without any human labelling.

성공한 수는 실제 위치를 알려준다: 핑거가 닫힐 때의 그립점이 링 중심이고, 링이 기둥에 들어갈 때의
그립점이 기둥 축이다. 그 점과 공칭 점의 차이가 국소 보정의 정답이므로, 그 수의 모든 프레임을
사람 라벨 없이 학습 데이터로 저장한다.
"""

from __future__ import annotations

import os
import time

import numpy as np


class MoveRecorder:
    """Collect frames of one move and write them with hindsight labels. / 한 수의 프레임 기록."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.images: list[np.ndarray] = []
        self.grips: list[np.ndarray] = []
        self.phase: list[int] = []  # 0 = looking at the object's peg, 1 = at the goal peg
        self.grasp_point: np.ndarray | None = None
        self.release_point: np.ndarray | None = None
        self.verified = False  # grasp confirmed held by the vision check / 파지 확인됨

    def frame(self, image: np.ndarray, grip_in_anchor: np.ndarray, holding: bool) -> None:
        self.images.append(np.asarray(image, dtype=np.uint8))
        self.grips.append(np.asarray(grip_in_anchor, dtype=np.float32))
        self.phase.append(int(holding))

    def grasped(self, grip_in_anchor: np.ndarray) -> None:
        self.grasp_point = np.asarray(grip_in_anchor, dtype=float)
        self.verified = False  # until the vision check confirms it / 비전 확인 전까지

    def released(self, grip_in_anchor: np.ndarray) -> None:
        self.release_point = np.asarray(grip_in_anchor, dtype=float)

    def save(
        self, object_nominal: np.ndarray, goal_nominal: np.ndarray, seated: bool, tag: str = ""
    ) -> str | None:
        """Write ``move_<time>.npz`` when the move gives usable labels. / 라벨이 생기면 저장.

        ``*_nominal`` are anchor-frame points; labels are metres in anchor axes.
        """
        if not self.images or self.grasp_point is None or not self.verified:
            return None  # no confirmed grasp: nothing trustworthy to label / 확인된 파지가 없으면 저장 안 함
        labels = np.full((len(self.images), 2), np.nan, dtype=np.float32)
        pick = (self.grasp_point - np.asarray(object_nominal, dtype=float))[:2]
        labels[np.array(self.phase) == 0] = pick
        if seated and self.release_point is not None:
            place = (self.release_point - np.asarray(goal_nominal, dtype=float))[:2]
            labels[np.array(self.phase) == 1] = place
        path = os.path.join(self.directory, f"move_{time.strftime('%Y%m%d_%H%M%S')}{tag}.npz")
        np.savez_compressed(
            path,
            images=np.stack(self.images),
            grips=np.stack(self.grips),
            phase=np.array(self.phase, dtype=np.int8),
            labels=labels,
            seated=np.array(seated),
        )
        return path
