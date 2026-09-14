"""Live telemetry of the real-robot loop. / 실기 루프 라이브 텔레메트리.

Every tick the loop hands over its signals (fiducial pose, raw and fused corrections, grasp
probability, skill phase, contact-search state, commanded target) and the images involved. This
module draws what the system *believes* over the camera frame and over the canonical view, and
writes everything to a directory that :mod:`factor.real.viewer` serves in a browser:

- ``camera.jpg``   the raw frame with detected markers, the anchor axes, the believed scene
  geometry (plate, pegs, every ring) and the corrected object / goal / held-ring estimate;
- ``canonical.jpg`` the estimator input with the believed and corrected outlines, like the
  simulation overlay videos;
- ``state.json``   the latest tick; ``log.jsonl`` every tick appended.

매 틱 루프가 넘겨주는 신호(앵커 포즈, 원시·융합 보정, 파지 확률, 스킬 단계, 접촉 탐색 상태, 목표)와
영상을 받아, 시스템이 믿고 있는 기하와 추정 기하를 원본 프레임·정규화 시야 위에 그리고, 뷰어가
읽는 디렉터리에 기록한다. Overlays reuse the drawing code of the simulation videos so that sim and
real are read the same way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from factor.perception.canonical import VirtualCamera
from factor.perception.overlay import (
    COLOUR_ESTIMATE,
    COLOUR_GOAL,
    camera_projector,
    canonical_projector,
    draw_peg,
    draw_polyline,
    draw_ring,
    label,
)

COLOUR_BELIEF = (0, 200, 255)  # believed (nominal) geometry, BGR / 믿는 기하
COLOUR_GRIP = (255, 255, 255)
COLOUR_MARKER = (255, 120, 0)


@dataclass
class SceneGeometry:
    """Static geometry to draw, in the anchor frame. / 앵커 프레임의 정적 기하."""

    plate: np.ndarray  # (4, 3) corners of the plate top / 판 윗면 모서리
    pegs: list[np.ndarray]  # peg base points / 페그 밑점
    peg_height: float
    rings: list[tuple[np.ndarray, float, float]] = field(default_factory=list)  # (centre, radius, height)


@dataclass
class MoveGeometry:
    """Geometry of the move in progress, in the anchor frame. / 진행 중인 수의 기하."""

    object_center: np.ndarray
    goal_center: np.ndarray
    radius: float
    height: float
    name: str = ""


class Telemetry:
    """Draw and publish one tick. / 한 틱을 그려서 기록한다.

    Parameters
    ----------
    directory
        Output directory (created). The viewer watches it.
    camera_matrix, distortion
        Real camera intrinsics for the frame overlay.
    virtual
        Canonical camera, for the canonical overlay.
    anchor_in_base
        Believed anchor pose; base-frame points are mapped back to the anchor frame for drawing.
    frame_scale
        Downscale factor of the published camera frame (the raw frame is large).
    """

    def __init__(
        self,
        directory: str,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        virtual: VirtualCamera,
        anchor_in_base: np.ndarray,
        frame_scale: float = 1.0 / 3.0,
        look_at: np.ndarray | None = None,
    ):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.distortion = np.asarray(distortion, dtype=float)
        self.virtual = virtual
        self.anchor_in_base = np.asarray(anchor_in_base, dtype=float)
        self.base_in_anchor = np.linalg.inv(self.anchor_in_base)
        self.frame_scale = frame_scale
        self.look_at = np.array([0.0, 0.0, 0.037]) if look_at is None else np.asarray(look_at, dtype=float)
        self.scene: SceneGeometry | None = None
        self.move: MoveGeometry | None = None
        self._log = open(os.path.join(directory, "log.jsonl"), "a", encoding="utf-8")
        self._t0 = time.time()

    # ------------------------------------------------------------------ frames
    def to_anchor(self, point_base: np.ndarray) -> np.ndarray:
        return self.base_in_anchor[:3, :3] @ np.asarray(point_base, dtype=float) + self.base_in_anchor[:3, 3]

    # ------------------------------------------------------------------ publish
    def publish(
        self,
        tick: dict,
        frame_rgb: np.ndarray | None,
        canonical_rgb: np.ndarray | None,
        anchor_in_camera: np.ndarray | None,
        marker_corners=None,
    ) -> None:
        """Write ``state.json``, the log line and the two overlay images. / 상태·로그·영상 기록."""
        tick = {"t": round(time.time() - self._t0, 3), **tick}
        if frame_rgb is not None and anchor_in_camera is not None:
            frame = self._draw_frame(frame_rgb, anchor_in_camera, tick, marker_corners)
            self._write_image("camera.jpg", frame)
        if canonical_rgb is not None and anchor_in_camera is not None:
            view = self._draw_canonical(canonical_rgb, anchor_in_camera, tick)
            self._write_image("canonical.jpg", view)
        self._log.write(json.dumps(tick) + "\n")
        self._log.flush()
        self._write_text("state.json", json.dumps(tick))

    def close(self) -> None:
        self._log.close()

    # ------------------------------------------------------------------ drawing
    def _draw_frame(self, rgb: np.ndarray, anchor_in_camera: np.ndarray, tick: dict, corners) -> np.ndarray:
        import cv2

        s = self.frame_scale
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        k = self.camera_matrix.copy()
        k[:2] *= s
        project = camera_projector(anchor_in_camera, k, self.distortion)

        if corners is not None:
            for c in corners:
                draw_polyline(image, np.asarray(c).reshape(-1, 2) * s, COLOUR_MARKER, thickness=2)
        axes = project(np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], dtype=float))
        for i, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            draw_polyline(image, axes[[0, i + 1]], colour, closed=False, thickness=2)

        if self.scene is not None:
            draw_polyline(image, project(self.scene.plate), COLOUR_BELIEF)
            for base in self.scene.pegs:
                draw_peg(image, project, base, self.scene.peg_height, COLOUR_BELIEF)
            for centre, radius, height in self.scene.rings:
                draw_ring(image, project, centre, radius, height, COLOUR_BELIEF)
        self._draw_move(image, project, tick)
        label(
            image, f"t {tick.get('t', 0):.1f}s  step {tick.get('step', 0)}  {tick.get('phase', '')}", (6, 16)
        )
        label(
            image,
            "orange: markers  yellow: believed scene  green: corrected object/goal  white: grip",
            (6, 32),
        )
        return image

    def _draw_canonical(self, rgb: np.ndarray, anchor_in_camera: np.ndarray, tick: dict) -> np.ndarray:
        import cv2

        scale = 4
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        camera_in_anchor = np.linalg.inv(anchor_in_camera)
        base = canonical_projector(camera_in_anchor[:3, 3], self.look_at, self.virtual)

        def project(points):
            return base(points) * scale

        if self.scene is not None:
            for peg in self.scene.pegs:
                draw_peg(image, project, peg, self.scene.peg_height, COLOUR_BELIEF)
        self._draw_move(image, project, tick)
        corr = tick.get("correction_mm_deg", [0, 0, 0])
        label(
            image, f"correction {corr[0]:+.1f} {corr[1]:+.1f} mm {corr[2]:+.2f} deg", (6, 16), COLOUR_ESTIMATE
        )
        label(image, f"grasped p {tick.get('grasped', 0):.2f}   {tick.get('phase', '')}", (6, 32))
        return image

    def _draw_move(self, image: np.ndarray, project, tick: dict) -> None:
        m = self.move
        if m is None:
            return
        draw_ring(image, project, m.object_center, m.radius, m.height, COLOUR_BELIEF)
        draw_ring(image, project, m.goal_center, m.radius, m.height, COLOUR_GOAL)
        obj_c = tick.get("object_corrected_anchor")
        goal_c = tick.get("goal_corrected_anchor")
        if obj_c is not None and not tick.get("holding", False):
            draw_ring(image, project, np.asarray(obj_c), m.radius, m.height, COLOUR_ESTIMATE)
        if goal_c is not None:
            draw_ring(image, project, np.asarray(goal_c), m.radius, m.height, COLOUR_ESTIMATE)
        grip = tick.get("grip_anchor")
        if grip is not None:
            grip = np.asarray(grip, dtype=float)
            px = project(grip.reshape(1, 3))[0]
            draw_polyline(image, np.array([px + [-6, 0], px + [6, 0]]), COLOUR_GRIP, closed=False)
            draw_polyline(image, np.array([px + [0, -6], px + [0, 6]]), COLOUR_GRIP, closed=False)
            if tick.get("holding", False):
                # The ring the system believes it holds, drawn at the grip point (occluded by the tool).
                # 들고 있다고 믿는 링을 그립점에 그린다 (툴에 가려진 추정 물체).
                draw_ring(image, project, grip, m.radius, m.height, COLOUR_ESTIMATE)

    # ------------------------------------------------------------------ files
    def _write_image(self, name: str, image: np.ndarray) -> None:
        import cv2

        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        path = os.path.join(self.directory, name)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data.tobytes())
        os.replace(tmp, path)

    def _write_text(self, name: str, text: str) -> None:
        path = os.path.join(self.directory, name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
