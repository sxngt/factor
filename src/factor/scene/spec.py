"""Scene specification data model.

A scene is everything both worlds must agree on: the robot and its tool frame, the fixed camera,
the fiducial layout that anchors the canonical views, and the manipulated objects with their
geometry. Task plugins extend it with task-specific placements.

장면 명세. 로봇과 툴 프레임, 고정 카메라, 정규화 시야를 앵커하는 피듀셜 배치, 조작 물체의 기하 등
시뮬과 실기가 반드시 공유해야 하는 정보만 담는다. 태스크별 배치는 태스크 플러그인이 확장한다.

Units are metres and radians unless a field name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole camera with lens distortion. / 핀홀 카메라와 렌즈 왜곡.

    ``fx, fy, cx, cy`` are in pixels for the full-resolution image ``width x height``.
    ``dist`` follows the OpenCV ordering (k1, k2, p1, p2, k3, ...).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix. / 3×3 내부 파라미터 행렬."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def distortion(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=float)


@dataclass(frozen=True)
class FiducialMarker:
    """One fiducial marker on the anchor board. / 앵커 보드의 마커 하나.

    ``center`` is the marker centre in the anchor frame; ``size`` is the black-square edge length.
    """

    marker_id: int
    center: Vec3
    size: float


@dataclass(frozen=True)
class FiducialLayout:
    """Planar fiducial board that defines the anchor frame. / 앵커 프레임을 정의하는 평면 피듀셜 보드.

    The anchor frame has its origin at the board centre, z pointing up out of the board.
    ``dictionary`` names the marker family (for example ``DICT_4X4_50``).
    """

    dictionary: str
    markers: tuple[FiducialMarker, ...]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(m.marker_id for m in self.markers)


@dataclass(frozen=True)
class ObjectSpec:
    """A manipulated object. / 조작 물체.

    ``asset`` points to the mesh or USD used by the simulation twin and by the kinematic
    renderer. ``grasp_half_gap`` is the finger half-opening that holds the object.
    """

    name: str
    asset: str
    mass: float
    grasp_half_gap: float
    extents: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RobotSpec:
    """Robot description and tool frame. / 로봇 모델과 툴 프레임.

    ``grip_offset`` is the vector from the tool tip to the grip point (where a held object's
    centre sits), expressed in the tool frame. ``kinematics`` and ``grasp`` are robot-specific
    mappings consumed by the robot package (for example DH parameters and the IK branch).
    """

    name: str
    urdf: str
    joint_names: tuple[str, ...]
    grip_offset: Vec3 = (0.0, 0.0, 0.0)
    kinematics: dict = field(default_factory=dict)
    grasp: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SceneSpec:
    """Complete scene. / 전체 장면.

    ``anchor_in_base`` is the 4x4 pose of the anchor (fiducial) frame in the robot base frame
    as currently believed; the perception layer estimates the residual error of this pose.
    ``camera_in_anchor`` is the nominal camera pose in the anchor frame used to build the
    simulation twin.
    """

    robot: RobotSpec
    camera: CameraSpec
    fiducials: FiducialLayout
    objects: tuple[ObjectSpec, ...]
    anchor_in_base: np.ndarray
    camera_in_anchor: np.ndarray
    extras: dict = field(default_factory=dict)

    def object(self, name: str) -> ObjectSpec:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(name)


def _pose_from_mapping(m: dict) -> np.ndarray:
    """Build a 4x4 pose from ``{xyz: [...], rpy_deg: [...]}``. / 위치·RPY(도) 사전 → 4×4 포즈."""
    xyz = np.asarray(m.get("xyz", (0.0, 0.0, 0.0)), dtype=float)
    roll, pitch, yaw = np.radians(np.asarray(m.get("rpy_deg", (0.0, 0.0, 0.0)), dtype=float))
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = xyz
    return pose


def load_scene(path: str | Path) -> SceneSpec:
    """Read a scene specification from YAML. / YAML 장면 명세를 읽는다.

    See ``configs/hanoi.yaml`` for the schema.
    """
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    cam = doc["camera"]
    camera = CameraSpec(
        width=int(cam["width"]),
        height=int(cam["height"]),
        fx=float(cam["fx"]),
        fy=float(cam["fy"]),
        cx=float(cam["cx"]),
        cy=float(cam["cy"]),
        dist=tuple(float(v) for v in cam.get("dist", ())),
    )

    fid = doc["fiducials"]
    markers = tuple(
        FiducialMarker(
            marker_id=int(m["id"]), center=tuple(float(v) for v in m["center"]), size=float(m["size"])
        )
        for m in fid["markers"]
    )
    fiducials = FiducialLayout(dictionary=str(fid["dictionary"]), markers=markers)

    rob = doc["robot"]
    robot = RobotSpec(
        name=str(rob["name"]),
        urdf=str(rob["urdf"]),
        joint_names=tuple(str(j) for j in rob["joint_names"]),
        grip_offset=tuple(float(v) for v in rob.get("grip_offset", (0.0, 0.0, 0.0))),
        kinematics=dict(rob.get("kinematics", {})),
        grasp=dict(rob.get("grasp", {})),
    )

    objects = tuple(
        ObjectSpec(
            name=str(o["name"]),
            asset=str(o["asset"]),
            mass=float(o["mass"]),
            grasp_half_gap=float(o["grasp_half_gap"]),
            extents=tuple(float(v) for v in o.get("extents", (0.0, 0.0, 0.0))),
        )
        for o in doc.get("objects", ())
    )

    return SceneSpec(
        robot=robot,
        camera=camera,
        fiducials=fiducials,
        objects=objects,
        anchor_in_base=_pose_from_mapping(doc["anchor_in_base"]),
        camera_in_anchor=_pose_from_mapping(doc["camera_in_anchor"]),
        extras=dict(doc.get("extras", {})),
    )
