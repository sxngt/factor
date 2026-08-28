"""Vectorised (torch) counterparts of the planning layer for the simulation twin.

The twin runs the same funnel planner, contact search and correction fusion as the real-robot
skill, but over thousands of environments at once. These helpers mirror
``factor.planning.funnel``, ``factor.planning.nudge`` and ``factor.perception.fusion`` and are
kept side by side with them so that any change is made in both places.

시뮬 트윈용 torch 벡터화 플래너. 실기 스킬과 같은 깔때기·NUDGE·보정 융합을 수천 env 에 대해 한 번에
계산한다. ``factor.planning`` / ``factor.perception.fusion`` 과 짝을 이루므로 한쪽을 바꾸면 다른 쪽도
바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from factor.planning.nudge import PATTERN as NUDGE_PATTERN


@dataclass(frozen=True)
class FunnelTolerances:
    pick_xy: float = 0.003
    place_xy: float = 0.002
    z: float = 0.002
    funnel_xy: float = 0.012
    funnel_z: float = 0.012
    hover_rise: float = 0.010
    freeze_xy_below: float = 0.0075  # above the object centre: fingers are around it, no lateral motion


def funnel_step(
    grip: torch.Tensor,
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    holding: torch.Tensor,
    hover_height: float,
    zone_top: float,
    max_step: float,
    tol: FunnelTolerances | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched funnel planner. / 배치 깔때기 플래너.

    Returns ``(step, close)``: the saturated grip-point displacement ``(N, 3)`` and the desired
    gripper state ``(N,)``. Frames must match between all position arguments.
    """
    tol = tol or FunnelTolerances()
    target_pos = torch.where(holding.unsqueeze(1), goal_pos, object_pos)
    tol_xy = torch.where(
        holding, torch.full_like(grip[:, 0], tol.place_xy), torch.full_like(grip[:, 0], tol.pick_xy)
    )
    hover = torch.full_like(grip[:, 0], hover_height)

    lateral = torch.norm(grip[:, :2] - target_pos[:, :2], dim=1)
    aligned = lateral <= tol_xy
    around = (~holding) & (grip[:, 2] < target_pos[:, 2] + tol.freeze_xy_below)  # fingers around the object
    entered = holding & (lateral <= tol.funnel_xy) & (grip[:, 2] < zone_top - 0.002)  # on the goal peg
    around = around | entered
    aligned = aligned | around
    far = lateral > tol.funnel_xy
    rise = torch.where(holding, torch.full_like(hover, 0.005), torch.full_like(hover, tol.hover_rise))
    low = grip[:, 2] < hover - rise

    floor = torch.where(holding, torch.full_like(hover, zone_top), target_pos[:, 2])
    z_funnel = torch.maximum(floor + tol.funnel_z, torch.minimum(grip[:, 2], hover))
    z_inside = torch.where(aligned, target_pos[:, 2], z_funnel)

    target = torch.where(
        (far & low).unsqueeze(1),
        torch.stack([grip[:, 0], grip[:, 1], hover], 1),
        torch.where(
            far.unsqueeze(1),
            torch.stack([target_pos[:, 0], target_pos[:, 1], hover], 1),
            torch.stack([target_pos[:, 0], target_pos[:, 1], z_inside], 1),
        ),
    )
    target = torch.where(around.unsqueeze(1), torch.stack([grip[:, 0], grip[:, 1], target[:, 2]], 1), target)
    at_height = (grip[:, 2] - target_pos[:, 2]).abs() < tol.z
    close = torch.where(holding, ~(aligned & at_height), aligned & at_height)
    step = (target - grip).clamp(-max_step, max_step)
    return step, close


class NudgeState:
    """Per-environment stall detector and search index. / env 별 정체 감지·탐색 인덱스."""

    def __init__(
        self, num_envs: int, device, amplitude: float = 0.0007, stall_steps: int = 3, progress: float = 0.0005
    ):
        self.amplitude = amplitude
        self.stall_steps = stall_steps
        self.progress = progress
        self.stall = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_height = torch.zeros(num_envs, device=device)
        self.retry = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.table = torch.tensor(NUDGE_PATTERN, dtype=torch.float32, device=device) * amplitude

    def reset(self, env_ids) -> None:
        self.stall[env_ids] = 0
        self.index[env_ids] = 0
        self.retry[env_ids] = False

    def update(
        self, grip_height: torch.Tensor, inserting: torch.Tensor, holding: torch.Tensor
    ) -> torch.Tensor:
        """Advance and return the ``(N, 2)`` lateral offsets. / 갱신 후 옆 오프셋 반환.

        The search index persists while the object is held and resets once it is released. When
        the whole pattern has been tried, ``retry`` is raised: the caller lifts the object above
        the latch height so the correction can refresh, then descends again.
        """
        moved = (self.prev_height - grip_height) >= self.progress
        self.stall = torch.where(inserting & ~moved, self.stall + 1, torch.zeros_like(self.stall))
        advance = self.stall >= self.stall_steps
        exhausted = advance & (self.index == len(NUDGE_PATTERN) - 1)
        self.retry |= exhausted
        self.index = torch.where(advance, (self.index + 1) % len(NUDGE_PATTERN), self.index)
        self.stall[advance] = 0
        self.index = torch.where(holding, self.index, torch.zeros_like(self.index))
        self.retry &= holding
        self.prev_height = grip_height.clone()
        return self.table[self.index]


class FusionState:
    """Batched EMA with reset flags and a height latch. / 배치 EMA + 초기화 플래그 + 높이 latch."""

    def __init__(
        self, num_envs: int, dim: int, device, alpha: float, latch_below: float, hover_height: float
    ):
        self.alpha = alpha
        self.latch_below = latch_below
        self.hover_height = hover_height
        self.value = torch.zeros(num_envs, dim, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids) -> None:
        self.fresh[env_ids] = True

    def update(self, correction: torch.Tensor, grip_height: torch.Tensor) -> torch.Tensor:
        high = grip_height > self.hover_height - self.latch_below
        smoothed = self.alpha * self.value + (1.0 - self.alpha) * correction
        self.value = torch.where(
            self.fresh.unsqueeze(1), correction, torch.where(high.unsqueeze(1), smoothed, self.value)
        )
        self.fresh[:] = False
        return self.value


def apply_correction(
    points: torch.Tensor, correction: torch.Tensor, pos_scale: float, yaw_scale: float
) -> torch.Tensor:
    """Rigid 2-D correction about the frame origin. / 원점 기준 강체 2D 보정.

    ``correction`` is ``(N, 3)`` in normalised units; ``points`` is ``(N, 3)`` in the frame whose
    origin the correction rotates about.
    """
    c = correction[:, :2] * pos_scale
    th = correction[:, 2] * yaw_scale
    dz = correction[:, 3] * pos_scale if correction.shape[1] > 3 else torch.zeros_like(th)
    cs, sn = torch.cos(th), torch.sin(th)
    return torch.stack(
        [
            cs * points[:, 0] - sn * points[:, 1] + c[:, 0],
            sn * points[:, 0] + cs * points[:, 1] + c[:, 1],
            points[:, 2] + dz,
        ],
        1,
    )
