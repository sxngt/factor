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


