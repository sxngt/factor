"""PPO fine-tuning of the residual skill in the twin. / 트윈에서 잔차 스킬 PPO 미세조정.

The perception stack is frozen: a trained estimator checkpoint (or, for ablation, the teacher
corrupted by Gaussian noise) supplies the pose correction exactly as on the robot. The residual
policy adds up to ``residual_scale`` metres to each planner step; its untrained output is zero,
so the first evaluation is the rule-based baseline.

Usage (Isaac Lab Python)::

    python -m factor.sim.isaaclab.train_skill --scene configs/hanoi.yaml --assets <repo root> \\
        --estimator estimator.pt --out skill.pt --headless

인식 층은 고정(학습된 추정기 체크포인트 또는 노이즈 섞은 교사)하고, 플래너 스텝에 더하는 잔차 정책만
PPO 로 학습한다. 초기 정책 출력은 0 이므로 첫 평가가 규칙 기준선이다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from factor.skills.ppo import PPOConfig, Rollout, ppo_update
from factor.skills.residual import FEATURE_DIM, ResidualConfig, ResidualPolicy, save_policy


class Perception:
    """Frozen correction source. / 고정된 보정 공급원."""

    def __init__(self, env, checkpoint: str, teacher_noise: float):
        self.env = env
        self.model = None
        self.noise = teacher_noise
        if checkpoint:
            from factor.perception.model import EstimatorConfig, StateEstimator

            ck = torch.load(checkpoint, map_location=env.device, weights_only=False)
            self.model = StateEstimator(EstimatorConfig(**ck["config"])).to(env.device)
            self.model.load_state_dict(ck["state_dict"], strict=False)
            self.model.eval()
            env.cfg.offset_scale = self.model.config.offset_scale  # the planner must use the model's scale

    @torch.no_grad()
    def action(self) -> torch.Tensor:
        env = self.env
        if self.model is None:
            teacher = env.teacher_action()
            teacher[:, :3] += torch.randn_like(teacher[:, :3]) * self.noise
            return teacher.clamp(-1.0, 1.0)
        out = self.model(env.last_frames["rgb"], env.last_frames["render"], env.last_frames["grip"])
        from factor.perception.train import action_from_output

        return action_from_output(out)


def shaped_reward(env, terminated, truncated, prev_distance, progress_coef: float) -> torch.Tensor:
    """Outcome reward plus potential-based progress. / 결과 보상 + 퍼텐셜 기반 진행 보상."""
    reason = env.last_term_reason
    r = reason["succeeded"].float() - (reason["dropped"] | reason["bad_release"]).float() - 0.005
    distance = env.target_distance()
    progress = (prev_distance - distance).clamp(-0.02, 0.02)
    progress = torch.where(terminated | truncated, torch.zeros_like(progress), progress)
    return r + progress_coef * progress


def count_outcomes(counts: dict, env, terminated, truncated) -> None:
    reason = env.last_term_reason
    seen = torch.zeros_like(terminated, dtype=torch.bool)
    for key in ("succeeded", "dropped", "bad_release"):
        m = reason[key] & ~seen
        counts[key] += int(m.sum())
        seen |= reason[key]
    counts["timeout"] += int((truncated & ~seen).sum())


def evaluate(env, perception: Perception, policy: ResidualPolicy | None, steps: int) -> dict:
    """Deterministic rollout; ``policy=None`` is the rule-based baseline. / 결정적 평가."""
    env.resample_belief = False
    env.reset()
    env.residual.zero_()
    counts = {"succeeded": 0, "dropped": 0, "bad_release": 0, "timeout": 0}
    for _ in range(steps):
        if policy is not None:
            env.residual = policy.residual(policy.act(env.skill_observation(), deterministic=True))
        _, _, terminated, truncated, _ = env.step(perception.action())
        count_outcomes(counts, env, terminated, truncated)
    env.residual.zero_()
    env.resample_belief = True
    env.reset()
    total = max(sum(counts.values()), 1)
    return {**counts, "success": counts["succeeded"] / total}


