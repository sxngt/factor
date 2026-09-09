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


def main() -> None:
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="configs/hanoi.yaml")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--estimator", default="", help="frozen estimator checkpoint (empty: noisy teacher)")
    ap.add_argument("--teacher_noise", type=float, default=0.15, help="normalised noise on the teacher")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--residual_scale", type=float, default=0.002)
    ap.add_argument("--progress_coef", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.001)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_steps", type=int, default=400)
    ap.add_argument("--anchor_err", type=float, default=0.008)
    ap.add_argument("--anchor_yaw_deg", type=float, default=1.0)
    ap.add_argument("--camera_rand", type=float, default=0.08)
    ap.add_argument(
        "--render", choices=("belief", "mask", "both", "none"), default="belief", help="render channels"
    )
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--view_width", type=float, default=0.14)
    ap.add_argument("--in_hand_offset", type=float, default=0.003)
    ap.add_argument("--offset_scale", type=float, default=0.02)
    ap.add_argument("--use_offset_estimate", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    from factor.scene import load_scene
    from factor.sim.contract import Randomisation
    from factor.sim.isaaclab.twin import HanoiTwin, make_env_cfg, settings_from_scene

    torch.manual_seed(args.seed)
    settings = settings_from_scene(
        load_scene(args.scene),
        args.assets,
        randomisation=Randomisation(
            anchor_xy=args.anchor_err, anchor_yaw=np.radians(args.anchor_yaw_deg), camera_xyz=args.camera_rand
        ),
        resolution=args.resolution,
        view_width=args.view_width,
        render_mask=(args.render in ("mask", "both")) and not args.no_mask,
        render_belief=(args.render in ("belief", "both")) and not args.no_mask,
        error_resample_steps=0,  # fixed per-episode belief error, as on the robot / 에피소드 고정 오차
        in_hand_offset=args.in_hand_offset,
        use_offset_estimate=args.use_offset_estimate,
    )
    env_cfg = make_env_cfg(settings, args.num_envs, args.device)
    env_cfg.offset_scale = args.offset_scale
    env = HanoiTwin(env_cfg)
    env.reset()
    perception = Perception(env, args.estimator, args.teacher_noise)
    policy = ResidualPolicy(ResidualConfig(residual_scale=args.residual_scale)).to(env.device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ppo = PPOConfig(horizon=args.horizon, lr=args.lr, entropy_coef=args.entropy_coef)
    rollout = Rollout(ppo.horizon, env.num_envs, FEATURE_DIM, env.device)

    baseline = evaluate(env, perception, None, args.eval_steps)
    print(f"[skill] baseline (rule skill) {baseline}  success {baseline['success']:.1%}", flush=True)
    best = baseline["success"]
    save_policy(policy, args.out, {"success": best, "baseline": baseline})

    env.resample_belief = False
    env.reset()
    prev_distance = env.target_distance()
    counts = {"succeeded": 0, "dropped": 0, "bad_release": 0, "timeout": 0}
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        policy.train()
        for _ in range(ppo.horizon):
            features = env.skill_observation()
            with torch.no_grad():
                dist, value = policy.distribution(features)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(1)
            env.residual = policy.residual(action)
            _, _, terminated, truncated, _ = env.step(perception.action())
            done = terminated | truncated
            reward = shaped_reward(env, terminated, truncated, prev_distance, args.progress_coef)
            prev_distance = env.target_distance()
            count_outcomes(counts, env, terminated, truncated)
            rollout.push(features, action, log_prob, value, reward, done)
        with torch.no_grad():
            _, last_value = policy.distribution(env.skill_observation())
        rollout.compute(last_value, ppo.gamma, ppo.lam)
        stats = ppo_update(policy, optimiser, rollout, ppo)
        if it % 5 == 0:
            total = max(sum(counts.values()), 1)
            print(
                f"[skill] it {it}  train success {counts['succeeded'] / total:.1%} over {total} eps  "
                f"loss pi {stats['policy']:.3f} v {stats['value']:.3f} ent {stats['entropy']:.2f} "
                f"clip {stats['clip_frac']:.2f}  std {policy.log_std.exp().mean():.3f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            counts = {k: 0 for k in counts}
        if it % args.eval_every == 0 or it == args.iterations:
            policy.eval()
            result = evaluate(env, perception, policy, args.eval_steps)
            print(f"[skill] it {it}  eval {result}  success {result['success']:.1%}", flush=True)
            if result["success"] >= best:
                best = result["success"]
                save_policy(policy, args.out, {"success": best, "baseline": baseline, "iteration": it})
                print(f"[skill] saved {args.out}  (best {best:.1%} vs baseline {baseline['success']:.1%})")
            env.resample_belief = False
            env.reset()
            prev_distance = env.target_distance()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
