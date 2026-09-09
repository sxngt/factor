"""Residual policy and PPO plumbing. / 잔차 정책·PPO 배관 검증."""

import pytest

torch = pytest.importorskip("torch")

from factor.skills.ppo import PPOConfig, Rollout, ppo_update  # noqa: E402
from factor.skills.residual import FEATURE_DIM, ResidualPolicy, skill_features  # noqa: E402


def _features(n=8):
    g = torch.zeros(n, 3)
    return skill_features(
        g,
        g + 0.01,
        g + 0.02,
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, 3),
        0.08,
        0.05,
        0.01,
    )


def test_feature_dim_and_zero_init():
    f = _features()
    assert f.shape == (8, FEATURE_DIM)
    policy = ResidualPolicy()
    assert torch.all(policy.act(f) == 0)  # untrained policy = rule-based skill
    assert torch.all(policy.residual(policy.act(f)) == 0)


def test_ppo_update_runs_and_changes_parameters():
    torch.manual_seed(0)
    policy = ResidualPolicy()
    cfg = PPOConfig(horizon=4, epochs=1, minibatches=2)
    rollout = Rollout(cfg.horizon, 8, FEATURE_DIM, "cpu")
    for t in range(cfg.horizon):
        f = _features()
        with torch.no_grad():
            dist, v = policy.distribution(f)
            a = dist.sample()
            lp = dist.log_prob(a).sum(1)
        rollout.push(f, a, lp, v, torch.ones(8) * t, torch.zeros(8, dtype=torch.bool))
    rollout.compute(torch.zeros(8), cfg.gamma, cfg.lam)
    before = policy.mean.weight.clone()
    stats = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-2), rollout, cfg)
    assert set(stats) == {"policy", "value", "entropy", "clip_frac"}
    assert not torch.allclose(before, policy.mean.weight)


def test_gae_masks_at_episode_end():
    rollout = Rollout(2, 1, FEATURE_DIM, "cpu")
    f = _features(1)
    for done in (True, False):
        rollout.push(
            f, torch.zeros(1, 3), torch.zeros(1), torch.zeros(1), torch.ones(1), torch.tensor([done])
        )
    rollout.compute(torch.full((1,), 100.0), 0.9, 1.0)
    # transition 0 ended an episode: no bootstrap from transition 1 or the last value
    assert float(rollout.returns[0, 0]) == pytest.approx(1.0)
    assert float(rollout.returns[1, 0]) == pytest.approx(1.0 + 0.9 * 100.0)
