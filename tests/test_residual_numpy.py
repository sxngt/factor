"""NumPy inference matches the torch policy. / NumPy 추론이 torch 정책과 일치."""

import numpy as np
import pytest

from factor.planning.funnel import FunnelPlanner
from factor.skills.pick_place import PickPlaceSkill, SkillObservation
from factor.skills.residual_np import FEATURE_DIM, NumpyResidualPolicy, skill_features


def test_numpy_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from factor.skills.residual import ResidualPolicy, export_numpy
    from factor.skills.residual import skill_features as torch_features

    torch.manual_seed(1)
    policy = ResidualPolicy()
    for p in policy.parameters():
        p.data.normal_(0, 0.5)
    export_numpy(policy, str(tmp_path / "skill.npz"))
    np_policy = NumpyResidualPolicy.from_file(str(tmp_path / "skill.npz"))

    grip = np.array([0.01, -0.02, 0.07])
    f_np = skill_features(
        grip, grip + 0.01, grip - 0.02, True, True, 3, 1, np.array([0.001, 0, 0]), 0.12, 0.06, 0.01
    )
    f_t = torch_features(
        torch.as_tensor(grip, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(grip + 0.01, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(grip - 0.02, dtype=torch.float32).unsqueeze(0),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([3]),
        torch.tensor([1]),
        torch.tensor([[0.001, 0.0, 0.0]]),
        0.12,
        0.06,
        0.01,
    )
    assert f_np.shape == (FEATURE_DIM,)
    assert np.allclose(f_np, f_t[0].numpy(), atol=1e-6)
    expected = policy.residual(policy.act(f_t))[0].numpy()
    assert np.allclose(np_policy(f_np), expected, atol=1e-6)


def test_skill_applies_residual_within_step_limit():
    planner = FunnelPlanner(hover_height=0.12, zone_top=0.06, max_step=0.01)
    skill = PickPlaceSkill(planner, np.zeros(3), residual=lambda f: np.array([0.05, 0.0, 0.0]))
    grip = np.array([0.0, 0.0, 0.12])
    cmd = skill.step(SkillObservation(grip, grip + [0.002, 0.0, -0.05], grip, False))
    assert cmd.step[0] == pytest.approx(0.01)  # 2 mm planner step + residual, saturated
