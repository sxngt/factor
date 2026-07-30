"""Correction fusion and the composed skill. / 보정 융합과 스킬 조합 검증."""

import numpy as np

from factor.perception.fusion import CorrectionFilter
from factor.planning.funnel import FunnelPlanner
from factor.planning.nudge import NudgeSearch
from factor.skills.pick_place import PickPlaceSkill, SkillObservation, apply_correction

HOVER = 0.1235
ZONE_TOP = 0.0655


def test_filter_latches_when_low():
    f = CorrectionFilter(alpha=0.9, latch_below=0.012, hover_height=HOVER)
    first = f.update(np.array([1.0, 0.0, 0.0]), HOVER)
    assert np.allclose(first, [1.0, 0.0, 0.0])
    low = f.update(np.array([-5.0, 0.0, 0.0]), HOVER - 0.05)
    assert np.allclose(low, first)
    high = f.update(np.array([0.0, 0.0, 0.0]), HOVER)
    assert np.allclose(high, [0.9, 0.0, 0.0])


def test_apply_correction_rotates_about_origin():
    origin = np.array([0.1, 0.2, 0.0])
    point = np.array([0.1 + 0.058, 0.2, 0.05])
    moved = apply_correction(point, np.array([0.0, 0.0, np.pi / 2]), origin)
    assert np.allclose(moved, [0.1, 0.2 + 0.058, 0.05])


def test_skill_with_true_correction_places_ring():
    """With the exact correction the skill must place a ring despite a 6 mm belief error."""
    planner = FunnelPlanner(hover_height=HOVER, zone_top=ZONE_TOP, max_step=0.01)
    origin = np.zeros(3)
    skill = PickPlaceSkill(planner, origin, CorrectionFilter(0.9, 0.012, HOVER), NudgeSearch())

    error = np.array([0.006, -0.004, 0.0])
    ring_true = np.array([0.058, 0.0, 0.0455])
    goal_true = np.array([-0.058, 0.0, 0.0125])
    ring_nominal, goal_nominal = ring_true - error, goal_true - error

    grip = np.array([0.058, 0.0, HOVER])
    holding = False
    released = False
    for _ in range(300):
        cmd = skill.step(SkillObservation(grip, ring_nominal, goal_nominal, holding, correction=error))
        grip = grip + cmd.step
        if (
            cmd.close
            and not holding
            and np.linalg.norm((grip - ring_true)[:2]) < 0.006
            and abs(grip[2] - ring_true[2]) < 0.006
        ):
            holding = True
        if holding and not cmd.close:
            released = True
            break
    assert released
    assert np.linalg.norm(grip[:2] - goal_true[:2]) <= planner.tol.place_xy + 1e-9
