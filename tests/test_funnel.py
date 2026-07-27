"""Funnel planner: a kinematic loop must pick and place. / 깔때기 플래너 운동학 루프 검증."""

import numpy as np

from factor.planning.funnel import FunnelPlanner

HOVER = 0.1235
ZONE_TOP = 0.0655


def run(planner: FunnelPlanner, grip, ring, goal, attach_xy=0.006, max_steps=300):
    grip, ring = np.array(grip, float), np.array(ring, float)
    holding = False
    events = []
    for k in range(max_steps):
        cmd = planner.plan(grip, ring, goal, holding)
        grip = grip + cmd.step
        if cmd.close and not holding:
            if np.linalg.norm((grip - ring)[:2]) < attach_xy and abs(grip[2] - ring[2]) < 0.006:
                holding = True
                events.append(("grasp", k))
        if holding and not cmd.close:
            events.append(("release", k, np.round(grip - goal, 4).tolist()))
            return events, grip
        if holding:
            ring = grip.copy()
    return events, grip


def test_pick_and_place_converges():
    planner = FunnelPlanner(hover_height=HOVER, zone_top=ZONE_TOP, max_step=0.01)
    ring = [0.058, 0.0, 0.0455]
    goal = [-0.058, 0.0, 0.0125]
    events, grip = run(planner, [0.058, 0.0, HOVER], ring, goal)
    assert events[0][0] == "grasp"
    assert events[-1][0] == "release"
    assert np.linalg.norm(grip[:2] - np.array(goal[:2])) <= planner.tol.place_xy
    assert abs(grip[2] - goal[2]) < planner.tol.z


def test_rises_before_travelling_when_low():
    planner = FunnelPlanner(hover_height=HOVER, zone_top=ZONE_TOP, max_step=0.01)
    cmd = planner.plan(
        np.array([0.0, 0.0, 0.05]), np.array([0.058, 0.0, 0.0455]), np.array([-0.058, 0.0, 0.0125]), False
    )
    assert cmd.step[2] > 0 and abs(cmd.step[0]) < 1e-9


def test_holds_above_object_until_aligned():
    planner = FunnelPlanner(hover_height=HOVER, zone_top=ZONE_TOP, max_step=0.01)
    ring = np.array([0.0, 0.0, 0.0455])
    grip = np.array([0.005, 0.0, ring[2] + 0.02])
    cmd = planner.plan(grip, ring, np.zeros(3), False)
    assert cmd.target[2] >= ring[2] + planner.tol.funnel_z - 1e-9
    assert not cmd.close


def test_no_lateral_motion_once_fingers_are_around_the_object():
    from factor.planning.funnel import FunnelPlanner

    planner = FunnelPlanner(hover_height=0.12, zone_top=0.06, max_step=0.01)
    obj = np.array([0.0, 0.0, 0.05])
    # 4 mm off laterally but already below the object top: descend straight, do not push sideways
    cmd = planner.plan(np.array([0.004, 0.0, 0.054]), obj, obj, holding=False)
    assert np.allclose(cmd.step[:2], 0.0)
    assert cmd.step[2] < 0
    # above the object the same lateral error still gets corrected
    cmd = planner.plan(np.array([0.004, 0.0, 0.07]), obj, obj, holding=False)
    assert cmd.step[0] < 0
