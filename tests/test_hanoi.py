"""Hanoi planner and geometry. / 하노이 플래너·기하 검증."""

import numpy as np

from factor.tasks.hanoi import HanoiLayout, HanoiPlanner, Ring

LAYOUT = HanoiLayout(
    peg_spacing=0.058,
    peg_height=0.055,
    plate_thickness=0.007,
    rings=tuple(
        Ring(name, r, 0.007) for name, r in (("a", 0.025), ("b", 0.0225), ("c", 0.020), ("d", 0.0175))
    ),
)


def test_classic_four_rings_is_fifteen_moves():
    planner = HanoiPlanner(LAYOUT)
    moves = planner.plan({"left": ["a", "b", "c", "d"], "middle": [], "right": []}, "right")
    assert len(moves) == 15
    assert moves[0].item == "d"


def test_arbitrary_state_and_legality():
    planner = HanoiPlanner(LAYOUT)
    state = {"left": ["a"], "middle": ["b", "d"], "right": ["c"]}
    moves = planner.plan(state, "left")
    order = {r.name: i for i, r in enumerate(LAYOUT.rings)}
    stacks = {p: list(v) for p, v in state.items()}
    for m in moves:
        assert stacks[m.source][-1] == m.item
        assert not stacks[m.destination] or order[stacks[m.destination][-1]] < order[m.item]
        stacks[m.destination].append(stacks[m.source].pop())
    assert stacks["left"] == ["a", "b", "c", "d"]


def test_nominal_targets():
    planner = HanoiPlanner(LAYOUT)
    state = {"left": ["a", "b"], "middle": [], "right": []}
    move = planner.plan(state, "right")[0]
    obj, goal = planner.nominal_targets(state, move)
    assert np.allclose(obj, [-0.058, 0.0, 0.007 + 0.007 * 1.5])
    assert np.allclose(goal[:2], LAYOUT.peg_xy(move.destination))
    assert goal[2] == 0.007 + 0.0035 + 0.002
