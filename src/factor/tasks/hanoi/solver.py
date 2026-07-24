"""Shortest-move Tower of Hanoi planner for arbitrary legal states.

Breadth-first search over stack configurations. States are small (3 pegs, at most a handful of
rings), so the search is instantaneous and handles any starting arrangement, not only the
classic all-on-one-peg case.

임의의 합법 상태에서 출발하는 최단 수 하노이 플래너(너비 우선 탐색). 상태 공간이 작아 즉시 풀리며
고전적 초기 배치뿐 아니라 어떤 배치든 다룬다.
"""

from __future__ import annotations

from collections import deque

from factor.planning.task import Move
from factor.tasks.hanoi.spec import PEGS, HanoiLayout

Stacks = dict[str, list[str]]


def _key(stacks: Stacks) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(stacks[p]) for p in PEGS)


def solve_hanoi(stacks: Stacks, goal: str, order: dict[str, int]) -> list[Move]:
    """Return the shortest move sequence that gathers every ring on ``goal``.

    ``stacks`` lists rings bottom-to-top per peg; ``order[name]`` is the ring size rank
    (0 = largest). A move is legal only onto an empty peg or a larger ring.

    모든 링을 ``goal`` 기둥에 모으는 최단 수. ``order`` 는 크기 순위(0 = 가장 큼).
    """
    start = {p: list(stacks.get(p, [])) for p in PEGS}
    total = sum(len(v) for v in start.values())
    target = _key(
        {p: ([r for r in sorted(sum(start.values(), []), key=order.get)] if p == goal else []) for p in PEGS}
    )

    frontier = deque([start])
    came_from: dict[tuple, tuple[tuple, Move] | None] = {_key(start): None}

    while frontier:
        state = frontier.popleft()
        key = _key(state)
        if key == target:
            break
        for src in PEGS:
            if not state[src]:
                continue
            ring = state[src][-1]
            for dst in PEGS:
                if dst == src:
                    continue
                if state[dst] and order[state[dst][-1]] > order[ring]:
                    continue
                nxt = {p: list(state[p]) for p in PEGS}
                nxt[src].pop()
                nxt[dst].append(ring)
                nk = _key(nxt)
                if nk not in came_from:
                    came_from[nk] = (key, Move(item=ring, source=src, destination=dst))
                    frontier.append(nxt)
    else:
        raise ValueError("goal state unreachable")

    moves: list[Move] = []
    key = target
    while came_from[key] is not None:
        key, move = came_from[key]
        moves.append(move)
    moves.reverse()
    assert len(moves) <= 2**total - 1
    return moves


class HanoiPlanner:
    """:class:`factor.planning.task.TaskPlanner` for the tower. / 탑용 태스크 플래너."""

    def __init__(self, layout: HanoiLayout):
        self.layout = layout
        self._order = {r.name: i for i, r in enumerate(layout.rings)}

    def plan(self, state: Stacks, goal: str) -> list[Move]:
        return solve_hanoi(state, goal, self._order)

    def nominal_targets(self, state: Stacks, move: Move):
        """Nominal grip targets for ``move`` in the anchor frame. / 이동의 공칭 목표(앵커 프레임).

        Returns ``(object_center, goal_center)``: the ring to pick sits on top of the source
        stack; it will land on top of the destination stack.
        """
        level_src = len(state[move.source]) - 1
        level_dst = len(state[move.destination])
        return self.layout.ring_center(move.source, level_src), self.layout.goal_center(
            move.destination, level_dst
        )
